"""Build owner-review evidence without approving or deploying a candidate."""

import argparse
import gc
import json
import os
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.classifier_guard import sha256_file, verify_classifier

SOURCES = ("FSIC", "MFS", "FSIBSIS", "WEO", "WGI")


def verify_inventory(root, manifest):
    for name, record in manifest["artifacts"].items():
        path = Path(root) / name
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Missing or wrong-sized artifact: {name}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Artifact checksum mismatch: {name}")


def country_coverage(root, source, cutoff):
    """Summarize populated observations by country, not dataset-wide maxima."""
    frame = pd.read_parquet(Path(root) / "cache" / f"{source}_cache.parquet")
    cutoff = pd.Timestamp(cutoff)
    if source in {"FSIC", "MFS", "WEO"}:
        frame["period"] = pd.to_datetime(frame["period"], errors="coerce")
        frame = frame.loc[frame["period"].le(cutoff) & frame["value"].notna()]
        result = frame.groupby("country_code").agg(
            observations=("period", "size"), indicators=("indicator_code", "nunique"),
            latest_observation=("period", "max"))
    elif source == "FSIBSIS":
        from src.data_loader import is_time_period_column, parse_period_label
        periods = sorted((c for c in frame if is_time_period_column(c)
                          and parse_period_label(c) <= cutoff), key=parse_period_label)
        frame["_count"] = 0
        frame["_latest"] = pd.NaT
        for column in periods:
            populated = pd.to_numeric(frame[column], errors="coerce").notna()
            frame.loc[populated, "_count"] += 1
            frame.loc[populated, "_latest"] = parse_period_label(column)
        frame = frame.loc[frame["_count"].gt(0)]
        result = frame.groupby("country_code").agg(
            observations=("_count", "sum"), indicators=("INDICATOR", "nunique"),
            latest_observation=("_latest", "max"))
    else:
        frame = frame.loc[pd.to_numeric(frame["year"], errors="coerce").le(cutoff.year)].copy()
        cols = [c for c in frame if c not in {"country_code", "year"}
                and pd.to_numeric(frame[c], errors="coerce").notna().any()]
        numeric = frame[cols].apply(pd.to_numeric, errors="coerce")
        frame["_count"] = numeric.notna().sum(axis=1)
        frame["_indicators"] = frame["_count"]
        frame["_latest"] = pd.to_datetime(frame["year"].astype(int).astype(str) + "-12-31")
        frame = frame.loc[frame["_count"].gt(0)]
        result = frame.groupby("country_code").agg(
            observations=("_count", "sum"), indicators=("_indicators", "max"),
            latest_observation=("_latest", "max"))
    result = result.reset_index()
    result["latest_observation"] = result["latest_observation"].dt.strftime("%Y-%m-%d")
    del frame
    gc.collect()
    return result


def compare_frames(before, after):
    """Compare complete country-level feature/score matrices, including missingness."""
    frames = []
    for frame, label in ((before, "baseline"), (after, "candidate")):
        if "country_code" not in frame or frame["country_code"].isna().any():
            raise RuntimeError("Country-level matrix has a missing country key")
        if frame["country_code"].duplicated().any():
            raise RuntimeError("Country-level matrix has duplicate country keys")
        frame = frame.drop(columns=["country_name"], errors="ignore")
        frames.append(frame.melt(id_vars="country_code", var_name="field", value_name=label))
    merged = frames[0].merge(frames[1], on=["country_code", "field"], how="outer",
                            indicator=True, validate="one_to_one")
    left, right = merged["baseline"], merged["candidate"]
    same = (left.isna() & right.isna()) | left.astype(str).eq(right.astype(str))
    a, b = pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")
    numeric = a.notna() & b.notna()
    same = same | (numeric & np.isclose(a, b, rtol=1e-10, atol=1e-12))
    merged["numeric_change"] = b - a
    return merged.loc[~same | merged["_merge"].ne("both")].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", default=".")
    parser.add_argument("--output", default="artifacts/review")
    args = parser.parse_args()
    baseline, candidate, output = map(Path, (args.baseline, args.candidate, args.output))
    output.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads((root / "artifacts/data_manifest.json").read_text())
                 for root in (baseline, candidate)]
    old, new = manifests
    for root, manifest in zip((baseline, candidate), manifests):
        verify_inventory(root, manifest)
    selected_manifest = baseline / "artifacts/data_manifest.json"
    classifier = verify_classifier(candidate, selected_manifest, load=True)
    if new.get("validation", {}).get("model_checks_failed") != 0:
        raise RuntimeError("Candidate model validation did not pass")
    if not new.get("model", {}).get("cutoff_verified"):
        raise RuntimeError("Candidate model cutoff is unverified")
    if new.get("snapshot_status") != "verified" or set(new.get("sources", {})) != set(SOURCES):
        raise RuntimeError("Candidate source/snapshot verification is incomplete")
    if pd.Timestamp(new["as_of_date"]) < pd.Timestamp(old["as_of_date"]):
        raise RuntimeError("Candidate cutoff precedes the selected baseline")

    coverage = []
    for source in SOURCES:
        a = country_coverage(baseline, source, old["as_of_date"])
        b = country_coverage(candidate, source, new["as_of_date"])
        delta = a.merge(b, on="country_code", how="outer", suffixes=("_baseline", "_candidate"),
                        indicator="country_presence", validate="one_to_one")
        delta.insert(0, "source", source)
        coverage.append(delta)
    pd.concat(coverage, ignore_index=True).to_csv(output / "country_source_coverage.csv", index=False)

    features = [pd.read_parquet(root / "cache/crisis_features.parquet") for root in (baseline, candidate)]
    feature_changes = compare_frames(*features)
    feature_changes.to_csv(output / "model_input_changes.csv", index=False)
    # These user-owned pickle bytes have already passed manifest checks above.
    models = []
    for root in (baseline, candidate):
        with (root / "cache/risk_model.pkl").open("rb") as stream:
            models.append(pickle.load(stream))
    scores = [model["country_scores"] for model in models]
    score_changes = compare_frames(*scores)
    score_changes.to_csv(output / "country_score_changes.csv", index=False)
    for label, score in zip(("baseline", "candidate"), scores):
        score.to_csv(output / f"{label}_country_scores.csv", index=False)
    old_countries, new_countries = [set(score["country_code"].astype(str)) for score in scores]
    dropped = sorted(old_countries - new_countries)
    if dropped:
        raise RuntimeError(f"Candidate loses scored countries: {dropped}")
    if new["model"]["countries_trained"] != len(scores[1]):
        raise RuntimeError("Candidate scoring universe differs from manifest")

    caveats = [
        "Classifier bytes preserved; PCA pillars and imputation are refitted by the existing refresh pipeline.",
        "Score changes reflect refreshed inputs and that normal refit; this is not a frozen-whole-model attribution study.",
        "Model-input changes include changed feature values and source-date fields. A full raw historical-observation revision diff is not performed.",
        "Country-source coverage counts populated eligible observations; these are not the manifest's normalized row counts.",
        "Technical review readiness is not owner approval, model-release approval, promotion, or deployment.",
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "baseline_commit": os.environ.get("BASELINE_COMMIT"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "baseline_cutoff": old["as_of_date"], "candidate_cutoff": new["as_of_date"],
        "status": "comparison_complete_awaiting_owner_review", "approved": False,
        "classifier": classifier, "candidate_artifacts_verified": len(new["artifacts"]),
        "baseline_scored_countries": len(old_countries), "candidate_scored_countries": len(new_countries),
        "added_scored_countries": sorted(new_countries - old_countries), "dropped_scored_countries": dropped,
        "changed_model_input_cells": len(feature_changes), "changed_score_cells": len(score_changes),
        "source_comparison": {name: {"baseline": old["sources"][name], "candidate": new["sources"][name]} for name in SOURCES},
        "limitations": caveats,
        "review_files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(output.glob("*.csv"))},
    }
    (output / "review.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ["# Classifier-preserving data candidate", "", "Status: AWAITING OWNER REVIEW — NOT APPROVED OR DEPLOYED.", "",
             f"Baseline: {old['as_of_date']}; candidate: {new['as_of_date']}",
             f"Classifier preserved: {classifier['sha256']}",
             f"Scored countries: {len(old_countries)} -> {len(new_countries)}", "",
             "## Review files", "", "country_source_coverage.csv: country-specific coverage and observation dates.",
             "model_input_changes.csv: changed model inputs, dates and missingness.",
             "country_score_changes.csv: changed scores and risk classifications.",
             "baseline_country_scores.csv / candidate_country_scores.csv: full score tables.", "", "## Interpretation", ""]
    lines.extend(caveats)
    (output / "REVIEW.md").write_text("\n\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
