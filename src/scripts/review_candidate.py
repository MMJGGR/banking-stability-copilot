"""Create an auditable review package; never promote or modify serving data."""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.classifier_integrity import load_validated_classifier, sha256_file, verify_classifier_file
from src.sources.sdmx_normalize import parse_period_label

SOURCES = ("FSIC", "MFS", "FSIBSIS", "WEO", "WGI")
KEYS = ["indicator", "frequency", "unit", "period"]


def read_manifest(root):
    return json.loads((root / "artifacts/data_manifest.json").read_text())


def verify_manifest(root, manifest):
    results = []
    for relative, entry in manifest.get("artifacts", {}).items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise RuntimeError(f"Unsafe manifest path: {relative}")
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"Manifest mismatch: {relative}")
        results.append(relative)
    return results


def observations(frame, name, cutoff):
    """Canonical non-null observations, cutoff-eligible, with explicit identities."""
    if name in {"FSIC", "MFS", "WEO"}:
        d = frame.copy()
        if name == "WEO" and "observation_status" in d:
            d = d[~d["observation_status"].isin(["estimate", "projection"])]
        d = d.rename(columns={"indicator_code": "indicator"})
        for field in ["frequency", "unit"]:
            if field not in d:
                d[field] = ""
        d = d[KEYS + ["value"]].copy()
        d["period"] = pd.to_datetime(d["period"], errors="coerce")
    elif name == "FSIBSIS":
        period_cols = [c for c in frame if pd.notna(parse_period_label(c))]
        d = frame.melt(id_vars=["INDICATOR"], value_vars=period_cols, var_name="period_label", value_name="value")
        d = d.rename(columns={"INDICATOR": "indicator"})
        # Original period label distinguishes annual/quarterly/monthly series.
        d["frequency"] = d["period_label"].astype(str).str.replace(r"[0-9]", "", regex=True)
        d["unit"] = ""
        mapping = {c: parse_period_label(c) for c in period_cols}
        d["period"] = pd.to_datetime(d["period_label"].map(mapping), errors="coerce")
    else:
        values = [c for c in frame if c not in {"country_code", "year"}]
        d = frame.melt(id_vars=["year"], value_vars=values, var_name="indicator", value_name="value")
        d["period"] = pd.to_datetime(d["year"].astype("Int64").astype(str) + "-12-31", errors="coerce")
        d["frequency"] = "A"
        d["unit"] = ""
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d[d["period"].le(cutoff) & d["value"].notna()].copy()
    for field in KEYS[:-1]:
        d[field] = d[field].fillna("").astype(str)
    return d[KEYS + ["value"]].drop_duplicates()


def source_diffs(baseline, candidate, output, baseline_cutoff, cutoff):
    rows = []
    detail_path = output / "source_value_changes.csv"
    detail_header = True
    for name in SOURCES:
        before = pd.read_parquet(baseline / f"cache/{name}_cache.parquet")
        after = pd.read_parquet(candidate / f"cache/{name}_cache.parquet")
        if "country_code" not in before or "country_code" not in after:
            raise RuntimeError(f"Unsupported {name} country schema")
        before_groups = before.groupby("country_code", sort=False).indices
        after_groups = after.groupby("country_code", sort=False).indices
        countries = sorted(set(before_groups) | set(after_groups))
        print(f"Comparing {name}: {len(countries)} countries", flush=True)
        for country in countries:
            a = observations(before.iloc[before_groups.get(country, [])], name, baseline_cutoff)
            b = observations(after.iloc[after_groups.get(country, [])], name, cutoff)
            latest_a = a["period"].max()
            latest_b = b["period"].max()
            # Never call changes in a non-unique identity a precise revision.
            ambiguous = pd.concat([a[a.duplicated(KEYS, keep=False)][KEYS], b[b.duplicated(KEYS, keep=False)][KEYS]]).drop_duplicates()
            ambiguous_count = len(ambiguous)
            if ambiguous_count:
                for label, frame in [("a", a), ("b", b)]:
                    cleaned = frame.merge(ambiguous.assign(ambiguous=True), on=KEYS, how="left")
                    cleaned = cleaned[cleaned["ambiguous"].isna()].drop(columns="ambiguous")
                    if label == "a":
                        a = cleaned
                    else:
                        b = cleaned
            merged = a.merge(b, on=KEYS, how="outer", suffixes=("_baseline", "_candidate"), indicator=True, validate="one_to_one")
            revised = merged["_merge"].eq("both") & ~np.isclose(merged["value_baseline"], merged["value_candidate"], rtol=1e-10, atol=1e-12, equal_nan=True)
            added = merged["_merge"].eq("right_only")
            removed = merged["_merge"].eq("left_only")
            row = {
                "source": name, "country_code": country,
                "baseline_latest_observation": "" if pd.isna(latest_a) else latest_a.date().isoformat(),
                "candidate_latest_observation": "" if pd.isna(latest_b) else latest_b.date().isoformat(),
                "baseline_comparable_observations": len(a), "candidate_comparable_observations": len(b),
                "new_period_observations": int((added & merged["period"].gt(baseline_cutoff)).sum()),
                "historical_backfills": int((added & merged["period"].le(baseline_cutoff)).sum()),
                "revised_values": int(revised.sum()), "removed_observations": int(removed.sum()),
                "ambiguous_identity_groups_excluded": ambiguous_count,
            }
            rows.append(row)
            changes = merged[added | removed | revised].copy()
            if not changes.empty:
                changes["change_type"] = np.select([added.loc[changes.index], removed.loc[changes.index]], ["added", "removed"], default="revised")
                changes.insert(0, "country_code", country)
                changes.insert(0, "source", name)
                changes.drop(columns="_merge").to_csv(detail_path, mode="w" if detail_header else "a", header=detail_header, index=False)
                detail_header = False
        del before, after
        gc.collect()
    table = pd.DataFrame(rows)
    table.to_csv(output / "country_source_freshness_and_changes.csv", index=False)
    if detail_header:
        pd.DataFrame(columns=["source", "country_code"] + KEYS + ["value_baseline", "value_candidate", "change_type"]).to_csv(detail_path, index=False)
    totals = table.groupby("source")[["new_period_observations", "historical_backfills", "revised_values", "removed_observations", "ambiguous_identity_groups_excluded"]].sum()
    return table, totals.reset_index().to_dict(orient="records")


def load_model(root):
    with (root / "cache/risk_model.pkl").open("rb") as stream:
        return pickle.load(stream)


def feature_diffs(a, b):
    a = a.set_index("country_code")
    b = b.set_index("country_code")
    countries = a.index.union(b.index)
    rows = []
    for field in a.columns.union(b.columns):
        if field in {"country_name"} or field.endswith("_period"):
            continue
        av = pd.to_numeric(a[field], errors="coerce").reindex(countries) if field in a else pd.Series(np.nan, index=countries)
        bv = pd.to_numeric(b[field], errors="coerce").reindex(countries) if field in b else pd.Series(np.nan, index=countries)
        changed = ~np.isclose(av, bv, rtol=1e-10, atol=1e-12, equal_nan=True)
        for country in countries[changed]:
            rows.append({"country_code": country, "feature": field, "baseline": av.loc[country], "candidate": bv.loc[country]})
    return pd.DataFrame(rows, columns=["country_code", "feature", "baseline", "candidate"])


def frozen_scores(features, pipeline, classifier):
    features = features.copy()
    X = features.copy()
    for field in classifier.feature_names_:
        if field not in X:
            X[field] = X["credit_to_gdp_relative"] if field == "credit_to_gdp_gap" and "credit_to_gdp_relative" in X else np.nan
        if X[field].isna().all():
            fill_values = getattr(classifier, "feature_fill_values_", None)
            fill = fill_values.get(field, np.nan) if fill_values is not None else np.nan
            X[field] = fill if pd.notna(fill) else 0.0
    probabilities = pd.Series(classifier.predict_proba(X[classifier.feature_names_]), index=features["country_code"], name="probability")
    names = pd.Series(dtype=object)
    pillars = pipeline.transform(features, names).set_index("country_code")
    probability = probabilities.reindex(pillars.index)
    return (pillars["risk_score"] + (0.1 * (1 + 9 * probability - pillars["risk_score"])).clip(lower=0)).clip(1, 10)


def main():
    parser = argparse.ArgumentParser()
    for name in ["baseline", "original", "candidate", "output"]:
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    baseline, original, candidate, output = [Path(getattr(args, name)).resolve() for name in ["baseline", "original", "candidate", "output"]]
    output.mkdir(parents=True, exist_ok=True)
    bm, om, cm = [read_manifest(root) for root in [baseline, original, candidate]]
    checked = {label: verify_manifest(root, manifest) for label, root, manifest in [("baseline", baseline, bm), ("original", original, om), ("candidate", candidate, cm)]}
    expected_classifier = verify_classifier_file()
    if sha256_file(baseline / "cache/crisis_classifier.pkl") != expected_classifier["sha256"]:
        raise RuntimeError("Baseline classifier does not match release lock")
    if cm.get("classifier_preservation", {}).get("mode") != "preserved":
        raise RuntimeError("Candidate is not classifier-preserving")
    if cm.get("validation", {}).get("model_checks_failed") != 0 or cm.get("snapshot_status") != "verified":
        raise RuntimeError("Candidate validation did not pass")
    identical_sources = {name: sha256_file(original / f"cache/{name}_cache.parquet") == sha256_file(candidate / f"cache/{name}_cache.parquet") for name in SOURCES}
    if not all(identical_sources.values()):
        raise RuntimeError("Controlled rebuild changed a source cache")
    old, prior, new = [load_model(root) for root in [baseline, original, candidate]]
    before = old["country_scores"].set_index("country_code")
    after = new["country_scores"].set_index("country_code")
    previous = prior["country_scores"].set_index("country_code")
    universe = before.index.union(after.index)
    scores = pd.DataFrame(index=universe)
    for field in ["risk_score", "risk_category", "crisis_prob", "economic_risk", "industry_risk", "data_confidence"]:
        if field in before:
            scores[f"baseline_{field}"] = before[field]
        if field in after:
            scores[f"candidate_{field}"] = after[field]
    scores["score_change"] = scores["candidate_risk_score"] - scores["baseline_risk_score"]
    scores["original_uncontrolled_score"] = previous["risk_score"]
    scores["correction_vs_uncontrolled_candidate"] = scores["candidate_risk_score"] - scores["original_uncontrolled_score"]
    scores["category_changed"] = scores["baseline_risk_category"] != scores["candidate_risk_category"]
    with (baseline / "cache/inference_pipeline.pkl").open("rb") as stream:
        pipeline = pickle.load(stream)["pillar_pipeline"]
    classifier = load_validated_classifier()
    frozen_old = frozen_scores(old["feature_values"], pipeline, classifier)
    frozen_new = frozen_scores(new["feature_values"], pipeline, classifier)
    scores["frozen_pipeline_baseline_replay"] = frozen_old
    scores["frozen_pipeline_candidate_score"] = frozen_new
    scores["input_effect_fixed_model"] = frozen_new - frozen_old
    scores["baseline_reproduction_residual"] = frozen_old - scores["baseline_risk_score"]
    scores["pillar_refit_effect"] = scores["candidate_risk_score"] - frozen_new
    decomposition = scores["input_effect_fixed_model"] + scores["baseline_reproduction_residual"] + scores["pillar_refit_effect"]
    if not np.allclose(decomposition.dropna(), scores.loc[decomposition.notna(), "score_change"], atol=1e-10):
        raise RuntimeError("Score movement decomposition failed to reconcile")
    scores.index.name = "country_code"
    scores.sort_values("score_change", key=lambda s: s.abs(), ascending=False).to_csv(output / "country_score_review.csv")
    changes = feature_diffs(old["feature_values"], new["feature_values"])
    changes.to_csv(output / "engineered_feature_changes.csv", index=False)
    controlled_changes = feature_diffs(prior["feature_values"], new["feature_values"])
    controlled_changes.to_csv(output / "corrected_vs_original_feature_changes.csv", index=False)
    non_classifier_changes = controlled_changes[controlled_changes["feature"] != "crisis_prob"]
    feature_coverage = []
    for field in sorted(set(old["feature_values"].columns) | set(new["feature_values"].columns)):
        if field == "country_code" or field.endswith("_year") or field.endswith("_period"):
            continue
        a = old["feature_values"][field].notna().sum() if field in old["feature_values"] else 0
        b = new["feature_values"][field].notna().sum() if field in new["feature_values"] else 0
        feature_coverage.append({"feature": field, "baseline_non_null": int(a), "candidate_non_null": int(b), "change": int(b-a)})
    pd.DataFrame(feature_coverage).to_csv(output / "feature_coverage.csv", index=False)
    source_table, source_totals = source_diffs(baseline, candidate, output, pd.Timestamp(bm["as_of_date"]), pd.Timestamp(cm["as_of_date"]))
    summary = {
        "status": "ready_for_human_review_not_promoted", "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_cutoff": bm["as_of_date"], "candidate_cutoff": cm["as_of_date"],
        "original_run_id": 34691442181, "original_artifact_id": 10297154977,
        "classifier": expected_classifier, "source_cache_identity": identical_sources,
        "manifest_checksums_verified": {label: len(paths) for label, paths in checked.items()},
        "countries_baseline": len(before), "countries_candidate": len(after),
        "countries_added": sorted(set(after.index)-set(before.index)), "countries_removed": sorted(set(before.index)-set(after.index)),
        "category_changes": int(scores["category_changed"].sum()),
        "max_absolute_score_change": float(scores["score_change"].abs().max()),
        "max_baseline_reproduction_residual": float(scores["baseline_reproduction_residual"].abs().max()),
        "non_classifier_feature_changes_vs_original": len(non_classifier_changes),
        "source_change_totals": source_totals,
        "model_validation": cm["validation"],
        "limitations": [
            "The data vintage remains 12 September 2026. This is a controlled rebuild, not a new full source download.",
            "Dataset and country latest dates do not mean every underlying indicator is equally current.",
            "Source comparisons exclude WEO estimates/projections and compare cutoff-eligible non-null observations; counts differ from raw normalized row totals.",
            "Conflicting duplicate observation identities are excluded from precise revision attribution and counted separately.",
            "Pillar transforms are refitted in the normal build. The frozen-pipeline decomposition reports input, refit and baseline-replay effects separately.",
            "Preserving the served legacy classifier does not renew its validation or authorize production promotion.",
            "Supplementary BOP/IIP, WDI/IDS and BIS source pipelines were not refreshed by this rebuild.",
        ],
    }
    (output / "review_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    cm_copy = output / "candidate_manifest.json"
    cm_copy.write_text(json.dumps(cm, indent=2, sort_keys=True))
    lines = ["# Banking Copilot — Classifier-Preserving Data Review", "", "**Ready for human review; not promoted.**", "", f"Serving baseline: {bm['as_of_date']}. Candidate source vintage: {cm['as_of_date']}.", f"Classifier SHA-256: `{expected_classifier['sha256']}` (unchanged).", f"All five normalized source caches are byte-identical to the original September candidate; {len(checked['candidate'])} candidate manifest checksums passed.", "", "## Score review", "", "| Country | Serving score | Corrected candidate | Change | Fixed-model input effect | Pillar-refit effect |", "|---|---:|---:|---:|---:|---:|"]
    focus = [c for c in ["KEN", "COD", "CMR", "BFA", "ETH", "UGA", "TZA", "NGA", "ZAF"] if c in scores.index]
    for country in focus:
        r = scores.loc[country]
        lines.append(f"| {country} | {r['baseline_risk_score']:.4f} | {r['candidate_risk_score']:.4f} | {r['score_change']:+.4f} | {r['input_effect_fixed_model']:+.4f} | {r['pillar_refit_effect']:+.4f} |")
    lines.extend(["", f"Countries scored: {len(before)} → {len(after)}. Category changes: {summary['category_changes']}. Maximum absolute score movement: {summary['max_absolute_score_change']:.4f}.", f"Baseline replay residual (maximum absolute): {summary['max_baseline_reproduction_residual']:.8f}. Non-classifier feature changes versus the original September run: {len(non_classifier_changes)}.", "", "## Review files", "", "`country_score_review.csv` contains all countries, category changes and the full reconciled score decomposition. `country_source_freshness_and_changes.csv` contains source dates and additions, backfills, revisions and removals by country. `source_value_changes.csv` lists the individual changes. Feature-level inputs and coverage are in the accompanying feature CSVs.", "", "## Review qualifications", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    (output / "REVIEW.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
