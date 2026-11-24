"""Read-only broad-source inventory. Never trains or promotes any model.

Usage:
    python -m src.scripts.audit_forecasting_sources --snapshot . \
        --output research-output/forecasting-inventory

The output directory must be new and outside protected source/serving paths.
Raw parquet data only is read: no pickle, online download, or source rewrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import pandas as pd

from src.forecasting.inventory import CORE_SOURCES, ForecastDataError, inventory_source


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def output_destination(snapshot: Path, output: Path) -> Path:
    snapshot, output = snapshot.resolve(), output.resolve()
    if output.exists():
        raise ForecastDataError("Output directory already exists; use a new run directory")
    if output == snapshot or snapshot.is_relative_to(output):
        raise ForecastDataError("Output cannot contain or replace the source snapshot")
    repo = Path(__file__).resolve().parents[2]
    for root in {snapshot, repo}:
        for folder in ("cache", "artifacts", "data", "src", "tests", ".git", ".github"):
            protected = root / folder
            if output == protected or output.is_relative_to(protected):
                raise ForecastDataError(f"Protected source/serving output path: {output}")
    return output


def run(snapshot: Path, output: Path) -> dict:
    snapshot = snapshot.resolve()
    output = output_destination(snapshot, output)
    manifest_path = snapshot / "artifacts/data_manifest.json"
    manifest_digest = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    paths = {s: snapshot / "cache" / f"{s}_cache.parquet" for s in CORE_SOURCES}
    hashes = {}
    # Verify all inputs before creating outputs; missing sources fail closed.
    for source, path in paths.items():
        if not path.resolve().is_relative_to(snapshot):
            raise ForecastDataError("Source path escapes snapshot")
        expected = manifest["artifacts"][str(path.relative_to(snapshot))]
        hashes[source] = sha256(path)
        if path.stat().st_size != expected["bytes"] or hashes[source] != expected["sha256"]:
            raise ForecastDataError(f"Source checksum mismatch: {source}")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "source_inventory_not_model_validation",
        "source_cutoff": manifest["as_of_date"],
        "manifest_sha256": manifest_digest, "source_sha256": hashes,
        "feature_limit": None, "sources": {},
        "python_version": platform.python_version(), "pandas_version": pd.__version__,
        "limitations": [
            "Counts include separate frequency/unit/dimension variants, not independent predictors or crisis episodes.",
            "Unambiguous retrospective cells are not certified point-in-time observations.",
            "Missing units, original source codes or collapsed upstream dimensions require metadata recovery.",
            "No targets are evaluated, no final holdout is consumed, and no serving artifacts are modified.",
        ],
    }
    try:
        report["code_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        report["code_commit"] = "unversioned_local_copy"
    try:
        for source, path in paths.items():
            frame = pd.read_parquet(path)
            result = inventory_source(frame, source, manifest["as_of_date"])
            report["sources"][source] = result.summary
            for name, table in (("registry", result.registry), ("conflicts", result.conflicts),
                                ("country-coverage", result.country_coverage), ("year-coverage", result.year_coverage)):
                table.to_csv(output / f"{source}-{name}.csv", index=False)
            print(source, json.dumps(result.summary, sort_keys=True), flush=True)
            del frame, result
        if sha256(manifest_path) != manifest_digest or any(sha256(paths[s]) != h for s, h in hashes.items()):
            raise ForecastDataError("Source/manifest changed during research audit")
        report["inputs_unchanged"] = True
        report["candidate_series"] = sum(x["features"] for x in report["sources"].values())
        (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        lines = ["# Broad-feature source inventory", "", "Status: research inventory, not forecast validation.",
                 f"Source cutoff: {manifest['as_of_date']}. No predictor-count cap was applied.", "",
                 "| Source | Candidate series | Countries | Conflict cells | Missing-unit series |",
                 "|---|---:|---:|---:|---:|"]
        for source, s in report["sources"].items():
            lines.append(f"| {source} | {s['features']:,} | {s['countries']} | {s['conflicting_observation_cells']:,} | {s['features_with_missing_units']:,} |")
        lines += ["", "## Reading the evidence", "",
                  "Registry files retain every observed source identity; admission is not yet assessed. Conflicts are recorded per country, feature and period rather than arbitrarily resolved. Separate coverage files describe distinct observed years without imputing missing outcomes.",
                  "", "Publication and vintage fields are counted separately. A historical period does not establish historical availability. The next data milestone must recover missing metadata and freeze the retrospective versus verified-vintage research policy.",
                  "", "All five source-file checksums and the source manifest were unchanged after the run. No model was trained, evaluated or promoted.", "", "## Limitations", ""]
        lines += ["- " + item for item in report["limitations"]]
        (output / "README.md").write_text("\n".join(lines) + "\n")
    except Exception as error:
        (output / "FAILED.json").write_text(json.dumps({"status": "failed_incomplete_inventory", "error": str(error)}, indent=2))
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.snapshot, args.output)


if __name__ == "__main__":
    main()
