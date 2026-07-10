"""Train and archive a dated model snapshot from local cached source data.

This is the offline counterpart to refresh_data.py. It does not fetch source
files; it uses the currently cached IMF/WGI data, applies the requested cutoff,
writes active serving artifacts, and archives a copy under artifacts/snapshots.
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR, CACHE_DIR
from src.data_loader import FSIBSISLoader, IMFDataLoader, WGILoader
from src.snapshot_manifest import build_snapshot_manifest, write_snapshot_manifest
from src.scripts.audit_model_policy import build_policy_audit
from train_model import BankingRiskModel, validate_model


ACTIVE_ARTIFACTS = [
    Path(CACHE_DIR) / "risk_model.pkl",
    Path(CACHE_DIR) / "inference_pipeline.pkl",
    Path(CACHE_DIR) / "crisis_classifier.pkl",
    Path(CACHE_DIR) / "crisis_features.parquet",
    Path(CACHE_DIR) / "imputed_features.parquet",
    Path(BASE_DIR) / "artifacts" / "model_policy_audit.json",
    Path(BASE_DIR) / "artifacts" / "data_manifest.json",
]


def _load_cached_sources():
    loader = IMFDataLoader()
    if not loader.load_from_cache():
        raise FileNotFoundError(
            "No FSIC/WEO/MFS parquet caches found. Run refresh_data.py or "
            "load source CSVs before building a local snapshot."
        )

    # Prime auxiliary caches so cutoff-aware feature extraction is deterministic.
    FSIBSISLoader().load()
    WGILoader().load()

    return (
        loader._data_cache.get("FSIC", pd.DataFrame()),
        loader._data_cache.get("WEO", pd.DataFrame()),
        loader._data_cache.get("MFS", pd.DataFrame()),
    )


def _write_policy_audit(model):
    audit = build_policy_audit(model.feature_values)
    output = Path(BASE_DIR) / "artifacts" / "model_policy_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return audit


def _archive_snapshot(as_of_date: str, manifest: dict, audit: dict) -> Path:
    snapshot_dir = Path(BASE_DIR) / "artifacts" / "snapshots" / as_of_date
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for source in ACTIVE_ARTIFACTS:
        if source.exists():
            shutil.copy2(source, snapshot_dir / source.name)

    snapshot_manifest = dict(manifest)
    snapshot_manifest["policy_audit"] = {
        "coverage_score_correlation": audit["baseline"][
            "coverage_score_correlation"
        ],
        "absolute_coverage_score_correlation": audit["baseline"][
            "absolute_coverage_score_correlation"
        ],
        "risk_floor_count": audit["baseline"]["risk_floor_count"],
    }
    (snapshot_dir / "snapshot_manifest.json").write_text(
        json.dumps(snapshot_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return snapshot_dir


def build_local_snapshot(as_of_date: str, retrain_classifier: bool = False) -> dict:
    cutoff = pd.Timestamp(as_of_date).normalize().date().isoformat()
    fsic_df, weo_df, mfs_df = _load_cached_sources()

    model = BankingRiskModel()
    results = model.train(
        fsic_df=fsic_df,
        weo_df=weo_df,
        mfs_df=mfs_df,
        as_of_date=cutoff,
        retrain_classifier=retrain_classifier,
    )
    passed_checks, failed_checks = validate_model(
        results, features_df=model.feature_values
    )
    model.save()

    audit = _write_policy_audit(model)
    manifest = build_snapshot_manifest(cutoff)
    manifest["validation"] = {
        "model_checks_passed": int(passed_checks),
        "model_checks_failed": int(failed_checks),
    }
    manifest["source_mode"] = "local_cached_sources"
    manifest_path = write_snapshot_manifest(
        manifest,
        Path(BASE_DIR) / "artifacts" / "data_manifest.json",
    )
    snapshot_dir = _archive_snapshot(cutoff, manifest, audit)

    return {
        "as_of_date": cutoff,
        "active_manifest": str(manifest_path),
        "snapshot_dir": str(snapshot_dir),
        "validation": manifest["validation"],
        "snapshot_status": manifest["snapshot_status"],
        "coverage_score_correlation": audit["baseline"][
            "coverage_score_correlation"
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True, help="Snapshot cutoff YYYY-MM-DD")
    parser.add_argument(
        "--retrain-classifier",
        action="store_true",
        help=(
            "Retrain the supervised crisis classifier. By default the existing "
            "validated classifier artifact is reused for faster dated snapshots."
        ),
    )
    args = parser.parse_args()

    result = build_local_snapshot(
        args.as_of,
        retrain_classifier=args.retrain_classifier,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
