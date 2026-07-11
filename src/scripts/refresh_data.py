"""Fetch, validate, normalize, train, and package a dated snapshot."""

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

from src.config import BASE_DIR
from src.data_loader import FSIBSISLoader, IMFDataLoader, WGILoader
from src.snapshot_manifest import build_snapshot_manifest, write_snapshot_manifest
from src.scripts.audit_model_policy import build_policy_audit
from src.sources import build_source_adapters
from src.sources.base import SourceResult
from src.sources.sdmx import build_sdmx_sources
from src.sources.sdmx_normalize import (
    normalize_fsibsis_cache,
    normalize_imf_long_cache,
    normalize_wgi_cache,
)
from train_model import BankingRiskModel, validate_model


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_existing_download(download_dir: Path, source_name: str) -> Path | None:
    patterns = {
        "WEO": "sdmx_WEO*.csv",
        "FSIC": "sdmx_FSIC*.csv",
        "MFS": "sdmx_MFS*.csv",
        "FSIBSIS": "sdmx_FSIBSIS*.csv",
        "WGI": "worldbank_WGI*.csv",
    }
    matches = [
        path for path in download_dir.glob(patterns[source_name])
        if path.is_file() and path.stat().st_size > 0
    ]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _source_result_from_existing(source_name: str, path: Path, source) -> SourceResult:
    remote_version = None
    if hasattr(source, "_version_from_csv"):
        remote_version = source._version_from_csv(path)
    return SourceResult(
        source=source_name,
        retrieval_method="reused_official_download",
        path=str(path.resolve()),
        bytes=path.stat().st_size,
        sha256=_sha256_file(path),
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        remote_version=remote_version,
    )


def _fetch_and_normalize_official_sources(download_dir: Path, reuse_downloads=False):
    sources = build_sdmx_sources()
    fetched = {}

    for name in ["WEO", "FSIC", "MFS", "FSIBSIS", "WGI"]:
        print(f"\nFetching official source: {name}")
        existing = (
            _latest_existing_download(download_dir, name)
            if reuse_downloads else None
        )
        if existing is not None:
            result = _source_result_from_existing(name, existing, sources[name])
        else:
            result = sources[name].fetch(download_dir)
        fetched[name] = result
        print(
            f"  {result.retrieval_method}: {result.bytes:,} bytes, "
            f"version={result.remote_version}"
        )

        if name in {"WEO", "FSIC", "MFS"}:
            df = normalize_imf_long_cache(name, result.path)
        elif name == "FSIBSIS":
            df = normalize_fsibsis_cache(result.path)
        elif name == "WGI":
            df = normalize_wgi_cache(result.path)
        else:
            continue

        print(
            f"  normalized {name}: {len(df):,} rows, "
            f"{df['country_code'].nunique() if 'country_code' in df else 'n/a'} countries"
        )

    loader = IMFDataLoader()
    if not loader.load_from_cache():
        raise RuntimeError("Official retrieval completed but IMF caches were not written")

    # Prime auxiliary caches from the newly normalized parquet files.
    FSIBSISLoader().load()
    WGILoader().load(force_refresh=False)

    return (
        fetched,
        loader._data_cache.get("FSIC"),
        loader._data_cache.get("WEO"),
        loader._data_cache.get("MFS"),
    )


def _fetch_legacy_sources(download_dir: Path):
    adapters = build_source_adapters()
    fetched = {
        name: adapter.fetch(download_dir, BASE_DIR)
        for name, adapter in adapters.items()
    }

    loader = IMFDataLoader()
    fsic_df = loader.load_fsic(fetched["FSIC"].path)
    weo_df = loader.load_weo(fetched["WEO"].path)
    mfs_df = loader.load_mfs(fetched["MFS"].path)
    loader.save_cache()

    FSIBSISLoader(fetched["FSIBSIS"].path).load()
    WGILoader(fetched["WGI"].path).load(force_refresh=True)

    return fetched, fsic_df, weo_df, mfs_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True, help="Snapshot cutoff YYYY-MM-DD")
    parser.add_argument(
        "--retrieval-mode",
        choices=("official", "legacy"),
        default="official",
        help=(
            "official fetches IMF SDMX/World Bank API data directly; legacy "
            "uses configured export URLs or local fallback files."
        ),
    )
    parser.add_argument(
        "--retrain-classifier",
        action="store_true",
        help=(
            "Retrain the supervised crisis classifier. By default the existing "
            "validated classifier artifact is reused for snapshot scoring."
        ),
    )
    parser.add_argument(
        "--reuse-downloads",
        action="store_true",
        help=(
            "For official mode, reuse the latest non-empty raw files in "
            "--download-dir instead of downloading again."
        ),
    )
    parser.add_argument(
        "--download-dir",
        default=str(Path(BASE_DIR) / "data" / "raw"),
    )
    parser.add_argument(
        "--manifest",
        default=str(Path(BASE_DIR) / "artifacts" / "data_manifest.json"),
    )
    args = parser.parse_args()

    download_dir = Path(args.download_dir)
    if args.retrieval_mode == "official":
        fetched, fsic_df, weo_df, mfs_df = _fetch_and_normalize_official_sources(
            download_dir,
            reuse_downloads=args.reuse_downloads,
        )
    else:
        fetched, fsic_df, weo_df, mfs_df = _fetch_legacy_sources(download_dir)

    from src.liquidity_features import assemble_liquidity_features
    extra_features = assemble_liquidity_features(as_of_date=args.as_of)

    model = BankingRiskModel()
    results = model.train(
        fsic_df,
        weo_df,
        mfs_df,
        as_of_date=args.as_of,
        retrain_classifier=args.retrain_classifier,
        extra_features=extra_features,
    )
    passed_checks, failed_checks = validate_model(
        results, features_df=model.feature_values
    )
    model.save()

    policy_audit = build_policy_audit(model.feature_values)
    policy_audit_path = Path(BASE_DIR) / "artifacts" / "model_policy_audit.json"
    policy_audit_path.parent.mkdir(parents=True, exist_ok=True)
    policy_audit_path.write_text(
        json.dumps(policy_audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest = build_snapshot_manifest(args.as_of)
    manifest["retrieval"] = {
        name: result.to_dict() for name, result in fetched.items()
    }
    manifest["source_mode"] = (
        "official_api_sdmx_worldbank"
        if args.retrieval_mode == "official"
        else "legacy_adapter"
    )
    manifest["validation"] = {
        "model_checks_passed": int(passed_checks),
        "model_checks_failed": int(failed_checks),
    }
    output = write_snapshot_manifest(manifest, args.manifest)
    print(f"Published candidate snapshot manifest: {output}")
    print(f"Snapshot status: {manifest['snapshot_status']}")
    print(
        f"Model validation: {passed_checks} passed, "
        f"{failed_checks} failed"
    )
    if failed_checks:
        raise RuntimeError(
            f"Candidate blocked by {failed_checks} model validation failure(s)"
        )


if __name__ == "__main__":
    main()
