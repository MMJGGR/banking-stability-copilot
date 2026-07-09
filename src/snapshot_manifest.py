"""Build auditable metadata for a dated serving snapshot."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.config import BASE_DIR, CACHE_DIR
from src.data_loader import is_time_period_column, parse_period_label
from src.model_store import load_model_artifact


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_long_cache(path: Path, cutoff: pd.Timestamp) -> dict:
    columns = ["country_code", "indicator_code", "period"]
    available_columns = set(pq.ParquetFile(path).schema.names)
    if "observation_status" in available_columns:
        columns.append("observation_status")

    frame = pd.read_parquet(path, columns=columns)
    frame["period"] = pd.to_datetime(frame["period"], errors="coerce")
    frame = frame.loc[
        frame["period"].notna() & (frame["period"] <= cutoff)
    ]

    summary = {
        "rows": int(len(frame)),
        "countries": int(frame["country_code"].nunique()),
        "indicators": int(frame["indicator_code"].nunique()),
        "latest_observation": (
            frame["period"].max().date().isoformat() if len(frame) else None
        ),
    }
    if "observation_status" in frame.columns:
        summary["observation_status_counts"] = {
            str(key): int(value)
            for key, value in frame["observation_status"].value_counts().items()
        }
    return summary


def summarize_fsibsis_cache(path: Path, cutoff: pd.Timestamp) -> dict:
    frame = pd.read_parquet(path)
    period_columns = [
        column
        for column in frame.columns
        if is_time_period_column(column)
        and parse_period_label(column) <= cutoff
    ]
    populated = {
        column: int(pd.to_numeric(frame[column], errors="coerce").notna().sum())
        for column in period_columns
    }
    populated = {key: value for key, value in populated.items() if value > 0}
    latest_label = (
        max(populated, key=lambda column: parse_period_label(column))
        if populated
        else None
    )
    has_observation = (
        frame[period_columns].notna().any(axis=1)
        if period_columns
        else pd.Series(False, index=frame.index)
    )
    return {
        "rows": int(len(frame)),
        "countries": int(frame.loc[has_observation, "country_code"].nunique()),
        "latest_observation": (
            parse_period_label(latest_label).date().isoformat()
            if latest_label
            else None
        ),
        "latest_period_label": latest_label,
    }


def summarize_wgi_cache(path: Path, cutoff: pd.Timestamp) -> dict:
    frame = pd.read_parquet(path)
    frame = frame.loc[pd.to_numeric(frame["year"], errors="coerce") <= cutoff.year]
    return {
        "rows": int(len(frame)),
        "countries": int(frame["country_code"].nunique()),
        "latest_observation": (
            f"{int(frame['year'].max())}-12-31" if len(frame) else None
        ),
    }


def build_snapshot_manifest(as_of_date, repository_root=None) -> dict:
    root = Path(repository_root or BASE_DIR)
    cache_dir = Path(CACHE_DIR)
    cutoff = pd.Timestamp(as_of_date).normalize()
    model = load_model_artifact(cache_dir / "risk_model.pkl")
    model_metadata = model.get("pca_info", {})

    sources = {}
    for source in ("FSIC", "MFS", "WEO"):
        path = cache_dir / f"{source}_cache.parquet"
        if path.exists():
            sources[source] = summarize_long_cache(path, cutoff)

    fsibsis_path = cache_dir / "FSIBSIS_cache.parquet"
    if fsibsis_path.exists():
        sources["FSIBSIS"] = summarize_fsibsis_cache(fsibsis_path, cutoff)

    wgi_path = cache_dir / "WGI_cache.parquet"
    if wgi_path.exists():
        sources["WGI"] = summarize_wgi_cache(wgi_path, cutoff)

    artifact_paths = [
        root / "artifacts" / "model_policy_audit.json",
        cache_dir / "risk_model.pkl",
        cache_dir / "inference_pipeline.pkl",
        cache_dir / "crisis_classifier.pkl",
        cache_dir / "crisis_features.parquet",
        cache_dir / "imputed_features.parquet",
        fsibsis_path,
        wgi_path,
    ]
    artifacts = {
        str(path.relative_to(root)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in artifact_paths
        if path.exists()
    }

    model_snapshot_date = model_metadata.get("snapshot_date")
    cutoff_verified = model_snapshot_date == cutoff.date().isoformat()

    return {
        "schema_version": 1,
        "snapshot_id": cutoff.strftime("%Y-%m-%d"),
        "as_of_date": cutoff.date().isoformat(),
        "snapshot_status": (
            "verified" if cutoff_verified else "legacy_model_unverified_cutoff"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "training_date": model["training_date"],
            "snapshot_date": model_snapshot_date,
            "countries_trained": int(model["countries_trained"]),
            "cutoff_verified": cutoff_verified,
        },
        "sources": sources,
        "artifacts": artifacts,
    }


def write_snapshot_manifest(manifest: dict, output_path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path
