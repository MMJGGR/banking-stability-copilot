"""Fetch and normalize the external-liquidity SDMX sources (ranks 5-14).

Consumes ``config/external_sources_discovery.json`` (produced by
``discover_external_sources.py``), downloads each resolved dataflow through
the existing chunk-safe SDMX client, normalizes the long CSV into a generic
observation Parquet under ``cache/external/``, and writes a coverage report
to ``artifacts/external_sources_report.json``.

These caches are STAGED inputs: they make BOP/IIP/IRFCL/CPIS/CDIS/FM/GFS
observations available for feature engineering (debt-service ratios, gross
external financing needs, reserves adequacy, portfolio flows), but no model
feature consumes them until that work is reviewed. Run where api.imf.org is
reachable:

    python -m src.scripts.fetch_external_sources --start-period 2005
    python -m src.scripts.fetch_external_sources --sources BOP IIP --keep-raw
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

import pandas as pd

from src.config import BASE_DIR, CACHE_DIR
from src.scripts.discover_external_sources import DISCOVERY_PATH
from src.sources.sdmx import SdmxDataflowSource, _parse_sdmx_period

EXTERNAL_CACHE_DIR = Path(CACHE_DIR) / "external"
REPORT_PATH = Path(BASE_DIR) / "artifacts" / "external_sources_report.json"

# Long-CSV columns that are metadata rather than key dimensions or values.
NON_DIMENSION_COLUMNS = {
    "STRUCTURE", "STRUCTURE_ID", "STRUCTURE_NAME", "ACTION", "TIME_PERIOD",
    "OBS_VALUE", "OBS_STATUS", "UNIT_MULT", "DECIMALS", "COMMENT",
}


def normalize_long_csv(csv_path, source_name: str) -> pd.DataFrame:
    """Long SDMX CSV -> generic observation table.

    Keeps every key dimension (they differ per dataflow) plus parsed period
    end, value, and observation status, so downstream feature work can slice
    without re-downloading.
    """
    frame = pd.read_csv(csv_path, low_memory=False)
    required = {"TIME_PERIOD", "OBS_VALUE"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source_name}: CSV lacks required columns {sorted(missing)}")

    country_column = next(
        (c for c in ("COUNTRY", "REF_AREA", "REPORTING_ECONOMY") if c in frame.columns),
        None,
    )
    if country_column is None:
        raise ValueError(f"{source_name}: no country dimension column found")

    normalized = pd.DataFrame(
        {
            "source": source_name,
            "dataset_version": frame.get("STRUCTURE_ID"),
            "country_code": frame[country_column].astype(str).str.strip(),
            "period_label": frame["TIME_PERIOD"].astype(str).str.strip(),
            "period": frame["TIME_PERIOD"].map(_parse_sdmx_period),
            "value": pd.to_numeric(frame["OBS_VALUE"], errors="coerce"),
            "observation_status": frame.get("OBS_STATUS"),
        }
    )
    for column in frame.columns:
        if column not in NON_DIMENSION_COLUMNS and column != country_column:
            if frame[column].dtype == object or str(frame[column].dtype).startswith("int"):
                normalized[f"dim_{column}"] = frame[column]

    normalized = normalized[
        normalized["value"].notna() & normalized["period"].notna()
    ].reset_index(drop=True)
    if normalized.empty:
        raise ValueError(f"{source_name}: no parsable observations after normalization")
    return normalized


def _coverage(normalized: pd.DataFrame) -> dict:
    indicator_columns = [c for c in normalized.columns if c.startswith("dim_")]
    indicator_column = next(
        (c for c in ("dim_INDICATOR", "dim_SERIES", "dim_ITEM") if c in normalized),
        indicator_columns[0] if indicator_columns else None,
    )
    return {
        "rows": int(len(normalized)),
        "countries": int(normalized["country_code"].nunique()),
        "indicators": (
            int(normalized[indicator_column].nunique())
            if indicator_column else None
        ),
        "earliest_period": str(normalized["period"].min().date()),
        "latest_period": str(normalized["period"].max().date()),
        "dataset_version": (
            str(normalized["dataset_version"].dropna().iloc[0])
            if normalized["dataset_version"].notna().any() else None
        ),
        "key_dimensions": indicator_columns,
    }


def fetch_all(discovery: dict, start_period=None, sources=None,
              keep_raw=False, download_dir=None) -> dict:
    download_dir = Path(
        download_dir
        or Path(BASE_DIR) / "data" / "raw"
        / f"external_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    EXTERNAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start_period": start_period,
        "sources": {},
    }
    for family, entry in discovery.get("families", {}).items():
        if sources and family not in sources:
            continue
        if entry.get("status") != "resolved":
            report["sources"][family] = {
                "status": "skipped_unresolved",
                "attempts": entry.get("attempts", []),
            }
            continue
        source = SdmxDataflowSource(
            name=family,
            agency=entry["agency"],
            dataflow_id=entry["dataflow_id"],
            dimensions=tuple(entry["dimensions"]),
        )
        try:
            result = source.fetch(download_dir, start_period=start_period)
            normalized = normalize_long_csv(result.path, family)
            cache_path = EXTERNAL_CACHE_DIR / f"{family}_observations.parquet"
            normalized.to_parquet(cache_path, index=False)
            report["sources"][family] = {
                "status": "ok",
                "cache_path": str(cache_path),
                "raw_bytes": result.bytes,
                "raw_sha256": result.sha256,
                "remote_version": result.remote_version,
                **_coverage(normalized),
            }
            if not keep_raw:
                Path(result.path).unlink(missing_ok=True)
        except Exception as error:  # noqa: BLE001 - report per source
            report["sources"][family] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
    if not keep_raw and download_dir.exists() and not any(download_dir.iterdir()):
        shutil.rmtree(download_dir, ignore_errors=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", default=str(DISCOVERY_PATH))
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--start-period", default="2005")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--download-dir", default=None)
    parser.add_argument("--output", default=str(REPORT_PATH))
    args = parser.parse_args()

    discovery_path = Path(args.discovery)
    if not discovery_path.exists():
        raise SystemExit(
            f"Discovery file not found: {discovery_path}. Run "
            "python -m src.scripts.discover_external_sources first."
        )
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))

    report = fetch_all(
        discovery,
        start_period=args.start_period,
        sources=args.sources,
        keep_raw=args.keep_raw,
        download_dir=args.download_dir,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Report written to: {output}")
    for family, entry in report["sources"].items():
        if entry["status"] == "ok":
            print(
                f"  {family}: {entry['rows']} rows, {entry['countries']} countries, "
                f"through {entry['latest_period']}"
            )
        else:
            print(f"  {family}: {entry['status']} {entry.get('error', '')}")
    failures = [
        family for family, entry in report["sources"].items()
        if entry["status"] == "failed"
    ]
    raise SystemExit(1 if failures and len(failures) == len(report["sources"]) else 0)


if __name__ == "__main__":
    main()
