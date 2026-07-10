"""Fetch and build staged external-liquidity challenger features.

This script performs bounded SDMX retrieval for the exact BOP/IIP series used
by the feature table. It avoids full dataflow downloads and writes:

- ``cache/external/external_feature_observations.parquet``
- ``cache/external/external_liquidity_features.parquet``
- ``artifacts/external_liquidity_features_report.json``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.external_liquidity import (
    EXTERNAL_FEATURE_OBSERVATIONS,
    EXTERNAL_FEATURE_REPORT,
    EXTERNAL_FEATURE_VALUES,
    build_external_liquidity_features,
    fetch_feature_observations,
    model_country_codes,
    write_external_liquidity_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="Fetch observations from IMF before building features.")
    parser.add_argument("--countries", nargs="*", default=None, help="Optional ISO3 country subset.")
    parser.add_argument("--start-period", default="2005")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--observations", default=str(EXTERNAL_FEATURE_OBSERVATIONS))
    parser.add_argument("--features", default=str(EXTERNAL_FEATURE_VALUES))
    parser.add_argument("--report", default=str(EXTERNAL_FEATURE_REPORT))
    args = parser.parse_args()

    observations_path = Path(args.observations)
    features_path = Path(args.features)
    report_path = Path(args.report)

    countries = [c.upper() for c in args.countries] if args.countries else model_country_codes()
    if args.fetch:
        observations = fetch_feature_observations(
            country_codes=countries,
            start_period=args.start_period,
            batch_size=args.batch_size,
        )
    else:
        if not observations_path.exists():
            raise SystemExit(
                f"Observation cache not found: {observations_path}. "
                "Run with --fetch first."
            )
        observations = pd.read_parquet(observations_path)

    features, report = build_external_liquidity_features(observations)
    write_external_liquidity_outputs(
        observations,
        features,
        report,
        observations_path=observations_path,
        features_path=features_path,
        report_path=report_path,
    )

    print(f"Observations written to: {observations_path}")
    print(f"Features written to: {features_path}")
    print(f"Report written to: {report_path}")
    print(json.dumps(report["feature_coverage"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
