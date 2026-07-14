"""Build staged general-government (fiscal) liquidity challenger features.

Unlike the external-liquidity builder, this script needs no IMF BOP/IIP or
World Bank API calls: the general-government WEO series it uses are already in
the local WEO cache. It writes:

- ``cache/government/government_liquidity_observations.parquet``
- ``cache/government/government_liquidity_features.parquet``
- ``artifacts/government_liquidity_features_report.json``

Pass ``--reference-dir data/reference`` to also package the compact outputs
into the app's reference directory so the hosted Streamlit app can surface
them without loading the large WEO cache at startup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.government_liquidity import (
    GOVT_FEATURE_OBSERVATIONS,
    GOVT_FEATURE_REPORT,
    GOVT_FEATURE_VALUES,
    model_country_codes,
    refresh_government_liquidity_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=None, help="Snapshot cutoff (e.g. 2026-06-30).")
    parser.add_argument("--countries", nargs="*", default=None, help="Optional ISO3 country subset.")
    parser.add_argument(
        "--include-projections",
        action="store_true",
        help="Include WEO projections (default: actuals and estimates only).",
    )
    parser.add_argument("--observations", default=str(GOVT_FEATURE_OBSERVATIONS))
    parser.add_argument("--features", default=str(GOVT_FEATURE_VALUES))
    parser.add_argument("--report", default=str(GOVT_FEATURE_REPORT))
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="Optional directory to also copy the compact outputs into (e.g. data/reference).",
    )
    args = parser.parse_args()

    countries = [c.upper() for c in args.countries] if args.countries else model_country_codes()

    observations_path = Path(args.observations)
    features_path = Path(args.features)
    report_path = Path(args.report)
    _, _, report = refresh_government_liquidity_outputs(
        as_of_date=args.as_of,
        model_countries=countries,
        include_projections=args.include_projections,
        observations_path=observations_path,
        features_path=features_path,
        report_path=report_path,
        reference_dir=Path(args.reference_dir) if args.reference_dir else None,
    )

    print(f"Observations written to: {observations_path}")
    print(f"Features written to: {features_path}")
    print(f"Report written to: {report_path}")
    print(json.dumps(report["feature_coverage"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
