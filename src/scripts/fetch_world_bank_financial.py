"""Fetch long World Bank financial-sector histories for crisis modelling."""

from __future__ import annotations

import argparse
import json

from src.world_bank_financial import (
    build_world_bank_financial_features,
    fetch_world_bank_financial_history,
    write_world_bank_financial_history,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1960)
    parser.add_argument("--end-year", type=int)
    args = parser.parse_args()

    observations = fetch_world_bank_financial_history(
        start_year=args.start_year,
        end_year=args.end_year,
    )
    reference_path, cache_path = write_world_bank_financial_history(observations)
    features = build_world_bank_financial_features(observations)
    summary = {
        "rows": int(len(observations)),
        "countries": int(observations["country_code"].nunique()),
        "indicators": int(observations["indicator_code"].nunique()),
        "first_year": int(observations["year"].min()),
        "last_year": int(observations["year"].max()),
        "feature_rows": int(len(features)),
        "reference_path": str(reference_path) if reference_path else None,
        "cache_path": str(cache_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
