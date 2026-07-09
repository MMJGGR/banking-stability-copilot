"""Generate reproducible sensitivity evidence for model-policy review."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR, CACHE_DIR
from src.crisis_labels import CrisisLabels
from src.pillar_pipeline import PillarInferencePipeline


def _score(features, anchor, **pipeline_options):
    pipeline = PillarInferencePipeline(**pipeline_options).fit(
        features,
        anchor,
    )
    return pipeline.transform(features).set_index("country_code")


def _compare(baseline, challenger):
    shared = baseline.index.intersection(challenger.index)
    base = baseline.loc[shared]
    other = challenger.loc[shared]
    delta = other["risk_score"] - base["risk_score"]
    return {
        "countries": int(len(shared)),
        "mean_absolute_score_change": float(delta.abs().mean()),
        "maximum_absolute_score_change": float(delta.abs().max()),
        "rank_correlation": float(
            base["risk_score"].corr(other["risk_score"], method="spearman")
        ),
        "countries_moving_at_least_one_point": int(
            (delta.abs() >= 1).sum()
        ),
    }


def build_policy_audit(features):
    anchor = features.set_index("country_code")["gdp_per_capita"]
    baseline = _score(features, anchor)
    scenarios = {
        "no_confidence_regression": _score(
            features,
            anchor,
            confidence_exponent=0.0,
        ),
        "no_risk_floors": _score(
            features,
            anchor,
            apply_risk_floors=False,
        ),
        "no_gdp_pca_input": _score(
            features.drop(columns=["gdp_per_capita"]),
            anchor,
        ),
        "no_gdp_orientation": _score(features, None),
    }
    labels = CrisisLabels()
    baseline_coverage_correlation = baseline["data_coverage"].corr(
        baseline["risk_score"]
    )
    return {
        "schema_version": 1,
        "baseline": {
            "countries": int(len(baseline)),
            "coverage_score_correlation": float(
                baseline_coverage_correlation
            ),
            "absolute_coverage_score_correlation": float(
                abs(baseline_coverage_correlation)
            ),
            "risk_floor_count": int(
                baseline["risk_floor_applied"].sum()
            ),
        },
        "scenarios": {
            name: _compare(baseline, result)
            for name, result in scenarios.items()
        },
        "crisis_labels": {
            "source_version": labels.SOURCE_VERSION,
            "source_coverage_end_year": labels.SOURCE_COVERAGE_END_YEAR,
            "systemic_countries": len(labels.SYSTEMIC_CRISES),
            "systemic_episodes": sum(
                len(periods)
                for periods in labels.SYSTEMIC_CRISES.values()
            ),
            "borderline_episodes_excluded": sum(
                len(periods)
                for periods in labels.BORDERLINE_CRISES.values()
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default=str(Path(CACHE_DIR) / "crisis_features.parquet"),
    )
    parser.add_argument(
        "--output",
        default=str(Path(BASE_DIR) / "artifacts" / "model_policy_audit.json"),
    )
    args = parser.parse_args()

    audit = build_policy_audit(pd.read_parquet(args.features))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Model policy audit written to: {output}")


if __name__ == "__main__":
    main()
