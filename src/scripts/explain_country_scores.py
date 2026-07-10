"""Publish a per-feature driver table for selected countries.

Reproduces each country's pillar scores from the active serving artifacts and
decomposes them into per-feature contributions, with raw versus imputed values
flagged, so a suspicious ranking (e.g. Kenya vs Mozambique) can be audited
without manual artifact inspection.

Usage:
    python -m src.scripts.explain_country_scores            # defaults KEN MOZ
    python -m src.scripts.explain_country_scores KEN MOZ TZA
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BASE_DIR, CACHE_DIR
from src.model_store import load_model_artifact


OUTPUT_PATH = Path(BASE_DIR) / "artifacts" / "score_drivers.json"
SUMMARY_COLUMNS = [
    "risk_score", "economic_pillar", "industry_pillar", "data_coverage",
    "economic_coverage", "industry_coverage", "crisis_prob", "crisis_uplift",
    "critical_missing_share", "critical_penalty", "risk_floor_applied",
    "risk_category",
]


def _load_pipeline():
    pipeline_path = Path(CACHE_DIR) / "inference_pipeline.pkl"
    with pipeline_path.open("rb") as handle:
        artifact = pickle.load(handle)
    pipeline = artifact["pillar_pipeline"]
    if pipeline is None or not pipeline.fitted_:
        raise ValueError("Fitted pillar pipeline is required for driver tables")
    return pipeline


def _pipeline_matrices(pipeline, features):
    """Reproduce the imputed/scaled/oriented matrices transform() uses."""
    indexed = pipeline._index_features(features)
    for column in pipeline.numeric_columns_:
        if column not in indexed.columns:
            indexed[column] = np.nan
    numeric = indexed[pipeline.numeric_columns_].apply(
        pd.to_numeric, errors="coerce"
    )
    eligible = numeric.loc[
        numeric.notna().mean(axis=1) >= pipeline.minimum_data_coverage
    ].copy()
    original_missing = eligible.isna()
    imputed = pd.DataFrame(
        pipeline.imputer_.transform(eligible[pipeline.imputed_columns_]),
        index=eligible.index,
        columns=pipeline.imputed_columns_,
    )
    for column in pipeline.empty_columns_:
        imputed[column] = 0.0
    imputed = imputed[pipeline.numeric_columns_]
    transformed = pipeline._apply_log_transforms(imputed)
    scaled = pd.DataFrame(
        pipeline.scaler_.transform(transformed),
        index=transformed.index,
        columns=transformed.columns,
    )
    oriented = pipeline._orient(scaled)
    return eligible, original_missing, imputed, oriented


def _pillar_drivers(pipeline, scorer, columns, pillar_name, country,
                    eligible, original_missing, imputed, oriented):
    weights = scorer.components_[0]
    directions = getattr(pipeline, "risk_directions_", {})
    critical = set(getattr(pipeline, "critical_columns_", []))
    row_oriented = oriented.loc[country, columns].to_numpy(dtype=float)
    # direction_signs_ maps the fitted component to safety orientation, so
    # -sign re-expresses contributions as risk (positive = riskier). This
    # keeps the table correct for both constrained pipelines (sign -1) and
    # legacy pipelines whose component orientation was anchor-determined.
    risk_sign = -float(pipeline.direction_signs_.get(pillar_name, -1.0))
    contributions = risk_sign * weights * (row_oriented - scorer.mean_)
    peer_percentiles = eligible[columns].rank(pct=True).loc[country]

    drivers = []
    for i, column in enumerate(columns):
        raw_value = eligible.loc[country, column]
        drivers.append({
            "pillar": pillar_name,
            "feature": column,
            "raw_value": None if pd.isna(raw_value) else float(raw_value),
            "used_value": float(imputed.loc[country, column]),
            "is_imputed": bool(original_missing.loc[country, column]),
            "is_critical": column in critical,
            "risk_direction": directions.get(column),
            "weight": float(weights[i]),
            "risk_contribution": float(contributions[i]),
            "peer_percentile_raw": (
                None if pd.isna(peer_percentiles[column])
                else round(float(peer_percentiles[column]), 3)
            ),
        })
    drivers.sort(key=lambda item: abs(item["risk_contribution"]), reverse=True)
    raw_pillar_score = float(contributions.sum())
    return drivers, raw_pillar_score


def build_driver_table(country_codes, model=None, pipeline=None) -> dict:
    model = model or load_model_artifact()
    pipeline = pipeline or _load_pipeline()
    features = model["feature_values"]
    scores = model["country_scores"].set_index("country_code")

    eligible, original_missing, imputed, oriented = _pipeline_matrices(
        pipeline, features
    )

    report = {
        "generated_from": {
            "training_date": model.get("training_date"),
            "snapshot_date": model.get("pca_info", {}).get("snapshot_date"),
        },
        "note": (
            "risk_contribution is on the oriented pillar-component scale: "
            "positive values push the country toward higher risk relative to "
            "the training mean. Contributions sum to the raw pillar score."
        ),
        "countries": {},
    }
    for country in country_codes:
        if country not in eligible.index:
            report["countries"][country] = {"error": "not scored (insufficient coverage)"}
            continue
        economic, economic_raw = _pillar_drivers(
            pipeline, pipeline.economic_pca_, pipeline.economic_columns_,
            "economic", country, eligible, original_missing, imputed, oriented,
        )
        industry, industry_raw = _pillar_drivers(
            pipeline, pipeline.industry_pca_, pipeline.industry_columns_,
            "industry", country, eligible, original_missing, imputed, oriented,
        )
        summary = {}
        if country in scores.index:
            row = scores.loc[country]
            for column in SUMMARY_COLUMNS:
                if column in row.index and pd.notna(row[column]):
                    value = row[column]
                    summary[column] = (
                        bool(value) if isinstance(value, (bool, np.bool_))
                        else value if isinstance(value, str)
                        else float(value)
                    )
        report["countries"][country] = {
            "summary": summary,
            "raw_pillar_components": {
                "economic": economic_raw,
                "industry": industry_raw,
            },
            "drivers": economic + industry,
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("countries", nargs="*", default=None)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()
    countries = [code.upper() for code in (args.countries or ["KEN", "MOZ"])]

    report = build_driver_table(countries)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    for country, payload in report["countries"].items():
        if "error" in payload:
            print(f"{country}: {payload['error']}")
            continue
        summary = payload["summary"]
        print(
            f"{country}: risk={summary.get('risk_score')} "
            f"econ={summary.get('economic_pillar'):.1f} "
            f"ind={summary.get('industry_pillar'):.1f} "
            f"crisis_prob={summary.get('crisis_prob', float('nan')):.2f} "
            f"critical_missing={summary.get('critical_missing_share', 0):.0%}"
        )
        top = payload["drivers"][:5]
        for driver in top:
            flag = " (imputed)" if driver["is_imputed"] else ""
            print(
                f"    {driver['pillar']}/{driver['feature']}: "
                f"contribution {driver['risk_contribution']:+.3f}{flag}"
            )
    print(f"Driver table written to: {output}")


if __name__ == "__main__":
    main()
