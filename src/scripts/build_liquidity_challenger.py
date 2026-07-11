"""Build a liquidity-feature challenger and compare it against production.

This wires the curated staged external / government liquidity features into the
real training pipeline (the pillar model plus the fixed crisis overlay) and
produces a challenger-vs-production comparison for governed review. It does NOT
call ``model.save()`` and never touches the active serving artifacts.

Two trains are run at the same cutoff, both reusing the cached classifier:

- control (no extra features), to confirm the retrain reproduces production and
  isolate the pure effect of the added features; and
- challenger (with the curated liquidity features).

Outputs:
- ``artifacts/liquidity_challenger_comparison.json``
- ``artifacts/snapshots/<cutoff>-challenger-liquidity/challenger_scores.parquet``
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BASE_DIR, CACHE_DIR
from src.lfs_resolver import ensure_lfs_file
from src.model_store import load_model_artifact


# Curated, data-backed, non-duplicative additions (see pillar_pipeline.py).
GOVT_CHALLENGER_FEATURES = ["govt_interest_to_revenue", "govt_debt_to_revenue"]
EXTERNAL_CHALLENGER_FEATURES = [
    "net_iip_gdp",
    "external_liabilities_gdp",
    "reserves_to_goods_services_imports",
    "gross_external_financing_need_proxy_gdp",
    "investment_income_debits_to_cxr",
]

TIER_BINS = [2, 4, 6, 8]


def _resolve_caches() -> None:
    for name in [
        "FSIC_cache.parquet", "WEO_cache.parquet", "MFS_cache.parquet",
        "FSIBSIS_cache.parquet", "WGI_cache.parquet",
        "risk_model.pkl", "crisis_classifier.pkl",
    ]:
        path = Path(CACHE_DIR) / name
        if path.exists():
            try:
                ensure_lfs_file(path)
            except Exception as exc:  # noqa: BLE001 - best-effort resolution
                print(f"  WARN could not resolve {name}: {exc}")


def _build_extra_features(as_of_date: str) -> pd.DataFrame:
    from src.government_liquidity import model_country_codes
    from src.liquidity_features import assemble_liquidity_features

    return assemble_liquidity_features(
        as_of_date=as_of_date, model_countries=model_country_codes()
    )


def _train_scores(fsic, weo, mfs, as_of_date, extra_features=None) -> pd.DataFrame:
    from train_model import BankingRiskModel

    model = BankingRiskModel()
    model.train(
        fsic_df=fsic, weo_df=weo, mfs_df=mfs,
        as_of_date=as_of_date,
        retrain_classifier=False,
        extra_features=extra_features,
    )
    scores = model.country_scores[["country_code", "risk_score"]].copy()
    scores["country_code"] = scores["country_code"].astype(str).str.upper()
    return scores


def _tier(score: pd.Series) -> pd.Series:
    return pd.cut(score, bins=[-np.inf, *TIER_BINS, np.inf], labels=[1, 2, 3, 4, 5]).astype(int)


def _compare(baseline: pd.DataFrame, challenger: pd.DataFrame, label: str) -> dict:
    from scipy.stats import spearmanr

    merged = baseline.merge(challenger, on="country_code", suffixes=("_base", "_chal"))
    merged = merged.dropna(subset=["risk_score_base", "risk_score_chal"])
    delta = merged["risk_score_chal"] - merged["risk_score_base"]
    tier_changes = int((_tier(merged["risk_score_chal"]) != _tier(merged["risk_score_base"])).sum())
    rho = float(spearmanr(merged["risk_score_base"], merged["risk_score_chal"]).correlation)
    movers = merged.assign(delta=delta).reindex(
        delta.abs().sort_values(ascending=False).index
    ).head(20)
    return {
        "label": label,
        "countries_compared": int(len(merged)),
        "mean_absolute_score_change": round(float(delta.abs().mean()), 3),
        "max_absolute_score_change": round(float(delta.abs().max()), 3),
        "countries_moving_at_least_one_point": int((delta.abs() >= 1).sum()),
        "rank_correlation_spearman": round(rho, 3),
        "risk_tier_changes": tier_changes,
        "largest_movements": [
            {
                "country_code": row.country_code,
                "base": round(float(row.risk_score_base), 1),
                "challenger": round(float(row.risk_score_chal), 1),
                "delta": round(float(row.delta), 1),
            }
            for row in movers.itertuples()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default="2026-06-30")
    parser.add_argument(
        "--report",
        default=str(Path(BASE_DIR) / "artifacts" / "liquidity_challenger_comparison.json"),
    )
    args = parser.parse_args()

    _resolve_caches()
    from src.scripts.build_local_snapshot import _load_cached_sources

    fsic, weo, mfs = _load_cached_sources()
    extra = _build_extra_features(args.as_of)
    added = [c for c in extra.columns if c != "country_code"]
    print(f"Added liquidity features: {added}")

    production = load_model_artifact()["country_scores"][["country_code", "risk_score"]].copy()
    production["country_code"] = production["country_code"].astype(str).str.upper()

    print("\n=== CONTROL TRAIN (no extra features) ===")
    control = _train_scores(fsic, weo, mfs, args.as_of)
    print("\n=== CHALLENGER TRAIN (with liquidity features) ===")
    challenger = _train_scores(fsic, weo, mfs, args.as_of, extra_features=extra)

    faithfulness = _compare(production, control, "control_vs_production")
    effect = _compare(control, challenger, "challenger_vs_control")
    headline = _compare(production, challenger, "challenger_vs_production")

    thresholds = {
        "mean_absolute_score_change": 0.5,
        "rank_correlation_spearman": 0.90,
        "risk_tier_changes": 15,
    }
    # Gate on the ISOLATED feature effect (challenger vs control), not the
    # headline vs production: the active cache/risk_model.pkl is stale relative
    # to the current pipeline, so the headline delta is confounded by that
    # staleness rather than by the added features.
    trips_review = (
        effect["mean_absolute_score_change"] > thresholds["mean_absolute_score_change"]
        or effect["rank_correlation_spearman"] < thresholds["rank_correlation_spearman"]
        or effect["risk_tier_changes"] > thresholds["risk_tier_changes"]
    )
    stale_active_artifact = (
        faithfulness["mean_absolute_score_change"] > thresholds["mean_absolute_score_change"]
    )

    report = {
        "generated": date.today().isoformat(),
        "cutoff": args.as_of,
        "added_features": added,
        "feature_directions": {
            "govt_interest_to_revenue": "+1 (higher = riskier)",
            "govt_debt_to_revenue": "+1 (higher = riskier)",
            "net_iip_gdp": "-1 (higher net creditor = safer)",
            "external_liabilities_gdp": "+1 (higher = riskier)",
            "reserves_to_goods_services_imports": "-1 (more import cover = safer)",
            "gross_external_financing_need_proxy_gdp": "+1 (higher = riskier)",
            "investment_income_debits_to_cxr": "+1 (higher income-service burden = riskier)",
        },
        "governance_thresholds": thresholds,
        "promotion_requires_owner_review": bool(trips_review),
        "review_basis": "feature_effect_challenger_vs_control",
        "stale_active_artifact_finding": bool(stale_active_artifact),
        "faithfulness_control_vs_production": faithfulness,
        "feature_effect_challenger_vs_control": effect,
        "headline_challenger_vs_production": headline,
        "notes": [
            "Staged challenger only. The active serving artifacts are unchanged; "
            "this script never calls model.save().",
            "Both trains reuse the cached crisis classifier, so the classifier "
            "overlay is held fixed and the delta isolates the pillar effect of "
            "the added liquidity features.",
            "control_vs_production should be near zero; a non-trivial value means "
            "the retrain itself does not reproduce production and the effect "
            "estimate is confounded. A large value indicates the active "
            "cache/risk_model.pkl is stale relative to the current pipeline and "
            "should be rebuilt independently of this challenger.",
            "Market/FDI/REER columns are omitted here because they are null until "
            "the next CI fetch of the external block.",
        ],
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    archive = Path(BASE_DIR) / "artifacts" / "snapshots" / f"{args.as_of}-challenger-liquidity"
    archive.mkdir(parents=True, exist_ok=True)
    challenger.to_parquet(archive / "challenger_scores.parquet", index=False)

    print("\n=== SUMMARY ===")
    print(f"Faithfulness (control vs production): mean|delta|="
          f"{faithfulness['mean_absolute_score_change']} "
          f"spearman={faithfulness['rank_correlation_spearman']}")
    print(f"Feature effect (challenger vs control): mean|delta|="
          f"{effect['mean_absolute_score_change']} "
          f">=1pt={effect['countries_moving_at_least_one_point']} "
          f"tier_changes={effect['risk_tier_changes']} "
          f"spearman={effect['rank_correlation_spearman']}")
    print(f"Headline (challenger vs production): mean|delta|="
          f"{headline['mean_absolute_score_change']} "
          f"tier_changes={headline['risk_tier_changes']} "
          f"spearman={headline['rank_correlation_spearman']}")
    print(f"Promotion requires owner review: {trips_review}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
