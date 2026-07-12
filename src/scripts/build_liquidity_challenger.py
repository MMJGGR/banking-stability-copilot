"""Build candidate risk overlays and compare them against production.

This wires the curated staged external / government liquidity features into the
real training pipeline (the pillar model plus the fixed crisis overlay) and
produces a challenger-vs-production comparison for governed review. It does NOT
call ``model.save()`` and never touches the active serving artifacts.

Candidate trains are run at the same cutoff, reusing the cached classifier:

- active-retrain (with the production-promoted liquidity features), to confirm
  the retrain reproduces production; and
- liquidity-only candidate train with monitored government/external liquidity
  candidates;
- commodity-only candidate train with monitored commodity vulnerability; and
- combined candidate train with all monitored candidates.

Outputs:
- ``artifacts/liquidity_challenger_comparison.json``
- ``artifacts/snapshots/<cutoff>-challenger-liquidity/challenger_scores.parquet``
- ``artifacts/snapshots/<cutoff>-challenger-commodity/challenger_scores.parquet``
- ``artifacts/snapshots/<cutoff>-challenger-combined/challenger_scores.parquet``
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


def _build_extra_features(as_of_date: str, include_candidates: bool = False) -> pd.DataFrame:
    from src.government_liquidity import model_country_codes
    from src.liquidity_features import assemble_liquidity_features

    return assemble_liquidity_features(
        as_of_date=as_of_date,
        model_countries=model_country_codes(),
        include_candidates=include_candidates,
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


def _with_candidate_columns(
    active_extra: pd.DataFrame,
    candidate_extra: pd.DataFrame,
    candidate_columns: list[str],
) -> pd.DataFrame:
    """Return active features plus one independent candidate block."""
    columns = ["country_code"] + [
        column for column in candidate_columns if column in candidate_extra.columns
    ]
    if len(columns) == 1:
        return active_extra.copy()
    additions = candidate_extra[columns].copy()
    return active_extra.merge(additions, on="country_code", how="outer")


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
    active_extra = _build_extra_features(args.as_of, include_candidates=False)
    candidate_extra = _build_extra_features(args.as_of, include_candidates=True)
    active_columns = [c for c in active_extra.columns if c != "country_code"]
    candidate_columns = [c for c in candidate_extra.columns if c != "country_code"]
    added = [c for c in candidate_columns if c not in set(active_columns)]
    candidate_groups = {
        "government_liquidity": [
            feature for feature in added if feature.startswith("govt_")
        ],
        "external_liquidity": [
            feature for feature in added
            if not feature.startswith("govt_")
            and feature != "commodity_export_share_pct"
        ],
        "external_vulnerability": [
            feature for feature in added
            if feature == "commodity_export_share_pct"
        ],
    }
    liquidity_candidates = (
        candidate_groups["government_liquidity"]
        + candidate_groups["external_liquidity"]
    )
    commodity_candidates = candidate_groups["external_vulnerability"]
    liquidity_extra = _with_candidate_columns(
        active_extra, candidate_extra, liquidity_candidates
    )
    commodity_extra = _with_candidate_columns(
        active_extra, candidate_extra, commodity_candidates
    )
    print(f"Active liquidity features: {active_columns}")
    print(f"Candidate liquidity features: {added}")

    production = load_model_artifact()["country_scores"][["country_code", "risk_score"]].copy()
    production["country_code"] = production["country_code"].astype(str).str.upper()

    print("\n=== ACTIVE RETRAIN (production liquidity features) ===")
    control = _train_scores(fsic, weo, mfs, args.as_of, extra_features=active_extra)
    print("\n=== LIQUIDITY CANDIDATE TRAIN (active + liquidity candidates, no commodity) ===")
    liquidity_challenger = _train_scores(
        fsic, weo, mfs, args.as_of, extra_features=liquidity_extra
    )
    print("\n=== COMMODITY CANDIDATE TRAIN (active + commodity vulnerability only) ===")
    commodity_challenger = _train_scores(
        fsic, weo, mfs, args.as_of, extra_features=commodity_extra
    )
    print("\n=== COMBINED CANDIDATE TRAIN (active + all monitored candidates) ===")
    challenger = _train_scores(fsic, weo, mfs, args.as_of, extra_features=candidate_extra)

    faithfulness = _compare(production, control, "active_retrain_vs_production")
    liquidity_effect = _compare(
        control, liquidity_challenger, "liquidity_candidate_vs_active_retrain"
    )
    commodity_effect = _compare(
        control, commodity_challenger, "commodity_candidate_vs_active_retrain"
    )
    effect = _compare(control, challenger, "combined_candidate_vs_active_retrain")
    headline = _compare(production, challenger, "candidate_vs_production")

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
        or min(
            liquidity_effect["rank_correlation_spearman"],
            commodity_effect["rank_correlation_spearman"],
            effect["rank_correlation_spearman"],
        ) < thresholds["rank_correlation_spearman"]
        or max(
            liquidity_effect["risk_tier_changes"],
            commodity_effect["risk_tier_changes"],
            effect["risk_tier_changes"],
        ) > thresholds["risk_tier_changes"]
        or max(
            liquidity_effect["mean_absolute_score_change"],
            commodity_effect["mean_absolute_score_change"],
            effect["mean_absolute_score_change"],
        ) > thresholds["mean_absolute_score_change"]
    )
    stale_active_artifact = (
        faithfulness["mean_absolute_score_change"] > thresholds["mean_absolute_score_change"]
    )

    report = {
        "generated": date.today().isoformat(),
        "cutoff": args.as_of,
        "active_features": active_columns,
        "candidate_features": added,
        "candidate_groups": candidate_groups,
        "feature_directions": {
            "govt_interest_to_revenue": "+1 (higher = riskier)",
            "govt_debt_to_revenue": "+1 (higher = riskier)",
            "govt_revenue_gdp": "-1 (higher revenue capacity = safer)",
            "govt_primary_deficit_gdp": "+1 (higher primary deficit = riskier)",
            "govt_interest_to_revenue_change_3y": "+1 (rising interest burden = riskier)",
            "govt_debt_to_revenue_change_3y": "+1 (rising debt burden = riskier)",
            "govt_primary_deficit_gdp_change_3y": "+1 (worsening primary deficit = riskier)",
            "govt_revenue_gdp_change_3y": "-1 (rising revenue base = safer)",
            "net_iip_gdp": "-1 (higher net creditor = safer)",
            "external_liabilities_gdp": "+1 (higher = riskier)",
            "reserves_to_goods_services_imports": "-1 (more import cover = safer)",
            "gross_external_financing_need_proxy_gdp": "+1 (higher = riskier)",
            "investment_income_debits_to_cxr": "+1 (higher income-service burden = riskier)",
            "reserves_to_current_account_payments": "-1 (more reserve cover = safer)",
            "portfolio_liabilities_gdp": "+1 (larger market funding stock = riskier)",
            "commodity_export_share_pct": "+1 (higher export concentration = riskier)",
            "wb_total_external_debt_service_gni_pct": "+1 (higher debt-service burden = riskier)",
            "wb_ppg_external_debt_service_gdp": "+1 (higher public external debt service = riskier)",
            "wb_public_financing_need_ext_debt_service_proxy_gdp": "+1 (higher public financing pressure = riskier)",
        },
        "governance_thresholds": thresholds,
        "promotion_requires_owner_review": bool(trips_review),
        "review_basis": "candidate_vs_active_retrain",
        "stale_active_artifact_finding": bool(stale_active_artifact),
        "faithfulness_active_retrain_vs_production": faithfulness,
        "scenario_effects_vs_active_retrain": {
            "liquidity": liquidity_effect,
            "commodity": commodity_effect,
            "combined": effect,
        },
        "liquidity_effect_vs_active_retrain": liquidity_effect,
        "commodity_effect_vs_active_retrain": commodity_effect,
        "candidate_effect_vs_active_retrain": effect,
        "headline_candidate_vs_production": headline,
        "notes": [
            "Candidate comparison only. The active serving artifacts are unchanged; "
            "this script never calls model.save().",
            "Both trains reuse the cached crisis classifier, so the classifier "
            "overlay is held fixed and the delta isolates the pillar effect of "
            "the monitored candidate features.",
            "active_retrain_vs_production should be near zero; a non-trivial value means "
            "the retrain itself does not reproduce production and the effect "
            "estimate is confounded. A large value indicates the active "
            "cache/risk_model.pkl is stale relative to the current pipeline and "
            "should be rebuilt independently of this challenger.",
            "FDI/REER columns are omitted from the candidate set until the external "
            "source refresh provides non-null cross-country coverage.",
        ],
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    archive_root = Path(BASE_DIR) / "artifacts" / "snapshots"
    archives = {
        "liquidity": (
            archive_root / f"{args.as_of}-challenger-liquidity",
            liquidity_challenger,
        ),
        "commodity": (
            archive_root / f"{args.as_of}-challenger-commodity",
            commodity_challenger,
        ),
        "combined": (
            archive_root / f"{args.as_of}-challenger-combined",
            challenger,
        ),
    }
    for archive, scores in archives.values():
        archive.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(archive / "challenger_scores.parquet", index=False)

    print("\n=== SUMMARY ===")
    print(f"Faithfulness (active retrain vs production): mean|delta|="
          f"{faithfulness['mean_absolute_score_change']} "
          f"spearman={faithfulness['rank_correlation_spearman']}")
    print(f"Liquidity effect (liquidity vs active retrain): mean|delta|="
          f"{liquidity_effect['mean_absolute_score_change']} "
          f">=1pt={liquidity_effect['countries_moving_at_least_one_point']} "
          f"tier_changes={liquidity_effect['risk_tier_changes']} "
          f"spearman={liquidity_effect['rank_correlation_spearman']}")
    print(f"Commodity effect (commodity vs active retrain): mean|delta|="
          f"{commodity_effect['mean_absolute_score_change']} "
          f">=1pt={commodity_effect['countries_moving_at_least_one_point']} "
          f"tier_changes={commodity_effect['risk_tier_changes']} "
          f"spearman={commodity_effect['rank_correlation_spearman']}")
    print(f"Combined effect (combined vs active retrain): mean|delta|="
          f"{effect['mean_absolute_score_change']} "
          f">=1pt={effect['countries_moving_at_least_one_point']} "
          f"tier_changes={effect['risk_tier_changes']} "
          f"spearman={effect['rank_correlation_spearman']}")
    print(f"Headline (candidate vs production): mean|delta|="
          f"{headline['mean_absolute_score_change']} "
          f"tier_changes={headline['risk_tier_changes']} "
          f"spearman={headline['rank_correlation_spearman']}")
    print(f"Promotion requires owner review: {trips_review}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
