"""Shared dashboard utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Weighted peer dimensions. The first group keeps peers close on model risk;
# the second prevents high-income/systemically large economies from defaulting
# to small countries with coincidentally similar two-pillar scores; the third
# adds banking, government-liquidity, and external-liquidity structure.
PEER_FEATURE_WEIGHTS = {
    "risk_score": 1.00,
    "economic_pillar": 0.90,
    "industry_pillar": 0.90,
    "data_coverage": 0.50,
    "nominal_gdp": 2.00,
    "gdp_per_capita": 2.00,
    "capital_adequacy": 0.45,
    "npl_ratio": 0.55,
    "liquid_assets_st_liab": 0.35,
    "customer_deposits_loans": 0.35,
    "govt_interest_to_revenue": 0.75,
    "govt_debt_to_revenue": 0.75,
    "govt_revenue_gdp": 0.40,
    "govt_primary_deficit_gdp": 0.45,
    "govt_interest_to_revenue_change_3y": 0.35,
    "govt_debt_to_revenue_change_3y": 0.35,
    "govt_primary_deficit_gdp_change_3y": 0.30,
    "govt_revenue_gdp_change_3y": 0.30,
    "net_iip_gdp": 0.65,
    "external_liabilities_gdp": 0.55,
    "reserves_to_goods_services_imports": 0.65,
    "reserves_to_current_account_payments": 0.55,
    "gross_external_financing_need_proxy_gdp": 0.75,
    "portfolio_liabilities_gdp": 0.45,
    "commodity_export_share_pct": 0.35,
    "wb_total_external_debt_service_gni_pct": 0.35,
    "wb_ppg_external_debt_service_gdp": 0.35,
    "wb_public_financing_need_ext_debt_service_proxy_gdp": 0.35,
    "current_account_gdp": 0.35,
    "govt_debt_gdp": 0.35,
}


def find_peers(
    target_country: str,
    scores_df: pd.DataFrame,
    n_peers: int = 4,
    feature_values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return model peers using score, scale, banking, and liquidity similarity.

    The previous implementation used Euclidean distance on only two columns
    (economic and industry pillars). That produced implausible defaults for
    large advanced economies because a small economy with similar pillar scores
    could appear closer than a structurally comparable banking system.

    This function still degrades gracefully when only ``scores_df`` is
    available, but when ``feature_values`` is supplied it also considers GDP
    scale, GDP per capita, selected banking ratios, government-liquidity
    affordability, and external-liquidity stress.
    """
    if scores_df is None or scores_df.empty or "country_code" not in scores_df:
        return pd.DataFrame()

    target_country = str(target_country).upper()
    scores = scores_df.copy()
    scores["country_code"] = scores["country_code"].astype(str).str.upper()
    if target_country not in set(scores["country_code"]):
        return pd.DataFrame()

    peer_frame = _merge_peer_features(scores, feature_values)
    target_row = peer_frame[peer_frame["country_code"] == target_country].iloc[0]
    candidates = peer_frame[peer_frame["country_code"] != target_country].copy()
    if candidates.empty:
        return pd.DataFrame()

    if "data_coverage" in candidates:
        candidates = candidates[
            pd.to_numeric(candidates["data_coverage"], errors="coerce").fillna(0) >= 0.75
        ]
        if candidates.empty:
            candidates = peer_frame[peer_frame["country_code"] != target_country].copy()

    candidates, peer_basis = _apply_structural_peer_filters(
        target_row,
        candidates,
        n_peers=n_peers,
    )

    distance_columns = [
        column for column in PEER_FEATURE_WEIGHTS
        if column in peer_frame.columns
        and (column in candidates.columns)
        and (
            pd.notna(target_row.get(column))
            or pd.to_numeric(candidates[column], errors="coerce").notna().any()
        )
    ]
    if not distance_columns:
        return pd.DataFrame()

    distance_frame = pd.concat(
        [target_row.to_frame().T, candidates],
        ignore_index=True,
    )
    distances = _weighted_robust_distances(
        distance_frame,
        distance_columns,
        target_index=0,
    )

    peers = candidates.copy()
    peers["distance"] = distances
    peers["peer_basis"] = peer_basis
    return peers.sort_values(["distance", "country_name"]).head(n_peers)


def _merge_peer_features(
    scores: pd.DataFrame,
    feature_values: pd.DataFrame | None,
) -> pd.DataFrame:
    if feature_values is None or feature_values.empty or "country_code" not in feature_values:
        return scores

    extra_columns = [
        column for column in PEER_FEATURE_WEIGHTS
        if column in feature_values.columns and column not in scores.columns
    ]
    if not extra_columns:
        return scores

    features = feature_values[["country_code", *extra_columns]].copy()
    features["country_code"] = features["country_code"].astype(str).str.upper()
    features = features.drop_duplicates("country_code")
    return scores.merge(features, on="country_code", how="left")


def _apply_structural_peer_filters(
    target_row: pd.Series,
    candidates: pd.DataFrame,
    n_peers: int,
) -> tuple[pd.DataFrame, str]:
    filtered = candidates.copy()
    applied = []

    target_income = _numeric_scalar(target_row.get("gdp_per_capita"))
    if target_income is not None and "gdp_per_capita" in filtered.columns:
        income = pd.to_numeric(filtered["gdp_per_capita"], errors="coerce")
        income_filtered = filtered[income.between(target_income * 0.35, target_income * 2.50)]
        if len(income_filtered) >= max(4, min(n_peers, 6)):
            filtered = income_filtered
            applied.append("income")

    target_size = _numeric_scalar(target_row.get("nominal_gdp"))
    if target_size is not None and "nominal_gdp" in filtered.columns:
        size = pd.to_numeric(filtered["nominal_gdp"], errors="coerce")
        scale_filtered = filtered[size.between(target_size / 15.0, target_size * 15.0)]
        if len(scale_filtered) >= max(4, min(n_peers, 6)):
            filtered = scale_filtered
            applied.append("scale")

    basis = " + ".join(applied) if applied else "global model-feature distance"
    return filtered, basis


def _weighted_robust_distances(
    frame: pd.DataFrame,
    columns: list[str],
    target_index: int = 0,
) -> np.ndarray:
    numeric = pd.DataFrame(index=frame.index)
    missing = pd.DataFrame(index=frame.index)

    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        if column in {"nominal_gdp", "gdp_per_capita"}:
            series = np.log10(series.clip(lower=1))
        lower, upper = series.quantile([0.02, 0.98])
        if pd.notna(lower) and pd.notna(upper) and lower < upper:
            series = series.clip(lower, upper)
        numeric[column] = series
        missing[column] = series.isna()

    medians = numeric.median(numeric_only=True)
    iqr = (numeric.quantile(0.75) - numeric.quantile(0.25)).replace(0, np.nan)
    scaled = ((numeric - medians) / iqr).fillna(0.0)

    target = scaled.iloc[target_index]
    weights = pd.Series(
        {column: PEER_FEATURE_WEIGHTS[column] for column in columns},
        dtype=float,
    )
    distances = np.sqrt(((scaled.iloc[1:] - target) ** 2).mul(weights).sum(axis=1))

    # Small penalty for asymmetric missingness so a sparse country is not
    # treated as a perfect match after median fill.
    target_missing = missing.iloc[target_index]
    missing_penalty = (
        missing.iloc[1:].ne(target_missing, axis=1).sum(axis=1).astype(float) * 0.03
    )
    return (distances + missing_penalty).to_numpy()


def _numeric_scalar(value) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric <= 0:
        return None
    return numeric


def driver_metric_value(
    summary: dict,
    score_row: pd.Series,
    column: str,
    drivers: list[dict] | None = None,
    pipeline=None,
) -> float | None:
    """Return Score Drivers summary metrics with legacy-artifact fallbacks."""
    value = summary.get(column) if isinstance(summary, dict) else None
    try:
        missing = value is None or pd.isna(value)
    except (TypeError, ValueError):
        missing = value is None
    if missing and column in score_row.index:
        value = score_row.get(column)
    try:
        if value is None or pd.isna(value):
            raise ValueError("missing")
        return float(value)
    except (TypeError, ValueError):
        pass

    drivers = drivers or []
    if column in {"critical_missing_share", "critical_penalty"}:
        critical_drivers = [
            driver for driver in drivers
            if driver.get("is_critical")
        ]
        if critical_drivers:
            missing_share = sum(
                1 for driver in critical_drivers
                if driver.get("is_imputed")
            ) / len(critical_drivers)
        else:
            missing_share = 0.0
        if column == "critical_missing_share":
            return float(missing_share)
        max_penalty = getattr(pipeline, "critical_missing_max_penalty", 0.0)
        return float(missing_share * max_penalty)

    if column == "crisis_uplift":
        return 0.0

    return None
