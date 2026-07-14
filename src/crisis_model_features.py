"""Feature contract shared by crisis-model backtests and deployment scoring."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_RISK_DIRECTIONS: dict[str, int] = {
    "gdp_growth_3y_avg": -1,
    "inflation": 1,
    "inflation_change_3y": 1,
    "ca_deficit_severity": 1,
    "current_account_change_3y": -1,
    "govt_debt_gdp": 1,
    "govt_debt_change_3y": 1,
    "fiscal_balance_gdp": -1,
    "govt_interest_to_revenue": 1,
    "govt_interest_to_revenue_change_3y": 1,
    "unemployment": 1,
    "crisis_recency_10y": 1,
    "bank_credit_gdp_change_3y": 1,
    "bank_credit_gdp_gap_10y": 1,
    "bis_private_credit_gdp": 1,
    "bis_bank_credit_gdp": 1,
    "bis_private_credit_to_gdp_gap": 1,
    "bis_private_debt_service_ratio": 1,
    "bis_household_debt_service_ratio": 1,
    "bis_corporate_debt_service_ratio": 1,
    "bis_real_house_price_growth_yoy": 1,
    "bank_credit_to_deposits": 1,
    "broad_money_to_reserves": 1,
    "reserves_months_imports": -1,
    "lending_interest_rate_change_3y": 1,
    "real_interest_rate": 1,
    "commodity_export_concentration": 1,
    "natural_resource_rents_gdp": 1,
    "terms_of_trade_deterioration_3y": 1,
    "commodity_shock_exposure": 1,
    "combined_npl_ratio": 1,
    "capital_adequacy": -1,
    "wb_bank_capital_assets": -1,
    "combined_bank_liquidity": -1,
    "bank_zscore": -1,
    "roe": -1,
    "macro_missing_share": 1,
    "credit_missing_share": 1,
    "banking_missing_share": 1,
}


MACRO_FEATURES = (
    "gdp_growth_3y_avg",
    "inflation",
    "inflation_change_3y",
    "ca_deficit_severity",
    "current_account_change_3y",
    "govt_debt_gdp",
    "govt_debt_change_3y",
    "fiscal_balance_gdp",
    "govt_interest_to_revenue",
    "govt_interest_to_revenue_change_3y",
    "unemployment",
    "crisis_recency_10y",
)

CREDIT_LIQUIDITY_FEATURES = (
    "bank_credit_gdp_change_3y",
    "bank_credit_gdp_gap_10y",
    "bank_credit_to_deposits",
    "broad_money_to_reserves",
    "reserves_months_imports",
    "lending_interest_rate_change_3y",
    "real_interest_rate",
)

EXTERNAL_VULNERABILITY_FEATURES = (
    "commodity_export_concentration",
    "natural_resource_rents_gdp",
    "terms_of_trade_deterioration_3y",
    "commodity_shock_exposure",
)

BANKING_FEATURES = (
    "combined_npl_ratio",
    "capital_adequacy",
    "wb_bank_capital_assets",
    "combined_bank_liquidity",
    "bank_zscore",
    "roe",
)

# Optional official BIS challenger fields. They are not added to the default
# feature sets because their country coverage is materially narrower; the
# dedicated variants make that choice explicit in validation.
BIS_CHALLENGER_FEATURES = (
    "bis_private_credit_gdp",
    "bis_bank_credit_gdp",
    "bis_private_credit_to_gdp_gap",
    "bis_private_debt_service_ratio",
    "bis_household_debt_service_ratio",
    "bis_corporate_debt_service_ratio",
    "bis_real_house_price_growth_yoy",
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "macro": MACRO_FEATURES,
    "macro_credit": (*MACRO_FEATURES, *CREDIT_LIQUIDITY_FEATURES),
    "macro_credit_commodity": (
        *MACRO_FEATURES,
        *CREDIT_LIQUIDITY_FEATURES,
        *EXTERNAL_VULNERABILITY_FEATURES,
    ),
    "full": (*MACRO_FEATURES, *CREDIT_LIQUIDITY_FEATURES, *BANKING_FEATURES),
    "full_commodity": (
        *MACRO_FEATURES,
        *CREDIT_LIQUIDITY_FEATURES,
        *BANKING_FEATURES,
        *EXTERNAL_VULNERABILITY_FEATURES,
    ),
    "full_bis": (
        *MACRO_FEATURES,
        *CREDIT_LIQUIDITY_FEATURES,
        *BANKING_FEATURES,
        *BIS_CHALLENGER_FEATURES,
    ),
    "full_commodity_bis": (
        *MACRO_FEATURES,
        *CREDIT_LIQUIDITY_FEATURES,
        *BANKING_FEATURES,
        *EXTERNAL_VULNERABILITY_FEATURES,
        *BIS_CHALLENGER_FEATURES,
    ),
    "full_with_missingness": (
        *MACRO_FEATURES,
        *CREDIT_LIQUIDITY_FEATURES,
        *BANKING_FEATURES,
        "macro_missing_share",
        "credit_missing_share",
        "banking_missing_share",
    ),
}


def _as_available(frame: pd.DataFrame, feature: str) -> pd.Series:
    raw = frame.get(feature)
    if raw is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(raw, errors="coerce")
    available = frame.get(f"{feature}__available")
    if available is not None:
        values = values.where(pd.Series(available, index=frame.index).fillna(False))
    return values


def _calendar_lag(
    frame: pd.DataFrame,
    values: pd.Series,
    years: int,
) -> pd.Series:
    lookup = pd.Series(
        values.to_numpy(),
        index=pd.MultiIndex.from_arrays(
            [frame["country_code"], frame["forecast_origin_year"]]
        ),
    )
    keys = pd.MultiIndex.from_arrays(
        [frame["country_code"], frame["forecast_origin_year"] - years]
    )
    return pd.Series(lookup.reindex(keys).to_numpy(), index=frame.index, dtype=float)


def _calendar_mean(frame: pd.DataFrame, values: pd.Series, years: int) -> pd.Series:
    parts = [values]
    parts.extend(_calendar_lag(frame, values, lag) for lag in range(1, years))
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def _years_since_crisis(frame: pd.DataFrame, labels) -> pd.Series:
    periods = getattr(labels, "crises", {})

    def one(code: str, origin: int) -> float:
        completed = [end for _, end in periods.get(code, []) if end < origin]
        if not completed:
            return 25.0
        return min(25.0, float(origin - max(completed)))

    return pd.Series(
        [
            one(str(code), int(origin))
            for code, origin in zip(
                frame["country_code"], frame["forecast_origin_year"]
            )
        ],
        index=frame.index,
        dtype=float,
    )


def derive_crisis_model_features(panel: pd.DataFrame, labels) -> pd.DataFrame:
    """Derive a compact governed feature set using calendar-year lags only."""

    required = {"country_code", "forecast_origin_year", "crisis_target"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Crisis panel missing required columns: {sorted(missing)}")
    frame = panel.sort_values(
        ["country_code", "forecast_origin_year"]
    ).reset_index(drop=True).copy()

    raw_features = {
        feature
        for feature in (
            "gdp_growth",
            "inflation",
            "current_account_gdp",
            "govt_debt_gdp",
            "fiscal_balance_gdp",
            "primary_balance_gdp",
            "unemployment",
            "govt_revenue_gdp",
            "capital_adequacy",
            "npl_ratio",
            "roe",
            "liquid_assets_total",
            "bank_credit_gdp_change_3y",
            "bank_credit_gdp_gap_10y",
            "bank_credit_to_deposits",
            "bank_zscore",
            "wb_bank_npl_ratio",
            "wb_bank_capital_assets",
            "wb_bank_liquid_reserves_assets",
            "broad_money_to_reserves",
            "reserves_months_imports",
            "lending_interest_rate_change_3y",
            "real_interest_rate",
            "commodity_export_concentration",
            "natural_resource_rents_gdp",
            "terms_of_trade_deterioration_3y",
            "commodity_shock_exposure",
            *BIS_CHALLENGER_FEATURES,
        )
    }
    values = {feature: _as_available(frame, feature) for feature in raw_features}

    frame["gdp_growth_3y_avg"] = _calendar_mean(frame, values["gdp_growth"], 3)
    frame["inflation"] = values["inflation"]
    frame["inflation_change_3y"] = values["inflation"] - _calendar_lag(
        frame, values["inflation"], 3
    )
    frame["ca_deficit_severity"] = (-values["current_account_gdp"]).clip(lower=0)
    frame["current_account_change_3y"] = values[
        "current_account_gdp"
    ] - _calendar_lag(frame, values["current_account_gdp"], 3)
    frame["govt_debt_gdp"] = values["govt_debt_gdp"]
    frame["govt_debt_change_3y"] = values["govt_debt_gdp"] - _calendar_lag(
        frame, values["govt_debt_gdp"], 3
    )
    frame["fiscal_balance_gdp"] = values["fiscal_balance_gdp"]
    revenue = values["govt_revenue_gdp"].where(values["govt_revenue_gdp"] > 0)
    interest_gdp = values["primary_balance_gdp"] - values["fiscal_balance_gdp"]
    frame["govt_interest_to_revenue"] = interest_gdp / revenue * 100
    frame["govt_interest_to_revenue_change_3y"] = frame[
        "govt_interest_to_revenue"
    ] - _calendar_lag(frame, frame["govt_interest_to_revenue"], 3)
    frame["unemployment"] = values["unemployment"]
    frame["crisis_recency_10y"] = (
        10.0 - _years_since_crisis(frame, labels)
    ).clip(lower=0)

    for feature in CREDIT_LIQUIDITY_FEATURES:
        frame[feature] = values[feature]
    for feature in EXTERNAL_VULNERABILITY_FEATURES:
        frame[feature] = values[feature]
    for feature in BIS_CHALLENGER_FEATURES:
        frame[feature] = values[feature]
    frame["combined_npl_ratio"] = values["npl_ratio"].combine_first(
        values["wb_bank_npl_ratio"]
    )
    frame["capital_adequacy"] = values["capital_adequacy"]
    frame["wb_bank_capital_assets"] = values["wb_bank_capital_assets"]
    frame["combined_bank_liquidity"] = values["liquid_assets_total"].combine_first(
        values["wb_bank_liquid_reserves_assets"]
    )
    frame["bank_zscore"] = values["bank_zscore"]
    frame["roe"] = values["roe"]

    frame["macro_missing_share"] = frame[list(MACRO_FEATURES)].isna().mean(axis=1)
    frame["credit_missing_share"] = frame[list(CREDIT_LIQUIDITY_FEATURES)].isna().mean(
        axis=1
    )
    frame["banking_missing_share"] = frame[list(BANKING_FEATURES)].isna().mean(axis=1)
    return frame


def model_matrix(
    feature_frame: pd.DataFrame,
    feature_names: Iterable[str],
    *,
    risk_oriented: bool = True,
) -> pd.DataFrame:
    names = list(feature_names)
    missing = sorted(set(names).difference(feature_frame.columns))
    if missing:
        raise ValueError(f"Missing crisis-model features: {missing}")
    values = feature_frame[names].apply(pd.to_numeric, errors="coerce").copy()
    if risk_oriented:
        unknown = sorted(set(names).difference(FEATURE_RISK_DIRECTIONS))
        if unknown:
            raise ValueError(f"Missing governed risk directions: {unknown}")
        for feature in names:
            values[feature] *= FEATURE_RISK_DIRECTIONS[feature]
    return values


__all__ = [
    "BIS_CHALLENGER_FEATURES",
    "BANKING_FEATURES",
    "CREDIT_LIQUIDITY_FEATURES",
    "FEATURE_RISK_DIRECTIONS",
    "FEATURE_SETS",
    "EXTERNAL_VULNERABILITY_FEATURES",
    "MACRO_FEATURES",
    "derive_crisis_model_features",
    "model_matrix",
]
