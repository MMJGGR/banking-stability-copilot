"""Canonical, Streamlit-free evidence inventory for active model inputs.

Active membership and pillar assignment come only from the persisted loading
maps.  The metadata catalog supplies presentation context, but never decides
whether a feature affects the served score.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


EVIDENCE_API_VERSION = 2


STATUS_REPORTED_DERIVED = "Reported/derived"
STATUS_IMPUTED = "Imputed for scoring"
STATUS_UNAVAILABLE = "Unavailable"
ACTIVE_MODEL_ROLE = "Active score input"

PILLAR_LABELS = {
    "economic": "Operating environment",
    "industry": "Banking system",
}


@dataclass(frozen=True)
class FeatureEvidenceMetadata:
    """Stable presentation metadata independent of active model membership."""

    label: str
    unit: str
    source_family: str


@dataclass(frozen=True)
class DirectActiveInputCoverage:
    """Direct, non-imputed coverage across the active loading-map universe."""

    numerator: int
    denominator: int

    @property
    def ratio(self) -> float | None:
        if self.denominator == 0:
            return None
        return self.numerator / self.denominator


@dataclass(frozen=True)
class ActiveInputInventory:
    """Country-level evidence rows plus their direct-coverage summary."""

    country_code: str
    rows: pd.DataFrame
    coverage: DirectActiveInputCoverage


def _metadata(label: str, unit: str, source_family: str) -> FeatureEvidenceMetadata:
    return FeatureEvidenceMetadata(label, unit, source_family)


# Presentation context for the 40 inputs in the active 2026-06-30 loading maps.
# Future features remain displayable through ``feature_metadata`` fallbacks.
FEATURE_EVIDENCE_METADATA: dict[str, FeatureEvidenceMetadata] = {
    "gdp_growth": _metadata("Real GDP growth", "%", "IMF WEO"),
    "inflation": _metadata("Consumer price inflation", "%", "IMF WEO"),
    "current_account_gdp": _metadata(
        "Current-account balance / GDP", "% of GDP", "IMF WEO"
    ),
    "gdp_per_capita": _metadata(
        "GDP per capita", "USD per person", "IMF WEO"
    ),
    "govt_debt_gdp": _metadata(
        "General government gross debt / GDP", "% of GDP", "IMF WEO"
    ),
    "fiscal_balance_gdp": _metadata(
        "General government fiscal balance / GDP", "% of GDP", "IMF WEO"
    ),
    "unemployment": _metadata(
        "Unemployment rate", "% of labor force", "IMF WEO"
    ),
    "credit_to_gdp_relative": _metadata(
        "Private credit / GDP versus cross-country median",
        "percentage points",
        "IMF MFS + IMF WEO (derived)",
    ),
    "sovereign_liability_to_reserves": _metadata(
        "Central-bank nonresident liabilities / reserve assets",
        "ratio (x)",
        "IMF MFS (derived)",
    ),
    "inflation_differential_3yr": _metadata(
        "Three-year inflation differential versus G7",
        "percentage points",
        "IMF WEO (derived)",
    ),
    "interest_cost_gdp": _metadata(
        "Implied interest balance / GDP (fiscal less primary)",
        "% of GDP",
        "IMF WEO (derived)",
    ),
    "interest_cost_trend_3yr": _metadata(
        "Three-year change in implied interest balance",
        "percentage points",
        "IMF WEO (derived)",
    ),
    "credit_growth_3yr": _metadata(
        "Private credit growth, three years", "%", "IMF MFS (derived)"
    ),
    "m2_to_reserves": _metadata(
        "Broad money / central-bank reserves", "ratio (x)", "IMF MFS (derived)"
    ),
    "ca_deficit_severity": _metadata(
        "Current-account deficit severity", "% of GDP", "IMF WEO (derived)"
    ),
    "tot_deterioration_3yr": _metadata(
        "Cumulative terms-of-trade change, three years",
        "%",
        "IMF WEO (derived)",
    ),
    "voice_accountability": _metadata(
        "Voice and accountability", "score (0-100)", "World Bank WGI"
    ),
    "political_stability": _metadata(
        "Political stability", "score (0-100)", "World Bank WGI"
    ),
    "govt_interest_to_revenue": _metadata(
        "Implied government interest burden / revenue",
        "% of revenue",
        "IMF WEO (derived)",
    ),
    "govt_debt_to_revenue": _metadata(
        "General government gross debt / revenue",
        "% of revenue",
        "IMF WEO (derived)",
    ),
    "net_iip_gdp": _metadata(
        "Net international investment position / GDP",
        "% of GDP",
        "IMF IIP + IMF WEO (derived)",
    ),
    "external_liabilities_gdp": _metadata(
        "External liabilities / GDP",
        "% of GDP",
        "IMF IIP + IMF WEO (derived)",
    ),
    "reserves_to_goods_services_imports": _metadata(
        "Reserve assets / annual goods and services imports",
        "% of annual imports",
        "IMF IIP + IMF BOP (derived)",
    ),
    "gross_external_financing_need_proxy_gdp": _metadata(
        "Gross external financing need proxy / GDP",
        "% of GDP",
        "IMF BOP + IMF WEO (derived proxy)",
    ),
    "investment_income_debits_to_cxr": _metadata(
        "Investment-income debits / current-account receipts",
        "% of current-account receipts",
        "IMF BOP (derived)",
    ),
    "govt_revenue_gdp": _metadata(
        "General government revenue / GDP", "% of GDP", "IMF WEO"
    ),
    "govt_interest_to_revenue_change_3y": _metadata(
        "Interest burden / revenue change, three years",
        "percentage points",
        "IMF WEO (derived)",
    ),
    "govt_debt_to_revenue_change_3y": _metadata(
        "Gross debt / revenue change, three years",
        "percentage points",
        "IMF WEO (derived)",
    ),
    "govt_revenue_gdp_change_3y": _metadata(
        "Government revenue / GDP change, three years",
        "percentage points",
        "IMF WEO (derived)",
    ),
    "capital_adequacy": _metadata(
        "Regulatory capital / risk-weighted assets",
        "% of risk-weighted assets",
        "IMF FSIC",
    ),
    "npl_ratio": _metadata(
        "Nonperforming loans / gross loans", "% of gross loans", "IMF FSIC"
    ),
    "roa": _metadata("Return on assets", "% of assets", "IMF FSIC"),
    "liquid_assets_st_liab": _metadata(
        "Liquid assets / short-term liabilities",
        "% of short-term liabilities",
        "IMF FSIC",
    ),
    "liquid_assets_total": _metadata(
        "Liquid assets / total assets", "% of total assets", "IMF FSIC"
    ),
    "customer_deposits_loans": _metadata(
        "Customer deposits / total loans", "% of total loans", "IMF FSIC"
    ),
    "npl_provisions": _metadata(
        "Provisions / nonperforming loans", "% of NPLs", "IMF FSIC"
    ),
    "loan_concentration": _metadata(
        "Loan concentration by economic activity", "%", "IMF FSIC"
    ),
    "sovereign_exposure_ratio": _metadata(
        "Bank claims on government / banking assets",
        "% of banking assets",
        "IMF MFS / FSIBSIS (derived or proxy)",
    ),
    "bank_liability_to_nfa": _metadata(
        "Bank nonresident liabilities / net foreign assets",
        "ratio (x)",
        "IMF MFS (derived)",
    ),
    "years_since_banking_crisis": _metadata(
        "Years since last completed systemic banking crisis",
        "years (capped at 25)",
        "IMF WP/26/94 crisis episodes",
    ),
}


REGISTRY_COLUMNS = [
    "feature",
    "label",
    "pillar",
    "pillar_label",
    "model_role",
    "loading",
    "unit",
    "source_family",
]

INVENTORY_COLUMNS = REGISTRY_COLUMNS + [
    "reported_value",
    "imputed_value",
    "value",
    "period",
    "status",
    "evidence_type",
    "is_direct",
]


def feature_metadata(feature: str) -> FeatureEvidenceMetadata:
    """Return explicit metadata or an honest presentation fallback."""

    code = str(feature)
    explicit = FEATURE_EVIDENCE_METADATA.get(code)
    if explicit is not None:
        return explicit
    return FeatureEvidenceMetadata(
        label=code.replace("_", " ").strip().title() or "Unnamed feature",
        unit="Source-defined",
        source_family="Unmapped source family",
    )


def build_active_feature_registry(pca_info: Mapping[str, Any] | None) -> pd.DataFrame:
    """Build the active feature registry from persisted loading maps only."""

    if pca_info is None:
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    if not isinstance(pca_info, Mapping):
        raise TypeError("pca_info must be a mapping or None")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pillar, key in (
        ("economic", "economic_loadings"),
        ("industry", "industry_loadings"),
    ):
        loadings = pca_info.get(key, {})
        if loadings is None:
            loadings = {}
        if not isinstance(loadings, Mapping):
            raise TypeError(f"pca_info[{key!r}] must be a mapping")
        for raw_feature, raw_loading in loadings.items():
            feature = str(raw_feature)
            if feature in seen:
                raise ValueError(
                    f"Active feature {feature!r} appears in both pillar loading maps"
                )
            seen.add(feature)
            metadata = feature_metadata(feature)
            try:
                loading = float(raw_loading)
            except (TypeError, ValueError):
                loading = np.nan
            rows.append(
                {
                    "feature": feature,
                    "label": metadata.label,
                    "pillar": pillar,
                    "pillar_label": PILLAR_LABELS[pillar],
                    "model_role": ACTIVE_MODEL_ROLE,
                    "loading": loading,
                    "unit": metadata.unit,
                    "source_family": metadata.source_family,
                }
            )
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def build_active_feature_coverage(
    scores: pd.DataFrame,
    model_features: pd.DataFrame,
    pca_info: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """Summarize direct input coverage across distinct served countries.

    Coverage uses the country universe in ``scores``. Duplicate feature rows do
    not inflate the numerator: a country is counted once when any of its rows
    carries a non-null value for the feature.
    """

    if not isinstance(scores, pd.DataFrame):
        raise TypeError("scores must be a pandas DataFrame")
    if not isinstance(model_features, pd.DataFrame):
        raise TypeError("model_features must be a pandas DataFrame")
    if "country_code" not in scores.columns:
        raise ValueError("scores must contain country_code")
    if "country_code" not in model_features.columns:
        raise ValueError("model_features must contain country_code")

    registry = build_active_feature_registry(pca_info)
    served_codes = set(scores["country_code"].astype(str).str.upper())
    country_total = len(served_codes)
    normalized = model_features.copy()
    normalized["_coverage_country_code"] = (
        normalized["country_code"].astype(str).str.upper()
    )
    normalized = normalized[
        normalized["_coverage_country_code"].isin(served_codes)
    ]

    rows: list[dict[str, Any]] = []
    for registry_row in registry.to_dict(orient="records"):
        feature = registry_row["feature"]
        direct_countries = (
            int(
                normalized.loc[
                    normalized[feature].notna(),
                    "_coverage_country_code",
                ].nunique()
            )
            if feature in normalized.columns
            else 0
        )
        rows.append(
            {
                **registry_row,
                "direct_countries": direct_countries,
                "direct_coverage": (
                    direct_countries / country_total if country_total else np.nan
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[*REGISTRY_COLUMNS, "direct_countries", "direct_coverage"],
    )


def _available(value: Any) -> bool:
    if value is None:
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return True
    return bool(not missing) if isinstance(missing, (bool, np.bool_)) else True


def _country_row(
    frame: pd.DataFrame | None,
    country_code: str,
    *,
    frame_name: str,
) -> pd.Series | None:
    if frame is None:
        return None
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{frame_name} must be a pandas DataFrame or None")
    if frame.empty:
        return None

    code = str(country_code).upper()
    if "country_code" in frame.columns:
        codes = frame["country_code"].astype(str).str.upper()
        matches = frame.loc[codes == code]
    else:
        codes = pd.Index(frame.index.astype(str).str.upper())
        matches = frame.loc[codes == code]
    if len(matches) > 1:
        raise ValueError(f"{frame_name} contains duplicate rows for {code}")
    return None if matches.empty else matches.iloc[0]


def _row_value(row: pd.Series | None, column: str) -> Any:
    if row is None or column not in row.index:
        return np.nan
    return row[column]


def _period_value(
    feature: str,
    raw_row: pd.Series | None,
    imputed_row: pd.Series | None,
) -> Any:
    column = f"{feature}_year"
    value = _row_value(raw_row, column)
    if not _available(value):
        value = _row_value(imputed_row, column)
    if not _available(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return int(value) if float(value).is_integer() else float(value)
    return str(value)


def _evidence_type(source_family: str, status: str) -> str:
    """Describe provenance at the finest level supported by the artifact.

    The promoted cross-section does not preserve upstream observation flags, so
    it would be misleading to infer reported, estimated, projected, or
    carried-forward status from a year alone. Derived and proxy roles are,
    however, explicit in the maintained source-family metadata.
    """
    if status == STATUS_IMPUTED:
        return "Imputed model value"
    if status == STATUS_UNAVAILABLE:
        return "Unavailable"
    lineage = str(source_family).lower()
    if "proxy" in lineage:
        return "Derived proxy"
    if "derived" in lineage:
        return "Derived feature"
    return "Source value; upstream status not retained"


def build_active_input_inventory(
    country_code: str,
    model_features: pd.DataFrame | None,
    pca_info: Mapping[str, Any] | None,
    imputed_features: pd.DataFrame | None = None,
) -> ActiveInputInventory:
    """Build one country's complete active-input evidence inventory.

    A non-null value in ``model_features`` is direct model evidence and is
    labelled ``Reported/derived`` because the cross-section does not preserve a
    finer reported-versus-derived status.  The optional natural-unit imputation
    sidecar supplies the scoring value only when the raw feature is missing.
    """

    code = str(country_code).upper()
    registry = build_active_feature_registry(pca_info)
    raw_row = _country_row(model_features, code, frame_name="model_features")
    imputed_row = _country_row(
        imputed_features,
        code,
        frame_name="imputed_features",
    )

    evidence_rows: list[dict[str, Any]] = []
    direct_count = 0
    for registry_row in registry.to_dict(orient="records"):
        feature = registry_row["feature"]
        reported_value = _row_value(raw_row, feature)
        imputed_value = _row_value(imputed_row, feature)
        if _available(reported_value):
            value = reported_value
            status = STATUS_REPORTED_DERIVED
            is_direct = True
            direct_count += 1
        elif _available(imputed_value):
            value = imputed_value
            status = STATUS_IMPUTED
            is_direct = False
        else:
            value = np.nan
            status = STATUS_UNAVAILABLE
            is_direct = False
        evidence_rows.append(
            {
                **registry_row,
                "reported_value": reported_value,
                "imputed_value": imputed_value,
                "value": value,
                "period": _period_value(feature, raw_row, imputed_row),
                "status": status,
                "evidence_type": _evidence_type(
                    registry_row["source_family"],
                    status,
                ),
                "is_direct": is_direct,
            }
        )

    rows = pd.DataFrame(evidence_rows, columns=INVENTORY_COLUMNS)
    if not rows.empty:
        # Keep unknown periods as ``None`` rather than allowing pandas to
        # coerce mixed integer/unknown years to floating-point values.
        rows["period"] = pd.Series(
            [row["period"] for row in evidence_rows],
            dtype="object",
        )
    coverage = DirectActiveInputCoverage(
        numerator=direct_count,
        denominator=len(registry),
    )
    return ActiveInputInventory(code, rows, coverage)


__all__ = [
    "ACTIVE_MODEL_ROLE",
    "ActiveInputInventory",
    "DirectActiveInputCoverage",
    "EVIDENCE_API_VERSION",
    "FEATURE_EVIDENCE_METADATA",
    "FeatureEvidenceMetadata",
    "STATUS_IMPUTED",
    "STATUS_REPORTED_DERIVED",
    "STATUS_UNAVAILABLE",
    "build_active_feature_registry",
    "build_active_feature_coverage",
    "build_active_input_inventory",
    "feature_metadata",
]
