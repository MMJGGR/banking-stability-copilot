"""Transparent mechanism-level evidence and alert-policy foundation.

The production pillar and crisis models use several overlapping feature
contracts.  This module gives those columns one analytical home without
changing either model:

* each governed feature belongs to one primary crisis mechanism;
* exact source/derivation substitutes are aliases for one signal and therefore
  cannot receive duplicate weight;
* observed values are converted to risk-oriented empirical percentiles;
* missing observations reduce evidence confidence, never the risk score; and
* the Amber/Red policy keeps probability, corroboration, confidence, and
  persistence as separate, auditable tests.

Inputs are the canonical feature values produced before pipeline-specific
scaling.  The sole transform-specific direction in the pillar contract,
``loan_concentration``, is therefore restored to its raw economic direction:
greater concentration means greater evidence of risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.crisis_model_features import (
    FEATURE_RISK_DIRECTIONS as CRISIS_FEATURE_RISK_DIRECTIONS,
)
from src.pillar_pipeline import (
    FEATURE_RISK_DIRECTIONS as PILLAR_FEATURE_RISK_DIRECTIONS,
)


def _canonical_risk_directions() -> dict[str, int]:
    """Merge both governed feature contracts and reject conflicting signs."""

    merged: dict[str, int] = {
        feature: int(direction)
        for feature, direction in PILLAR_FEATURE_RISK_DIRECTIONS.items()
    }
    for feature, direction in CRISIS_FEATURE_RISK_DIRECTIONS.items():
        sign = int(direction)
        if feature in merged and merged[feature] != sign:
            raise ValueError(
                f"Conflicting risk direction for {feature!r}: "
                f"{merged[feature]} versus {sign}"
            )
        merged[feature] = sign
    return merged


CANONICAL_RISK_DIRECTIONS = _canonical_risk_directions()
# ``pillar_pipeline`` transforms raw concentration to -log1p(abs(value)) before
# applying its direction.  Evidence is calculated from the raw feature frame,
# where greater concentration is economically riskier.
CANONICAL_RISK_DIRECTIONS["loan_concentration"] = 1
MISSINGNESS_FEATURES = frozenset(
    feature
    for feature in CANONICAL_RISK_DIRECTIONS
    if feature.endswith("_missing_share")
)


@dataclass(frozen=True)
class SignalSpec:
    """One analytical signal, optionally supplied by ordered feature aliases."""

    key: str
    label: str
    features: tuple[str, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class MechanismSpec:
    """A governed group of economically related crisis signals."""

    key: str
    label: str
    description: str
    signals: tuple[SignalSpec, ...]
    corroboration_eligible: bool = True


def _signal(
    key: str,
    label: str,
    *features: str,
    weight: float = 1.0,
) -> SignalSpec:
    return SignalSpec(key=key, label=label, features=tuple(features), weight=weight)


# Ordered for stable display and deterministic tie-breaking.  Aliases are
# ordered from the most directly modelled/combined field to the closest
# production alternative.  Every feature occurs in exactly one signal.
MECHANISM_TAXONOMY: tuple[MechanismSpec, ...] = (
    MechanismSpec(
        key="credit_property",
        label="Credit and property cycle",
        description="Credit depth, acceleration, debt-service pressure, and property exposure.",
        signals=(
            _signal(
                "credit_depth",
                "Credit depth",
                "credit_to_gdp",
                "bis_private_credit_gdp",
                "bis_bank_credit_gdp",
            ),
            _signal(
                "credit_gap",
                "Credit-to-GDP gap",
                "bis_private_credit_to_gdp_gap",
                "bank_credit_gdp_gap_10y",
            ),
            _signal(
                "credit_acceleration",
                "Credit acceleration",
                "bank_credit_gdp_change_3y",
                "credit_growth_3yr",
            ),
            _signal(
                "relative_credit_depth",
                "Relative credit depth",
                "credit_to_gdp_relative",
            ),
            _signal(
                "debt_service",
                "Private debt-service burden",
                "bis_private_debt_service_ratio",
                "bis_household_debt_service_ratio",
                "bis_corporate_debt_service_ratio",
                "debt_service_gdp",
            ),
            _signal(
                "property_credit_growth",
                "Property-credit growth",
                "real_estate_credit_growth_3yr",
            ),
            _signal(
                "property_price_growth",
                "Residential property-price growth",
                "bis_real_house_price_growth_yoy",
            ),
            _signal("property_exposure", "Property-loan exposure", "real_estate_loans"),
        ),
    ),
    MechanismSpec(
        key="bank_solvency_asset_quality",
        label="Bank solvency and asset quality",
        description=(
            "Capital buffers, impaired assets, profitability, and "
            "balance-sheet resilience."
        ),
        signals=(
            _signal(
                "nonperforming_loans",
                "Nonperforming loans",
                "combined_npl_ratio",
                "npl_ratio",
            ),
            _signal("regulatory_capital", "Regulatory capital", "capital_adequacy"),
            _signal("capital_assets", "Capital-to-assets", "wb_bank_capital_assets"),
            _signal("tier1_capital", "Tier 1 capital", "tier1_capital"),
            _signal("npl_provisions", "NPL provisioning", "npl_provisions"),
            _signal("return_on_equity", "Return on equity", "roe"),
            _signal("return_on_assets", "Return on assets", "roa"),
            _signal("bank_zscore", "Bank Z-score", "bank_zscore"),
            _signal("loan_concentration", "Loan concentration", "loan_concentration"),
        ),
    ),
    MechanismSpec(
        key="funding_liquidity",
        label="Bank funding and liquidity",
        description="Deposit funding, credit-to-deposit pressure, and liquid-asset buffers.",
        signals=(
            _signal(
                "bank_liquid_assets",
                "Bank liquid-asset buffer",
                "combined_bank_liquidity",
                "liquid_assets_total",
            ),
            _signal(
                "short_term_liquidity",
                "Liquid assets to short-term liabilities",
                "liquid_assets_st_liab",
            ),
            _signal(
                "deposit_funding",
                "Customer deposit funding",
                "customer_deposits_loans",
            ),
            _signal(
                "credit_to_deposits",
                "Bank credit to deposits",
                "bank_credit_to_deposits",
            ),
        ),
    ),
    MechanismSpec(
        key="sovereign_liquidity_market_access",
        label="Sovereign liquidity and market access",
        description=(
            "Debt affordability, fiscal flow pressure, financing needs, and "
            "bank-sovereign exposure."
        ),
        signals=(
            _signal("government_debt", "Government debt", "govt_debt_gdp"),
            _signal("government_debt_change", "Government debt change", "govt_debt_change_3y"),
            _signal("fiscal_balance", "Fiscal balance", "fiscal_balance_gdp"),
            _signal(
                "interest_to_revenue",
                "Interest to revenue",
                "govt_interest_to_revenue",
            ),
            _signal(
                "interest_to_revenue_change",
                "Interest-to-revenue change",
                "govt_interest_to_revenue_change_3y",
            ),
            _signal("debt_to_revenue", "Debt to revenue", "govt_debt_to_revenue"),
            _signal(
                "debt_to_revenue_change",
                "Debt-to-revenue change",
                "govt_debt_to_revenue_change_3y",
            ),
            _signal("government_revenue", "Government revenue base", "govt_revenue_gdp"),
            _signal(
                "government_revenue_change",
                "Government revenue change",
                "govt_revenue_gdp_change_3y",
            ),
            _signal(
                "primary_deficit",
                "Primary deficit",
                "govt_primary_deficit_gdp",
            ),
            _signal(
                "primary_deficit_change",
                "Primary-deficit change",
                "govt_primary_deficit_gdp_change_3y",
            ),
            _signal("interest_cost", "Interest cost to GDP", "interest_cost_gdp"),
            _signal(
                "interest_cost_trend",
                "Interest-cost trend",
                "interest_cost_trend_3yr",
            ),
            _signal(
                "public_external_financing_need",
                "Public external financing-need proxy",
                "wb_public_financing_need_ext_debt_service_proxy_gdp",
            ),
            _signal(
                "sovereign_bank_exposure",
                "Bank exposure to the sovereign",
                "sovereign_exposure_ratio",
            ),
        ),
    ),
    MechanismSpec(
        key="external_fx",
        label="External and foreign-exchange stress",
        description="External balance-sheet, reserve, rollover, flow, and FX vulnerabilities.",
        signals=(
            _signal(
                "current_account",
                "Current-account pressure",
                "ca_deficit_severity",
                "current_account_gdp",
            ),
            _signal(
                "current_account_change",
                "Current-account deterioration",
                "current_account_change_3y",
            ),
            _signal("external_debt", "External debt", "external_debt_gdp"),
            _signal(
                "external_liabilities",
                "External liabilities",
                "external_liabilities_gdp",
            ),
            _signal("net_iip", "Net international investment position", "net_iip_gdp"),
            _signal(
                "reserves_import_cover",
                "Reserve import cover",
                "reserves_to_goods_services_imports",
                "reserves_months_imports",
            ),
            _signal(
                "reserves_payments_cover",
                "Reserves to current-account payments",
                "reserves_to_current_account_payments",
            ),
            _signal(
                "money_to_reserves",
                "Broad money to reserves",
                "broad_money_to_reserves",
                "m2_to_reserves",
            ),
            _signal(
                "sovereign_liabilities_to_reserves",
                "Sovereign liabilities to reserves",
                "sovereign_liability_to_reserves",
            ),
            _signal(
                "gross_external_financing_need",
                "Gross external financing-need proxy",
                "gross_external_financing_need_proxy_gdp",
            ),
            _signal(
                "external_income_service",
                "External income-service burden",
                "investment_income_debits_to_cxr",
            ),
            _signal(
                "portfolio_liabilities",
                "Portfolio liabilities",
                "portfolio_liabilities_gdp",
            ),
            _signal(
                "total_external_debt_service",
                "Total external debt service",
                "wb_total_external_debt_service_gni_pct",
            ),
            _signal(
                "public_external_debt_service",
                "Public external debt service",
                "wb_ppg_external_debt_service_gdp",
            ),
            _signal("fx_loans", "Foreign-currency loans", "fx_loan_exposure"),
            _signal(
                "bank_external_funding",
                "Bank external funding",
                "bank_liability_to_nfa",
            ),
        ),
    ),
    MechanismSpec(
        key="macro_commodity_global_triggers",
        label="Macro, commodity, and global triggers",
        description="Growth, price, rates, labour-market, commodity, and terms-of-trade shocks.",
        signals=(
            _signal(
                "growth",
                "Economic growth",
                "gdp_growth_3y_avg",
                "gdp_growth",
            ),
            _signal("inflation", "Inflation", "inflation"),
            _signal("inflation_change", "Inflation change", "inflation_change_3y"),
            _signal(
                "inflation_differential",
                "Inflation differential",
                "inflation_differential_3yr",
            ),
            _signal("unemployment", "Unemployment", "unemployment"),
            _signal("real_interest_rate", "Real interest rate", "real_interest_rate"),
            _signal(
                "lending_rate_change",
                "Lending-rate change",
                "lending_interest_rate_change_3y",
            ),
            _signal(
                "commodity_concentration",
                "Commodity export concentration",
                "commodity_export_concentration",
                "commodity_export_share_pct",
            ),
            _signal(
                "natural_resource_dependence",
                "Natural-resource dependence",
                "natural_resource_rents_gdp",
            ),
            _signal(
                "terms_of_trade_shock",
                "Terms-of-trade deterioration",
                "terms_of_trade_deterioration_3y",
                "tot_deterioration_3yr",
            ),
            _signal(
                "commodity_shock",
                "Commodity-shock exposure",
                "commodity_shock_exposure",
            ),
        ),
    ),
    MechanismSpec(
        key="structural_resilience",
        label="Structural resilience",
        description="Income, institutional strength, governance, and crisis-history resilience.",
        corroboration_eligible=False,
        signals=(
            _signal("income_level", "Income level", "gdp_per_capita"),
            _signal("voice_accountability", "Voice and accountability", "voice_accountability"),
            _signal("political_stability", "Political stability", "political_stability"),
            _signal("government_effectiveness", "Government effectiveness", "govt_effectiveness"),
            _signal("regulatory_quality", "Regulatory quality", "regulatory_quality"),
            _signal("rule_of_law", "Rule of law", "rule_of_law"),
            _signal("control_corruption", "Control of corruption", "control_corruption"),
            _signal(
                "crisis_history",
                "Recent banking-crisis history",
                "crisis_recency_10y",
                "years_since_banking_crisis",
            ),
        ),
    ),
)


def _validate_taxonomy(taxonomy: Sequence[MechanismSpec]) -> None:
    mechanism_keys: set[str] = set()
    signal_keys: set[str] = set()
    feature_owner: dict[str, str] = {}
    for mechanism in taxonomy:
        if mechanism.key in mechanism_keys:
            raise ValueError(f"Duplicate mechanism key: {mechanism.key}")
        mechanism_keys.add(mechanism.key)
        for signal in mechanism.signals:
            qualified_key = f"{mechanism.key}.{signal.key}"
            if qualified_key in signal_keys:
                raise ValueError(f"Duplicate signal key: {qualified_key}")
            signal_keys.add(qualified_key)
            if not signal.features:
                raise ValueError(f"Signal {qualified_key} has no feature aliases")
            if not np.isfinite(signal.weight) or signal.weight <= 0:
                raise ValueError(f"Signal {qualified_key} has invalid weight")
            for feature in signal.features:
                if feature in MISSINGNESS_FEATURES:
                    raise ValueError(
                        f"Missingness field {feature!r} cannot be risk evidence"
                    )
                if feature not in CANONICAL_RISK_DIRECTIONS:
                    raise ValueError(
                        f"Taxonomy feature {feature!r} has no governed risk direction"
                    )
                if feature in feature_owner:
                    raise ValueError(
                        f"Feature {feature!r} belongs to both "
                        f"{feature_owner[feature]} and {qualified_key}"
                    )
                feature_owner[feature] = qualified_key


_validate_taxonomy(MECHANISM_TAXONOMY)


def feature_mechanism_map(
    taxonomy: Sequence[MechanismSpec] = MECHANISM_TAXONOMY,
) -> dict[str, str]:
    """Return the unique primary mechanism for every canonical feature alias."""

    return {
        feature: mechanism.key
        for mechanism in taxonomy
        for signal in mechanism.signals
        for feature in signal.features
    }


@dataclass(frozen=True)
class MechanismEvidenceResult:
    """Tidy evidence tables plus one row-level summary per input record."""

    mechanism_evidence: pd.DataFrame
    signal_evidence: pd.DataFrame
    summary: pd.DataFrame


def _numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values.where(np.isfinite(values))


def _risk_percentile(value: float, reference: np.ndarray, direction: int) -> float:
    """Empirical mid-percentile oriented so 100 always means higher risk."""

    finite = np.asarray(reference, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not np.isfinite(value) or finite.size == 0:
        return np.nan
    if np.all(finite == finite[0]):
        percentile = 50.0
    else:
        below = float(np.count_nonzero(finite < value))
        equal = float(np.count_nonzero(finite == value))
        percentile = 100.0 * (below + 0.5 * equal) / finite.size
    if direction < 0:
        percentile = 100.0 - percentile
    return float(np.clip(percentile, 0.0, 100.0))


def _identifier_columns(
    values: pd.DataFrame,
    requested: Iterable[str] | None,
) -> list[str]:
    if requested is None:
        requested = ("country_code", "forecast_origin_year", "period", "as_of_date")
    return [column for column in requested if column in values.columns]


def calculate_mechanism_evidence(
    values: pd.DataFrame,
    *,
    reference: pd.DataFrame | None = None,
    taxonomy: Sequence[MechanismSpec] = MECHANISM_TAXONOMY,
    identifier_columns: Iterable[str] | None = None,
    minimum_reference_observations: int = 3,
    minimum_dominant_confidence: float = 0.5,
) -> MechanismEvidenceResult:
    """Calculate normalized mechanism evidence without imputing missing values.

    Parameters
    ----------
    values:
        Records to score.  Rows may be countries, country-years, or any other
        observations; canonical feature columns are detected from the taxonomy.
    reference:
        Cohort used for empirical normalization.  Defaults to ``values``.  A
        separate, frozen training/reference cohort is recommended for deployed
        scoring so percentiles do not drift with the selected screen.
    minimum_reference_observations:
        A signal is eligible only when at least this many finite values exist
        for one of its feature aliases in the reference cohort.
    minimum_dominant_confidence:
        Minimum within-mechanism coverage required before that mechanism can be
        selected as the dominant mechanism.

    Returns
    -------
    MechanismEvidenceResult
        ``signal_evidence`` explains every observed signal and chosen alias;
        ``mechanism_evidence`` contains 0--100 evidence and separate 0--1
        confidence; ``summary`` identifies the dominant sufficiently evidenced
        mechanism and total evidence coverage for each input row.
    """

    if not isinstance(values, pd.DataFrame):
        raise TypeError("values must be a pandas DataFrame")
    if reference is None:
        reference = values
    if not isinstance(reference, pd.DataFrame):
        raise TypeError("reference must be a pandas DataFrame")
    if minimum_reference_observations < 1:
        raise ValueError("minimum_reference_observations must be at least 1")
    if not 0 <= minimum_dominant_confidence <= 1:
        raise ValueError("minimum_dominant_confidence must be between 0 and 1")
    _validate_taxonomy(taxonomy)

    ids = _identifier_columns(values, identifier_columns)
    signal_rows: list[dict] = []
    mechanism_rows: list[dict] = []
    summary_rows: list[dict] = []

    # Eligibility and normalization are feature-alias-specific.  The first
    # eligible, observed alias wins for each row, preventing duplicate source
    # versions from receiving multiple votes.
    reference_cache: dict[str, np.ndarray] = {}
    for feature in feature_mechanism_map(taxonomy):
        if feature in reference.columns:
            reference_cache[feature] = _numeric(reference[feature]).dropna().to_numpy()

    for row_position, (_, row) in enumerate(values.iterrows()):
        row_mechanisms: list[dict] = []
        total_contract_weight = 0.0
        total_eligible_weight = 0.0
        total_observed_weight = 0.0

        for mechanism in taxonomy:
            mechanism_signal_rows: list[dict] = []
            contract_weight = float(sum(signal.weight for signal in mechanism.signals))
            eligible_weight = 0.0
            observed_weight = 0.0

            for signal in mechanism.signals:
                eligible_aliases = [
                    feature
                    for feature in signal.features
                    if feature in reference_cache
                    and len(reference_cache[feature]) >= minimum_reference_observations
                ]
                if not eligible_aliases:
                    continue
                eligible_weight += signal.weight
                chosen_feature = next(
                    (
                        feature
                        for feature in eligible_aliases
                        if feature in values.columns
                        and pd.notna(pd.to_numeric(row.get(feature), errors="coerce"))
                        and np.isfinite(float(pd.to_numeric(row.get(feature), errors="coerce")))
                    ),
                    None,
                )
                if chosen_feature is None:
                    continue

                raw_value = float(pd.to_numeric(row[chosen_feature], errors="coerce"))
                direction = CANONICAL_RISK_DIRECTIONS[chosen_feature]
                risk_evidence = _risk_percentile(
                    raw_value,
                    reference_cache[chosen_feature],
                    direction,
                )
                observed_weight += signal.weight
                detail = {
                    "row_position": row_position,
                    "record_id": values.index[row_position],
                    "mechanism": mechanism.key,
                    "mechanism_label": mechanism.label,
                    "signal": signal.key,
                    "signal_label": signal.label,
                    "feature": chosen_feature,
                    "raw_value": raw_value,
                    "risk_direction": direction,
                    "risk_evidence": risk_evidence,
                    "weight": signal.weight,
                }
                detail.update({column: row[column] for column in ids})
                signal_rows.append(detail)
                mechanism_signal_rows.append(detail)

            total_contract_weight += contract_weight
            total_eligible_weight += eligible_weight
            total_observed_weight += observed_weight
            if observed_weight:
                weighted_risk = sum(
                    detail["risk_evidence"] * detail["weight"]
                    for detail in mechanism_signal_rows
                ) / observed_weight
                dominant_signal = max(
                    mechanism_signal_rows,
                    key=lambda detail: (detail["risk_evidence"], -mechanism.signals.index(
                        next(s for s in mechanism.signals if s.key == detail["signal"])
                    )),
                )
            else:
                weighted_risk = np.nan
                dominant_signal = None
            # Coverage is measured against the complete governed taxonomy, not
            # merely the subset of signals that happens to exist in this data
            # vintage.  The latter is useful as a separate utilisation metric,
            # but calling it coverage can report 100% with most signal families
            # absent from the package.
            confidence = observed_weight / contract_weight if contract_weight else np.nan
            source_coverage = eligible_weight / contract_weight if contract_weight else np.nan
            supported_utilisation = (
                observed_weight / eligible_weight if eligible_weight else np.nan
            )
            mechanism_row = {
                "row_position": row_position,
                "record_id": values.index[row_position],
                "mechanism": mechanism.key,
                "mechanism_label": mechanism.label,
                "mechanism_description": mechanism.description,
                "corroboration_eligible": mechanism.corroboration_eligible,
                "risk_evidence": weighted_risk,
                "evidence_confidence": confidence,
                "source_coverage": source_coverage,
                "supported_source_utilisation": supported_utilisation,
                "observed_signals": len(mechanism_signal_rows),
                "eligible_signals": sum(
                    1
                    for signal in mechanism.signals
                    if any(
                        feature in reference_cache
                        and len(reference_cache[feature]) >= minimum_reference_observations
                        for feature in signal.features
                    )
                ),
                "taxonomy_signals": len(mechanism.signals),
                "dominant_signal": (
                    dominant_signal["signal"] if dominant_signal else None
                ),
                "dominant_signal_label": (
                    dominant_signal["signal_label"] if dominant_signal else None
                ),
                "dominant_signal_evidence": (
                    dominant_signal["risk_evidence"] if dominant_signal else np.nan
                ),
            }
            mechanism_row.update({column: row[column] for column in ids})
            mechanism_rows.append(mechanism_row)
            row_mechanisms.append(mechanism_row)

        dominant_candidates = [
            mechanism
            for mechanism in row_mechanisms
            if pd.notna(mechanism["risk_evidence"])
            and pd.notna(mechanism["evidence_confidence"])
            and mechanism["evidence_confidence"] >= minimum_dominant_confidence
        ]
        dominant = (
            max(
                dominant_candidates,
                key=lambda mechanism: (
                    mechanism["risk_evidence"],
                    -next(
                        index
                        for index, spec in enumerate(taxonomy)
                        if spec.key == mechanism["mechanism"]
                    ),
                ),
            )
            if dominant_candidates
            else None
        )
        summary = {
            "row_position": row_position,
            "record_id": values.index[row_position],
            "dominant_mechanism": dominant["mechanism"] if dominant else None,
            "dominant_mechanism_label": (
                dominant["mechanism_label"] if dominant else None
            ),
            "dominant_mechanism_evidence": (
                dominant["risk_evidence"] if dominant else np.nan
            ),
            "overall_evidence_confidence": (
                total_observed_weight / total_contract_weight
                if total_contract_weight
                else np.nan
            ),
            "overall_source_coverage": (
                total_eligible_weight / total_contract_weight
                if total_contract_weight
                else np.nan
            ),
            "supported_source_utilisation": (
                total_observed_weight / total_eligible_weight
                if total_eligible_weight
                else np.nan
            ),
            "observed_signal_weight": total_observed_weight,
            "eligible_signal_weight": total_eligible_weight,
            "taxonomy_signal_weight": total_contract_weight,
        }
        summary.update({column: row[column] for column in ids})
        summary_rows.append(summary)

    mechanism_columns = [
        "row_position",
        "record_id",
        "mechanism",
        "mechanism_label",
        "mechanism_description",
        "corroboration_eligible",
        "risk_evidence",
        "evidence_confidence",
        "source_coverage",
        "supported_source_utilisation",
        "observed_signals",
        "eligible_signals",
        "taxonomy_signals",
        "dominant_signal",
        "dominant_signal_label",
        "dominant_signal_evidence",
        *ids,
    ]
    signal_columns = [
        "row_position",
        "record_id",
        "mechanism",
        "mechanism_label",
        "signal",
        "signal_label",
        "feature",
        "raw_value",
        "risk_direction",
        "risk_evidence",
        "weight",
        *ids,
    ]
    summary_columns = [
        "row_position",
        "record_id",
        "dominant_mechanism",
        "dominant_mechanism_label",
        "dominant_mechanism_evidence",
        "overall_evidence_confidence",
        "overall_source_coverage",
        "supported_source_utilisation",
        "observed_signal_weight",
        "eligible_signal_weight",
        "taxonomy_signal_weight",
        *ids,
    ]
    return MechanismEvidenceResult(
        mechanism_evidence=pd.DataFrame(mechanism_rows, columns=mechanism_columns),
        signal_evidence=pd.DataFrame(signal_rows, columns=signal_columns),
        summary=pd.DataFrame(summary_rows, columns=summary_columns),
    )


__all__ = [
    "CANONICAL_RISK_DIRECTIONS",
    "MECHANISM_TAXONOMY",
    "MISSINGNESS_FEATURES",
    "MechanismEvidenceResult",
    "MechanismSpec",
    "SignalSpec",
    "calculate_mechanism_evidence",
    "feature_mechanism_map",
]
