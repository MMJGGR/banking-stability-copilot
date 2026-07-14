import numpy as np
import pandas as pd
import pytest

from src.mechanism_evidence import (
    CANONICAL_RISK_DIRECTIONS,
    MECHANISM_TAXONOMY,
    MISSINGNESS_FEATURES,
    calculate_mechanism_evidence,
    feature_mechanism_map,
)


def _mechanism(result, row_position, mechanism):
    rows = result.mechanism_evidence
    return rows[
        (rows["row_position"] == row_position) & (rows["mechanism"] == mechanism)
    ].iloc[0]


def test_taxonomy_covers_each_governed_risk_feature_once_but_not_missingness():
    mapping = feature_mechanism_map()
    expected = set(CANONICAL_RISK_DIRECTIONS).difference(MISSINGNESS_FEATURES)

    assert set(mapping) == expected
    assert not set(mapping).intersection(MISSINGNESS_FEATURES)
    assert len(mapping) == sum(
        len(signal.features)
        for mechanism in MECHANISM_TAXONOMY
        for signal in mechanism.signals
    )


def test_active_government_and_external_liquidity_features_have_primary_groups():
    mapping = feature_mechanism_map()

    assert mapping["govt_interest_to_revenue"] == "sovereign_liquidity_market_access"
    assert mapping["govt_debt_to_revenue"] == "sovereign_liquidity_market_access"
    assert mapping["net_iip_gdp"] == "external_fx"
    assert mapping["external_liabilities_gdp"] == "external_fx"
    assert mapping["reserves_to_goods_services_imports"] == "external_fx"
    assert mapping["gross_external_financing_need_proxy_gdp"] == "external_fx"
    assert mapping["commodity_export_share_pct"] == "macro_commodity_global_triggers"


def test_evidence_is_risk_oriented_and_confidence_is_separate_from_risk():
    reference = pd.DataFrame(
        {
            "credit_to_gdp": [20.0, 40.0, 60.0, 80.0, 100.0],
            "bank_credit_gdp_gap_10y": [-10.0, -5.0, 0.0, 5.0, 10.0],
            "capital_adequacy": [8.0, 10.0, 12.0, 14.0, 16.0],
        }
    )
    values = pd.DataFrame(
        {
            "country_code": ["LOW", "MISS", "HIGH"],
            "credit_to_gdp": [20.0, 100.0, 100.0],
            "bank_credit_gdp_gap_10y": [np.nan, np.nan, 10.0],
            "capital_adequacy": [16.0, 8.0, 8.0],
        }
    )

    result = calculate_mechanism_evidence(values, reference=reference)
    low_credit = _mechanism(result, 0, "credit_property")
    missing_credit = _mechanism(result, 1, "credit_property")
    high_credit = _mechanism(result, 2, "credit_property")
    low_solvency = _mechanism(result, 0, "bank_solvency_asset_quality")
    high_solvency = _mechanism(result, 2, "bank_solvency_asset_quality")

    assert low_credit.risk_evidence < missing_credit.risk_evidence
    assert missing_credit.risk_evidence == pytest.approx(90.0)
    assert high_credit.risk_evidence == pytest.approx(90.0)
    assert missing_credit.evidence_confidence == pytest.approx(1 / 8)
    assert high_credit.evidence_confidence == pytest.approx(2 / 8)
    assert missing_credit.supported_source_utilisation == pytest.approx(0.5)
    assert high_credit.supported_source_utilisation == pytest.approx(1.0)
    assert low_solvency.risk_evidence < high_solvency.risk_evidence


def test_missing_signal_does_not_create_risk_evidence_or_missingness_penalty():
    reference = pd.DataFrame(
        {
            "credit_to_gdp": [20.0, 40.0, 60.0, 80.0, 100.0],
            "bank_credit_gdp_gap_10y": [-10.0, -5.0, 0.0, 5.0, 10.0],
        }
    )
    values = pd.DataFrame(
        {
            "credit_to_gdp": [80.0, 80.0],
            "bank_credit_gdp_gap_10y": [np.nan, np.nan],
            # Deliberately high; missingness fields are excluded from evidence.
            "credit_missing_share": [0.0, 1.0],
        }
    )

    result = calculate_mechanism_evidence(values, reference=reference)
    first = _mechanism(result, 0, "credit_property")
    second = _mechanism(result, 1, "credit_property")

    assert first.risk_evidence == second.risk_evidence
    assert first.evidence_confidence == second.evidence_confidence == 1 / 8
    assert (
        first.supported_source_utilisation == second.supported_source_utilisation == 0.5
    )
    assert "credit_missing_share" not in result.signal_evidence["feature"].tolist()


def test_exact_source_aliases_count_as_one_signal_and_prefer_combined_field():
    reference = pd.DataFrame(
        {
            "combined_npl_ratio": [1.0, 2.0, 3.0, 4.0, 5.0],
            "npl_ratio": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    values = pd.DataFrame(
        {
            "combined_npl_ratio": [5.0],
            "npl_ratio": [1.0],
        }
    )

    result = calculate_mechanism_evidence(values, reference=reference)
    npl = result.signal_evidence[
        result.signal_evidence["signal"] == "nonperforming_loans"
    ]
    solvency = _mechanism(result, 0, "bank_solvency_asset_quality")

    assert len(npl) == 1
    assert npl.iloc[0]["feature"] == "combined_npl_ratio"
    assert npl.iloc[0]["risk_evidence"] == pytest.approx(90.0)
    assert solvency.observed_signals == 1
    assert solvency.eligible_signals == 1


def test_dominant_mechanism_requires_coverage_of_the_full_governed_contract():
    reference = pd.DataFrame(
        {
            "credit_to_gdp": [20.0, 40.0, 60.0, 80.0, 100.0],
            "npl_ratio": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    values = pd.DataFrame(
        {
            "credit_to_gdp": [100.0],
            "npl_ratio": [2.0],
        }
    )

    result = calculate_mechanism_evidence(values, reference=reference)

    assert result.summary.iloc[0]["dominant_mechanism"] is None
    total_weight = sum(
        signal.weight
        for mechanism in MECHANISM_TAXONOMY
        for signal in mechanism.signals
    )
    assert result.summary.iloc[0]["overall_evidence_confidence"] == pytest.approx(
        2 / total_weight
    )
    assert result.summary.iloc[0]["supported_source_utilisation"] == pytest.approx(1.0)
