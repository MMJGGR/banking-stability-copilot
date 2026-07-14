import numpy as np
import pandas as pd
import pytest

from src.mechanism_evidence import (
    AlertPolicyConfig,
    CANONICAL_RISK_DIRECTIONS,
    MECHANISM_TAXONOMY,
    MechanismSpec,
    MISSINGNESS_FEATURES,
    SignalSpec,
    apply_alert_policy,
    calculate_mechanism_evidence,
    feature_mechanism_map,
)


def _mechanism(result, row_position, mechanism):
    rows = result.mechanism_evidence
    return rows[
        (rows["row_position"] == row_position)
        & (rows["mechanism"] == mechanism)
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
    assert first.supported_source_utilisation == second.supported_source_utilisation == 0.5
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


def test_alert_policy_separates_probability_corroboration_and_persistence():
    reference = pd.DataFrame(
        {
            "credit_to_gdp": [20.0, 40.0, 60.0, 80.0, 100.0],
            "npl_ratio": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    values = pd.DataFrame(
        {
            "country_code": ["NONE", "AMBER", "RED", "WAIT"],
            "credit_to_gdp": [20.0, 120.0, 120.0, 120.0],
            "npl_ratio": [1.0, 8.0, 8.0, 8.0],
        }
    )
    compact_taxonomy = (
        MechanismSpec(
            key="credit_property",
            label="Credit",
            description="Credit",
            signals=(SignalSpec("credit_depth", "Credit depth", ("credit_to_gdp",)),),
        ),
        MechanismSpec(
            key="bank_solvency_asset_quality",
            label="Solvency",
            description="Solvency",
            signals=(SignalSpec("npl", "NPL", ("npl_ratio",)),),
        ),
    )
    evidence = calculate_mechanism_evidence(
        values, reference=reference, taxonomy=compact_taxonomy
    )
    policy = AlertPolicyConfig(
        amber_hazard_threshold=0.10,
        red_hazard_threshold=0.20,
        mechanism_evidence_threshold=70.0,
        minimum_corroborating_mechanisms=2,
        minimum_persistent_periods=2,
    )

    alerts = apply_alert_policy(
        [0.05, 0.15, 0.30, 0.30],
        evidence,
        persistence_periods=[2, 2, 2, 1],
        config=policy,
    )

    assert alerts["alert_level"].tolist() == ["none", "amber", "red", "amber"]
    assert alerts.loc[2, "corroborating_mechanism_count"] == 2
    assert set(alerts.loc[2, "corroborating_mechanisms"]) == {
        "credit_property",
        "bank_solvency_asset_quality",
    }
    assert alerts.loc[3, "red_blockers"] == ("persistence_unconfirmed",)


def test_high_hazard_with_missing_mechanism_evidence_is_not_forced_to_alert():
    reference = pd.DataFrame(
        {"credit_to_gdp": [20.0, 40.0, 60.0, 80.0, 100.0]}
    )
    values = pd.DataFrame({"credit_to_gdp": [np.nan]})
    evidence = calculate_mechanism_evidence(values, reference=reference)

    alerts = apply_alert_policy(0.50, evidence, persistence_periods=3)

    assert alerts.loc[0, "alert_level"] == "insufficient_evidence"
    assert "insufficient_evidence" in alerts.loc[0, "red_blockers"]


def test_invalid_alert_threshold_order_is_rejected():
    with pytest.raises(ValueError, match="red_hazard_threshold"):
        AlertPolicyConfig(
            amber_hazard_threshold=0.30,
            red_hazard_threshold=0.20,
        )
