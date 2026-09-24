import numpy as np
import pandas as pd
import pytest

from src.forecasting.phase6_rating import (
    ThreeAxisRatingPolicy,
    add_peer_relative_index,
    add_strict_history_index,
    combine_event_probabilities,
    risk_category,
)
from src.forecasting.phase6_event_labels import (
    EventTargetPolicy,
    build_forward_event_targets,
    stock_panel_to_episode_sensitivity,
)


def panel():
    rows = []
    for entity, shift in (("AAA", 0.0), ("BBB", 0.2), ("CCC", -0.2)):
        for year in range(2010, 2020):
            rows.append({
                "entity_code": entity,
                "forecast_origin_year": year,
                "structural_stress_probability_2y": 0.2 + shift + 0.03 * (year - 2010),
                "banking_or_sovereign_event_probability_1_3y": 0.02 + 0.01 * (year - 2010),
                "observed_share": 0.7,
                "state_uncertainty_proxy": 0.2,
            })
    return pd.DataFrame(rows)


def test_peer_index_is_same_year_relative():
    result = add_peer_relative_index(panel(), signal_col="structural_stress_probability_2y")
    one_year = result.loc[result.forecast_origin_year.eq(2014)].set_index("entity_code")
    assert one_year.loc["BBB", "peer_relative_risk_index"] > one_year.loc["AAA", "peer_relative_risk_index"]
    assert one_year.loc["AAA", "peer_reference_count"] == 3


def test_history_index_uses_strictly_prior_rows_only():
    original = panel()
    scored = add_strict_history_index(
        original,
        signal_col="structural_stress_probability_2y",
        minimum_prior_years=3,
    )
    key = scored.set_index(["entity_code", "forecast_origin_year"])
    before = key.loc[("AAA", 2015), "own_history_stress_index"]
    modified = original.copy()
    modified.loc[
        (modified.entity_code.eq("AAA")) & (modified.forecast_origin_year.gt(2015)),
        "structural_stress_probability_2y",
    ] = 1000
    rescored = add_strict_history_index(
        modified,
        signal_col="structural_stress_probability_2y",
        minimum_prior_years=3,
    ).set_index(["entity_code", "forecast_origin_year"])
    assert rescored.loc[("AAA", 2015), "own_history_stress_index"] == before
    assert key.loc[("AAA", 2012), "own_history_supported"] == False


def test_event_combination_preserves_missing_head_and_sensitivities():
    bank = pd.Series([0.10, 0.20, np.nan])
    sovereign = pd.Series([0.30, np.nan, np.nan])
    result = combine_event_probabilities(bank, sovereign)
    assert result.loc[0, "event_probability_max_lower_bound"] == pytest.approx(0.30)
    assert result.loc[0, "event_probability_independence_sensitivity"] == pytest.approx(0.37)
    assert result.loc[1, "event_head_coverage"] == "banking_only"
    assert result.loc[1, "banking_or_sovereign_event_probability_1_3y"] == pytest.approx(0.20)
    assert np.isnan(result.loc[2, "banking_or_sovereign_event_probability_1_3y"])


def test_event_overlay_is_upward_only_and_confidence_does_not_change_score():
    data = panel()
    policy = ThreeAxisRatingPolicy(
        peer_weight=0.5,
        event_overlay_strength=1.0,
        minimum_prior_years=3,
        observed_share_floor=0.5,
    ).fit_event_reference(np.linspace(0, 0.2, 100))
    scored = policy.transform(data)
    supported = scored.loc[scored.own_history_supported].copy()
    assert (supported.replacement_risk_index >= supported.structural_risk_index - 1e-12).all()
    changed = data.copy()
    changed["observed_share"] = 0.01
    rescored = policy.transform(changed)
    np.testing.assert_allclose(
        scored.replacement_risk_score,
        rescored.replacement_risk_score,
        equal_nan=True,
    )
    assert rescored.rating_support_status.eq("provisional_low_coverage").all()


def test_event_overlay_cannot_derisk_country():
    data = panel().loc[lambda x: x.entity_code.eq("AAA")].copy()
    policy = ThreeAxisRatingPolicy(
        peer_weight=1.0,
        event_overlay_strength=0.75,
        minimum_prior_years=2,
    ).fit_event_reference(np.linspace(0, 1, 101))
    low = data.copy()
    low["banking_or_sovereign_event_probability_1_3y"] = 0.01
    high = data.copy()
    high["banking_or_sovereign_event_probability_1_3y"] = 0.99
    low_score = policy.transform(low).replacement_risk_score
    high_score = policy.transform(high).replacement_risk_score
    assert (high_score >= low_score).all()


def test_categories_match_production_convention():
    assert risk_category(2.0) == "1-2: Very Low Risk"
    assert risk_category(4.0) == "3-4: Low Risk"
    assert risk_category(6.0) == "5-6: Moderate Risk"
    assert risk_category(8.0) == "7-8: High Risk"
    assert risk_category(9.0) == "9-10: Very High Risk"


def test_event_target_builder_separates_banking_and_sovereign():
    origins = pd.DataFrame(
        [("AAA", y) for y in range(2010, 2021)],
        columns=["entity_code", "forecast_origin_year"],
    )
    episodes = pd.DataFrame([
        {
            "country_code": "AAA", "event_type": "banking",
            "start_year": 2015, "end_year": 2016,
            "source_name": "bank", "source_version": "1",
        },
        {
            "country_code": "AAA", "event_type": "sovereign",
            "start_year": 2020, "end_year": 2020,
            "source_name": "sov", "source_version": "1",
        },
    ])
    targets = build_forward_event_targets(
        origins,
        episodes,
        policy=EventTargetPolicy(label_coverage_end_year=2022),
    ).set_index("forecast_origin_year")
    assert targets.loc[2012, "banking_crisis_target_1_3y"] == 1
    assert targets.loc[2017, "sovereign_crisis_target_1_3y"] == 1
    assert targets.loc[2017, "banking_or_sovereign_event_target_1_3y"] == 1
    assert targets.loc[2015, "event_target_eligible"] == False


def test_stock_adapter_is_sensitivity_only_and_detects_entry():
    stock = pd.DataFrame({
        "country_code": ["AAA"] * 6,
        "year": range(2010, 2016),
        "debt_in_default_usd": [0, 0, 10, 20, 0, 5],
    })
    episodes = stock_panel_to_episode_sensitivity(stock, source_version="2025")
    assert episodes[["start_year", "end_year"]].to_records(index=False).tolist() == [(2012, 2013), (2015, 2015)]
    assert episodes.event_derivation.str.contains("sensitivity").all()
