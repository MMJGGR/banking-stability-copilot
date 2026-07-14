import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.crisis_hazard import (
    ExpertRoutingConfig,
    HazardExpertRouter,
    RegularizedDiscreteTimeHazardModel,
    build_crisis_target_hierarchy,
    cumulative_incidence_from_annual_hazards,
    event_balanced_sample_weights,
    regularized_cloglog_hazard,
    regularized_logit_hazard,
)


def test_target_hierarchy_uses_conditional_at_risk_intervals_and_stable_event_id():
    panel = pd.DataFrame(
        {
            "country_code": ["USA"] * 4,
            "forecast_origin_year": [2001, 2002, 2003, 2004],
        }
    )
    events = pd.DataFrame(
        {
            "country_code": ["USA"],
            "crisis_start_year": [2005],
            "crisis_end_year": [2006],
            "crisis_event_id": ["USA-2005-2006"],
        }
    )

    targets = build_crisis_target_hierarchy(panel, events).set_index(
        "forecast_origin_year"
    )

    assert pd.isna(targets.loc[2001, "years_to_onset"])
    assert targets.loc[2002, "crisis_hazard_year_3"] == 1
    assert targets.loc[2003, "crisis_hazard_year_2"] == 1
    assert targets.loc[2004, "crisis_hazard_1y"] == 1
    assert pd.isna(targets.loc[2004, "crisis_hazard_year_2"])
    assert not targets.loc[2004, "at_risk_year_2"]
    assert targets.loc[2004, "crisis_onset_within_3y"] == 1
    assert targets.loc[2004, "crisis_onset_in_years_2_3"] == 0
    assert targets.loc[2003, "hazard_event_id"] == "USA-2005-2006"


def test_target_hierarchy_marks_unresolved_followup_missing_not_negative():
    panel = pd.DataFrame(
        {"country_code": ["USA"], "forecast_origin_year": [2024]}
    )
    targets = build_crisis_target_hierarchy(
        panel,
        pd.DataFrame(columns=["country_code", "crisis_start_year"]),
        label_coverage_end_year=2025,
    )

    assert targets.loc[0, "crisis_hazard_1y"] == 0
    assert pd.isna(targets.loc[0, "crisis_hazard_year_2"])
    assert pd.isna(targets.loc[0, "crisis_onset_within_3y"])


def test_cumulative_incidence_preserves_survival_arithmetic():
    result = cumulative_incidence_from_annual_hazards([[0.10, 0.20, 0.30]]).iloc[0]

    assert result["probability_1y"] == pytest.approx(0.10)
    assert result["probability_within_2y"] == pytest.approx(0.28)
    assert result["probability_years_2_3"] == pytest.approx(0.396)
    assert result["probability_within_3y"] == pytest.approx(0.496)


def test_event_balancing_gives_each_episode_equal_total_positive_weight():
    frame = pd.DataFrame(
        {
            "crisis_onset_within_3y": [1, 1, 1, 1, 0, 0],
            "hazard_event_id": ["A", "A", "A", "B", pd.NA, pd.NA],
        }
    )
    weights = event_balanced_sample_weights(frame)

    assert weights.groupby(frame["hazard_event_id"], dropna=True).sum().to_dict() == {
        "A": pytest.approx(2.0),
        "B": pytest.approx(2.0),
    }
    assert weights.iloc[4:].eq(1.0).all()
    assert weights.mean() == pytest.approx(1.0)


@pytest.mark.parametrize("link", ["logit", "cloglog"])
def test_regularized_hazard_model_is_cloneable_missing_safe_and_serializable(link):
    rng = np.random.default_rng(14)
    X = pd.DataFrame(
        {
            "credit_gap": rng.normal(size=500),
            "liquidity_stress": rng.normal(size=500),
        }
    )
    linear = -2.3 + 1.2 * X["credit_gap"] + 0.8 * X["liquidity_stress"]
    probability = 1.0 / (1.0 + np.exp(-linear))
    y = rng.binomial(1, probability)
    X.loc[::17, "liquidity_stress"] = np.nan

    model = clone(
        RegularizedDiscreteTimeHazardModel(
            link=link,
            C=2.0,
            class_weight=None,
            metadata={
                "target": "crisis_hazard_1y",
                "training_rows": np.int64(len(X)),
                "as_of": pd.Timestamp("2026-06-30"),
            },
        )
    ).fit(X, y)
    probabilities = model.predict_hazard(X.iloc[:20])
    restored = RegularizedDiscreteTimeHazardModel.from_json(model.to_json())

    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0) & (probabilities < 1)).all()
    np.testing.assert_allclose(
        probabilities, restored.predict_hazard(X.iloc[:20]), rtol=1e-12, atol=1e-12
    )
    assert json.loads(model.to_json())["metadata"]["target"] == "crisis_hazard_1y"
    assert json.loads(model.to_json())["metadata"]["as_of"] == "2026-06-30T00:00:00"
    assert set(model.coefficient_frame()["feature"]) == {
        "credit_gap",
        "liquidity_stress",
    }


def test_regularization_uses_penalized_likelihood_scale_not_sample_size_penalty():
    """A clear signal must not collapse to the unconditional base rate.

    This guards the ``C`` scaling used by the custom optimiser.  The data loss
    is averaged by total sample weight, so the L2 term must be averaged by the
    same amount to retain conventional penalised-likelihood semantics.
    """

    rng = np.random.default_rng(23)
    feature = rng.normal(size=600)
    X = pd.DataFrame({"signal": feature})
    y = (feature > 1.0).astype(int)

    model = RegularizedDiscreteTimeHazardModel(
        link="logit", C=0.25, class_weight=None
    ).fit(X, y)
    probabilities = model.predict_hazard(X)

    assert model.coef_[0, 0] > 1.0
    assert probabilities[y == 1].mean() > probabilities[y == 0].mean() + 0.50
    assert probabilities.max() - probabilities.min() > 0.80


def test_champion_factories_expose_common_interface():
    assert regularized_logit_hazard().link == "logit"
    assert regularized_cloglog_hazard().link == "cloglog"


class ConstantHazard:
    def __init__(self, probability):
        self.probability = probability

    def predict_hazard(self, X):
        return np.full(len(X), self.probability)


def test_router_selects_experts_and_surfaces_current_horizon_probabilities():
    config = ExpertRoutingConfig(
        historical_features=("macro_a", "macro_b"),
        modern_incremental_features=("bank_a", "bank_b"),
        historical_min_coverage=0.5,
        modern_min_coverage=0.5,
    )
    frame = pd.DataFrame(
        {
            "macro_a": [1.0, 1.0, np.nan],
            "macro_b": [2.0, 2.0, np.nan],
            "bank_a": [3.0, np.nan, 3.0],
            "bank_b": [np.nan, np.nan, 4.0],
        },
        index=["modern", "historical", "insufficient"],
    )
    router = HazardExpertRouter(config)
    output = router.predict_current(
        frame,
        historical_models={
            1: ConstantHazard(0.10),
            2: ConstantHazard(0.20),
            3: ConstantHazard(0.30),
        },
        modern_models={
            1: ConstantHazard(0.05),
            2: ConstantHazard(0.10),
            3: ConstantHazard(0.15),
        },
    )

    assert output.loc["modern", "selected_expert"] == "modern_full"
    assert output.loc["historical", "selected_expert"] == "historical_core"
    assert output.loc["insufficient", "selected_expert"] == "insufficient_evidence"
    assert output.loc["modern", "probability_1y"] == pytest.approx(0.05)
    assert output.loc["modern", "probability_years_2_3"] == pytest.approx(0.22325)
    assert output.loc["modern", "probability_within_3y"] == pytest.approx(0.27325)
    assert pd.isna(output.loc["insufficient", "probability_within_3y"])
    assert output["horizon_model_basis"].eq("horizon_specific").all()


def test_availability_flags_override_non_null_values_in_routing():
    config = ExpertRoutingConfig(
        historical_features=("macro",),
        modern_incremental_features=("bank",),
        historical_min_coverage=1.0,
        modern_min_coverage=1.0,
    )
    frame = pd.DataFrame(
        {
            "macro": [1.0],
            "macro__available": [True],
            "bank": [99.0],
            "bank__available": [False],
        }
    )
    route = HazardExpertRouter(config).route(frame)

    assert route.loc[0, "selected_expert"] == "historical_core"
    assert route.loc[0, "modern_incremental_coverage"] == 0.0
