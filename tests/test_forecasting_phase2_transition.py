from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.phase2_full_execution import deterministic_selection_sample
from src.forecasting.phase2_transition import (
    build_transition_pairs,
    run_transition_development,
)
from src.forecasting.phase2_weighted_measurement import row_feature_balance


def synthetic_states(seed=11):
    rng = np.random.default_rng(seed)
    entities = [f"E{i:02d}" for i in range(24)]
    years = list(range(2003, 2024))
    rows = []
    for entity_index, entity in enumerate(entities):
        state = rng.normal(scale=0.8, size=4)
        for year in years:
            common = np.array(
                [
                    0.25 * np.sin((year - 2000) / 3),
                    0.15 * np.cos((year - 2000) / 4),
                    0.0,
                    0.0,
                ]
            )
            state = 0.55 * state + common + rng.normal(scale=0.08, size=4)
            rows.append(
                {
                    "entity_code": entity,
                    "forecast_origin_year": year,
                    **{f"state_{j+1}": state[j] for j in range(4)},
                }
            )
    states = pd.DataFrame(rows)
    information = states[["entity_code", "forecast_origin_year"]].copy()
    information["observed_share"] = 0.55 + 0.4 * rng.random(len(information))
    information["state_uncertainty_proxy"] = 1.0 / (
        2.0 + 10.0 * information.observed_share
    )
    information["effective_information"] = 1.0 / (
        information.state_uncertainty_proxy**2
    )
    return states, information


def test_transition_pairs_are_exact_calendar_and_complete_state():
    states, information = synthetic_states()
    pairs = build_transition_pairs(states, information, 2)
    assert (pairs.target_year - pairs.forecast_origin_year).eq(2).all()
    state_columns = [
        c
        for c in pairs
        if c.startswith("future_state_")
        and c.removeprefix("future_state_").isdigit()
    ]
    assert len(state_columns) == 4
    assert pairs.current_state_uncertainty_proxy.notna().all()
    assert pairs.future_state_uncertainty_proxy.notna().all()


def test_transition_development_can_beat_no_change_on_known_dynamics():
    states, information = synthetic_states()
    result = run_transition_development(states, information)
    assert result["supervised_banking_targets_read"] == 0
    assert result["crisis_labels_read"] == 0
    assert not result["final_confirmation_evaluated"]
    assert len(result["winners"]) == 2
    assert any(row["challenger_beats_no_change"] for row in result["winners"])


def test_selection_sampling_retains_every_row_and_feature():
    states, _ = synthetic_states()
    records = []
    for row in states.itertuples(index=False):
        for feature in ["a", "b", "c"]:
            records.append(
                {
                    "entity_code": row.entity_code,
                    "forecast_origin_year": row.forecast_origin_year,
                    "predictor_id": feature,
                    "feature_id": feature,
                    "model_value": float(row.state_1),
                }
            )
    cells = pd.DataFrame(records)
    sample = deterministic_selection_sample(
        cells,
        1200,
        minimum_per_row=1,
        minimum_per_feature=2,
    )
    assert set(
        zip(sample.entity_code, sample.forecast_origin_year)
    ) == set(zip(cells.entity_code, cells.forecast_origin_year))
    assert set(sample.predictor_id) == set(cells.predictor_id)


def test_row_feature_balance_is_positive_and_normalized():
    states, _ = synthetic_states()
    cells = states[["entity_code", "forecast_origin_year"]].copy()
    cells["predictor_id"] = np.where(np.arange(len(cells)) % 3, "dense", "sparse")
    weight = row_feature_balance(cells)
    assert np.isfinite(weight).all()
    assert weight.gt(0).all()
    assert abs(weight.mean() - 1.0) < 1e-12


def test_phase2_full_execution_has_target_firewall():
    source = Path("src/forecasting/phase2_full_execution.py").read_text()
    transition = Path("src/forecasting/phase2_transition.py").read_text()
    for text in (source, transition):
        assert "target-pairs.csv" not in text
        assert "crisis_classifier.pkl" not in text
        assert "risk_model.pkl" not in text
