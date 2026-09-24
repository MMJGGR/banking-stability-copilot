from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.phase4_interpretation import varimax
from src.forecasting.phase4_crisis import (
    build_crisis_overlay_panel,
    run_crisis_overlay_development,
)


def test_varimax_preserves_geometry_and_measurement():
    rng = np.random.default_rng(4)
    loadings = rng.normal(size=(120, 6))
    states = rng.normal(size=(30, 6))
    rotated, rotation, diagnostics = varimax(loadings, max_iter=200)
    assert diagnostics["orthogonality_max_abs_error"] < 1e-10
    before = np.linalg.norm(states[:, None, :] - states[None, :, :], axis=2)
    rotated_states = states @ rotation
    after = np.linalg.norm(
        rotated_states[:, None, :] - rotated_states[None, :, :], axis=2
    )
    np.testing.assert_allclose(before, after, atol=1e-9)
    np.testing.assert_allclose(
        states @ loadings.T,
        rotated_states @ rotated.T,
        atol=1e-9,
    )


def _small_state_data():
    rows, information = [], []
    for country, shift in (("AAA", 0.0), ("BBB", 0.5), ("CCC", -0.5)):
        for year in range(2010, 2026):
            rows.append({
                "entity_code": country,
                "forecast_origin_year": year,
                "state_1": shift + (year - 2010) * 0.1,
                "state_2": shift * 0.5,
            })
            information.append({
                "entity_code": country,
                "forecast_origin_year": year,
                "state_uncertainty_proxy": 0.2,
                "observed_share": 0.7,
                "effective_information": 3.0,
            })
    return pd.DataFrame(rows), pd.DataFrame(information)


def test_crisis_panel_excludes_active_cooldown_and_censoring():
    states, information = _small_state_data()
    episodes = pd.DataFrame([
        {
            "country_code": "AAA",
            "start_year": 2014,
            "label_end_year": 2015,
            "classification": "systemic",
        },
        {
            "country_code": "BBB",
            "start_year": 2022,
            "label_end_year": 2022,
            "classification": "borderline",
        },
    ])
    panel, excluded, summary = build_crisis_overlay_panel(
        states, information, episodes
    )
    aaa = excluded.loc[excluded.entity_code.eq("AAA")]
    assert set(aaa.loc[aaa.active_crisis, "forecast_origin_year"]) == {2014, 2015}
    assert set(
        aaa.loc[aaa.post_crisis_cooldown, "forecast_origin_year"]
    ) >= {2016, 2017, 2018}
    assert excluded.loc[excluded.right_censored, "forecast_origin_year"].min() == 2023
    assert panel.loc[
        panel.entity_code.eq("AAA") & panel.forecast_origin_year.eq(2011),
        "crisis_target",
    ].iloc[0] == 1
    assert summary["borderline_episodes_used"] == 0
    assert summary["provider_projection_rows_read"] == 0


def _synthetic_overlay_panel(seed=17):
    rng = np.random.default_rng(seed)
    rows = []
    entity_effect = rng.normal(size=60)
    for index in range(60):
        country = f"E{index:02d}"
        previous = None
        for year in range(1980, 2023):
            common = 0.8 * np.sin((year - 1980) / 4)
            state = np.array([
                entity_effect[index] + common + rng.normal(scale=0.25),
                0.5 * entity_effect[index] - common + rng.normal(scale=0.35),
                rng.normal(scale=0.7),
                rng.normal(scale=0.5),
            ])
            velocity = np.zeros(4) if previous is None else state - previous
            linear = 1.25 * state[0] - 0.65 * state[1] + 0.35 * velocity[0] - 1.4
            probability = 1 / (1 + np.exp(-linear))
            target = int(rng.random() < probability * 0.35)
            row = {
                "entity_code": country,
                "forecast_origin_year": year,
                "crisis_target": target,
                "state_uncertainty_proxy": 0.25 + abs(rng.normal(scale=0.05)),
                "observed_share": 0.6 + rng.random() * 0.3,
                "velocity_observed": int(previous is not None),
            }
            for component, value in enumerate(state, 1):
                row[f"state_{component}"] = value
                row[f"velocity_state_{component}"] = velocity[component - 1]
            rows.append(row)
            previous = state
    return pd.DataFrame(rows)


def test_state_overlay_runs_time_ordered_and_beats_event_rate():
    result = run_crisis_overlay_development(
        _synthetic_overlay_panel(),
        minimum_training_positives=10,
        minimum_test_positives=1,
    )
    assert result["status"] == "completed_phase4c_crisis_overlay_development"
    aggregate = result["aggregate_metrics"].set_index("model")
    assert "event_rate" in aggregate.index
    assert aggregate.loc[
        ["state_only", "state_velocity_uncertainty"], "brier"
    ].min() < aggregate.loc["event_rate", "brier"]
    assert result["state_refit_with_crisis_labels"] is False
    assert result["provider_projection_rows_read"] == 0


def test_phase4_sources_do_not_use_provider_projections_or_production_scores():
    root = Path("src/forecasting")
    for name in (
        "phase4_interpretation.py",
        "phase4_outcomes.py",
        "phase4_crisis.py",
    ):
        source = (root / name).read_text()
        assert "risk_model.pkl" not in source
        assert "crisis_classifier.pkl" not in source
