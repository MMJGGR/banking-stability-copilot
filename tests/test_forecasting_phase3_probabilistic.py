import numpy as np
import pandas as pd

from src.forecasting.phase3_probabilistic import (
    StateErrorCalibrator,
    add_residual_decomposition,
    build_transition_pairs,
    fit_fixed_point_model,
    leave_target_year_out_calibration,
    reproduce_phase2_windows,
    rolling_point_forecasts,
    simulate_distribution,
    make_residual_pool,
)


def synthetic_state(seed=4):
    rng = np.random.default_rng(seed)
    entities = [f"E{i:02d}" for i in range(30)]
    years = range(1998, 2025)
    rows = []
    infos = []
    for entity in entities:
        state = rng.normal(scale=.5, size=6)
        for year in years:
            common = np.array([.12*np.sin((year-1998)/3), .08*np.cos((year-1998)/4), 0, 0, 0, 0])
            state = .65*state + common + rng.normal(scale=.10, size=6)
            rows.append({"entity_code": entity, "forecast_origin_year": year, **{f"state_{j+1}": state[j] for j in range(6)}})
            observed = int(rng.integers(30, 500))
            uncertainty = 1/np.sqrt(observed)
            infos.append({"entity_code": entity, "forecast_origin_year": year, "observed_model_features": observed, "observed_share": observed/500, "state_uncertainty_proxy": uncertainty, "effective_information": 1/uncertainty**2})
    return pd.DataFrame(rows), pd.DataFrame(infos)


def masking(seed=5):
    rng = np.random.default_rng(seed)
    uncertainty = np.exp(rng.uniform(np.log(.03), np.log(.6), size=200))
    error = .15 + 1.2*uncertainty + rng.normal(scale=.03, size=200)
    return pd.DataFrame({"uncertainty_proxy": uncertainty, "state_error": np.maximum(error, .01)})


def test_pairs_and_models():
    states, info = synthetic_state()
    pairs = build_transition_pairs(states, info, 2)
    assert (pairs.target_year-pairs.forecast_origin_year).eq(2).all()
    columns = [f"state_{i}" for i in range(1, 7)]
    model = fit_fixed_point_model(pairs[pairs.target_year < 2018], 2, columns)
    prediction = model.predict_raw(pairs.loc[pairs.forecast_origin_year.eq(2018), columns].to_numpy())
    assert prediction.shape[1] == 6
    assert np.isfinite(model.sensitivity_raw()).all()


def test_rolling_and_calibration():
    states, info = synthetic_state()
    rolling, _ = rolling_point_forecasts(states, info, minimum_train_rows=150, minimum_test_rows=10)
    columns = [f"state_{i}" for i in range(1, 7)]
    rolling = add_residual_decomposition(rolling, columns)
    detail, summary = leave_target_year_out_calibration(rolling, columns)
    assert len(detail) > 100
    assert set(summary.breakdown) == {"overall", "uncertainty", "coverage"}
    assert detail.covered_95.mean() > .75


def test_phase2_window_reproduction_executes():
    states, info = synthetic_state()
    result = reproduce_phase2_windows(states, info, windows=((2014, 2016), (2017, 2019)))
    assert len(result) == 4
    assert result.relative_rmse_improvement.notna().all()


def test_simulation_reproducible():
    states, info = synthetic_state()
    rolling, fitted = rolling_point_forecasts(states, info, minimum_train_rows=150, minimum_test_rows=10)
    columns = [f"state_{i}" for i in range(1, 7)]
    rolling = add_residual_decomposition(rolling, columns)
    row = rolling.iloc[-1]
    pool = make_residual_pool(rolling, columns, int(row.horizon))
    calibrator = StateErrorCalibrator(masking())
    current = row[[f"current_{c}" for c in columns]].to_numpy(dtype=float)
    point = row[[f"point_{c}" for c in columns]].to_numpy(dtype=float)
    model = fitted[(int(row.horizon), int(row.forecast_origin_year))]
    first, diagnostics_first = simulate_distribution(current, point, model.sensitivity_raw(), float(row.current_state_uncertainty_proxy), int(row.uncertainty_group), calibrator, pool, draws=40, rng=np.random.default_rng(99))
    second, diagnostics_second = simulate_distribution(current, point, model.sensitivity_raw(), float(row.current_state_uncertainty_proxy), int(row.uncertainty_group), calibrator, pool, draws=40, rng=np.random.default_rng(99))
    np.testing.assert_allclose(first, second)
    assert diagnostics_first == diagnostics_second
