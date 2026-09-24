"""Phase 3 probabilistic future-state research execution.

Consumes immutable Phase 2 evidence, keeps the 96-dimensional measurement state
fixed, and produces calibrated one-/two-year future-state distributions. Current-
vintage WEO provider projections remain outside the baseline model in a separate
scenario/benchmark lane. Production remains unchanged.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .phase3_common import DEFAULT_SEED, FIXED_POINT_MODELS, Phase3Error, StateScale, RidgeDelta, DiagonalAR, FittedPointModel, _json_ready, _state_columns, add_residual_decomposition, build_transition_pairs, fit_fixed_point_model, point_metrics, reproduce_phase2_windows, rolling_point_forecasts, sha256, write_json
from .phase3_distribution import StateErrorCalibrator, ResidualPool, leave_target_year_out_calibration, make_residual_pool, simulate_distribution, uncertainty_component_summary
from .phase3_latest import fit_latest_models, historical_analogues, latest_joint_simulation, observable_implications


def run_phase3(phase2_results: Path, output: Path, *, calibration_draws: int=160, latest_draws: int=800, peer_draws: int=250, seed: int=DEFAULT_SEED) -> dict:
    phase2_results = Path(phase2_results).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise Phase3Error('Output directory must be new')
    output.mkdir(parents=True)
    required = {'country-year-states.csv.gz', 'country-year-information.csv.gz', 'measurement-loadings.csv.gz', 'uncertainty-masking-rows.csv.gz', 'phase2a-summary.json'}
    missing = [name for name in required if not (phase2_results / name).is_file()]
    if missing:
        raise Phase3Error(f'Missing Phase 2 evidence: {missing}')
    states = pd.read_csv(phase2_results / 'country-year-states.csv.gz')
    information = pd.read_csv(phase2_results / 'country-year-information.csv.gz')
    loadings = pd.read_csv(phase2_results / 'measurement-loadings.csv.gz', low_memory=False)
    masking = pd.read_csv(phase2_results / 'uncertainty-masking-rows.csv.gz')
    phase2_summary = json.loads((phase2_results / 'phase2a-summary.json').read_text())
    state_cols = _state_columns(states)
    if len(state_cols) != int(phase2_summary['state_rank']):
        raise Phase3Error('Phase 2 state rank mismatch')
    reproduction = reproduce_phase2_windows(states, information)
    reproduction.to_csv(output / 'phase2-point-model-reproduction.csv', index=False)
    reproduction_max_difference = None
    expected_path = phase2_results / 'transition-fold-metrics.csv'
    if expected_path.is_file():
        expected = pd.read_csv(expected_path)
        expected = expected.loc[expected.horizon.eq(1) & expected.model.eq('ridge_delta') | expected.horizon.eq(2) & expected.model.eq('diagonal_ar')]
        compare_fields = ['state_rmse', 'mean_row_rmse', 'fraction_rows_beating_no_change', 'mean_movement_cosine']
        merged = reproduction.merge(expected[['horizon', 'outer_start', 'outer_end', *compare_fields]], on=['horizon', 'outer_start', 'outer_end'], suffixes=('_reproduced', '_phase2'), validate='one_to_one')
        differences = []
        for field in compare_fields:
            differences.extend(np.abs(merged[f'{field}_reproduced'].to_numpy(dtype=float) - merged[f'{field}_phase2'].to_numpy(dtype=float)).tolist())
        reproduction_max_difference = float(np.nanmax(differences))
        if reproduction_max_difference > 1e-09:
            raise Phase3Error(f'Phase 2 point-model reproduction mismatch: {reproduction_max_difference}')
    rolling, fitted = rolling_point_forecasts(states, information)
    rolling = add_residual_decomposition(rolling, state_cols)
    rolling_rows = len(rolling)
    rolling_target_years = int(rolling.target_year.nunique())
    rolling.to_pickle(output / 'rolling-point-forecasts.pkl')
    calibration_detail, calibration_summary = leave_target_year_out_calibration(rolling, state_cols)
    calibration_detail.to_csv(output / 'calibration-row-results.csv.gz', index=False)
    calibration_summary.to_csv(output / 'calibration-summary.csv', index=False)
    calibration_rows = len(calibration_detail)
    component_summary = uncertainty_component_summary(rolling, masking, state_cols)
    component_summary.to_csv(output / 'uncertainty-component-summary.csv', index=False)
    latest, latest_models, _ = fit_latest_models(states, information)
    latest_summary, coordinate_quantiles, peer_frequency, simulation_draws, point_by_horizon = latest_joint_simulation(states, information, rolling, masking, latest_models, draws=latest_draws, peer_draws=peer_draws, seed=seed)
    latest_summary.to_csv(output / 'latest-forecast-summary.csv', index=False)
    coordinate_quantiles.to_csv(output / 'latest-state-coordinate-quantiles.csv.gz', index=False)
    peer_frequency.to_csv(output / 'nearest-peer-frequencies.csv.gz', index=False)
    analogue = historical_analogues(states, information, latest_summary, point_by_horizon)
    analogue.to_csv(output / 'historical-analogues.csv.gz', index=False)
    latest_states = states.sort_values('forecast_origin_year').groupby('entity_code', observed=True).tail(1)
    latest_states = latest_states.loc[latest_states.forecast_origin_year.eq(latest_states.forecast_origin_year.max())].copy()
    del rolling, fitted, latest_models, calibration_detail
    gc.collect()
    implications, implication_summary = observable_implications(latest_states, latest_summary, loadings, simulation_draws, point_by_horizon)
    implications.to_csv(output / 'observable-implications-top.csv.gz', index=False)
    implication_summary.to_csv(output / 'observable-feature-summary.csv.gz', index=False)
    overall_calibration = calibration_summary[calibration_summary.breakdown.eq('overall')].copy()
    acceptance = {}
    for row in overall_calibration.itertuples(index=False):
        key = f'{row.model}_{int(row.horizon)}y'
        acceptance[key] = {'coverage_50': float(row.coverage_50), 'coverage_80': float(row.coverage_80), 'coverage_95': float(row.coverage_95), 'mean_radius_80': float(row.mean_radius_80), 'mean_radius_95': float(row.mean_radius_95), 'calibration_pass_80': bool(row.coverage_80 >= 0.76), 'calibration_pass_95': bool(row.coverage_95 >= 0.92)}
    transition_pass = all((value['calibration_pass_80'] and value['calibration_pass_95'] for key, value in acceptance.items() if key.startswith('transition_')))
    report = {'status': 'completed_phase3_probabilistic_future_state_development' if transition_pass else 'phase3_stopped_for_probability_recalibration', 'phase2_state_rank': len(state_cols), 'phase2_measurement_state_fixed': True, 'rolling_forecast_rows': rolling_rows, 'rolling_target_years': rolling_target_years, 'calibration_rows': calibration_rows, 'calibration_method': 'leave-target-year-out empirical radial conformal regions', 'uncertainty_components_reported': int(len(component_summary)), 'latest_country_states': int(latest_summary.entity_code.nunique()), 'phase2_point_model_reproduction_max_abs_difference': reproduction_max_difference, 'latest_joint_draws_per_horizon': latest_draws, 'peer_frequency_draws_per_horizon': min(peer_draws, latest_draws), 'fixed_point_models': FIXED_POINT_MODELS, 'calibration': acceptance, 'transition_distribution_acceptance_passed': transition_pass, 'targets_read': 0, 'crisis_labels_read': 0, 'production_scores_read': 0, 'provider_projection_rows_read': 0, 'provider_projection_policy': 'Current-vintage WEO 2026-2031 projections are excluded from the baseline state, transition fitting, calibration and realized outcomes. They are retained in a separate optional scenario/benchmark lane.', 'production_modified': False, 'crisis_classifier_retrained': False, 'final_confirmation_evaluated': False, 'limitations': ['Retrospective latest-vintage state evidence, not a vintage-clean historical backtest.', 'Current-state perturbation directions use empirical transition-residual directions because Phase 2 preserved masking error magnitudes but not full masking error vectors.', 'Observable implications are standardized changes through measurement loadings, not exact raw future indicator levels.', 'Probabilistic calibration is development evidence; a later untouched confirmation period remains required.']}
    write_json(output / 'phase3-summary.json', report)
    checksums = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob('*')) if path.is_file()}
    write_json(output / 'output-checksums.json', checksums)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase2-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-draws', type=int, default=160)
    parser.add_argument('--latest-draws', type=int, default=800)
    parser.add_argument('--peer-draws', type=int, default=250)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    report = run_phase3(args.phase2_results, args.output, calibration_draws=args.calibration_draws, latest_draws=args.latest_draws, peer_draws=args.peer_draws, seed=args.seed)
    print('PHASE3_STATUS', json.dumps(_json_ready(report), sort_keys=True))


if __name__ == "__main__":
    main()
