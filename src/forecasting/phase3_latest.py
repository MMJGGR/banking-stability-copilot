"""Phase 3 latest joint simulations, peers, analogues and implications."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from .phase3_common import Phase3Error, DEFAULT_SEED, FittedPointModel, _rms_radius, _safe_cosine, _state_columns, build_transition_pairs, fit_fixed_point_model
from .phase3_distribution import StateErrorCalibrator, _empirical_radius_quantiles, _matched_indices, make_residual_pool


def fit_latest_models(states: pd.DataFrame, information: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, FittedPointModel], dict[int, pd.DataFrame]]:
    state_cols = _state_columns(states)
    latest = states.sort_values('forecast_origin_year').groupby('entity_code', observed=True).tail(1)
    latest = latest.merge(information, on=['entity_code', 'forecast_origin_year'], how='left', validate='one_to_one').sort_values('entity_code').reset_index(drop=True)
    models = {}
    pair_tables = {}
    for horizon in (1, 2):
        pairs = build_transition_pairs(states, information, horizon)
        latest_origin = int(latest.forecast_origin_year.max())
        train = pairs[pairs.target_year <= latest_origin].copy()
        models[horizon] = fit_fixed_point_model(train, horizon, state_cols)
        pair_tables[horizon] = pairs
    return (latest, models, pair_tables)


def _forecast_quality_flags(latest: pd.DataFrame) -> pd.Series:
    uncertainty = latest.state_uncertainty_proxy
    observed = latest.observed_model_features
    q50_u, q90_u = uncertainty.quantile([0.5, 0.9])
    q50_o = observed.quantile(0.5)
    flags = np.where((uncertainty <= q50_u) & (observed >= q50_o), 'high', np.where((uncertainty <= q90_u) & (observed >= 20), 'standard', 'limited'))
    return pd.Series(flags, index=latest.index)


def latest_joint_simulation(states: pd.DataFrame, information: pd.DataFrame, rolling: pd.DataFrame, masking: pd.DataFrame, fitted_latest: dict[int, FittedPointModel], *, draws: int=600, peer_draws: int=180, seed: int=DEFAULT_SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, np.ndarray], dict[int, np.ndarray]]:
    state_cols = _state_columns(states)
    latest = states.sort_values('forecast_origin_year').groupby('entity_code', observed=True).tail(1)
    latest_year = int(latest.forecast_origin_year.max())
    latest = latest.loc[latest.forecast_origin_year.eq(latest_year)].copy()
    latest = latest.merge(information, on=['entity_code', 'forecast_origin_year'], validate='one_to_one').sort_values('entity_code').reset_index(drop=True)
    latest['forecast_quality'] = _forecast_quality_flags(latest)
    supported = latest.forecast_quality.ne('limited').to_numpy()
    if supported.sum() < 10:
        raise Phase3Error('Too few supported latest states for peer simulation')
    calibrator = StateErrorCalibrator(masking)
    peer_scale = states[state_cols].std(ddof=0).replace(0, 1).to_numpy(dtype=float)
    current_values = latest[state_cols].to_numpy(dtype=float)
    current_center = current_values[supported].mean(axis=0)
    current_distance = _rms_radius((current_values - current_center) / peer_scale)
    current_percentile = np.full(len(latest), np.nan)
    current_percentile[supported] = pd.Series(current_distance[supported]).rank(pct=True, method='average').to_numpy()
    forecast_rows = []
    coordinate_rows = []
    peer_rows = []
    draws_by_horizon: dict[int, np.ndarray] = {}
    point_by_horizon: dict[int, np.ndarray] = {}
    for horizon in (1, 2):
        model = fitted_latest[horizon]
        point = model.predict_raw(current_values)
        point_by_horizon[horizon] = point
        pool = make_residual_pool(rolling, state_cols, horizon, model='transition')
        sensitivity = model.sensitivity_raw()
        rng = np.random.default_rng(seed + horizon * 10000)
        n_country = len(latest)
        n_state = len(state_cols)
        uncertainty_edges = rolling.loc[rolling.horizon.eq(horizon), 'current_state_uncertainty_proxy'].quantile([0.25, 0.5, 0.75]).to_numpy()
        uncertainty_groups = np.searchsorted(uncertainty_edges, latest.state_uncertainty_proxy.to_numpy(dtype=float), side='right').astype(int)
        current_error = np.empty((draws, n_country, n_state), dtype=np.float32)
        idio = np.empty_like(current_error)
        for i, row in latest.iterrows():
            country_rng = np.random.default_rng(seed + horizon * 1000003 + i * 997)
            direction_idx = country_rng.integers(0, len(pool.directions), size=draws)
            magnitude = calibrator.sample_magnitude(float(row.state_uncertainty_proxy), draws, country_rng)
            current_error[:, i, :] = (pool.directions[direction_idx] * magnitude[:, None]).astype(np.float32)
            idio_idx = _matched_indices(pool.idio_groups, int(uncertainty_groups[i]), draws, country_rng)
            idio[:, i, :] = pool.idio[idio_idx].astype(np.float32)
        propagated = (current_error.reshape(-1, n_state) @ sensitivity).reshape(draws, n_country, n_state)
        common_draw = pool.common[rng.integers(0, len(pool.common), size=draws)][:, None, :]
        deviation = propagated + common_draw + idio
        raw_radius = _rms_radius(deviation)
        raw_q80 = np.quantile(raw_radius, 0.8, axis=0)
        raw_q95 = np.quantile(raw_radius, 0.95, axis=0)
        inflation = np.ones(n_country, dtype=float)
        for i, group in enumerate(uncertainty_groups):
            conformal = _empirical_radius_quantiles(pool, int(group))
            inflation[i] = max(1.0, conformal[0.8] / max(float(raw_q80[i]), 1e-08), conformal[0.95] / max(float(raw_q95[i]), 1e-08))
        deviation *= inflation[None, :, None]
        simulation = (point[None, :, :] + deviation).astype(np.float32)
        draws_by_horizon[horizon] = simulation
        future_center = simulation[:, supported, :].mean(axis=1)
        centered = (simulation - future_center[:, None, :]) / peer_scale[None, None, :]
        future_distance = _rms_radius(centered)
        future_percentile = np.full_like(future_distance, np.nan)
        supported_indices = np.flatnonzero(supported)
        for draw_index in range(draws):
            future_percentile[draw_index, supported] = pd.Series(future_distance[draw_index, supported]).rank(pct=True, method='average').to_numpy()
        contemporary_distance = _rms_radius((simulation - current_center[None, None, :]) / peer_scale[None, None, :])
        movement_radius = _rms_radius(simulation - current_values[None, :, :])
        peer_counts = [dict() for _ in range(n_country)]
        used_peer_draws = min(peer_draws, draws)
        supported_entities = latest.loc[supported, 'entity_code'].to_numpy()
        peer_values = (simulation[:used_peer_draws, supported, :] / peer_scale[None, None, :]).astype(np.float32, copy=False)
        batch_size = 32
        nearest_batches = []
        for start in range(0, used_peer_draws, batch_size):
            values = peer_values[start:start + batch_size]
            norms = np.sum(values * values, axis=2)
            gram = np.einsum('dnp,dmp->dnm', values, values, optimize=True)
            distance2 = np.maximum(norms[:, :, None] + norms[:, None, :] - 2.0 * gram, 0.0)
            diagonal = np.arange(distance2.shape[1])
            distance2[:, diagonal, diagonal] = np.inf
            nearest_batches.append(np.argmin(distance2, axis=2))
        nearest = np.concatenate(nearest_batches, axis=0)
        for draw_index in range(used_peer_draws):
            for local_i, global_i in enumerate(supported_indices):
                neighbour = supported_entities[nearest[draw_index, local_i]]
                peer_counts[global_i][neighbour] = peer_counts[global_i].get(neighbour, 0) + 1
        for i, row in latest.iterrows():
            state_draws = simulation[:, i, :]
            quantile = np.quantile(state_draws, [0.1, 0.25, 0.5, 0.75, 0.9], axis=0)
            for j, col in enumerate(state_cols):
                coordinate_rows.append({'entity_code': row.entity_code, 'current_state_year': int(row.forecast_origin_year), 'forecast_year': int(row.forecast_origin_year + horizon), 'horizon': horizon, 'state_coordinate': col, 'point': float(point[i, j]), 'q10': float(quantile[0, j]), 'q25': float(quantile[1, j]), 'q50': float(quantile[2, j]), 'q75': float(quantile[3, j]), 'q90': float(quantile[4, j])})
            radius_q = np.quantile(movement_radius[:, i], [0.5, 0.8, 0.95])
            if supported[i]:
                pct_q = np.nanquantile(future_percentile[:, i], [0.1, 0.5, 0.9])
                prob_improve = float(np.nanmean(future_percentile[:, i] < current_percentile[i]))
                prob_deteriorate = float(np.nanmean(future_percentile[:, i] > current_percentile[i]))
                prob_farther = float(np.mean(contemporary_distance[:, i] > current_distance[i]))
                prob_closer = float(np.mean(contemporary_distance[:, i] < current_distance[i]))
            else:
                pct_q = [np.nan, np.nan, np.nan]
                prob_improve = prob_deteriorate = prob_farther = prob_closer = np.nan
            forecast_rows.append({'entity_code': row.entity_code, 'current_state_year': int(row.forecast_origin_year), 'forecast_year': int(row.forecast_origin_year + horizon), 'horizon': horizon, 'forecast_quality': row.forecast_quality, 'observed_model_features': int(row.observed_model_features), 'observed_share': float(row.observed_share), 'state_uncertainty_proxy': float(row.state_uncertainty_proxy), 'uncertainty_group': int(uncertainty_groups[i]), 'calibration_inflation': float(inflation[i]), 'movement_radius_q50': float(radius_q[0]), 'movement_radius_q80': float(radius_q[1]), 'movement_radius_q95': float(radius_q[2]), 'current_peer_distance_percentile': float(current_percentile[i]) if supported[i] else np.nan, 'future_peer_percentile_q10': float(pct_q[0]), 'future_peer_percentile_q50': float(pct_q[1]), 'future_peer_percentile_q90': float(pct_q[2]), 'probability_relative_percentile_improves': prob_improve, 'probability_relative_percentile_deteriorates': prob_deteriorate, 'probability_farther_from_contemporary_peer_center': prob_farther, 'probability_closer_to_contemporary_peer_center': prob_closer})
            if supported[i]:
                for neighbour, count in sorted(peer_counts[i].items(), key=lambda item: (-item[1], item[0]))[:10]:
                    peer_rows.append({'entity_code': row.entity_code, 'horizon': horizon, 'forecast_year': int(row.forecast_origin_year + horizon), 'peer_entity': neighbour, 'frequency': count, 'probability': count / used_peer_draws, 'peer_draws': used_peer_draws})
    return (pd.DataFrame(forecast_rows), pd.DataFrame(coordinate_rows), pd.DataFrame(peer_rows), draws_by_horizon, point_by_horizon)


def historical_analogues(states: pd.DataFrame, information: pd.DataFrame, latest_summary: pd.DataFrame, point_by_horizon: dict[int, np.ndarray], *, neighbours: int=8) -> pd.DataFrame:
    state_cols = _state_columns(states)
    data = states.merge(information, on=['entity_code', 'forecast_origin_year'], validate='one_to_one').sort_values(['entity_code', 'forecast_origin_year']).reset_index(drop=True)
    previous = data[['entity_code', 'forecast_origin_year', *state_cols]].copy()
    previous['forecast_origin_year'] += 1
    previous = previous.rename(columns={c: f'previous_{c}' for c in state_cols})
    data = data.merge(previous, on=['entity_code', 'forecast_origin_year'], how='left')
    current = data[state_cols].to_numpy(dtype=float)
    prior = data[[f'previous_{c}' for c in state_cols]].to_numpy(dtype=float)
    velocity = current - prior
    velocity[~np.isfinite(velocity).all(axis=1)] = 0.0
    scale_state = np.where(np.std(current, axis=0) > 1e-08, np.std(current, axis=0), 1.0)
    scale_velocity = np.where(np.std(velocity, axis=0) > 1e-08, np.std(velocity, axis=0), 1.0)
    design = np.column_stack([current / scale_state, velocity / scale_velocity])
    latest_entities = set(latest_summary.entity_code.astype(str))
    latest_rows = data.sort_values('forecast_origin_year').groupby('entity_code', observed=True).tail(1)
    latest_rows = latest_rows.loc[latest_rows.entity_code.astype(str).isin(latest_entities)]
    latest_indices = latest_rows.index.to_numpy()
    latest_design = design[latest_indices]
    latest_year = int(latest_rows.forecast_origin_year.max())
    candidate_mask = (data.forecast_origin_year <= latest_year - 2) & (data.observed_model_features >= 20)
    candidates = data.loc[candidate_mask].copy()
    candidate_design = design[np.flatnonzero(candidate_mask.to_numpy())]
    if len(candidates) <= neighbours:
        raise Phase3Error('Insufficient analogue candidates')
    search = NearestNeighbors(n_neighbors=min(neighbours + 10, len(candidates)), algorithm='brute').fit(candidate_design)
    distance, index = search.kneighbors(latest_design)
    candidate_records = candidates.reset_index(drop=True)
    state_lookup = states.set_index(['entity_code', 'forecast_origin_year'])[state_cols]
    latest_order = latest_rows.entity_code.tolist()
    records = []
    for i, entity in enumerate(latest_order):
        emitted = 0
        for d, idx in zip(distance[i], index[i]):
            row = candidate_records.iloc[int(idx)]
            if row.entity_code == entity and row.forecast_origin_year == latest_rows.iloc[i].forecast_origin_year:
                continue
            base = row[state_cols].to_numpy(dtype=float)
            record = {'entity_code': entity, 'analogue_entity': row.entity_code, 'analogue_origin_year': int(row.forecast_origin_year), 'analogue_distance': float(d / np.sqrt(design.shape[1])), 'analogue_observed_share': float(row.observed_share), 'analogue_state_uncertainty': float(row.state_uncertainty_proxy), 'same_entity_history': bool(row.entity_code == entity)}
            for horizon in (1, 2):
                key = (row.entity_code, int(row.forecast_origin_year + horizon))
                if key in state_lookup.index:
                    future = state_lookup.loc[key].to_numpy(dtype=float)
                    movement = future - base
                    record[f'realized_{horizon}y_movement_radius'] = float(_rms_radius(movement[None, :])[0])
                    point_current = point_by_horizon[horizon][i] - latest_rows.iloc[i][state_cols].to_numpy(dtype=float)
                    record[f'realized_{horizon}y_direction_cosine_vs_current_forecast'] = float(_safe_cosine(movement[None, :], point_current[None, :])[0])
                else:
                    record[f'realized_{horizon}y_movement_radius'] = np.nan
                    record[f'realized_{horizon}y_direction_cosine_vs_current_forecast'] = np.nan
            records.append(record)
            emitted += 1
            if emitted >= neighbours:
                break
    return pd.DataFrame(records)


def observable_implications(latest_states: pd.DataFrame, latest_summary: pd.DataFrame, loadings: pd.DataFrame, draws_by_horizon: dict[int, np.ndarray], point_by_horizon: dict[int, np.ndarray], *, top_each_direction: int=12, projection_draws: int=250) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_cols = _state_columns(latest_states)
    load_state_cols = [c for c in state_cols if c in loadings.columns]
    if load_state_cols != state_cols:
        raise Phase3Error('Measurement loading state schema mismatch')
    loading_matrix = loadings[state_cols].to_numpy(dtype=float)
    metadata = [c for c in ['predictor_id', 'feature_id', 'source', 'INDICATOR', 'indicator_label', 'UNIT', 'unit_policy'] if c in loadings.columns]
    latest = latest_states.sort_values('entity_code').reset_index(drop=True)
    current = latest[state_cols].to_numpy(dtype=float)
    records = []
    feature_summary = []
    for horizon in (1, 2):
        point_delta = point_by_horizon[horizon] - current
        implication = point_delta @ loading_matrix.T
        selected_all = set()
        for i, entity in enumerate(latest.entity_code):
            order = np.argsort(implication[i])
            selected = np.unique(np.concatenate([order[:top_each_direction], order[-top_each_direction:]]))
            selected_all.update(selected.tolist())
            draw_count = min(projection_draws, draws_by_horizon[horizon].shape[0])
            draw_delta = draws_by_horizon[horizon][:draw_count, i, :] - current[i][None, :]
            projected = draw_delta @ loading_matrix[selected].T
            q = np.quantile(projected, [0.1, 0.5, 0.9], axis=0)
            for position, feature_index in enumerate(selected):
                meta = loadings.iloc[int(feature_index)]
                record = {'entity_code': entity, 'horizon': horizon, 'forecast_year': int(latest.iloc[i].forecast_origin_year + horizon), 'expected_standardized_change': float(implication[i, feature_index]), 'q10_standardized_change': float(q[0, position]), 'q50_standardized_change': float(q[1, position]), 'q90_standardized_change': float(q[2, position]), 'direction': 'increase' if implication[i, feature_index] >= 0 else 'decrease'}
                record.update({c: meta[c] for c in metadata})
                records.append(record)
        values = implication[:, sorted(selected_all)]
        for position, feature_index in enumerate(sorted(selected_all)):
            meta = loadings.iloc[int(feature_index)]
            record = {'horizon': horizon, 'median_expected_standardized_change': float(np.median(values[:, position])), 'q10_country_expected_change': float(np.quantile(values[:, position], 0.1)), 'q90_country_expected_change': float(np.quantile(values[:, position], 0.9)), 'share_countries_positive': float(np.mean(values[:, position] > 0))}
            record.update({c: meta[c] for c in metadata})
            feature_summary.append(record)
    return (pd.DataFrame(records), pd.DataFrame(feature_summary))
