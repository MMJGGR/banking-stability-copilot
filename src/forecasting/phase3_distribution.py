"""Phase 3 empirical uncertainty decomposition and calibration."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from .phase3_common import Phase3Error, _rms_radius, _safe_cosine, _state_columns

class StateErrorCalibrator:
    """Empirical monotone mapping from state-information uncertainty to error."""

    def __init__(self, masking: pd.DataFrame):
        required = {'uncertainty_proxy', 'state_error'}
        if not required <= set(masking.columns):
            raise Phase3Error('Incomplete masking calibration evidence')
        clean = masking.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required)).copy()
        if len(clean) < 20:
            raise Phase3Error('Insufficient masking calibration rows')
        order = clean.sort_values('uncertainty_proxy')
        self.model_ = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(order.uncertainty_proxy.to_numpy(dtype=float), order.state_error.to_numpy(dtype=float))
        self.edges_ = np.unique(np.quantile(order.uncertainty_proxy, [0, 0.25, 0.5, 0.75, 1]))
        if len(self.edges_) < 3:
            self.edges_ = np.array([-np.inf, np.inf])
        self.data_ = order
        self.global_errors_ = order.state_error.to_numpy(dtype=float)

    def expected(self, uncertainty: np.ndarray | float) -> np.ndarray:
        values = np.asarray(uncertainty, dtype=float)
        return np.asarray(self.model_.predict(values.reshape(-1))).reshape(values.shape)

    def group(self, uncertainty: float) -> int:
        if len(self.edges_) <= 2:
            return 0
        return int(np.clip(np.searchsorted(self.edges_[1:-1], uncertainty, side='right'), 0, 3))

    def sample_magnitude(self, uncertainty: float, size: int, rng: np.random.Generator) -> np.ndarray:
        group = self.group(float(uncertainty))
        if len(self.edges_) > 2:
            low = self.edges_[group]
            high = self.edges_[group + 1]
            if group == len(self.edges_) - 2:
                subset = self.data_.loc[self.data_.uncertainty_proxy.between(low, high, inclusive='both'), 'state_error'].to_numpy(dtype=float)
            else:
                subset = self.data_.loc[(self.data_.uncertainty_proxy >= low) & (self.data_.uncertainty_proxy < high), 'state_error'].to_numpy(dtype=float)
        else:
            subset = self.global_errors_
        if len(subset) < 10:
            subset = self.global_errors_
        sampled = rng.choice(subset, size=size, replace=True)
        expected = float(self.expected(np.array([uncertainty]))[0])
        denominator = max(float(np.mean(subset)), 1e-08)
        return sampled * expected / denominator

def _unit_direction_pool(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1)
    valid = norm > 1e-10
    if not valid.any():
        raise Phase3Error('No nonzero residual direction pool')
    return vectors[valid] / norm[valid, None]

@dataclass
class ResidualPool:
    common: np.ndarray
    idio: np.ndarray
    idio_groups: np.ndarray
    total_radii: np.ndarray
    radius_groups: np.ndarray
    directions: np.ndarray

def make_residual_pool(data: pd.DataFrame, state_cols: list[str], horizon: int, *, model: str='transition', exclude_target_year: int | None=None) -> ResidualPool:
    subset = data[data.horizon.eq(horizon)].copy()
    if exclude_target_year is not None:
        subset = subset[~subset.target_year.eq(exclude_target_year)]
    if subset.empty:
        raise Phase3Error('Empty residual pool')
    prefix = 'residual' if model == 'transition' else 'no_change_residual'
    common_cols = [f'{prefix}_common_{c}' for c in state_cols]
    idio_cols = [f'{prefix}_idio_{c}' for c in state_cols]
    total_cols = [f'{prefix}_{c}' for c in state_cols]
    common = subset[['target_year', *common_cols]].drop_duplicates('target_year')[common_cols].to_numpy(dtype=float)
    idio = subset[idio_cols].to_numpy(dtype=float)
    total = subset[total_cols].to_numpy(dtype=float)
    groups = subset.uncertainty_group.to_numpy(dtype=int)
    return ResidualPool(common=common, idio=idio, idio_groups=groups, total_radii=_rms_radius(total), radius_groups=groups, directions=_unit_direction_pool(idio))

def _matched_indices(groups: np.ndarray, requested_group: int, size: int, rng: np.random.Generator) -> np.ndarray:
    candidates = np.flatnonzero(groups == requested_group)
    if len(candidates) < 30:
        candidates = np.arange(len(groups))
    return rng.choice(candidates, size=size, replace=True)

def _empirical_radius_quantiles(pool: ResidualPool, group: int) -> dict[float, float]:
    radius = pool.total_radii[pool.radius_groups == group]
    if len(radius) < 30:
        radius = pool.total_radii
    q = np.quantile(radius, [0.5, 0.8, 0.95])
    return {0.5: float(q[0]), 0.8: float(q[1]), 0.95: float(q[2])}

def simulate_distribution(current: np.ndarray, point: np.ndarray, sensitivity: np.ndarray, current_uncertainty: float, uncertainty_group: int, calibrator: StateErrorCalibrator, pool: ResidualPool, *, draws: int, rng: np.random.Generator, conformal_floor: bool=True) -> tuple[np.ndarray, dict]:
    direction_idx = rng.integers(0, len(pool.directions), size=draws)
    state_magnitude = calibrator.sample_magnitude(current_uncertainty, draws, rng)
    current_error = pool.directions[direction_idx] * state_magnitude[:, None]
    propagated = current_error @ sensitivity
    common = pool.common[rng.integers(0, len(pool.common), size=draws)]
    idio_idx = _matched_indices(pool.idio_groups, uncertainty_group, draws, rng)
    idio = pool.idio[idio_idx]
    deviation = propagated + common + idio
    raw_radius = _rms_radius(deviation)
    raw_q = dict(zip((0.5, 0.8, 0.95), np.quantile(raw_radius, [0.5, 0.8, 0.95])))
    conformal = _empirical_radius_quantiles(pool, uncertainty_group)
    inflation = 1.0
    if conformal_floor:
        inflation = max(1.0, conformal[0.8] / max(float(raw_q[0.8]), 1e-08), conformal[0.95] / max(float(raw_q[0.95]), 1e-08))
    deviation *= inflation
    draws_state = point[None, :] + deviation
    final_radius = _rms_radius(deviation)
    final_q = dict(zip((0.5, 0.8, 0.95), np.quantile(final_radius, [0.5, 0.8, 0.95])))
    return (draws_state, {'inflation': float(inflation), 'raw_q50': float(raw_q[0.5]), 'raw_q80': float(raw_q[0.8]), 'raw_q95': float(raw_q[0.95]), 'q50': float(final_q[0.5]), 'q80': float(final_q[0.8]), 'q95': float(final_q[0.95]), 'conformal_q80': conformal[0.8], 'conformal_q95': conformal[0.95]})

def _deterministic_group_sample(data: pd.DataFrame, maximum_rows: int) -> pd.DataFrame:
    if len(data) <= maximum_rows:
        return data.copy()
    work = data.copy()
    work['_sample_hash'] = [int.from_bytes(hashlib.sha256(f'{entity}|{int(origin)}|{int(target)}'.encode()).digest()[:8], 'big') for entity, origin, target in zip(work.entity_code, work.forecast_origin_year, work.target_year)]
    quota = max(1, maximum_rows // max(work.uncertainty_group.nunique(), 1))
    selected = work.sort_values('_sample_hash').groupby('uncertainty_group', observed=True, group_keys=False).head(quota)
    if len(selected) < maximum_rows:
        remainder = work.loc[~work.index.isin(selected.index)].sort_values('_sample_hash')
        selected = pd.concat([selected, remainder.head(maximum_rows - len(selected))], axis=0)
    return selected.drop(columns='_sample_hash').head(maximum_rows)

def _simulate_batch_radii(points: np.ndarray, sensitivity: np.ndarray, uncertainties: np.ndarray, groups: np.ndarray, calibrator: StateErrorCalibrator, pool: ResidualPool, *, draws: int, rng: np.random.Generator) -> dict:
    points = np.asarray(points, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)
    groups = np.asarray(groups, dtype=int)
    n_rows, n_state = points.shape
    current_error = np.empty((draws, n_rows, n_state), dtype=np.float32)
    idio = np.empty_like(current_error)
    for i in range(n_rows):
        direction_idx = rng.integers(0, len(pool.directions), size=draws)
        magnitude = calibrator.sample_magnitude(float(uncertainties[i]), draws, rng)
        current_error[:, i, :] = (pool.directions[direction_idx] * magnitude[:, None]).astype(np.float32)
        idio_idx = _matched_indices(pool.idio_groups, int(groups[i]), draws, rng)
        idio[:, i, :] = pool.idio[idio_idx].astype(np.float32)
    propagated = (current_error.reshape(-1, n_state) @ sensitivity).reshape(draws, n_rows, n_state)
    common_idx = rng.integers(0, len(pool.common), size=(draws, n_rows))
    common = pool.common[common_idx]
    deviation = propagated + common + idio
    radius = _rms_radius(deviation)
    raw_q = np.quantile(radius, [0.5, 0.8, 0.95], axis=0)
    conformal = np.empty((3, n_rows), dtype=float)
    for i in range(n_rows):
        q = _empirical_radius_quantiles(pool, int(groups[i]))
        conformal[:, i] = [q[0.5], q[0.8], q[0.95]]
    inflation = np.maximum.reduce([np.ones(n_rows), conformal[1] / np.maximum(raw_q[1], 1e-08), conformal[2] / np.maximum(raw_q[2], 1e-08)])
    final_q = raw_q * inflation[None, :]
    return {'raw_q': raw_q, 'conformal_q': conformal, 'inflation': inflation, 'q': final_q}

def leave_target_year_out_calibration(rolling: pd.DataFrame, state_cols: list[str], *, max_rows_per_target_year: int | None=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-target-year-out empirical radial calibration."""
    rows = []
    for model_kind in ('transition', 'no_change'):
        prefix = 'row_error_radius' if model_kind == 'transition' else 'no_change_error_radius'
        for horizon in (1, 2):
            subset = rolling[rolling.horizon.eq(horizon)].copy()
            for target_year, full_year_data in subset.groupby('target_year', sort=True):
                if len(full_year_data) < 10:
                    continue
                year_data = full_year_data
                if max_rows_per_target_year is not None:
                    year_data = _deterministic_group_sample(full_year_data, max_rows_per_target_year)
                pool = subset.loc[~subset.target_year.eq(target_year)].copy()
                if pool.empty:
                    continue
                for row in year_data.itertuples(index=False):
                    matched = pool.loc[pool.uncertainty_group.eq(int(row.uncertainty_group)), prefix].to_numpy(dtype=float)
                    if len(matched) < 30:
                        matched = pool[prefix].to_numpy(dtype=float)
                    q50, q80, q95 = np.quantile(matched, [0.5, 0.8, 0.95])
                    actual_radius = float(getattr(row, prefix))
                    rows.append({'model': model_kind, 'horizon': horizon, 'entity_code': row.entity_code, 'forecast_origin_year': int(row.forecast_origin_year), 'target_year': int(target_year), 'uncertainty_group': int(row.uncertainty_group), 'coverage_group': int(row.coverage_group), 'actual_radius': actual_radius, 'inflation': 1.0, 'q50': float(q50), 'q80': float(q80), 'q95': float(q95), 'covered_50': bool(actual_radius <= q50), 'covered_80': bool(actual_radius <= q80), 'covered_95': bool(actual_radius <= q95)})
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise Phase3Error('No calibration rows')
    summaries = []
    groupings = [('overall', ['model', 'horizon']), ('uncertainty', ['model', 'horizon', 'uncertainty_group']), ('coverage', ['model', 'horizon', 'coverage_group'])]
    for dimension, keys in groupings:
        for values, data in detail.groupby(keys, observed=True):
            if not isinstance(values, tuple):
                values = (values,)
            record = {'breakdown': dimension, **dict(zip(keys, values)), 'rows': len(data)}
            for nominal in (50, 80, 95):
                record[f'coverage_{nominal}'] = float(data[f'covered_{nominal}'].mean())
                record[f'mean_radius_{nominal}'] = float(data[f'q{nominal}'].mean())
            record['mean_actual_radius'] = float(data.actual_radius.mean())
            record['mean_inflation'] = 1.0
            summaries.append(record)
    return (detail, pd.DataFrame(summaries))

def uncertainty_component_summary(rolling: pd.DataFrame, masking: pd.DataFrame, state_cols: list[str]) -> pd.DataFrame:
    records = []
    state_error = masking.state_error.to_numpy(dtype=float)
    records.append({'horizon': 0, 'component': 'current_state_identification', 'rows': len(state_error), 'radius_mean': float(state_error.mean()), 'radius_q50': float(np.quantile(state_error, 0.5)), 'radius_q80': float(np.quantile(state_error, 0.8)), 'radius_q95': float(np.quantile(state_error, 0.95))})
    for horizon in (1, 2):
        data = rolling[rolling.horizon.eq(horizon)]
        common_cols = [f'residual_common_{c}' for c in state_cols]
        idio_cols = [f'residual_idio_{c}' for c in state_cols]
        common = data[['target_year', *common_cols]].drop_duplicates('target_year')
        components = {'shared_future_shock': _rms_radius(common[common_cols].to_numpy(dtype=float)), 'country_specific_transition': _rms_radius(data[idio_cols].to_numpy(dtype=float)), 'total_transition_error': data.row_error_radius.to_numpy(dtype=float)}
        for name, radius in components.items():
            records.append({'horizon': horizon, 'component': name, 'rows': len(radius), 'radius_mean': float(radius.mean()), 'radius_q50': float(np.quantile(radius, 0.5)), 'radius_q80': float(np.quantile(radius, 0.8)), 'radius_q95': float(np.quantile(radius, 0.95))})
    return pd.DataFrame(records)
