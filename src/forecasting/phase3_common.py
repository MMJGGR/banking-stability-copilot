"""Phase 3 common state-transition models and rolling point forecasts."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

STATE_PREFIX = "state_"
FIXED_POINT_MODELS = {1: {"model": "ridge_delta", "alpha": 100.0}, 2: {"model": "diagonal_ar", "alpha": 0.1}}
DEFAULT_SEED = 37

class Phase3Error(ValueError):
    """Invalid Phase 3 data or configuration."""

def _state_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if c.startswith(STATE_PREFIX)]
    try:
        cols = sorted(cols, key=lambda c: int(c.split('_')[-1]))
    except ValueError as exc:
        raise Phase3Error('Invalid state column naming') from exc
    if not cols:
        raise Phase3Error('No state columns')
    return cols

def _rms_radius(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.sqrt(np.mean(values * values, axis=-1))

def _safe_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    valid = (na > 1e-12) & (nb > 1e-12)
    out = np.full(len(a), np.nan, dtype=float)
    out[valid] = np.sum(a[valid] * b[valid], axis=1) / (na[valid] * nb[valid])
    return out

def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return value

def write_json(path: Path, value: dict) -> None:
    Path(path).write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + '\n')

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

@dataclass
class StateScale:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> 'StateScale':
        values = np.asarray(values, dtype=float)
        mean = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0)
        scale = np.where(scale > 1e-08, scale, 1.0)
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.mean

class RidgeDelta:

    def __init__(self, alpha: float):
        self.alpha = float(alpha)

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight=None):
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(x, y - x, sample_weight=sample_weight)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x + self.model_.predict(x)

    def sensitivity(self) -> np.ndarray:
        coef = np.asarray(self.model_.coef_, dtype=float)
        return np.eye(coef.shape[1]) + coef.T

class DiagonalAR:

    def __init__(self, alpha: float):
        self.alpha = float(alpha)

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight=None):
        self.models_ = []
        for j in range(x.shape[1]):
            model = Ridge(alpha=self.alpha)
            model.fit(x[:, [j]], y[:, j], sample_weight=sample_weight)
            self.models_.append(model)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([model.predict(x[:, [j]]) for j, model in enumerate(self.models_)])

    def sensitivity(self) -> np.ndarray:
        return np.diag([float(model.coef_[0]) for model in self.models_])

@dataclass
class FittedPointModel:
    horizon: int
    model_name: str
    scale: StateScale
    model: object
    state_columns: list[str]

    def predict_raw(self, values: np.ndarray) -> np.ndarray:
        return self.scale.inverse(self.model.predict(self.scale.transform(values)))

    def sensitivity_raw(self) -> np.ndarray:
        a = self.model.sensitivity()
        return a * self.scale.scale[None, :] / self.scale.scale[:, None]

def build_transition_pairs(states: pd.DataFrame, information: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon not in (1, 2):
        raise Phase3Error('Only one- and two-year horizons are registered')
    required = {'entity_code', 'forecast_origin_year'}
    if not required <= set(states.columns) or not required <= set(information.columns):
        raise Phase3Error('Incomplete states/information schema')
    state_cols = _state_columns(states)
    if states.duplicated(['entity_code', 'forecast_origin_year']).any():
        raise Phase3Error('Duplicate country-year state')
    current = states[['entity_code', 'forecast_origin_year', *state_cols]].copy()
    future = current.copy()
    future['forecast_origin_year'] -= horizon
    future = future.rename(columns={c: f'future_{c}' for c in state_cols})
    pairs = current.merge(future, on=['entity_code', 'forecast_origin_year'], how='inner', validate='one_to_one')
    pairs['target_year'] = pairs.forecast_origin_year + horizon
    pairs['horizon'] = horizon
    info_cols = ['entity_code', 'forecast_origin_year', 'observed_model_features', 'observed_share', 'state_uncertainty_proxy', 'effective_information']
    missing = [c for c in info_cols if c not in information]
    if missing:
        raise Phase3Error(f'Missing information fields: {missing}')
    current_info = information[info_cols].copy()
    pairs = pairs.merge(current_info.add_prefix('current_').rename(columns={'current_entity_code': 'entity_code', 'current_forecast_origin_year': 'forecast_origin_year'}), on=['entity_code', 'forecast_origin_year'], how='left', validate='one_to_one')
    future_info = current_info.copy()
    future_info['forecast_origin_year'] -= horizon
    pairs = pairs.merge(future_info.add_prefix('future_').rename(columns={'future_entity_code': 'entity_code', 'future_forecast_origin_year': 'forecast_origin_year'}), on=['entity_code', 'forecast_origin_year'], how='left', validate='one_to_one')
    if pairs[[f'future_{c}' for c in state_cols]].isna().any().any():
        raise Phase3Error('Incomplete future state')
    return pairs.sort_values(['forecast_origin_year', 'entity_code']).reset_index(drop=True)

def _transition_arrays(pairs: pd.DataFrame, state_cols: list[str], scale: StateScale):
    x_raw = pairs[state_cols].to_numpy(dtype=float)
    y_raw = pairs[[f'future_{c}' for c in state_cols]].to_numpy(dtype=float)
    x = scale.transform(x_raw)
    y = scale.transform(y_raw)
    uncertainty = pairs.current_state_uncertainty_proxy.to_numpy(dtype=float) ** 2 + pairs.future_state_uncertainty_proxy.to_numpy(dtype=float) ** 2
    weight = 1.0 / np.maximum(uncertainty, 0.0001)
    lo, hi = np.quantile(weight, [0.02, 0.98])
    weight = np.clip(weight, lo, hi)
    weight /= weight.mean()
    return (x_raw, y_raw, x, y, weight)

def fit_fixed_point_model(train: pd.DataFrame, horizon: int, state_cols: list[str]) -> FittedPointModel:
    spec = FIXED_POINT_MODELS[horizon]
    scale = StateScale.fit(train[state_cols].to_numpy(dtype=float))
    _, _, x, y, weight = _transition_arrays(train, state_cols, scale)
    if spec['model'] == 'ridge_delta':
        model = RidgeDelta(spec['alpha']).fit(x, y, sample_weight=weight)
    elif spec['model'] == 'diagonal_ar':
        model = DiagonalAR(spec['alpha']).fit(x, y, sample_weight=weight)
    else:
        raise Phase3Error('Unknown fixed point model')
    return FittedPointModel(horizon, spec['model'], scale, model, state_cols)

def point_metrics(current: np.ndarray, actual: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - actual
    baseline_error = current - actual
    row_rmse = _rms_radius(error)
    baseline = _rms_radius(baseline_error)
    true_move = actual - current
    predicted_move = prediction - current
    cosine = _safe_cosine(true_move, predicted_move)
    return {'rows': len(current), 'state_rmse': float(np.sqrt(np.mean(error * error))), 'mean_row_rmse': float(row_rmse.mean()), 'no_change_state_rmse': float(np.sqrt(np.mean(baseline_error * baseline_error))), 'relative_rmse_improvement': float((np.sqrt(np.mean(baseline_error * baseline_error)) - np.sqrt(np.mean(error * error))) / np.sqrt(np.mean(baseline_error * baseline_error))), 'fraction_rows_beating_no_change': float(np.mean(row_rmse < baseline)), 'mean_movement_cosine': float(np.nanmean(cosine)) if np.isfinite(cosine).any() else None}

def reproduce_phase2_windows(states: pd.DataFrame, information: pd.DataFrame, windows=((2016, 2018), (2019, 2021))) -> pd.DataFrame:
    """Reproduce the registered Phase 2 point models on the original scale used there."""
    state_cols = _state_columns(states)
    records = []
    for horizon in (1, 2):
        pairs = build_transition_pairs(states, information, horizon)
        for start, end in windows:
            train = pairs[(pairs.forecast_origin_year < start) & (pairs.target_year < start)].copy()
            test = pairs[pairs.forecast_origin_year.between(start, end)].copy()
            model = fit_fixed_point_model(train, horizon, state_cols)
            current_raw = test[state_cols].to_numpy(dtype=float)
            actual_raw = test[[f'future_{c}' for c in state_cols]].to_numpy(dtype=float)
            prediction_raw = model.predict_raw(current_raw)
            current = model.scale.transform(current_raw)
            actual = model.scale.transform(actual_raw)
            prediction = model.model.predict(current)
            scaled = point_metrics(current, actual, prediction)
            raw = point_metrics(current_raw, actual_raw, prediction_raw)
            records.append({'horizon': horizon, 'outer_start': start, 'outer_end': end, 'model': model.model_name, **scaled, **{f'raw_{key}': value for key, value in raw.items() if key != 'rows'}})
    return pd.DataFrame(records)

def rolling_point_forecasts(states: pd.DataFrame, information: pd.DataFrame, *, minimum_train_rows: int=300, minimum_train_target_years: int=5, minimum_test_rows: int=10) -> tuple[pd.DataFrame, dict[tuple[int, int], FittedPointModel]]:
    state_cols = _state_columns(states)
    frames = []
    fitted: dict[tuple[int, int], FittedPointModel] = {}
    for horizon in (1, 2):
        pairs = build_transition_pairs(states, information, horizon)
        for origin in sorted(pairs.forecast_origin_year.unique()):
            test = pairs[pairs.forecast_origin_year.eq(origin)].copy()
            train = pairs[(pairs.forecast_origin_year < origin) & (pairs.target_year <= origin)].copy()
            if len(train) < minimum_train_rows or train.target_year.nunique() < minimum_train_target_years or len(test) < minimum_test_rows:
                continue
            model = fit_fixed_point_model(train, horizon, state_cols)
            current = test[state_cols].to_numpy(dtype=float)
            actual = test[[f'future_{c}' for c in state_cols]].to_numpy(dtype=float)
            prediction = model.predict_raw(current)
            residual = actual - prediction
            no_change_residual = actual - current
            base = test[['entity_code', 'forecast_origin_year', 'target_year', 'horizon', 'current_observed_model_features', 'current_observed_share', 'current_state_uncertainty_proxy', 'current_effective_information']].reset_index(drop=True)
            arrays = np.column_stack([current, actual, prediction, residual, no_change_residual])
            columns = [f'current_{c}' for c in state_cols] + [f'actual_{c}' for c in state_cols] + [f'point_{c}' for c in state_cols] + [f'residual_{c}' for c in state_cols] + [f'no_change_residual_{c}' for c in state_cols]
            frame = pd.concat([base, pd.DataFrame(arrays, columns=columns)], axis=1)
            frame['row_error_radius'] = _rms_radius(residual)
            frame['no_change_error_radius'] = _rms_radius(no_change_residual)
            frame['movement_cosine'] = _safe_cosine(actual - current, prediction - current)
            frames.append(frame)
            fitted[horizon, int(origin)] = model
    if not frames:
        raise Phase3Error('No rolling forecasts generated')
    return (pd.concat(frames, ignore_index=True), fitted)

def add_residual_decomposition(rolling: pd.DataFrame, state_cols: list[str]) -> pd.DataFrame:
    data = rolling.copy()
    additions = []
    for prefix in ('residual', 'no_change_residual'):
        cols = [f'{prefix}_{c}' for c in state_cols]
        common = data.groupby(['horizon', 'target_year'], observed=True)[cols].transform('mean')
        common.columns = [f'{prefix}_common_{c}' for c in state_cols]
        idio = data[cols].to_numpy(dtype=float) - common.to_numpy(dtype=float)
        idio = pd.DataFrame(idio, columns=[f'{prefix}_idio_{c}' for c in state_cols], index=data.index)
        additions.extend([common, idio])
    data = pd.concat([data, *additions], axis=1).copy()
    data['uncertainty_group'] = -1
    data['coverage_group'] = -1
    for _, idx in data.groupby('horizon').groups.items():
        idx = list(idx)
        for field, output in (('current_state_uncertainty_proxy', 'uncertainty_group'), ('current_observed_share', 'coverage_group')):
            ranked = data.loc[idx, field].rank(method='first', pct=True)
            data.loc[idx, output] = np.minimum((ranked * 4).astype(int), 3)
    return data
