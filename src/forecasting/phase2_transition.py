"""Phase 2B: target-independent transition modelling for the learned state.

This module forecasts the complete Phase 2A state, not selected banking
variables. It compares no-change, simple autoregressive, regularized,
reduced-rank, shared-condition and historical-analogue transitions on
strictly later development windows. Results remain retrospective research.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

from .inventory import ForecastDataError


WINDOWS = ((2016, 2018), (2019, 2021))
ALPHAS = (0.1, 1.0, 10.0, 100.0)
ANALOGUE_K = (5, 10, 20)


def _state_columns(frame: pd.DataFrame) -> list[str]:
    columns = [c for c in frame if c.startswith("state_")]
    try:
        return sorted(columns, key=lambda c: int(c.split("_")[-1]))
    except ValueError as exc:
        raise ForecastDataError("Invalid state-column naming") from exc


def build_transition_pairs(
    states: pd.DataFrame,
    information: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Create exact-calendar state transitions and prior velocity.

    Every model for a horizon is evaluated on the same pair rows. No outcome
    other than the future learned state is read.
    """
    if horizon not in {1, 2}:
        raise ForecastDataError("Only registered one- and two-year horizons")
    required = {"entity_code", "forecast_origin_year"}
    if not required <= set(states) or not required <= set(information):
        raise ForecastDataError("Incomplete state/information schema")
    state_cols = _state_columns(states)
    if not state_cols:
        raise ForecastDataError("No state dimensions")
    if states.duplicated(list(required)).any():
        raise ForecastDataError("Duplicate country-year state")

    current = states[["entity_code", "forecast_origin_year", *state_cols]].copy()
    future = current.copy()
    future["forecast_origin_year"] -= horizon
    future = future.rename(columns={c: f"future_{c}" for c in state_cols})
    pairs = current.merge(
        future,
        on=["entity_code", "forecast_origin_year"],
        how="inner",
        validate="one_to_one",
    )
    pairs["target_year"] = pairs.forecast_origin_year + horizon
    pairs["horizon"] = horizon

    previous = current.copy()
    previous["forecast_origin_year"] += 1
    previous = previous.rename(columns={c: f"previous_{c}" for c in state_cols})
    pairs = pairs.merge(
        previous,
        on=["entity_code", "forecast_origin_year"],
        how="left",
        validate="one_to_one",
    )

    info = information[
        [
            "entity_code",
            "forecast_origin_year",
            "observed_share",
            "state_uncertainty_proxy",
            "effective_information",
        ]
    ].copy()
    pairs = pairs.merge(
        info.add_prefix("current_").rename(
            columns={
                "current_entity_code": "entity_code",
                "current_forecast_origin_year": "forecast_origin_year",
            }
        ),
        on=["entity_code", "forecast_origin_year"],
        how="left",
        validate="one_to_one",
    )
    future_info = info.copy()
    future_info["forecast_origin_year"] -= horizon
    future_info = future_info.add_prefix("future_").rename(
        columns={
            "future_entity_code": "entity_code",
            "future_forecast_origin_year": "forecast_origin_year",
        }
    )
    pairs = pairs.merge(
        future_info,
        on=["entity_code", "forecast_origin_year"],
        how="left",
        validate="one_to_one",
    )

    current_values = pairs[state_cols].to_numpy(dtype=float)
    previous_values = pairs[[f"previous_{c}" for c in state_cols]].to_numpy(dtype=float)
    velocity = current_values - previous_values
    missing_velocity = ~np.isfinite(velocity).all(axis=1)
    velocity[missing_velocity] = 0.0
    for j, column in enumerate(state_cols):
        pairs[f"velocity_{column}"] = velocity[:, j]
    pairs["velocity_observed"] = ~missing_velocity

    if not np.isfinite(
        pairs[state_cols + [f"future_{c}" for c in state_cols]].to_numpy(dtype=float)
    ).all():
        raise ForecastDataError("Nonfinite transition state")
    return pairs.sort_values(["forecast_origin_year", "entity_code"]).reset_index(drop=True)


@dataclass
class StateScale:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "StateScale":
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0, ddof=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


class DiagonalAR:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, x, y, sample_weight=None):
        self.models_ = []
        for j in range(x.shape[1]):
            model = Ridge(alpha=self.alpha)
            model.fit(x[:, [j]], y[:, j], sample_weight=sample_weight)
            self.models_.append(model)
        return self

    def predict(self, x):
        return np.column_stack(
            [model.predict(x[:, [j]]) for j, model in enumerate(self.models_)]
        )


class DeltaRidge:
    def __init__(self, alpha: float, use_global: bool = False):
        self.alpha = alpha
        self.use_global = use_global

    def fit(self, x, y, global_state=None, sample_weight=None):
        design = x
        if self.use_global:
            if global_state is None:
                raise ForecastDataError("Global state required")
            design = np.column_stack([x, global_state])
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(design, y - x, sample_weight=sample_weight)
        return self

    def predict(self, x, global_state=None):
        design = x
        if self.use_global:
            if global_state is None:
                raise ForecastDataError("Global state required")
            design = np.column_stack([x, global_state])
        return x + self.model_.predict(design)


class ReducedRankDelta:
    """Ridge delta forecast projected onto a data-selected output subspace."""

    def __init__(self, alpha: float, rank: int):
        self.alpha = alpha
        self.rank = rank

    def fit(self, x, y, sample_weight=None):
        if self.rank < 1 or self.rank > y.shape[1]:
            raise ForecastDataError("Invalid transition rank")
        self.model_ = Ridge(alpha=self.alpha)
        delta = y - x
        self.model_.fit(x, delta, sample_weight=sample_weight)
        fitted = self.model_.predict(x)
        if sample_weight is None:
            weight = np.ones(len(x))
        else:
            weight = np.asarray(sample_weight, dtype=float)
        self.delta_mean_ = np.average(fitted, axis=0, weights=weight)
        centered = (fitted - self.delta_mean_) * np.sqrt(weight)[:, None]
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.output_basis_ = vt[: self.rank].T
        return self

    def predict(self, x):
        fitted = self.model_.predict(x)
        projected = self.delta_mean_ + (
            (fitted - self.delta_mean_)
            @ self.output_basis_
            @ self.output_basis_.T
        )
        return x + projected


class AnalogueDelta:
    def __init__(self, neighbours: int):
        self.neighbours = neighbours

    def fit(self, x, y, velocity):
        if self.neighbours < 1 or self.neighbours >= len(x):
            raise ForecastDataError("Invalid analogue neighbour count")
        self.design_ = np.column_stack([x, velocity])
        self.delta_ = y - x
        self.search_ = NearestNeighbors(
            n_neighbors=self.neighbours,
            algorithm="brute",
            metric="euclidean",
        ).fit(self.design_)
        return self

    def predict(self, x, velocity):
        design = np.column_stack([x, velocity])
        distance, index = self.search_.kneighbors(design)
        weight = 1.0 / np.maximum(distance, 1e-8)
        weight /= weight.sum(axis=1, keepdims=True)
        delta = np.sum(self.delta_[index] * weight[:, :, None], axis=1)
        return x + delta

    def predict_training_loo(self, x, velocity):
        design = np.column_stack([x, velocity])
        search = NearestNeighbors(
            n_neighbors=min(self.neighbours + 1, len(x)),
            algorithm="brute",
            metric="euclidean",
        ).fit(self.design_)
        distance, index = search.kneighbors(design)
        distance = distance[:, 1:]
        index = index[:, 1:]
        weight = 1.0 / np.maximum(distance, 1e-8)
        weight /= weight.sum(axis=1, keepdims=True)
        delta = np.sum(self.delta_[index] * weight[:, :, None], axis=1)
        return x + delta


def _global_state(pairs: pd.DataFrame, state_cols: list[str], scale: StateScale) -> np.ndarray:
    means = pairs.groupby("forecast_origin_year", observed=True)[state_cols].transform("mean")
    return scale.transform(means.to_numpy(dtype=float))


def _arrays(pairs: pd.DataFrame, scale: StateScale):
    state_cols = [c for c in pairs if c.startswith("state_")]
    state_cols = sorted(state_cols, key=lambda c: int(c.split("_")[-1]))
    x_raw = pairs[state_cols].to_numpy(dtype=float)
    y_raw = pairs[[f"future_{c}" for c in state_cols]].to_numpy(dtype=float)
    velocity_raw = pairs[[f"velocity_{c}" for c in state_cols]].to_numpy(dtype=float)
    x = scale.transform(x_raw)
    y = scale.transform(y_raw)
    velocity = velocity_raw / scale.scale
    global_state = _global_state(pairs, state_cols, scale)
    uncertainty = (
        pairs.current_state_uncertainty_proxy.to_numpy(dtype=float) ** 2
        + pairs.future_state_uncertainty_proxy.to_numpy(dtype=float) ** 2
    )
    weight = 1.0 / np.maximum(uncertainty, 1e-4)
    lower, upper = np.quantile(weight, [0.02, 0.98])
    weight = np.clip(weight, lower, upper)
    weight /= np.mean(weight)
    return x, y, velocity, global_state, weight


def transition_metrics(x, y, prediction, training_residual_radius=None):
    error = prediction - y
    row_rmse = np.sqrt(np.mean(error**2, axis=1))
    baseline = np.sqrt(np.mean((x - y) ** 2, axis=1))
    true_move = y - x
    predicted_move = prediction - x
    true_norm = np.linalg.norm(true_move, axis=1)
    predicted_norm = np.linalg.norm(predicted_move, axis=1)
    good = (true_norm > 1e-10) & (predicted_norm > 1e-10)
    cosine = np.full(len(x), np.nan)
    cosine[good] = np.sum(true_move[good] * predicted_move[good], axis=1) / (
        true_norm[good] * predicted_norm[good]
    )
    result = {
        "rows": len(x),
        "state_rmse": float(np.sqrt(np.mean(error**2))),
        "state_mae": float(np.mean(np.abs(error))),
        "mean_row_rmse": float(np.mean(row_rmse)),
        "fraction_rows_beating_no_change": float(np.mean(row_rmse < baseline)),
        "mean_movement_cosine": float(np.nanmean(cosine)) if good.any() else None,
        "movement_magnitude_rmse": float(
            np.sqrt(np.mean((predicted_norm - true_norm) ** 2))
        ),
    }
    if training_residual_radius is not None and len(training_residual_radius):
        q80, q95 = np.quantile(training_residual_radius, [0.80, 0.95])
        result.update(
            {
                "residual_radius_q80": float(q80),
                "residual_radius_q95": float(q95),
                "test_coverage_80": float(np.mean(row_rmse <= q80)),
                "test_coverage_95": float(np.mean(row_rmse <= q95)),
            }
        )
    return result


def _fit_model(name, params, x, y, velocity, global_state, weight):
    if name == "diagonal_ar":
        return DiagonalAR(params["alpha"]).fit(x, y, sample_weight=weight)
    if name == "ridge_delta":
        return DeltaRidge(params["alpha"]).fit(x, y, sample_weight=weight)
    if name == "global_ridge_delta":
        return DeltaRidge(params["alpha"], use_global=True).fit(
            x,
            y,
            global_state=global_state,
            sample_weight=weight,
        )
    if name == "reduced_rank_delta":
        return ReducedRankDelta(params["alpha"], params["rank"]).fit(
            x,
            y,
            sample_weight=weight,
        )
    if name == "analogue_delta":
        return AnalogueDelta(params["neighbours"]).fit(x, y, velocity)
    raise ForecastDataError(f"Unknown transition model {name}")


def _predict_model(name, model, x, velocity, global_state):
    if name == "global_ridge_delta":
        return model.predict(x, global_state=global_state)
    if name == "analogue_delta":
        return model.predict(x, velocity)
    return model.predict(x)


def _training_prediction(name, model, x, velocity, global_state):
    if name == "analogue_delta":
        return model.predict_training_loo(x, velocity)
    return _predict_model(name, model, x, velocity, global_state)


def _candidate_parameters(state_rank: int):
    candidates = []
    for alpha in ALPHAS:
        candidates.extend(
            [
                ("diagonal_ar", {"alpha": alpha}),
                ("ridge_delta", {"alpha": alpha}),
                ("global_ridge_delta", {"alpha": alpha}),
            ]
        )
        for rank in (2, 4, 8, 16, 32, 64, 96):
            if rank <= state_rank:
                candidates.append(
                    ("reduced_rank_delta", {"alpha": alpha, "rank": rank})
                )
    for neighbours in ANALOGUE_K:
        candidates.append(("analogue_delta", {"neighbours": neighbours}))
    return candidates


def run_transition_development(
    states: pd.DataFrame,
    information: pd.DataFrame,
    *,
    windows=WINDOWS,
) -> dict:
    """Nested, time-ordered development comparisons for full future states."""
    state_cols = _state_columns(states)
    fold_records = []
    prediction_records = []
    tuning_records = []
    skipped = []

    for horizon in (1, 2):
        pairs = build_transition_pairs(states, information, horizon)
        for start, end in windows:
            train = pairs[
                (pairs.forecast_origin_year < start)
                & (pairs.target_year < start)
            ].copy()
            test = pairs[
                pairs.forecast_origin_year.between(start, end)
            ].copy()
            inner_start = start - 3
            inner_train = train[
                (train.forecast_origin_year < inner_start)
                & (train.target_year < inner_start)
            ].copy()
            inner_test = train[
                train.forecast_origin_year.between(inner_start, start - 1)
                & (train.target_year < start)
            ].copy()
            identity = {
                "horizon": horizon,
                "outer_start": start,
                "outer_end": end,
                "train_rows": len(train),
                "test_rows": len(test),
                "inner_train_rows": len(inner_train),
                "inner_test_rows": len(inner_test),
            }
            if min(len(train), len(inner_train)) < 40 or min(len(test), len(inner_test)) < 5:
                skipped.append({**identity, "reason": "insufficient_transition_rows"})
                continue

            inner_scale = StateScale.fit(
                inner_train[state_cols].to_numpy(dtype=float)
            )
            ix, iy, iv, ig, iw = _arrays(inner_train, inner_scale)
            vx, vy, vv, vg, _ = _arrays(inner_test, inner_scale)
            candidate_results = []
            candidates = _candidate_parameters(len(state_cols))
            for name, params in candidates:
                if name == "analogue_delta" and params["neighbours"] >= len(ix):
                    continue
                model = _fit_model(name, params, ix, iy, iv, ig, iw)
                prediction = _predict_model(name, model, vx, vv, vg)
                metric = transition_metrics(vx, vy, prediction)
                candidate_results.append(
                    {
                        "model": name,
                        "params": json.dumps(params, sort_keys=True),
                        **metric,
                    }
                )
            tuning = pd.DataFrame(candidate_results).sort_values(
                ["state_rmse", "model", "params"]
            )
            if tuning.empty:
                skipped.append({**identity, "reason": "no_transition_candidate"})
                continue
            chosen = (
                tuning.groupby("model", as_index=False)
                .first()
                .sort_values("model")
            )
            tuning_records.extend(
                [{**identity, **row} for row in tuning.to_dict("records")]
            )

            scale = StateScale.fit(train[state_cols].to_numpy(dtype=float))
            tx, ty, tv, tg, tw = _arrays(train, scale)
            ex, ey, ev, eg, _ = _arrays(test, scale)

            baseline_prediction = ex.copy()
            baseline_train = tx.copy()
            baseline_radius = np.sqrt(np.mean((baseline_train - ty) ** 2, axis=1))
            baseline_metric = transition_metrics(
                ex,
                ey,
                baseline_prediction,
                baseline_radius,
            )
            fold_records.append({**identity, "model": "no_change", **baseline_metric})

            baseline_frame = test[["entity_code", "forecast_origin_year", "target_year"]].copy()
            baseline_frame["model"] = "no_change"
            baseline_frame["row_rmse"] = np.sqrt(
                np.mean((baseline_prediction - ey) ** 2, axis=1)
            )
            baseline_frame["baseline_row_rmse"] = baseline_frame.row_rmse
            prediction_records.append(baseline_frame)

            for row in chosen.itertuples(index=False):
                name = row.model
                params = json.loads(row.params)
                if name == "analogue_delta" and params["neighbours"] >= len(tx):
                    continue
                model = _fit_model(name, params, tx, ty, tv, tg, tw)
                train_prediction = _training_prediction(name, model, tx, tv, tg)
                test_prediction = _predict_model(name, model, ex, ev, eg)
                train_radius = np.sqrt(
                    np.mean((train_prediction - ty) ** 2, axis=1)
                )
                metric = transition_metrics(ex, ey, test_prediction, train_radius)
                fold_records.append(
                    {
                        **identity,
                        "model": name,
                        "params": json.dumps(params, sort_keys=True),
                        **metric,
                    }
                )
                frame = test[["entity_code", "forecast_origin_year", "target_year"]].copy()
                frame["model"] = name
                frame["params"] = json.dumps(params, sort_keys=True)
                frame["row_rmse"] = np.sqrt(
                    np.mean((test_prediction - ey) ** 2, axis=1)
                )
                frame["baseline_row_rmse"] = np.sqrt(
                    np.mean((ex - ey) ** 2, axis=1)
                )
                prediction_records.append(frame)

    if not fold_records:
        raise ForecastDataError("No transition development fold executed")

    folds = pd.DataFrame(fold_records)
    predictions = pd.concat(prediction_records, ignore_index=True)
    tuning = pd.DataFrame(tuning_records)
    aggregate = []
    for (horizon, model), data in folds.groupby(["horizon", "model"], observed=True):
        weight = data.test_rows.to_numpy(dtype=float)
        record = {
            "horizon": int(horizon),
            "model": model,
            "folds": len(data),
            "rows": int(data.test_rows.sum()),
        }
        for field in [
            "state_rmse",
            "state_mae",
            "mean_row_rmse",
            "fraction_rows_beating_no_change",
            "movement_magnitude_rmse",
            "test_coverage_80",
            "test_coverage_95",
        ]:
            if field in data and data[field].notna().any():
                record[field] = float(np.average(data[field], weights=weight))
        if "mean_movement_cosine" in data and data.mean_movement_cosine.notna().any():
            valid = data.mean_movement_cosine.notna()
            record["mean_movement_cosine"] = float(
                np.average(data.loc[valid, "mean_movement_cosine"], weights=weight[valid])
            )
        aggregate.append(record)

    aggregate_frame = pd.DataFrame(aggregate)
    winners = []
    for horizon, data in aggregate_frame.groupby("horizon"):
        baseline = float(data.loc[data.model.eq("no_change"), "state_rmse"].iloc[0])
        challengers = data.loc[~data.model.eq("no_change")].sort_values("state_rmse")
        best = challengers.iloc[0]
        winners.append(
            {
                "horizon": int(horizon),
                "no_change_state_rmse": baseline,
                "best_challenger": best.model,
                "best_challenger_state_rmse": float(best.state_rmse),
                "relative_rmse_improvement": float((baseline - best.state_rmse) / baseline),
                "challenger_beats_no_change": bool(best.state_rmse < baseline),
            }
        )

    return {
        "status": "completed_phase2b_retrospective_state_transition_development",
        "fold_metrics": folds,
        "aggregate_metrics": aggregate_frame,
        "predictions": predictions,
        "tuning": tuning,
        "winners": winners,
        "skipped": skipped,
        "state_rank": len(state_cols),
        "supervised_banking_targets_read": 0,
        "crisis_labels_read": 0,
        "final_confirmation_evaluated": False,
        "representation_caveat": (
            "Phase 2A loadings are fitted on the retrospective research panel; "
            "this is development evidence, not a real-time transition backtest."
        ),
    }
