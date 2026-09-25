"""Phase 6 time-ordered hazard models and replacement risk rating.

The module compares transparent regularized binary-event models on fixed
Phase 2 states, exact trajectory measures and registered crisis-specific raw
measurements. It also converts the selected combined-event model into the
three-view rating requested by the owner: peer-relative risk, own-history risk
and absolute event imminence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


DEFAULT_WINDOWS = ((2001, 2007), (2008, 2014), (2015, 2021))
C_GRID = (0.03, 0.3)
PEER_WEIGHT_GRID = tuple(np.round(np.linspace(0.0, 1.0, 11), 1))
MIN_TRAIN_POSITIVES = 15
MIN_TEST_POSITIVES = 2
MIN_FEATURE_OBSERVATIONS = 20
RECALL_FLOOR = 0.60


def _state_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [
            column for column in frame.columns
            if str(column).startswith("state_")
            and str(column).split("_")[-1].isdigit()
        ],
        key=lambda column: int(str(column).split("_")[-1]),
    )


def _velocity_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [
            column for column in frame.columns
            if str(column).startswith("velocity_state_")
            and str(column).split("_")[-1].isdigit()
        ],
        key=lambda column: int(str(column).split("_")[-1]),
    )


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def candidate_feature_groups(
    frame: pd.DataFrame,
    raw_groups: Mapping[str, Sequence[str]],
    event_family: str,
) -> dict[str, list[str]]:
    """Return auditable candidate feature groups without target screening."""
    states = _state_columns(frame)
    velocity = _velocity_columns(frame)
    summary = [
        column for column in (
            "state_norm",
            "velocity_norm",
            "acceleration_norm",
            "own_state_anomaly",
            "own_state_distance",
            "own_history_years",
            "own_history_supported",
            "state_uncertainty_proxy",
            "observed_share",
            "effective_information",
            "velocity_observed_share",
            "acceleration_observed_share",
        )
        if column in frame.columns
    ]
    quality = [
        column for column in (
            "state_uncertainty_proxy", "observed_share", "effective_information"
        )
        if column in frame.columns
    ]
    if event_family == "banking":
        raw = list(raw_groups.get("banking_raw", ()))
    elif event_family == "sovereign":
        raw = list(raw_groups.get("sovereign_raw", ()))
    elif event_family == "combined":
        raw = list(raw_groups.get("all_raw", ()))
    else:
        raise ValueError(f"Unknown event family: {event_family}")
    raw = [column for column in raw if column in frame.columns]
    return {
        "state_only": _unique([*states, *quality]),
        "state_trajectory": _unique([*states, *velocity, *summary]),
        "raw_targeted": _unique([*raw, *summary]),
        "hybrid": _unique([*states, *velocity, *raw, *summary]),
    }


def _country_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("country_code", observed=True).country_code.transform("size")
    weights = 1.0 / counts.to_numpy(dtype=float)
    return weights / np.mean(weights)


@dataclass
class PreparedLogit:
    feature_columns: list[str]
    active_columns: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    model: LogisticRegression

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.reindex(columns=self.active_columns).apply(
            pd.to_numeric, errors="coerce"
        )
        transformed = self.imputer.transform(values)
        transformed = self.scaler.transform(transformed)
        return self.model.predict_proba(transformed)[:, 1]


def fit_regularized_logit(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    *,
    c_value: float,
    minimum_observations: int = MIN_FEATURE_OBSERVATIONS,
) -> PreparedLogit:
    """Fit L2 logistic regression after fold-local admission and imputation."""
    if c_value <= 0:
        raise ValueError("c_value must be positive")
    y = pd.to_numeric(train[target_column], errors="raise").astype(int)
    if y.nunique() != 2:
        raise ValueError("Binary model requires both target classes")
    values = train.reindex(columns=list(feature_columns)).apply(
        pd.to_numeric, errors="coerce"
    )
    observed = values.notna().sum()
    distinct = values.nunique(dropna=True)
    active = observed.index[
        observed.ge(min(minimum_observations, max(2, len(train) // 20)))
        & distinct.gt(1)
    ].tolist()
    if not active:
        raise ValueError("No learnable feature columns in training fold")
    active_values = values[active]
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    transformed = imputer.fit_transform(active_values)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(transformed)
    solver = "liblinear" if transformed.shape[1] <= 500 else "saga"
    model = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver=solver,
        max_iter=2500,
        tol=1e-4,
        random_state=17,
        n_jobs=2 if solver == "saga" else None,
    )
    model.fit(
        transformed,
        y.to_numpy(),
        sample_weight=_country_balanced_weights(train),
    )
    return PreparedLogit(list(feature_columns), active, imputer, scaler, model)


@dataclass
class ProbabilityCalibrator:
    model: LogisticRegression | None

    def transform(self, probability: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        if self.model is None:
            return p
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        return self.model.predict_proba(logit)[:, 1]


def fit_platt_calibrator(y: Sequence[int], probability: Sequence[float]) -> ProbabilityCalibrator:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives < 3 or negatives < 10 or len(np.unique(p)) < 3:
        return ProbabilityCalibrator(None)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(
        C=1e6,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=17,
    ).fit(logit, y)
    return ProbabilityCalibrator(model)


def event_metrics(y: Sequence[int], probability: Sequence[float]) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    result = {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "event_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_probability": float(p.mean()),
    }
    if len(np.unique(y)) == 2:
        result["pr_auc"] = float(average_precision_score(y, p))
        result["roc_auc"] = float(roc_auc_score(y, p))
    else:
        result["pr_auc"] = np.nan
        result["roc_auc"] = np.nan
    return result


def review_threshold(y: Sequence[int], probability: Sequence[float], recall_floor=RECALL_FLOOR) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    if y.sum() == 0:
        return 1.0
    precision, recall, thresholds = precision_recall_curve(y, p)
    candidates = [
        (float(precision[index]), float(threshold))
        for index, threshold in enumerate(thresholds)
        if recall[index] >= recall_floor
    ]
    if not candidates:
        return float(np.quantile(p, 0.90))
    return max(candidates, key=lambda item: (item[0], item[1]))[1]


def alert_metrics(y: Sequence[int], probability: Sequence[float], threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    alert = p >= threshold
    true_positive = int(np.sum(alert & (y == 1)))
    false_positive = int(np.sum(alert & (y == 0)))
    positives = int(np.sum(y == 1))
    alerts = int(alert.sum())
    return {
        "threshold": float(threshold),
        "alerts": alerts,
        "alert_rate": float(alerts / len(y)) if len(y) else np.nan,
        "recall": float(true_positive / positives) if positives else np.nan,
        "precision": float(true_positive / alerts) if alerts else np.nan,
        "false_alerts": false_positive,
        "false_alerts_per_true_alert": (
            float(false_positive / true_positive) if true_positive else np.nan
        ),
    }


def _percentile_against(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    if not len(reference):
        return np.full(len(values), np.nan)
    return np.searchsorted(reference, np.asarray(values, dtype=float), side="right") / len(reference)


def _peer_percentile(frame: pd.DataFrame, probability_column: str) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, index in frame.groupby("forecast_origin_year", observed=True).groups.items():
        values = frame.loc[index, probability_column].to_numpy(dtype=float)
        order = pd.Series(values).rank(method="average", pct=True).to_numpy()
        result.loc[index] = order
    return result


def _history_percentile(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    probability_column: str,
    minimum_history: int = 3,
) -> pd.Series:
    result = pd.Series(np.nan, index=current.index, dtype=float)
    by_country = {
        country: group.sort_values("forecast_origin_year")
        for country, group in reference.groupby("country_code", observed=True)
    }
    for index, row in current.iterrows():
        history = by_country.get(row.country_code)
        if history is None:
            continue
        values = history.loc[
            history.forecast_origin_year < row.forecast_origin_year,
            probability_column,
        ].dropna().to_numpy(dtype=float)
        if len(values) < minimum_history:
            continue
        result.loc[index] = _percentile_against(values, [row[probability_column]])[0]
    return result


def rating_components(
    reference_predictions: pd.DataFrame,
    current_predictions: pd.DataFrame,
    *,
    peer_weight: float,
    probability_column: str = "probability",
) -> pd.DataFrame:
    if not 0 <= peer_weight <= 1:
        raise ValueError("peer_weight must be in [0,1]")
    result = current_predictions.copy()
    result["peer_risk_percentile"] = _peer_percentile(result, probability_column)
    result["own_history_risk_percentile"] = _history_percentile(
        reference_predictions, result, probability_column
    )
    result["own_history_supported"] = result.own_history_risk_percentile.notna()
    history = result.own_history_risk_percentile.fillna(result.peer_risk_percentile)
    result["relative_risk_component"] = (
        peer_weight * result.peer_risk_percentile + (1 - peer_weight) * history
    )
    result["imminence_component"] = _percentile_against(
        reference_predictions[probability_column].dropna().to_numpy(dtype=float),
        result[probability_column].to_numpy(dtype=float),
    )
    result["risk_intensity"] = np.maximum(
        result.relative_risk_component, result.imminence_component
    )
    result["replacement_risk_score"] = (
        1 + 9 * result.risk_intensity
    ).clip(1, 10).round(1)
    result["replacement_risk_category"] = result.replacement_risk_score.map(
        risk_category
    )
    result["peer_weight"] = peer_weight
    return result


def select_peer_weight(
    reference_predictions: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    *,
    target_column: str,
    probability_column: str = "probability",
    grid=PEER_WEIGHT_GRID,
) -> tuple[float, pd.DataFrame]:
    records = []
    for weight in grid:
        scored = rating_components(
            reference_predictions,
            validation_predictions,
            peer_weight=float(weight),
            probability_column=probability_column,
        )
        y = validation_predictions[target_column].to_numpy(dtype=int)
        brier = brier_score_loss(y, scored.risk_intensity.to_numpy(dtype=float))
        records.append({
            "peer_weight": float(weight),
            "brier": float(brier),
            "own_history_supported_share": float(scored.own_history_supported.mean()),
        })
    trials = pd.DataFrame(records)
    minimum = trials.brier.min()
    candidates = trials.loc[np.isclose(trials.brier, minimum, rtol=0, atol=1e-12)].copy()
    candidates["balance_distance"] = (candidates.peer_weight - 0.5).abs()
    selected = candidates.sort_values(["balance_distance", "peer_weight"]).iloc[0]
    return float(selected.peer_weight), trials


def risk_category(score: float) -> str:
    if pd.isna(score):
        return "Unrated"
    if score <= 2:
        return "1-2: Very Low Risk"
    if score <= 4:
        return "3-4: Low Risk"
    if score <= 6:
        return "5-6: Moderate Risk"
    if score <= 8:
        return "7-8: High Risk"
    return "9-10: Very High Risk"


def _inner_split(train: pd.DataFrame, outer_start: int, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    inner_start = outer_start - 5
    inner_train = train.loc[
        train.forecast_origin_year + horizon < inner_start
    ].copy()
    inner_validation = train.loc[
        train.forecast_origin_year.between(inner_start, outer_start - 1)
    ].copy()
    return inner_train, inner_validation


def _eligible_panel(
    frame: pd.DataFrame,
    target_prefix: str,
    horizon: int,
) -> tuple[pd.DataFrame, str, str]:
    target_column = f"{target_prefix}_{horizon}y"
    eligible_column = f"{target_prefix}_{horizon}y_eligible"
    if target_column not in frame or eligible_column not in frame:
        raise ValueError(f"Missing target fields for {target_prefix} {horizon}y")
    data = frame.loc[frame[eligible_column].astype(bool)].copy()
    data[target_column] = pd.to_numeric(data[target_column], errors="raise").astype(int)
    return data, target_column, eligible_column


def evaluate_event_family(
    frame: pd.DataFrame,
    *,
    event_family: str,
    target_prefix: str,
    raw_groups: Mapping[str, Sequence[str]],
    horizons=(1, 2, 3),
    windows=DEFAULT_WINDOWS,
    c_grid=C_GRID,
    minimum_train_positives=MIN_TRAIN_POSITIVES,
    minimum_test_positives=MIN_TEST_POSITIVES,
) -> dict:
    """Run nested time-ordered hazard comparisons for one event family."""
    feature_groups = candidate_feature_groups(frame, raw_groups, event_family)
    fold_rows, tuning_rows, test_predictions, reference_predictions = [], [], [], []
    weight_rows, skipped = [], []

    for horizon in horizons:
        data, target_column, _ = _eligible_panel(frame, target_prefix, horizon)
        for outer_start, outer_end in windows:
            train = data.loc[
                data.forecast_origin_year + horizon < outer_start
            ].copy()
            test = data.loc[
                data.forecast_origin_year.between(outer_start, outer_end)
            ].copy()
            inner_train, inner_validation = _inner_split(train, outer_start, horizon)
            identity = {
                "event_family": event_family,
                "target_prefix": target_prefix,
                "horizon": int(horizon),
                "outer_start": int(outer_start),
                "outer_end": int(outer_end),
                "train_rows": len(train),
                "train_positives": int(train[target_column].sum()),
                "test_rows": len(test),
                "test_positives": int(test[target_column].sum()),
                "inner_train_rows": len(inner_train),
                "inner_train_positives": int(inner_train[target_column].sum()),
                "inner_validation_rows": len(inner_validation),
                "inner_validation_positives": int(inner_validation[target_column].sum()),
            }
            if (
                identity["train_positives"] < minimum_train_positives
                or identity["test_positives"] < minimum_test_positives
                or identity["inner_train_positives"] < max(8, minimum_train_positives // 2)
                or identity["inner_validation_positives"] < 1
                or len(inner_validation) < 20
            ):
                skipped.append({**identity, "reason": "insufficient_time_ordered_events"})
                continue

            baseline_rate = float(train[target_column].mean())
            baseline_test = np.full(len(test), baseline_rate)
            fold_rows.append({
                **identity,
                "model": "event_rate",
                "selected_c": np.nan,
                "active_features": 0,
                **event_metrics(test[target_column], baseline_test),
            })
            baseline_frame = test[[
                "country_code", "forecast_origin_year", target_column
            ]].copy()
            baseline_frame = baseline_frame.rename(columns={target_column: "target"})
            baseline_frame["event_family"] = event_family
            baseline_frame["target_prefix"] = target_prefix
            baseline_frame["horizon"] = horizon
            baseline_frame["outer_start"] = outer_start
            baseline_frame["model"] = "event_rate"
            baseline_frame["probability"] = baseline_test
            baseline_frame["peer_weight"] = 0.5
            test_predictions.append(baseline_frame)

            for model_name, feature_columns in feature_groups.items():
                if not feature_columns:
                    skipped.append({**identity, "model": model_name, "reason": "no_features"})
                    continue
                candidates = []
                fitted_inner = {}
                for c_value in c_grid:
                    try:
                        model = fit_regularized_logit(
                            inner_train, feature_columns, target_column, c_value=c_value
                        )
                        raw_probability = model.predict_proba(inner_validation)
                        calibrator = fit_platt_calibrator(
                            inner_validation[target_column], raw_probability
                        )
                        probability = calibrator.transform(raw_probability)
                        metric = event_metrics(inner_validation[target_column], probability)
                        record = {
                            **identity,
                            "model": model_name,
                            "c_value": float(c_value),
                            "active_features": len(model.active_columns),
                            **metric,
                        }
                        candidates.append(record)
                        fitted_inner[float(c_value)] = (model, calibrator, probability)
                    except (ValueError, FloatingPointError):
                        continue
                if not candidates:
                    skipped.append({**identity, "model": model_name, "reason": "fit_failed"})
                    continue
                tuning_rows.extend(candidates)
                selected = sorted(
                    candidates,
                    key=lambda row: (
                        row["brier"], row["log_loss"],
                        -row["pr_auc"] if np.isfinite(row["pr_auc"]) else math.inf,
                        row["c_value"],
                    ),
                )[0]
                selected_c = float(selected["c_value"])
                inner_model, calibrator, inner_probability = fitted_inner[selected_c]
                inner_reference = inner_train[[
                    "country_code", "forecast_origin_year", target_column
                ]].copy().rename(columns={target_column: "target"})
                inner_reference["probability"] = calibrator.transform(
                    inner_model.predict_proba(inner_train)
                )
                inner_current = inner_validation[[
                    "country_code", "forecast_origin_year", target_column
                ]].copy().rename(columns={target_column: "target"})
                inner_current["probability"] = inner_probability
                selected_weight, trials = select_peer_weight(
                    inner_reference,
                    inner_current,
                    target_column="target",
                )
                trials["event_family"] = event_family
                trials["target_prefix"] = target_prefix
                trials["horizon"] = horizon
                trials["outer_start"] = outer_start
                trials["model"] = model_name
                weight_rows.append(trials)

                final_model = fit_regularized_logit(
                    train, feature_columns, target_column, c_value=selected_c
                )
                train_probability = calibrator.transform(final_model.predict_proba(train))
                test_probability = calibrator.transform(final_model.predict_proba(test))
                threshold = review_threshold(
                    inner_validation[target_column], inner_probability
                )
                alerts = alert_metrics(test[target_column], test_probability, threshold)
                fold_rows.append({
                    **identity,
                    "model": model_name,
                    "selected_c": selected_c,
                    "active_features": len(final_model.active_columns),
                    "peer_weight": selected_weight,
                    **event_metrics(test[target_column], test_probability),
                    **{f"review_{key}": value for key, value in alerts.items()},
                })

                current_frame = test[[
                    "country_code", "forecast_origin_year", target_column
                ]].copy().rename(columns={target_column: "target"})
                current_frame["event_family"] = event_family
                current_frame["target_prefix"] = target_prefix
                current_frame["horizon"] = horizon
                current_frame["outer_start"] = outer_start
                current_frame["model"] = model_name
                current_frame["probability"] = test_probability
                current_frame["peer_weight"] = selected_weight
                test_predictions.append(current_frame)

                reference_frame = train[[
                    "country_code", "forecast_origin_year", target_column
                ]].copy().rename(columns={target_column: "target"})
                reference_frame["event_family"] = event_family
                reference_frame["target_prefix"] = target_prefix
                reference_frame["horizon"] = horizon
                reference_frame["outer_start"] = outer_start
                reference_frame["model"] = model_name
                reference_frame["probability"] = train_probability
                reference_frame["peer_weight"] = selected_weight
                reference_predictions.append(reference_frame)

    folds = pd.DataFrame(fold_rows)
    tuning = pd.DataFrame(tuning_rows)
    predictions = pd.concat(test_predictions, ignore_index=True) if test_predictions else pd.DataFrame()
    references = pd.concat(reference_predictions, ignore_index=True) if reference_predictions else pd.DataFrame()
    weights = pd.concat(weight_rows, ignore_index=True) if weight_rows else pd.DataFrame()
    aggregate = []
    if not predictions.empty:
        for (horizon, model_name), group in predictions.groupby(
            ["horizon", "model"], observed=True
        ):
            aggregate.append({
                "event_family": event_family,
                "target_prefix": target_prefix,
                "horizon": int(horizon),
                "model": model_name,
                **event_metrics(group.target, group.probability),
            })
    aggregate_frame = pd.DataFrame(aggregate)
    if not aggregate_frame.empty:
        baseline = aggregate_frame.loc[
            aggregate_frame.model.eq("event_rate"), ["horizon", "brier", "log_loss", "pr_auc"]
        ].rename(columns={
            "brier": "event_rate_brier",
            "log_loss": "event_rate_log_loss",
            "pr_auc": "event_rate_pr_auc",
        })
        aggregate_frame = aggregate_frame.merge(baseline, on="horizon", how="left")
        aggregate_frame["brier_skill_vs_event_rate"] = (
            aggregate_frame.event_rate_brier - aggregate_frame.brier
        ) / aggregate_frame.event_rate_brier
        aggregate_frame["log_loss_skill_vs_event_rate"] = (
            aggregate_frame.event_rate_log_loss - aggregate_frame.log_loss
        ) / aggregate_frame.event_rate_log_loss
        aggregate_frame["passes_event_rate_gate"] = (
            aggregate_frame.brier.lt(aggregate_frame.event_rate_brier)
            & aggregate_frame.log_loss.lt(aggregate_frame.event_rate_log_loss)
            & aggregate_frame.pr_auc.ge(0.95 * aggregate_frame.event_rate_pr_auc)
        )
    return {
        "fold_metrics": folds,
        "aggregate_metrics": aggregate_frame,
        "test_predictions": predictions,
        "reference_predictions": references,
        "tuning": tuning,
        "weight_trials": weights,
        "feature_groups": feature_groups,
        "skipped": skipped,
    }


def select_development_model(aggregate: pd.DataFrame, horizon: int) -> dict:
    data = aggregate.loc[
        aggregate.horizon.eq(horizon) & ~aggregate.model.eq("event_rate")
    ].copy()
    if data.empty:
        return {"status": "no_challenger", "model": "event_rate"}
    passing = data.loc[data.passes_event_rate_gate].copy()
    pool = passing if not passing.empty else data
    selected = pool.sort_values(["brier", "log_loss", "model"]).iloc[0]
    return {
        "status": (
            "passes_event_rate_gate" if bool(selected.passes_event_rate_gate)
            else "best_challenger_does_not_pass_event_rate_gate"
        ),
        "model": str(selected.model),
        "brier": float(selected.brier),
        "log_loss": float(selected.log_loss),
        "pr_auc": float(selected.pr_auc),
        "roc_auc": float(selected.roc_auc),
        "brier_skill_vs_event_rate": float(selected.brier_skill_vs_event_rate),
    }


def build_oof_replacement_ratings(
    evaluation: dict,
    *,
    selected_model: str,
    horizon: int = 3,
) -> pd.DataFrame:
    tests = evaluation["test_predictions"]
    references = evaluation["reference_predictions"]
    if tests.empty or references.empty:
        return pd.DataFrame()
    tests = tests.loc[
        tests.horizon.eq(horizon) & tests.model.eq(selected_model)
    ].copy()
    references = references.loc[
        references.horizon.eq(horizon) & references.model.eq(selected_model)
    ].copy()
    results = []
    for outer_start, current in tests.groupby("outer_start", observed=True):
        reference = references.loc[references.outer_start.eq(outer_start)].copy()
        if reference.empty:
            continue
        peer_weight = float(current.peer_weight.dropna().iloc[0])
        scored = rating_components(reference, current, peer_weight=peer_weight)
        results.append(scored)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def category_diagnostics(ratings: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if ratings.empty:
        return pd.DataFrame(), {"monotonic_event_rate": False}
    order = [
        "1-2: Very Low Risk",
        "3-4: Low Risk",
        "5-6: Moderate Risk",
        "7-8: High Risk",
        "9-10: Very High Risk",
    ]
    grouped = ratings.groupby("replacement_risk_category", observed=True).agg(
        rows=("target", "size"),
        events=("target", "sum"),
        event_rate=("target", "mean"),
        mean_score=("replacement_risk_score", "mean"),
    ).reindex(order).dropna(subset=["rows"]).reset_index()
    monotonic = bool(
        np.all(np.diff(grouped.event_rate.to_numpy(dtype=float)) >= -1e-12)
    ) if len(grouped) >= 2 else False
    low = ratings.loc[ratings.replacement_risk_score.le(4), "target"]
    high = ratings.loc[ratings.replacement_risk_score.gt(6), "target"]
    summary = {
        "monotonic_event_rate": monotonic,
        "low_risk_event_rate": float(low.mean()) if len(low) else np.nan,
        "high_risk_event_rate": float(high.mean()) if len(high) else np.nan,
        "high_vs_low_event_rate_ratio": (
            float(high.mean() / low.mean())
            if len(high) and len(low) and low.mean() > 0 else np.nan
        ),
    }
    return grouped, summary


def latest_event_probabilities(
    frame: pd.DataFrame,
    *,
    event_family: str,
    target_prefix: str,
    raw_groups: Mapping[str, Sequence[str]],
    evaluation: dict,
    latest_year: int,
    horizons=(1, 2, 3),
) -> tuple[pd.DataFrame, dict]:
    feature_groups = candidate_feature_groups(frame, raw_groups, event_family)
    latest = frame.loc[frame.forecast_origin_year.eq(latest_year)].copy()
    outputs, model_records = [], {}
    for horizon in horizons:
        data, target_column, _ = _eligible_panel(frame, target_prefix, horizon)
        selection = select_development_model(evaluation["aggregate_metrics"], horizon)
        model_name = selection["model"]
        if model_name == "event_rate":
            probability = np.full(len(latest), float(data[target_column].mean()))
            reference_probability = np.full(len(data), float(data[target_column].mean()))
            selected_c = np.nan
            peer_weight = 0.5
        else:
            fold_metrics = evaluation["fold_metrics"].loc[
                evaluation["fold_metrics"].horizon.eq(horizon)
                & evaluation["fold_metrics"].model.eq(model_name)
            ]
            selected_c = float(fold_metrics.selected_c.dropna().mode().iloc[0])
            peer_weight = float(fold_metrics.peer_weight.dropna().median())
            columns = feature_groups[model_name]
            model = fit_regularized_logit(
                data, columns, target_column, c_value=selected_c
            )
            probability = model.predict_proba(latest)
            reference_probability = model.predict_proba(data)
        current = latest[["country_code", "forecast_origin_year"]].copy()
        current["probability"] = probability
        current["event_family"] = event_family
        current["horizon"] = horizon
        current["model"] = model_name
        current["selected_c"] = selected_c
        current["model_status"] = selection["status"]
        if horizon == 3:
            reference = data[["country_code", "forecast_origin_year"]].copy()
            reference["probability"] = reference_probability
            current = rating_components(
                reference, current, peer_weight=peer_weight
            )
        outputs.append(current)
        model_records[str(horizon)] = {
            **selection,
            "selected_c": None if pd.isna(selected_c) else selected_c,
            "peer_weight": peer_weight,
            "training_rows": len(data),
            "training_positives": int(data[target_column].sum()),
        }
    return pd.concat(outputs, ignore_index=True), model_records


def compare_latest_with_production(
    latest_ratings: pd.DataFrame,
    production_reference: pd.DataFrame,
) -> pd.DataFrame:
    production = production_reference.copy()
    if "entity_code" not in production and "country_code" in production:
        production = production.rename(columns={"country_code": "entity_code"})
    keep = [
        column for column in (
            "entity_code", "production_country_name", "production_risk_score",
            "production_risk_category", "production_crisis_probability",
        )
        if column in production
    ]
    production = production[keep].drop_duplicates("entity_code")
    challenger = latest_ratings.rename(columns={"country_code": "entity_code"})
    merged = challenger.merge(production, on="entity_code", how="left", validate="one_to_one")
    merged["score_difference"] = (
        merged.replacement_risk_score - merged.production_risk_score
    )
    merged["category_changed"] = (
        merged.replacement_risk_category != merged.production_risk_category
    ) & merged.production_risk_category.notna()
    return merged
