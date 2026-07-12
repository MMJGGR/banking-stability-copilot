"""Leakage-resistant validation for rare-event crisis classifiers.

The functions in this module deliberately separate model development from model
assessment.  For every outer fold they:

1. generate out-of-fold predictions inside the outer training sample;
2. cross-fit an optional probability calibrator on those predictions;
3. select the operating threshold from the cross-fitted training predictions;
4. refit preprocessing and the estimator on the complete outer training sample;
5. freeze the calibrator and threshold before touching the outer test sample.

Two public evaluation designs are provided:

``evaluate_country_grouped``
    Every country is held out exactly once.  No country's observations can be in
    both sides of an outer fold.

``evaluate_forward_temporal``
    Expanding-window evaluation.  Every test origin is strictly later than its
    training observations; inner threshold selection is expanding-window too.

``evaluate_outer_split`` is useful for an official, pre-declared holdout.  It is
also the smallest surface on which to audit that test labels cannot influence
the selected threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EstimatorFactory = Callable[[], BaseEstimator]
PreprocessorFactory = Callable[[pd.DataFrame], TransformerMixin]


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration shared by grouped and forward validation.

    ``calibration`` may be ``"none"``, ``"sigmoid"`` (Platt scaling), or
    ``"isotonic"``.  Calibration is learned from predictions that were already
    out-of-fold within the outer training sample.
    """

    outer_splits: int = 5
    inner_splits: int = 4
    calibration: str = "sigmoid"
    recall_floor: float = 0.60
    random_state: int = 42
    bootstrap_iterations: int = 200
    confidence_level: float = 0.95
    temporal_outer_splits: int = 4
    temporal_inner_splits: int = 3
    temporal_min_train_periods: int | None = None
    temporal_gap_periods: int = 0
    purge_overlapping_events: bool = True

    def __post_init__(self) -> None:
        if self.outer_splits < 2 or self.inner_splits < 2:
            raise ValueError("outer_splits and inner_splits must both be at least 2")
        if self.temporal_outer_splits < 1 or self.temporal_inner_splits < 1:
            raise ValueError("temporal split counts must be positive")
        if self.calibration not in {"none", "sigmoid", "isotonic"}:
            raise ValueError("calibration must be 'none', 'sigmoid', or 'isotonic'")
        if not 0 < self.recall_floor <= 1:
            raise ValueError("recall_floor must be in (0, 1]")
        if self.bootstrap_iterations < 0:
            raise ValueError("bootstrap_iterations cannot be negative")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.temporal_gap_periods < 0:
            raise ValueError("temporal_gap_periods cannot be negative")


@dataclass
class ValidationResult:
    """Auditable result of one validation design."""

    design: str
    summary: dict[str, Any]
    fold_metrics: pd.DataFrame
    ledger: pd.DataFrame
    tuning_ledger: pd.DataFrame
    bootstrap_cis: pd.DataFrame
    fold_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, include_ledgers: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "design": self.design,
            "summary": self.summary,
            "fold_metrics": self.fold_metrics.to_dict(orient="records"),
            "bootstrap_cis": self.bootstrap_cis.to_dict(orient="records"),
            "fold_details": self.fold_details,
        }
        if include_ledgers:
            payload["ledger"] = self.ledger.to_dict(orient="records")
            payload["tuning_ledger"] = self.tuning_ledger.to_dict(
                orient="records"
            )
        return payload


@dataclass
class _PreparedData:
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    times: np.ndarray | None
    metadata: pd.DataFrame
    source_index: np.ndarray


class _ProbabilityCalibrator:
    """Small calibration adapter fitted only to OOF training predictions."""

    def __init__(self, method: str, random_state: int) -> None:
        self.method = method
        self.random_state = random_state
        self.model: Any | None = None
        self.fitted_method = "none"

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "_ProbabilityCalibrator":
        probabilities = np.asarray(probabilities, dtype=float)
        y = np.asarray(y, dtype=int)
        valid = np.isfinite(probabilities)
        probabilities, y = probabilities[valid], y[valid]

        # Returning an identity mapping is safer than fitting an unstable or
        # undefined calibrator to a single class or a tiny tuning sample.
        if self.method == "none" or len(y) < 4 or np.unique(y).size < 2:
            self.fitted_method = "none"
            return self

        if self.method == "sigmoid":
            self.model = LogisticRegression(
                solver="liblinear",
                C=1.0,
                max_iter=1_000,
                random_state=self.random_state,
            )
            self.model.fit(self._logit(probabilities), y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(probabilities, y)
        else:  # ValidationConfig checks this; retain a defensive guard.
            raise ValueError(f"Unsupported calibration method: {self.method}")
        self.fitted_method = self.method
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=float)
        if self.model is None:
            calibrated = probabilities
        elif self.fitted_method == "sigmoid":
            calibrated = self.model.predict_proba(self._logit(probabilities))[:, 1]
        else:
            calibrated = self.model.predict(probabilities)
        return np.clip(np.asarray(calibrated, dtype=float), 1e-6, 1 - 1e-6)


def _default_preprocessor(X_train: pd.DataFrame) -> TransformerMixin:
    numeric_columns = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [c for c in X_train.columns if c not in numeric_columns]
    transformers: list[tuple[str, TransformerMixin, list[str]]] = []

    if numeric_columns:
        numeric = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(strategy="median", keep_empty_features=True),
                ),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric, numeric_columns))
    if categorical_columns:
        categorical = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="most_frequent", keep_empty_features=True
                    ),
                ),
                (
                    "encode",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical, categorical_columns))
    if not transformers:
        raise ValueError("X must contain at least one feature column")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _new_pipeline(
    X_train: pd.DataFrame,
    estimator_factory: EstimatorFactory,
    preprocessor_factory: PreprocessorFactory | None,
) -> Pipeline:
    preprocessor = (
        preprocessor_factory(X_train)
        if preprocessor_factory is not None
        else _default_preprocessor(X_train)
    )
    estimator = estimator_factory()
    if not hasattr(estimator, "fit"):
        raise TypeError("estimator_factory must return an unfitted sklearn estimator")
    return Pipeline([("preprocess", preprocessor), ("estimator", estimator)])


def _positive_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        values = np.asarray(model.predict_proba(X), dtype=float)
        if values.ndim == 2:
            classes = np.asarray(model.classes_)
            positive = np.flatnonzero(classes == 1)
            if positive.size != 1:
                raise ValueError("estimator classes must contain binary class 1")
            return values[:, positive[0]]
        return values
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        return 1 / (1 + np.exp(-np.clip(scores, -35, 35)))
    raise TypeError("estimator must expose predict_proba or decision_function")


def _prepare_inputs(
    X: pd.DataFrame | np.ndarray,
    y: Sequence[int] | pd.Series | np.ndarray,
    groups: Sequence[Any] | pd.Series | np.ndarray,
    times: Sequence[Any] | pd.Series | np.ndarray | None,
    metadata: pd.DataFrame | Mapping[str, Sequence[Any]] | None,
) -> _PreparedData:
    if isinstance(X, pd.DataFrame):
        source_index = X.index.to_numpy(copy=True)
        X_frame = X.copy().reset_index(drop=True)
    else:
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional array or DataFrame")
        X_frame = pd.DataFrame(array, columns=[f"x{i}" for i in range(array.shape[1])])
        source_index = np.arange(len(X_frame))

    y_array = np.asarray(y)
    groups_array = np.asarray(groups, dtype=object)
    n_rows = len(X_frame)
    if len(y_array) != n_rows or len(groups_array) != n_rows:
        raise ValueError("X, y, and groups must have identical row counts")
    if pd.isna(y_array).any() or not set(np.unique(y_array)).issubset({0, 1, False, True}):
        raise ValueError("y must be non-missing and binary (0/1)")
    y_array = y_array.astype(int)
    if np.unique(y_array).size < 2:
        raise ValueError("y must contain both crisis and non-crisis observations")
    if pd.isna(groups_array).any() or pd.unique(groups_array).size < 2:
        raise ValueError("groups must contain at least two non-missing countries")

    times_array: np.ndarray | None = None
    if times is not None:
        times_array = np.asarray(times)
        if len(times_array) != n_rows or pd.isna(times_array).any():
            raise ValueError("times must match X and contain no missing values")

    if metadata is None:
        metadata_frame = pd.DataFrame(index=np.arange(n_rows))
    else:
        metadata_frame = pd.DataFrame(metadata).copy().reset_index(drop=True)
        if len(metadata_frame) != n_rows:
            raise ValueError("metadata must have the same number of rows as X")

    return _PreparedData(
        X=X_frame,
        y=y_array,
        groups=groups_array,
        times=times_array,
        metadata=metadata_frame,
        source_index=source_index,
    )


def _inner_group_splits(
    y: np.ndarray,
    groups: np.ndarray,
    requested_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_groups = pd.unique(groups).size
    n_splits = min(requested_splits, n_groups)
    if n_splits < 2:
        raise ValueError("at least two training countries are needed for inner CV")

    # Stratification is confined to the outer training sample.  It improves the
    # chance that rare events are represented without exposing outer-test labels.
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    usable = [(tr, va) for tr, va in splits if np.unique(y[tr]).size == 2]
    if len(usable) < 2:
        fallback = GroupKFold(n_splits=n_splits)
        splits = list(fallback.split(np.zeros(len(y)), y, groups))
        usable = [(tr, va) for tr, va in splits if np.unique(y[tr]).size == 2]
    if len(usable) < 2:
        raise ValueError(
            "inner grouped CV needs at least two folds with both classes in training"
        )
    return usable


def _ordered_unique(values: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(sorted(pd.unique(values)))
    except TypeError as exc:
        raise ValueError("times must be mutually orderable") from exc


def _forward_splits(
    times: np.ndarray,
    requested_splits: int,
    min_train_periods: int | None,
    gap_periods: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    periods = _ordered_unique(times)
    if len(periods) < 3:
        raise ValueError("forward evaluation requires at least three time periods")

    if min_train_periods is None:
        first_test_position = max(2, len(periods) - requested_splits)
    else:
        first_test_position = max(1, int(min_train_periods) + gap_periods)
    candidate_positions = list(range(first_test_position, len(periods)))
    if requested_splits:
        candidate_positions = candidate_positions[-requested_splits:]

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for test_position in candidate_positions:
        train_end = test_position - gap_periods
        if train_end <= 0:
            continue
        train_periods = periods[:train_end]
        test_period = periods[test_position]
        train = np.flatnonzero(np.isin(times, train_periods))
        test = np.flatnonzero(times == test_period)
        if train.size and test.size:
            splits.append((train, test))
    if not splits:
        raise ValueError("temporal configuration produced no forward folds")
    return splits


def _inner_forward_splits(
    y: np.ndarray,
    times: np.ndarray,
    requested_splits: int,
    gap_periods: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = _forward_splits(
        times,
        requested_splits=requested_splits,
        min_train_periods=None,
        gap_periods=gap_periods,
    )
    usable = [(tr, va) for tr, va in splits if np.unique(y[tr]).size == 2]
    if not usable:
        raise ValueError(
            "inner forward CV has no fold with both classes in the training window"
        )
    validation_rows = np.concatenate([va for _, va in usable])
    if np.unique(y[validation_rows]).size < 2:
        raise ValueError(
            "inner forward validation windows must collectively contain both classes"
        )
    return usable


def select_review_threshold(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    recall_floor: float = 0.60,
) -> dict[str, float | int | str]:
    """Maximise precision subject to a minimum recall, with deterministic ties."""

    y = np.asarray(y_true, dtype=int)
    proba = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(proba)
    y, proba = y[valid], proba[valid]
    if len(y) == 0 or np.unique(y).size < 2 or y.sum() == 0:
        raise ValueError("threshold selection requires finite scores and both classes")
    if not 0 < recall_floor <= 1:
        raise ValueError("recall_floor must be in (0, 1]")

    precision_values, recall_values, thresholds = precision_recall_curve(y, proba)
    # sklearn returns one additional terminal precision/recall point that has
    # no associated threshold.
    precision_values = precision_values[:-1]
    recall_values = recall_values[:-1]
    f1_values = np.divide(
        2 * precision_values * recall_values,
        precision_values + recall_values,
        out=np.zeros_like(precision_values),
        where=(precision_values + recall_values) > 0,
    )
    eligible = np.flatnonzero(recall_values + 1e-12 >= recall_floor)
    if eligible.size == 0:
        raise RuntimeError("no threshold satisfies the requested recall floor")

    sorted_probabilities = np.sort(proba)
    alert_counts = len(proba) - np.searchsorted(
        sorted_probabilities, thresholds, side="left"
    )
    selected_index = max(
        eligible,
        key=lambda index: (
            float(precision_values[index]),
            float(recall_values[index]),
            float(f1_values[index]),
            -int(alert_counts[index]),
            float(thresholds[index]),
        ),
    )
    selected = {
        "threshold": float(thresholds[selected_index]),
        "precision": float(precision_values[selected_index]),
        "recall": float(recall_values[selected_index]),
        "f1": float(f1_values[selected_index]),
        "alert_burden": float(alert_counts[selected_index] / len(proba)),
        "alerts": int(alert_counts[selected_index]),
    }
    return {
        **selected,
        "policy": "review",
        "description": (
            "maximum inner-OOF precision subject to "
            f"recall >= {recall_floor:.0%}"
        ),
    }


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Return discrimination, calibration, classification, and burden metrics."""

    y = np.asarray(y_true, dtype=int)
    proba = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    pred = np.asarray(predictions, dtype=int)
    if not (len(y) == len(proba) == len(pred)):
        raise ValueError("metric inputs must have identical lengths")
    both_classes = np.unique(y).size == 2
    roc_auc = float(roc_auc_score(y, proba)) if both_classes else np.nan
    average_precision = (
        float(average_precision_score(y, proba)) if y.sum() > 0 else np.nan
    )
    if both_classes:
        precision_curve, recall_curve, _ = precision_recall_curve(y, proba)
        pr_auc = float(auc(recall_curve[::-1], precision_curve[::-1]))
    else:
        pr_auc = np.nan
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    alert_count = int(pred.sum())
    true_events = int(y.sum())
    return {
        "n": int(len(y)),
        "positive_count": true_events,
        "prevalence": float(y.mean()) if len(y) else np.nan,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "pr_auc": pr_auc,
        "brier": float(brier_score_loss(y, proba)),
        "log_loss": float(log_loss(y, proba, labels=[0, 1])),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "alert_count": alert_count,
        "alert_burden": float(pred.mean()) if len(pred) else np.nan,
        "alerts_per_positive": (
            float(alert_count / true_events) if true_events else np.nan
        ),
        "confusion": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


def _cross_fitted_calibration(
    raw_oof: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    method: str,
    random_state: int,
) -> tuple[np.ndarray, _ProbabilityCalibrator]:
    valid = np.isfinite(raw_oof) & (fold_ids >= 0)
    if not valid.any():
        raise ValueError("inner CV generated no OOF probabilities")
    calibrated = np.full(len(y), np.nan, dtype=float)

    if method == "none":
        calibrated[valid] = np.clip(raw_oof[valid], 1e-6, 1 - 1e-6)
    else:
        for fold_id in np.unique(fold_ids[valid]):
            held_out = valid & (fold_ids == fold_id)
            calibration_train = valid & (fold_ids != fold_id)
            calibrator = _ProbabilityCalibrator(method, random_state + int(fold_id))
            calibrator.fit(raw_oof[calibration_train], y[calibration_train])
            calibrated[held_out] = calibrator.predict(raw_oof[held_out])

    final_calibrator = _ProbabilityCalibrator(method, random_state)
    final_calibrator.fit(raw_oof[valid], y[valid])
    return calibrated, final_calibrator


def _metadata_value(
    data: _PreparedData, column: str, indices: np.ndarray, fallback: np.ndarray | None
) -> np.ndarray:
    if column in data.metadata.columns:
        return data.metadata.iloc[indices][column].to_numpy()
    if fallback is None:
        return np.full(len(indices), pd.NA, dtype=object)
    return np.asarray(fallback)[indices]


def _prediction_ledger(
    data: _PreparedData,
    indices: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    design: str,
    outer_fold: int,
    inner_fold: np.ndarray | int | None = None,
) -> pd.DataFrame:
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    ledger = pd.DataFrame(
        {
            "design": design,
            "outer_fold": outer_fold,
            "row_id": indices,
            "source_index": data.source_index[indices],
            "country": _metadata_value(data, "country", indices, data.groups),
            "origin": _metadata_value(data, "origin", indices, data.times),
            "event_id": _metadata_value(data, "event_id", indices, None),
            "y": data.y[indices].astype(int),
            "proba": np.asarray(probabilities, dtype=float),
            "threshold": float(threshold),
            "pred": pred,
        }
    )
    # Preserve caller-supplied audit fields without allowing them to overwrite
    # the canonical validation columns above.
    for column in data.metadata.columns:
        if column in ledger.columns:
            continue
        ledger[column] = data.metadata.iloc[indices][column].to_numpy()
    if inner_fold is not None:
        if np.isscalar(inner_fold):
            ledger["inner_fold"] = int(inner_fold)
        else:
            ledger["inner_fold"] = np.asarray(inner_fold, dtype=int)
    return ledger


def _evaluate_outer_fold(
    *,
    data: _PreparedData,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    estimator_factory: EstimatorFactory,
    config: ValidationConfig,
    preprocessor_factory: PreprocessorFactory | None,
    design: str,
    outer_fold: int,
    inner_strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError("outer train and test indices must be disjoint")
    if np.unique(data.y[train_indices]).size < 2:
        raise ValueError(f"outer fold {outer_fold} training sample has only one class")

    X_train = data.X.iloc[train_indices]
    y_train = data.y[train_indices]
    groups_train = data.groups[train_indices]
    times_train = data.times[train_indices] if data.times is not None else None

    inner_event_purge_counts: list[int]
    if inner_strategy == "grouped":
        inner_splits = _inner_group_splits(
            y_train,
            groups_train,
            requested_splits=config.inner_splits,
            random_state=config.random_state + outer_fold,
        )
        inner_event_purge_counts = [0] * len(inner_splits)
    elif inner_strategy == "temporal":
        if times_train is None:
            raise ValueError("times are required for temporal inner validation")
        inner_splits = _inner_forward_splits(
            y_train,
            times_train,
            requested_splits=config.temporal_inner_splits,
            gap_periods=config.temporal_gap_periods,
        )
        if config.purge_overlapping_events and "event_id" in data.metadata:
            purged_splits: list[tuple[np.ndarray, np.ndarray]] = []
            purge_counts: list[int] = []
            for inner_train, inner_valid in inner_splits:
                global_train = train_indices[inner_train]
                global_valid = train_indices[inner_valid]
                retained_global = _purge_event_overlap(
                    data, global_train, global_valid
                )
                retained = inner_train[np.isin(global_train, retained_global)]
                if retained.size and np.unique(y_train[retained]).size == 2:
                    purged_splits.append((retained, inner_valid))
                    purge_counts.append(int(len(inner_train) - len(retained)))
            inner_splits = purged_splits
            inner_event_purge_counts = purge_counts
            if not inner_splits:
                raise ValueError(
                    "event-overlap purging left no usable inner temporal fold"
                )
        else:
            inner_event_purge_counts = [0] * len(inner_splits)
    else:
        raise ValueError("inner_strategy must be 'grouped' or 'temporal'")

    raw_oof = np.full(len(train_indices), np.nan, dtype=float)
    inner_fold_ids = np.full(len(train_indices), -1, dtype=int)
    inner_audit: list[dict[str, Any]] = []
    for inner_fold, (inner_train, inner_valid) in enumerate(inner_splits):
        model = _new_pipeline(
            X_train.iloc[inner_train], estimator_factory, preprocessor_factory
        )
        model.fit(X_train.iloc[inner_train], y_train[inner_train])
        raw_oof[inner_valid] = _positive_probability(model, X_train.iloc[inner_valid])
        inner_fold_ids[inner_valid] = inner_fold
        audit: dict[str, Any] = {
            "inner_fold": inner_fold,
            "train_n": int(len(inner_train)),
            "valid_n": int(len(inner_valid)),
            "train_source_indices": data.source_index[train_indices[inner_train]].tolist(),
            "valid_source_indices": data.source_index[train_indices[inner_valid]].tolist(),
        }
        audit["purged_event_rows"] = inner_event_purge_counts[inner_fold]
        if times_train is not None:
            audit["train_max_time"] = _python_scalar(np.max(times_train[inner_train]))
            audit["valid_min_time"] = _python_scalar(np.min(times_train[inner_valid]))
        inner_audit.append(audit)

    calibrated_oof, final_calibrator = _cross_fitted_calibration(
        raw_oof,
        y_train,
        inner_fold_ids,
        method=config.calibration,
        random_state=config.random_state + outer_fold,
    )
    tuning_mask = np.isfinite(calibrated_oof)
    threshold_info = select_review_threshold(
        y_train[tuning_mask],
        calibrated_oof[tuning_mask],
        recall_floor=config.recall_floor,
    )
    threshold = float(threshold_info["threshold"])

    final_model = _new_pipeline(X_train, estimator_factory, preprocessor_factory)
    final_model.fit(X_train, y_train)
    raw_test = _positive_probability(final_model, data.X.iloc[test_indices])
    test_proba = final_calibrator.predict(raw_test)

    ledger = _prediction_ledger(
        data,
        test_indices,
        test_proba,
        threshold,
        design,
        outer_fold,
    )
    tuning_indices = train_indices[tuning_mask]
    tuning_ledger = _prediction_ledger(
        data,
        tuning_indices,
        calibrated_oof[tuning_mask],
        threshold,
        design,
        outer_fold,
        inner_fold=inner_fold_ids[tuning_mask],
    )
    tuning_ledger["raw_proba"] = raw_oof[tuning_mask]

    metrics = classification_metrics(ledger["y"], ledger["proba"], ledger["pred"])
    metrics.update(
        {
            "design": design,
            "outer_fold": outer_fold,
            "threshold": threshold,
            "calibration": config.calibration,
            "inner_tuning_n": int(tuning_mask.sum()),
            "inner_tuning_precision": float(threshold_info["precision"]),
            "inner_tuning_recall": float(threshold_info["recall"]),
        }
    )
    detail: dict[str, Any] = {
        "outer_fold": outer_fold,
        "train_n": int(len(train_indices)),
        "test_n": int(len(test_indices)),
        "train_source_indices": data.source_index[train_indices].tolist(),
        "test_source_indices": data.source_index[test_indices].tolist(),
        "train_groups": sorted(map(str, pd.unique(data.groups[train_indices]))),
        "test_groups": sorted(map(str, pd.unique(data.groups[test_indices]))),
        "threshold": threshold,
        "threshold_policy": threshold_info,
        "calibration_requested": config.calibration,
        "calibration_fitted": final_calibrator.fitted_method,
        "inner_folds": inner_audit,
    }
    if data.times is not None:
        detail["train_max_time"] = _python_scalar(np.max(data.times[train_indices]))
        detail["test_min_time"] = _python_scalar(np.min(data.times[test_indices]))
    return ledger, tuning_ledger, metrics, detail


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _flatten_fold_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in metrics.items() if key != "confusion"}
    flat.update(metrics["confusion"])
    return flat


def _summarise_ledger(ledger: pd.DataFrame, design: str) -> dict[str, Any]:
    metrics = classification_metrics(ledger["y"], ledger["proba"], ledger["pred"])
    metrics.update(
        {
            "design": design,
            "folds": int(ledger["outer_fold"].nunique()),
            "countries": int(ledger["country"].nunique()),
            "threshold_policy": "review",
            "note": (
                "Pooled untouched outer-test predictions; each fold threshold was "
                "selected and frozen using inner OOF training predictions only."
            ),
        }
    )
    return metrics


_BOOTSTRAP_METRICS = (
    "roc_auc",
    "average_precision",
    "pr_auc",
    "brier",
    "log_loss",
    "precision",
    "recall",
    "f1",
    "alert_burden",
)


def bootstrap_confidence_intervals(
    ledger: pd.DataFrame,
    *,
    iterations: int = 200,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Deterministic cluster bootstrap by country and, when present, event.

    Country clustering preserves within-country serial dependence.  Event
    clustering keeps repeated positive windows for the same crisis together;
    non-event rows are treated as singleton clusters in that second view.
    Thresholds and predictions remain frozen in every resample.
    """

    columns = ["cluster", "metric", "estimate", "lower", "upper", "valid_draws"]
    if iterations <= 0 or ledger.empty:
        return pd.DataFrame(columns=columns)

    cluster_specs: list[tuple[str, pd.Series]] = [
        ("country", ledger["country"].astype(str))
    ]
    if "event_id" in ledger and ledger["event_id"].notna().any():
        event_cluster = ledger["event_id"].astype("string")
        missing = event_cluster.isna()
        singleton = "non_event::" + ledger.index.astype(str)
        event_cluster = event_cluster.where(~missing, singleton)
        cluster_specs.append(("event_id", event_cluster.astype(str)))

    base = classification_metrics(ledger["y"], ledger["proba"], ledger["pred"])
    alpha = (1 - confidence_level) / 2
    output: list[dict[str, Any]] = []
    for spec_number, (cluster_name, cluster_ids) in enumerate(cluster_specs):
        frame = ledger.copy()
        frame["__cluster"] = cluster_ids.to_numpy()
        unique_clusters = frame["__cluster"].drop_duplicates().to_numpy()
        if len(unique_clusters) < 2:
            continue
        rng = np.random.default_rng(random_state + 10_000 * spec_number)
        draws: dict[str, list[float]] = {metric: [] for metric in _BOOTSTRAP_METRICS}
        grouped = {key: group for key, group in frame.groupby("__cluster", sort=False)}
        for _ in range(iterations):
            sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
            sample = pd.concat([grouped[key] for key in sampled], ignore_index=True)
            values = classification_metrics(
                sample["y"], sample["proba"], sample["pred"]
            )
            for metric in _BOOTSTRAP_METRICS:
                value = float(values[metric])
                if np.isfinite(value):
                    draws[metric].append(value)
        for metric, values in draws.items():
            interval = np.asarray(values, dtype=float)
            output.append(
                {
                    "cluster": cluster_name,
                    "metric": metric,
                    "estimate": float(base[metric]),
                    "lower": (
                        float(np.quantile(interval, alpha)) if interval.size else np.nan
                    ),
                    "upper": (
                        float(np.quantile(interval, 1 - alpha))
                        if interval.size
                        else np.nan
                    ),
                    "valid_draws": int(interval.size),
                }
            )
    return pd.DataFrame(output, columns=columns)


def _assemble_result(
    design: str,
    ledgers: list[pd.DataFrame],
    tuning_ledgers: list[pd.DataFrame],
    metrics: list[dict[str, Any]],
    details: list[dict[str, Any]],
    config: ValidationConfig,
) -> ValidationResult:
    ledger = pd.concat(ledgers, ignore_index=True)
    tuning = pd.concat(tuning_ledgers, ignore_index=True)
    fold_metrics = pd.DataFrame([_flatten_fold_metrics(row) for row in metrics])
    summary = _summarise_ledger(ledger, design)
    cis = bootstrap_confidence_intervals(
        ledger,
        iterations=config.bootstrap_iterations,
        confidence_level=config.confidence_level,
        random_state=config.random_state,
    )
    return ValidationResult(
        design=design,
        summary=summary,
        fold_metrics=fold_metrics,
        ledger=ledger,
        tuning_ledger=tuning,
        bootstrap_cis=cis,
        fold_details=details,
    )


def evaluate_outer_split(
    estimator_factory: EstimatorFactory,
    X: pd.DataFrame | np.ndarray,
    y: Sequence[int] | pd.Series | np.ndarray,
    groups: Sequence[Any] | pd.Series | np.ndarray,
    train_indices: Sequence[int] | np.ndarray,
    test_indices: Sequence[int] | np.ndarray,
    *,
    times: Sequence[Any] | pd.Series | np.ndarray | None = None,
    metadata: pd.DataFrame | Mapping[str, Sequence[Any]] | None = None,
    config: ValidationConfig | None = None,
    preprocessor_factory: PreprocessorFactory | None = None,
    inner_strategy: str = "grouped",
) -> ValidationResult:
    """Evaluate one pre-declared holdout with a frozen inner-trained threshold."""

    config = config or ValidationConfig()
    data = _prepare_inputs(X, y, groups, times, metadata)
    ledger, tuning, metrics, detail = _evaluate_outer_fold(
        data=data,
        train_indices=np.asarray(train_indices, dtype=int),
        test_indices=np.asarray(test_indices, dtype=int),
        estimator_factory=estimator_factory,
        config=config,
        preprocessor_factory=preprocessor_factory,
        design="fixed_outer_holdout",
        outer_fold=0,
        inner_strategy=inner_strategy,
    )
    return _assemble_result(
        "fixed_outer_holdout", [ledger], [tuning], [metrics], [detail], config
    )


def evaluate_country_grouped(
    estimator_factory: EstimatorFactory,
    X: pd.DataFrame | np.ndarray,
    y: Sequence[int] | pd.Series | np.ndarray,
    groups: Sequence[Any] | pd.Series | np.ndarray,
    *,
    times: Sequence[Any] | pd.Series | np.ndarray | None = None,
    metadata: pd.DataFrame | Mapping[str, Sequence[Any]] | None = None,
    config: ValidationConfig | None = None,
    preprocessor_factory: PreprocessorFactory | None = None,
) -> ValidationResult:
    """Outer country-grouped validation with nested threshold selection."""

    config = config or ValidationConfig()
    data = _prepare_inputs(X, y, groups, times, metadata)
    n_splits = min(config.outer_splits, pd.unique(data.groups).size)
    if n_splits < 2:
        raise ValueError("country-grouped evaluation requires at least two countries")
    # GroupKFold is intentionally label-independent.  This makes fold membership
    # pre-declarable and makes it impossible for outer labels to tune the split.
    splits = GroupKFold(n_splits=n_splits).split(data.X, data.y, data.groups)
    ledgers: list[pd.DataFrame] = []
    tuning_ledgers: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for outer_fold, (train, test) in enumerate(splits):
        ledger, tuning, fold_metrics, detail = _evaluate_outer_fold(
            data=data,
            train_indices=train,
            test_indices=test,
            estimator_factory=estimator_factory,
            config=config,
            preprocessor_factory=preprocessor_factory,
            design="country_grouped",
            outer_fold=outer_fold,
            inner_strategy="grouped",
        )
        ledgers.append(ledger)
        tuning_ledgers.append(tuning)
        metrics.append(fold_metrics)
        details.append(detail)
    return _assemble_result(
        "country_grouped", ledgers, tuning_ledgers, metrics, details, config
    )


def _purge_event_overlap(
    data: _PreparedData, train: np.ndarray, test: np.ndarray
) -> np.ndarray:
    if "event_id" not in data.metadata:
        return train
    train_events = data.metadata.iloc[train]["event_id"]
    test_events = set(data.metadata.iloc[test]["event_id"].dropna().astype(str))
    if not test_events:
        return train
    keep = ~train_events.astype("string").isin(test_events).to_numpy()
    return train[keep]


def evaluate_forward_temporal(
    estimator_factory: EstimatorFactory,
    X: pd.DataFrame | np.ndarray,
    y: Sequence[int] | pd.Series | np.ndarray,
    groups: Sequence[Any] | pd.Series | np.ndarray,
    times: Sequence[Any] | pd.Series | np.ndarray,
    *,
    metadata: pd.DataFrame | Mapping[str, Sequence[Any]] | None = None,
    config: ValidationConfig | None = None,
    preprocessor_factory: PreprocessorFactory | None = None,
) -> ValidationResult:
    """Expanding-window validation with temporally nested threshold selection."""

    config = config or ValidationConfig()
    data = _prepare_inputs(X, y, groups, times, metadata)
    assert data.times is not None
    splits = _forward_splits(
        data.times,
        requested_splits=config.temporal_outer_splits,
        min_train_periods=config.temporal_min_train_periods,
        gap_periods=config.temporal_gap_periods,
    )
    ledgers: list[pd.DataFrame] = []
    tuning_ledgers: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for outer_fold, (train, test) in enumerate(splits):
        unpurged_train_n = len(train)
        if config.purge_overlapping_events:
            train = _purge_event_overlap(data, train, test)
        ledger, tuning, fold_metrics, detail = _evaluate_outer_fold(
            data=data,
            train_indices=train,
            test_indices=test,
            estimator_factory=estimator_factory,
            config=config,
            preprocessor_factory=preprocessor_factory,
            design="forward_temporal",
            outer_fold=outer_fold,
            inner_strategy="temporal",
        )
        detail["purged_outer_event_rows"] = int(unpurged_train_n - len(train))
        ledgers.append(ledger)
        tuning_ledgers.append(tuning)
        metrics.append(fold_metrics)
        details.append(detail)
    return _assemble_result(
        "forward_temporal", ledgers, tuning_ledgers, metrics, details, config
    )


def evaluate_crisis_model(
    estimator_factory: EstimatorFactory,
    X: pd.DataFrame | np.ndarray,
    y: Sequence[int] | pd.Series | np.ndarray,
    groups: Sequence[Any] | pd.Series | np.ndarray,
    times: Sequence[Any] | pd.Series | np.ndarray,
    *,
    metadata: pd.DataFrame | Mapping[str, Sequence[Any]] | None = None,
    config: ValidationConfig | None = None,
    preprocessor_factory: PreprocessorFactory | None = None,
) -> dict[str, ValidationResult]:
    """Run the complementary country-generalisation and forward-time designs."""

    config = config or ValidationConfig()
    return {
        "country_grouped": evaluate_country_grouped(
            estimator_factory,
            X,
            y,
            groups,
            times=times,
            metadata=metadata,
            config=config,
            preprocessor_factory=preprocessor_factory,
        ),
        "forward_temporal": evaluate_forward_temporal(
            estimator_factory,
            X,
            y,
            groups,
            times,
            metadata=metadata,
            config=config,
            preprocessor_factory=preprocessor_factory,
        ),
    }


__all__ = [
    "ValidationConfig",
    "ValidationResult",
    "bootstrap_confidence_intervals",
    "classification_metrics",
    "evaluate_country_grouped",
    "evaluate_crisis_model",
    "evaluate_forward_temporal",
    "evaluate_outer_split",
    "select_review_threshold",
]
