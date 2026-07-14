"""Leakage-resistant offline cross-validation for crisis hazard candidates.

The snapshot builder intentionally stays small and deterministic.  This module
provides the separate *development* surface needed to compare a bounded set of
transparent hazard candidates without reading the locked forward-test sample.

The central time coordinate is the year in which a target becomes observable,
not merely the forecast-origin year::

    label_available_year = forecast_origin_year + horizon

Consequently, every training label in a fold is known before the first
validation label, and every row used anywhere in CV becomes observable before
``locked_test_start_year``.  Crisis event identifiers are purged across each
train/validation boundary as an additional safeguard for stacked or otherwise
overlapping horizon data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.crisis_hazard import (
    RegularizedDiscreteTimeHazardModel,
    event_balanced_sample_weights,
)


CalibrationMethod = Literal["none", "sigmoid"]
PositiveWeight = float | Literal["balanced"]


def _json_safe(value: Any) -> Any:
    """Return only strict-JSON primitives; non-finite values become ``None``."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True)
class HazardCandidateSpec:
    """A deliberately bounded transparent hazard-model candidate.

    ``positive_class_weight="balanced"`` calculates the negative-to-positive
    ratio inside each fold, then caps it at ``positive_class_weight_cap``.
    Numeric weights are capped the same way.  This permits class-imbalance
    experiments without allowing extreme inverse-prevalence weights to enter
    the optimiser silently.

    ``complexity_rank`` is an explicit governance choice.  Lower values are
    preferred whenever candidates are within the selection tolerance.
    """

    name: str
    feature_names: tuple[str, ...]
    link: Literal["logit", "cloglog"] = "logit"
    C: float = 0.3
    positive_class_weight: PositiveWeight = 1.0
    positive_class_weight_cap: float = 20.0
    nonnegative_coefficients: bool = False
    minimum_feature_coverage: float = 0.05
    calibration: CalibrationMethod = "none"
    minimum_calibration_positives: int = 10
    complexity_rank: int = 0
    random_state: int = 42

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("candidate name cannot be blank")
        if not self.feature_names:
            raise ValueError("candidate needs at least one feature")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("candidate feature names must be unique")
        if self.link not in {"logit", "cloglog"}:
            raise ValueError("link must be 'logit' or 'cloglog'")
        if self.C <= 0:
            raise ValueError("C must be positive")
        if self.positive_class_weight != "balanced":
            weight = float(self.positive_class_weight)
            if not math.isfinite(weight) or weight < 1:
                raise ValueError("numeric positive class weight must be at least one")
        if not math.isfinite(self.positive_class_weight_cap):
            raise ValueError("positive class-weight cap must be finite")
        if self.positive_class_weight_cap < 1:
            raise ValueError("positive class-weight cap must be at least one")
        if not 0 < self.minimum_feature_coverage <= 1:
            raise ValueError("minimum feature coverage must be in (0, 1]")
        if self.calibration not in {"none", "sigmoid"}:
            raise ValueError("calibration must be 'none' or 'sigmoid'")
        if self.minimum_calibration_positives < 1:
            raise ValueError("minimum calibration positives must be positive")
        if self.complexity_rank < 0:
            raise ValueError("complexity rank cannot be negative")

    def effective_positive_weight(self, y: Sequence[int]) -> float:
        target = np.asarray(y, dtype=int)
        positives = int(target.sum())
        negatives = int(len(target) - positives)
        if positives <= 0 or negatives <= 0:
            raise ValueError("class weighting requires both target classes")
        requested = (
            float(negatives / positives)
            if self.positive_class_weight == "balanced"
            else float(self.positive_class_weight)
        )
        return float(min(max(1.0, requested), self.positive_class_weight_cap))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "name": self.name,
                "feature_names": self.feature_names,
                "link": self.link,
                "C": self.C,
                "positive_class_weight": self.positive_class_weight,
                "positive_class_weight_cap": self.positive_class_weight_cap,
                "nonnegative_coefficients": self.nonnegative_coefficients,
                "minimum_feature_coverage": self.minimum_feature_coverage,
                "calibration": self.calibration,
                "minimum_calibration_positives": self.minimum_calibration_positives,
                "complexity_rank": self.complexity_rank,
                "random_state": self.random_state,
            }
        )


@dataclass(frozen=True)
class OutcomeYearFold:
    """One expanding fold represented by immutable source-row positions."""

    fold: int
    validation_start_year: int
    validation_end_year: int
    train_positions: tuple[int, ...]
    validation_positions: tuple[int, ...]
    purged_train_rows: int = 0
    purged_event_ids: tuple[str, ...] = ()

    def to_dict(self, *, include_positions: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "fold": self.fold,
            "validation_start_year": self.validation_start_year,
            "validation_end_year": self.validation_end_year,
            "train_rows": len(self.train_positions),
            "validation_rows": len(self.validation_positions),
            "purged_train_rows": self.purged_train_rows,
            "purged_event_ids": self.purged_event_ids,
        }
        if include_positions:
            payload["train_positions"] = self.train_positions
            payload["validation_positions"] = self.validation_positions
        return _json_safe(payload)


@dataclass(frozen=True)
class SigmoidCalibrationArtifact:
    """JSON-safe Platt-style mapping from a raw probability to a probability."""

    intercept: float
    coefficient: float
    training_rows: int
    positive_rows: int

    @staticmethod
    def _logit(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    def predict(self, probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
        linear = self.intercept + self.coefficient * self._logit(probabilities)[:, 0]
        return np.clip(1.0 / (1.0 + np.exp(-np.clip(linear, -35, 35))), 1e-6, 1 - 1e-6)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "type": "sigmoid_on_logit_probability",
                "intercept": self.intercept,
                "coefficient": self.coefficient,
                "training_rows": self.training_rows,
                "positive_rows": self.positive_rows,
            }
        )


@dataclass
class HazardCVResult:
    """Auditable candidate result whose public serialization is strict JSON."""

    candidate: HazardCandidateSpec
    horizon: int
    locked_test_start_year: int
    development_rows: int
    excluded_locked_or_later_rows: int
    pooled_metrics: dict[str, Any]
    raw_pooled_metrics: dict[str, Any]
    per_fold_metrics: list[dict[str, Any]]
    stability: dict[str, Any]
    calibration: dict[str, Any]
    fold_details: list[dict[str, Any]]
    ledger: pd.DataFrame = field(repr=False)

    def to_dict(self, *, include_ledger: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "candidate": self.candidate.to_dict(),
            "horizon": self.horizon,
            "locked_test_start_year": self.locked_test_start_year,
            "development_rows": self.development_rows,
            "excluded_locked_or_later_rows": self.excluded_locked_or_later_rows,
            "pooled_metrics": self.pooled_metrics,
            "raw_pooled_metrics": self.raw_pooled_metrics,
            "per_fold_metrics": self.per_fold_metrics,
            "stability": self.stability,
            "calibration": self.calibration,
            "fold_details": self.fold_details,
        }
        if include_ledger:
            payload["ledger"] = self.ledger.to_dict(orient="records")
        return _json_safe(payload)

    def to_json(self, *, include_ledger: bool = False, **json_kwargs: Any) -> str:
        return json.dumps(
            self.to_dict(include_ledger=include_ledger),
            allow_nan=False,
            **json_kwargs,
        )


def _development_target(
    frame: pd.DataFrame,
    *,
    target_col: str,
    development_positions: np.ndarray,
) -> pd.Series:
    """Validate only development labels; locked labels are deliberately unread."""

    raw = frame.iloc[development_positions][target_col]
    numeric = pd.to_numeric(raw, errors="coerce")
    invalid_numeric = raw.notna() & numeric.isna()
    if invalid_numeric.any():
        raise ValueError(f"{target_col} contains non-numeric development labels")
    invalid_binary = numeric.dropna()[~numeric.dropna().isin([0, 1])]
    if not invalid_binary.empty:
        raise ValueError(f"{target_col} must contain only 0, 1, or missing")
    return pd.Series(numeric.to_numpy(), index=development_positions, dtype=float)


def make_expanding_outcome_folds(
    frame: pd.DataFrame,
    *,
    horizon: int,
    locked_test_start_year: int,
    validation_blocks: Sequence[tuple[int, int]],
    origin_col: str = "forecast_origin_year",
    target_col: str = "crisis_hazard_year_1",
    event_id_col: str = "hazard_event_id",
) -> list[OutcomeYearFold]:
    """Create explicit expanding folds using target-availability years.

    A row is eligible for development only when ``origin + horizon`` is
    strictly earlier than ``locked_test_start_year``.  Target values and event
    identifiers on all later rows are never read.
    """

    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not validation_blocks:
        raise ValueError("at least one validation block is required")
    required = {origin_col, target_col, event_id_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame missing required columns: {sorted(missing)}")

    origins = pd.to_numeric(frame[origin_col], errors="raise").astype(int).to_numpy()
    label_year = origins + int(horizon)
    development_positions = np.flatnonzero(label_year < int(locked_test_start_year))
    if development_positions.size == 0:
        raise ValueError("no labels become observable before the locked test start")
    target = _development_target(
        frame, target_col=target_col, development_positions=development_positions
    )
    event_ids = frame.iloc[development_positions][event_id_col].astype("string")
    event_ids.index = development_positions

    normalised_blocks: list[tuple[int, int]] = []
    previous_end: int | None = None
    for raw_start, raw_end in validation_blocks:
        start, end = int(raw_start), int(raw_end)
        if start > end:
            raise ValueError("validation block start cannot exceed its end")
        if end >= locked_test_start_year:
            raise ValueError("validation blocks must end before locked test start")
        if previous_end is not None and start <= previous_end:
            raise ValueError("validation blocks must be ordered and non-overlapping")
        normalised_blocks.append((start, end))
        previous_end = end

    observed_positions = target.index[target.notna()].to_numpy(dtype=int)
    folds: list[OutcomeYearFold] = []
    for fold_number, (start, end) in enumerate(normalised_blocks):
        train = observed_positions[label_year[observed_positions] < start]
        validation = observed_positions[
            (label_year[observed_positions] >= start)
            & (label_year[observed_positions] <= end)
        ]
        if train.size == 0 or validation.size == 0:
            raise ValueError(
                f"validation block {start}-{end} has an empty train or validation set"
            )

        validation_events = set(
            event_ids.reindex(validation).dropna().astype(str).str.strip()
        )
        validation_events.discard("")
        train_event_values = event_ids.reindex(train).astype("string")
        purge_mask = train_event_values.isin(validation_events).fillna(False).to_numpy()
        purged_rows = int(purge_mask.sum())
        purged_ids = tuple(
            sorted(
                set(
                    train_event_values.loc[purge_mask]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
            )
        )
        train = train[~purge_mask]
        if train.size == 0:
            raise ValueError(f"event purge emptied training block before {start}")

        folds.append(
            OutcomeYearFold(
                fold=fold_number,
                validation_start_year=start,
                validation_end_year=end,
                train_positions=tuple(map(int, train)),
                validation_positions=tuple(map(int, validation)),
                purged_train_rows=purged_rows,
                purged_event_ids=purged_ids,
            )
        )
    return folds


def _binary_metrics(y: Sequence[int], probability: Sequence[float]) -> dict[str, Any]:
    target = np.asarray(y, dtype=int)
    scores = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    if len(target) != len(scores) or len(target) == 0:
        raise ValueError("metric inputs must be non-empty and have equal length")
    if not np.isfinite(scores).all():
        raise ValueError("probabilities must be finite")
    prevalence = float(target.mean())
    both_classes = np.unique(target).size == 2
    average_precision = (
        float(average_precision_score(target, scores)) if target.sum() else None
    )
    return _json_safe(
        {
            "rows": len(target),
            "positive_rows": int(target.sum()),
            "prevalence": prevalence,
            "roc_auc": float(roc_auc_score(target, scores)) if both_classes else None,
            "average_precision": average_precision,
            "ap_lift": (
                average_precision / prevalence
                if average_precision is not None and prevalence > 0
                else None
            ),
            "brier": float(brier_score_loss(target, scores)),
        }
    )


def _metric_stability(per_fold: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"folds": len(per_fold)}
    for metric in ("roc_auc", "average_precision", "ap_lift", "brier"):
        values = np.asarray(
            [row[metric] for row in per_fold if row.get(metric) is not None],
            dtype=float,
        )
        output[metric] = {
            "usable_folds": int(len(values)),
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "standard_deviation": float(values.std(ddof=0)) if len(values) else None,
            "minimum": float(values.min()) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None,
        }
    return _json_safe(output)


def _fit_sigmoid(
    probability: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
) -> SigmoidCalibrationArtifact:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(
        solver="liblinear", C=1.0, max_iter=1_000, random_state=random_state
    )
    model.fit(logit, y)
    return SigmoidCalibrationArtifact(
        intercept=float(model.intercept_[0]),
        coefficient=float(model.coef_[0, 0]),
        training_rows=int(len(y)),
        positive_rows=int(y.sum()),
    )


def _cross_fitted_sigmoid(
    raw_probability: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    *,
    minimum_positives: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_probability = np.asarray(raw_probability, dtype=float)
    y = np.asarray(y, dtype=int)
    fold_ids = np.asarray(fold_ids, dtype=int)
    if not (len(raw_probability) == len(y) == len(fold_ids)):
        raise ValueError("calibration inputs must have equal length")
    if not len(y):
        raise ValueError("calibration inputs cannot be empty")

    # Fold identifiers are assigned in chronological validation-block order by
    # ``make_expanding_outcome_folds``.  A held-out fold may therefore use only
    # OOF predictions from *smaller* fold identifiers.  Symmetric cross-fitting
    # (all folds except the held-out fold) would allow later crisis outcomes to
    # calibrate an earlier fold and would invalidate the outcome-year embargo.
    unique_folds = np.unique(fold_ids)
    fold_artifacts: list[dict[str, Any]] = []
    calibrated = raw_probability.copy()
    calibrated_folds: list[int] = []
    fallback_folds: list[int] = []

    for fold_id in unique_folds:
        held_out = fold_ids == fold_id
        earlier = fold_ids < fold_id
        training_folds = sorted(int(value) for value in np.unique(fold_ids[earlier]))
        train_y = y[earlier]
        positives = int(train_y.sum())
        negatives = int(len(train_y) - positives)

        if not earlier.any():
            fallback_folds.append(int(fold_id))
            fold_artifacts.append(
                {
                    "held_out_fold": int(fold_id),
                    "applied": "none",
                    "reason": "no earlier OOF folds are available",
                    "training_folds": training_folds,
                    "training_rows": 0,
                    "positive_rows": 0,
                    "negative_rows": 0,
                }
            )
            continue

        if positives < minimum_positives or negatives < minimum_positives:
            fallback_folds.append(int(fold_id))
            fold_artifacts.append(
                {
                    "held_out_fold": int(fold_id),
                    "applied": "none",
                    "reason": (
                        "earlier OOF folds lack the minimum positive or "
                        "negative calibration rows"
                    ),
                    "training_folds": training_folds,
                    "training_rows": int(earlier.sum()),
                    "positive_rows": positives,
                    "negative_rows": negatives,
                }
            )
            continue

        artifact = _fit_sigmoid(
            raw_probability[earlier],
            train_y,
            random_state=random_state + int(fold_id),
        )
        calibrated[held_out] = artifact.predict(raw_probability[held_out])
        calibrated_folds.append(int(fold_id))
        fold_artifacts.append(
            {
                "held_out_fold": int(fold_id),
                "applied": "sigmoid",
                "reason": "fit only on chronologically earlier OOF folds",
                "training_folds": training_folds,
                "negative_rows": negatives,
                **artifact.to_dict(),
            }
        )

    if not np.isfinite(calibrated).all():
        raise RuntimeError("forward-only calibration returned non-finite predictions")

    # This artifact is for a later, explicitly governed development refit.  It
    # is intentionally not applied to any reported OOF probability or metric.
    total_positives = int(y.sum())
    total_negatives = int(len(y) - total_positives)
    final_artifact = None
    if (
        total_positives >= minimum_positives
        and total_negatives >= minimum_positives
    ):
        final_artifact = _fit_sigmoid(
            raw_probability, y, random_state=random_state + 10_000
        ).to_dict()

    if calibrated_folds:
        applied = "forward_only_sigmoid_with_raw_fallback"
        reason = (
            "each calibrated fold used only earlier OOF predictions; "
            "underpowered folds retained raw probabilities"
        )
    else:
        applied = "none"
        reason = (
            "no fold had enough earlier OOF outcomes for forward-only calibration"
        )
    return calibrated, {
        "requested": "sigmoid",
        "applied": applied,
        "reason": reason,
        "minimum_positive_rows": minimum_positives,
        "calibrated_folds": calibrated_folds,
        "fallback_folds": fallback_folds,
        "cross_fitted_models": fold_artifacts,
        "final_model": final_artifact,
        "final_model_usage": (
            "development-refit artifact only; not applied to reported OOF metrics"
        ),
    }


def _selected_features(
    train_frame: pd.DataFrame,
    feature_names: Sequence[str],
    minimum_coverage: float,
) -> tuple[list[str], dict[str, float]]:
    coverage: dict[str, float] = {}
    selected: list[str] = []
    for feature in feature_names:
        values = pd.to_numeric(train_frame[feature], errors="coerce")
        share = float(values.notna().mean())
        coverage[feature] = share
        if share >= minimum_coverage:
            selected.append(feature)
    return selected, coverage


def _event_weights(
    y: pd.Series,
    event_ids: pd.Series,
) -> np.ndarray:
    event = event_ids.astype("string").where(y.eq(1))
    temporary = pd.DataFrame({"target": y.astype(int), "event_id": event})
    return event_balanced_sample_weights(
        temporary, target_col="target", event_id_col="event_id"
    ).to_numpy(dtype=float)


def _evaluate_with_folds(
    frame: pd.DataFrame,
    *,
    candidate: HazardCandidateSpec,
    folds: Sequence[OutcomeYearFold],
    horizon: int,
    locked_test_start_year: int,
    origin_col: str,
    target_col: str,
    event_id_col: str,
) -> HazardCVResult:
    missing_features = sorted(set(candidate.feature_names).difference(frame.columns))
    if missing_features:
        raise ValueError(f"frame missing candidate features: {missing_features}")

    origin = pd.to_numeric(frame[origin_col], errors="raise").astype(int)
    label_year = origin + horizon
    development_mask = label_year.lt(locked_test_start_year)
    ledger_parts: list[pd.DataFrame] = []
    fold_details: list[dict[str, Any]] = []

    for fold in folds:
        train_positions = np.asarray(fold.train_positions, dtype=int)
        validation_positions = np.asarray(fold.validation_positions, dtype=int)
        train = frame.iloc[train_positions]
        validation = frame.iloc[validation_positions]
        y_train = pd.to_numeric(train[target_col], errors="raise").astype(int)
        y_validation = pd.to_numeric(validation[target_col], errors="raise").astype(int)
        if y_train.nunique() < 2:
            raise ValueError(f"fold {fold.fold} training target has fewer than two classes")

        selected, coverage = _selected_features(
            train, candidate.feature_names, candidate.minimum_feature_coverage
        )
        if not selected:
            raise ValueError(f"fold {fold.fold} has no eligible training feature")
        effective_weight = candidate.effective_positive_weight(y_train)
        class_weight = {0: 1.0, 1: effective_weight}
        model = RegularizedDiscreteTimeHazardModel(
            link=candidate.link,
            C=candidate.C,
            class_weight=class_weight,
            nonnegative_coefficients=candidate.nonnegative_coefficients,
            metadata={
                "candidate": candidate.name,
                "fold": fold.fold,
                "horizon": horizon,
                "locked_test_start_year": locked_test_start_year,
                "effective_positive_class_weight": effective_weight,
            },
        )
        sample_weight = _event_weights(y_train, train[event_id_col])
        model.fit(train[selected], y_train.to_numpy(), sample_weight=sample_weight)
        raw_probability = model.predict_hazard(validation[selected])

        fold_details.append(
            {
                **fold.to_dict(),
                "train_label_available_year_max": int(label_year.iloc[train_positions].max()),
                "validation_label_available_year_min": int(
                    label_year.iloc[validation_positions].min()
                ),
                "selected_features": selected,
                "training_feature_coverage": coverage,
                "effective_positive_class_weight": effective_weight,
                "training_positive_rows": int(y_train.sum()),
                "optimization_success": bool(model.optimization_success_),
            }
        )
        ledger_parts.append(
            pd.DataFrame(
                {
                    "source_position": validation_positions,
                    "fold": fold.fold,
                    "label_available_year": label_year.iloc[
                        validation_positions
                    ].to_numpy(dtype=int),
                    "y": y_validation.to_numpy(dtype=int),
                    "raw_probability": raw_probability,
                }
            )
        )

    ledger = pd.concat(ledger_parts, ignore_index=True).sort_values(
        ["fold", "source_position"]
    ).reset_index(drop=True)
    y_oof = ledger["y"].to_numpy(dtype=int)
    raw_probability = ledger["raw_probability"].to_numpy(dtype=float)
    fold_ids = ledger["fold"].to_numpy(dtype=int)
    if np.unique(y_oof).size < 2:
        raise ValueError("pooled OOF target needs both classes")

    if candidate.calibration == "sigmoid":
        probability, calibration = _cross_fitted_sigmoid(
            raw_probability,
            y_oof,
            fold_ids,
            minimum_positives=candidate.minimum_calibration_positives,
            random_state=candidate.random_state,
        )
    else:
        probability = raw_probability.copy()
        calibration = {
            "requested": "none",
            "applied": "none",
            "reason": "candidate requested no probability calibration",
            "minimum_positive_rows": candidate.minimum_calibration_positives,
            "cross_fitted_models": [],
            "final_model": None,
        }
    ledger["probability"] = probability

    per_fold_metrics: list[dict[str, Any]] = []
    for fold_id, group in ledger.groupby("fold", sort=True):
        metrics = _binary_metrics(group["y"], group["probability"])
        detail = fold_details[int(fold_id)]
        per_fold_metrics.append(
            {
                "fold": int(fold_id),
                "validation_start_year": detail["validation_start_year"],
                "validation_end_year": detail["validation_end_year"],
                **metrics,
            }
        )

    return HazardCVResult(
        candidate=candidate,
        horizon=horizon,
        locked_test_start_year=locked_test_start_year,
        development_rows=int(development_mask.sum()),
        excluded_locked_or_later_rows=int((~development_mask).sum()),
        pooled_metrics=_binary_metrics(y_oof, probability),
        raw_pooled_metrics=_binary_metrics(y_oof, raw_probability),
        per_fold_metrics=per_fold_metrics,
        stability=_metric_stability(per_fold_metrics),
        calibration=_json_safe(calibration),
        fold_details=_json_safe(fold_details),
        ledger=ledger,
    )


def evaluate_hazard_candidate(
    frame: pd.DataFrame,
    candidate: HazardCandidateSpec,
    *,
    horizon: int,
    locked_test_start_year: int,
    validation_blocks: Sequence[tuple[int, int]],
    origin_col: str = "forecast_origin_year",
    target_col: str = "crisis_hazard_year_1",
    event_id_col: str = "hazard_event_id",
) -> HazardCVResult:
    """Evaluate one candidate without reading locked-period targets or features."""

    folds = make_expanding_outcome_folds(
        frame,
        horizon=horizon,
        locked_test_start_year=locked_test_start_year,
        validation_blocks=validation_blocks,
        origin_col=origin_col,
        target_col=target_col,
        event_id_col=event_id_col,
    )
    return _evaluate_with_folds(
        frame,
        candidate=candidate,
        folds=folds,
        horizon=horizon,
        locked_test_start_year=locked_test_start_year,
        origin_col=origin_col,
        target_col=target_col,
        event_id_col=event_id_col,
    )


def evaluate_hazard_candidates(
    frame: pd.DataFrame,
    candidates: Iterable[HazardCandidateSpec],
    *,
    horizon: int,
    locked_test_start_year: int,
    validation_blocks: Sequence[tuple[int, int]],
    origin_col: str = "forecast_origin_year",
    target_col: str = "crisis_hazard_year_1",
    event_id_col: str = "hazard_event_id",
) -> list[HazardCVResult]:
    """Evaluate a predeclared candidate list on identical expanding folds."""

    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("at least one candidate is required")
    names = [candidate.name for candidate in candidate_list]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    folds = make_expanding_outcome_folds(
        frame,
        horizon=horizon,
        locked_test_start_year=locked_test_start_year,
        validation_blocks=validation_blocks,
        origin_col=origin_col,
        target_col=target_col,
        event_id_col=event_id_col,
    )
    return [
        _evaluate_with_folds(
            frame,
            candidate=candidate,
            folds=folds,
            horizon=horizon,
            locked_test_start_year=locked_test_start_year,
            origin_col=origin_col,
            target_col=target_col,
            event_id_col=event_id_col,
        )
        for candidate in candidate_list
    ]


def select_hazard_candidate(
    results: Sequence[HazardCVResult],
    *,
    metric: Literal["average_precision", "roc_auc", "ap_lift"] = "average_precision",
    tolerance: float = 1e-6,
) -> HazardCVResult:
    """Select the best candidate, preferring the simpler specification on ties."""

    if not results:
        raise ValueError("at least one CV result is required")
    if tolerance < 0:
        raise ValueError("selection tolerance cannot be negative")
    scored: list[tuple[HazardCVResult, float]] = []
    for result in results:
        value = result.pooled_metrics.get(metric)
        if value is not None and math.isfinite(float(value)):
            scored.append((result, float(value)))
    if not scored:
        raise ValueError(f"no candidate has a finite {metric}")
    best_value = max(value for _, value in scored)
    tied = [result for result, value in scored if value >= best_value - tolerance]
    return min(
        tied,
        key=lambda result: (
            result.candidate.complexity_rank,
            len(result.candidate.feature_names),
            result.candidate.name,
        ),
    )


__all__ = [
    "HazardCVResult",
    "HazardCandidateSpec",
    "OutcomeYearFold",
    "SigmoidCalibrationArtifact",
    "evaluate_hazard_candidate",
    "evaluate_hazard_candidates",
    "make_expanding_outcome_folds",
    "select_hazard_candidate",
]
