"""Broad dense-ridge baseline with train-only preprocessing, research only.

No cap or univariate target screen is imposed. Columns completely unobserved
in training remain in the input contract and are recorded as unlearnable in
that fold; future observations cannot retroactively make them learnable.
Hyperparameter selection and final predictive validation are separate work.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .inventory import ForecastDataError

FORBIDDEN_COLUMNS = {
    "country_code", "country_name", "forecast_origin", "forecast_origin_year",
    "feature_cutoff_year", "observation_period", "available_at", "vintage_at",
    "retrieved_at", "target", "target_end", "target_available_at", "crisis_target",
}


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ForecastDataError("A nonempty DataFrame with unique feature columns is required")
    if not frame.index.is_unique:
        raise ForecastDataError("Nonunique row index; use unique country-origin IDs")
    if any(not isinstance(c, str) or not c.strip() for c in frame.columns):
        raise ForecastDataError("Feature names must be nonempty strings")
    if FORBIDDEN_COLUMNS & set(frame.columns) or any(c.startswith("target_") for c in frame):
        raise ForecastDataError("Keep targets and temporal/country metadata outside predictors")
    if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in frame):
        raise ForecastDataError("Explicit numeric feature matrix required")
    data = frame.astype(float)
    if np.isinf(data.to_numpy()).any():
        raise ForecastDataError("Infinite predictor values")
    return data


class BroadRidgeRegressor(RegressorMixin, BaseEstimator):
    """Median-impute, standardize and regularize using training data only."""

    def __init__(self, alpha: float = 10.0, add_missing_indicators: bool = True):
        self.alpha = alpha
        self.add_missing_indicators = add_missing_indicators

    def fit(self, X: pd.DataFrame, y: pd.Series):
        data = _validate(X)
        if not isinstance(y, pd.Series) or not y.index.equals(data.index):
            raise ForecastDataError("Target index must exactly match predictor index")
        target = pd.to_numeric(y, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(target).all() or len(data) < 2:
            raise ForecastDataError("At least two rows with observed finite targets required")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ForecastDataError("Ridge alpha must be positive and finite")
        feature_names = data.columns.tolist()
        active = data.columns[data.notna().any()].tolist()
        inactive = sorted(set(feature_names) - set(active))
        if not active:
            raise ForecastDataError("No observed training features")
        medians = data[active].median()
        values = data[active].fillna(medians).to_numpy()
        if self.add_missing_indicators:
            values = np.column_stack([values, data[active].isna().to_numpy(dtype=float)])
        scaler = StandardScaler().fit(values)
        model = Ridge(alpha=self.alpha, solver="lsqr", tol=1e-8).fit(scaler.transform(values), target)
        # Assign learned state only after a successful fit.
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.n_features_in_ = len(feature_names)
        self.active_features_, self.unlearnable_features_ = active, inactive
        self.medians_, self.scaler_, self.model_ = medians, scaler, model
        self.training_audit_ = {
            "candidate_features": len(feature_names), "learnable_features": len(active),
            "unlearnable_all_missing_in_training": inactive,
            "feature_cap": None, "preprocessing": "fit_on_training_only",
            "alpha": float(self.alpha), "predictive_validation": "not_established",
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")
        data = _validate(X)
        if set(data.columns) != set(self.feature_names_in_):
            raise ForecastDataError("Prediction feature schema differs from training")
        active = data[self.active_features_]
        values = active.fillna(self.medians_).to_numpy()
        if self.add_missing_indicators:
            values = np.column_stack([values, active.isna().to_numpy(dtype=float)])
        return self.model_.predict(self.scaler_.transform(values))


def compare_development_fold(X: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame,
                             persistence: pd.Series, compact_features: list[str],
                             validation_start: str, validation_end: str, *,
                             alpha: float = 10.0, embargo_days: int = 0) -> dict:
    """Run matched persistence/compact/broad benchmarks on a declared fold.

    This is a development comparison, not hyperparameter selection or final
    confirmation. Input panel construction/vintage evidence is upstream.
    No rows/features are selected using validation outcomes.
    """
    from .temporal import purged_forward_split
    if not all(isinstance(v, (pd.Series, pd.DataFrame)) and v.index.equals(X.index)
               for v in (y, metadata, persistence)):
        raise ForecastDataError("Predictors, targets, metadata and persistence must align")
    if not compact_features or len(set(compact_features)) != len(compact_features):
        raise ForecastDataError("Compact comparator needs unique declared predictors")
    if not set(compact_features) <= set(X.columns):
        raise ForecastDataError("Unknown compact predictor")
    split = purged_forward_split(metadata, validation_start, validation_end, embargo_days=embargo_days)
    train, test = split.train_positions, split.validation_positions
    actual = y.iloc[test].to_numpy(dtype=float)
    persistent = persistence.iloc[test].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(persistent).all():
        raise ForecastDataError("Matched observed outcomes and persistence inputs required")
    models = {
        "compact_ridge": BroadRidgeRegressor(alpha=alpha).fit(X.iloc[train][compact_features], y.iloc[train]),
        "broad_ridge": BroadRidgeRegressor(alpha=alpha).fit(X.iloc[train], y.iloc[train]),
    }
    predictions = pd.DataFrame({
        "actual": actual, "persistence": persistent,
        "compact_ridge": models["compact_ridge"].predict(X.iloc[test][compact_features]),
        "broad_ridge": models["broad_ridge"].predict(X.iloc[test]),
    }, index=X.index[test])
    metrics = {}
    for name in ("persistence", "compact_ridge", "broad_ridge"):
        error = predictions[name].to_numpy() - actual
        metrics[name] = {"mae": float(np.abs(error).mean()),
                         "rmse": float(np.sqrt(np.mean(error ** 2))), "rows": len(test)}
    return {"status": "development_only_not_promotion_evidence", "predictions": predictions,
            "metrics": metrics, "split_audit": split.audit,
            "model_audits": {k: v.training_audit_ for k, v in models.items()}}
