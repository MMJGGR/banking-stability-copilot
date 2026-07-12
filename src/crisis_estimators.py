"""Governed estimator candidates for the crisis early-warning model."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class SignConstrainedLogisticRegression(ClassifierMixin, BaseEstimator):
    """L2 logistic regression whose feature coefficients cannot be negative.

    All input columns must first be oriented so a larger value represents more
    risk.  Non-negative bounds then make the final probability monotonic in
    every governed feature, including after serialization and deployment.
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        class_weight: str | dict | None = "balanced",
        fit_intercept: bool = True,
        max_iter: int = 1_000,
        tol: float = 1e-8,
    ) -> None:
        self.C = C
        self.class_weight = class_weight
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y, sample_weight=None):
        if self.C <= 0:
            raise ValueError("C must be positive")
        X, y = check_X_y(X, y, dtype=float, force_all_finite=True)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("SignConstrainedLogisticRegression requires two classes")
        self.classes_ = classes
        binary_y = (y == classes[1]).astype(float)

        weights = compute_sample_weight(self.class_weight, y).astype(float)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape != (len(y),):
                raise ValueError("sample_weight must have one value per row")
            weights *= sample_weight
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("sample weights must be finite, non-negative, and non-zero")
        weight_total = float(weights.sum())

        n_features = X.shape[1]
        n_parameters = n_features + int(self.fit_intercept)
        initial = np.zeros(n_parameters, dtype=float)
        if self.fit_intercept:
            prevalence = np.clip(np.average(binary_y, weights=weights), 1e-6, 1 - 1e-6)
            initial[0] = np.log(prevalence / (1 - prevalence))

        def unpack(parameters):
            if self.fit_intercept:
                return parameters[0], parameters[1:]
            return 0.0, parameters

        def objective(parameters):
            intercept, coefficients = unpack(parameters)
            linear = intercept + X @ coefficients
            data_loss = np.sum(
                weights * (np.logaddexp(0.0, linear) - binary_y * linear)
            ) / weight_total
            penalty = 0.5 * np.dot(coefficients, coefficients) / self.C
            probabilities = expit(linear)
            residual = weights * (probabilities - binary_y) / weight_total
            coefficient_gradient = X.T @ residual + coefficients / self.C
            if self.fit_intercept:
                gradient = np.concatenate(([residual.sum()], coefficient_gradient))
            else:
                gradient = coefficient_gradient
            return float(data_loss + penalty), gradient

        bounds = (
            [(None, None)] + [(0.0, None)] * n_features
            if self.fit_intercept
            else [(0.0, None)] * n_features
        )
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": int(self.max_iter), "ftol": float(self.tol)},
        )
        if not result.success:
            warnings.warn(
                f"Sign-constrained logistic optimization did not fully converge: {result.message}",
                RuntimeWarning,
                stacklevel=2,
            )
        intercept, coefficients = unpack(result.x)
        self.coef_ = np.asarray(coefficients, dtype=float).reshape(1, -1)
        self.intercept_ = np.asarray([intercept], dtype=float)
        self.n_features_in_ = n_features
        self.n_iter_ = np.asarray([int(result.nit)], dtype=int)
        self.optimization_success_ = bool(result.success)
        self.optimization_message_ = str(result.message)
        return self

    def decision_function(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        X = check_array(X, dtype=float, force_all_finite=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; expected {self.n_features_in_}"
            )
        return X @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X):
        positive = expit(self.decision_function(X))
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def logistic_baseline(*, C: float = 0.25, random_state: int = 42):
    return LogisticRegression(
        C=C,
        class_weight="balanced",
        penalty="l2",
        solver="liblinear",
        max_iter=2_000,
        random_state=random_state,
    )


def sign_constrained_logistic(*, C: float = 1.0):
    return SignConstrainedLogisticRegression(C=C, class_weight="balanced")


def monotonic_hist_gradient_boosting(
    n_features: int,
    *,
    learning_rate: float = 0.05,
    max_iter: int = 150,
    max_leaf_nodes: int = 7,
    min_samples_leaf: int = 30,
    l2_regularization: float = 5.0,
    random_state: int = 42,
):
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        class_weight="balanced",
        monotonic_cst=[1] * n_features,
        random_state=random_state,
    )


def monotonic_xgboost(
    n_features: int,
    *,
    n_estimators: int = 150,
    max_depth: int = 2,
    learning_rate: float = 0.03,
    random_state: int = 42,
):
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - training extra is optional in serving
        raise RuntimeError("xgboost is not installed in this environment") from exc
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=10.0,
        gamma=0.5,
        eval_metric="logloss",
        monotone_constraints=tuple([1] * n_features),
        random_state=random_state,
        n_jobs=1,
    )


__all__ = [
    "SignConstrainedLogisticRegression",
    "logistic_baseline",
    "monotonic_hist_gradient_boosting",
    "monotonic_xgboost",
    "sign_constrained_logistic",
]
