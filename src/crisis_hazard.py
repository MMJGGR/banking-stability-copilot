"""Transparent discrete-time hazard foundations for systemic-crisis warning.

This module deliberately separates five concerns that were previously easy to
conflate:

* the observed crisis-onset targets at each forecast horizon;
* conversion of conditional annual hazards into cumulative incidence;
* episode-balanced training weights;
* a small, serialisable regularised hazard estimator; and
* routing between a long-history core expert and a data-richer modern expert.

Nothing here selects an operating threshold or claims that data coverage is
statistical confidence.  Threshold governance remains in ``crisis_validation``
and the router labels its output explicitly as *evidence* confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted

from src.crisis_model_features import (
    BANKING_FEATURES,
    CREDIT_LIQUIDITY_FEATURES,
    EXTERNAL_VULNERABILITY_FEATURES,
    MACRO_FEATURES,
)
from src.crisis_panel import crisis_event_frame


HISTORICAL_CORE_FEATURES: tuple[str, ...] = tuple(
    dict.fromkeys(
        (*MACRO_FEATURES, *CREDIT_LIQUIDITY_FEATURES, *EXTERNAL_VULNERABILITY_FEATURES)
    )
)
MODERN_INCREMENTAL_FEATURES: tuple[str, ...] = tuple(BANKING_FEATURES)
MODERN_FULL_FEATURES: tuple[str, ...] = tuple(
    dict.fromkeys((*HISTORICAL_CORE_FEATURES, *MODERN_INCREMENTAL_FEATURES))
)


def _json_safe(value: Any) -> Any:
    """Recursively reduce common scientific-Python values to JSON primitives."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    return value


@dataclass(frozen=True)
class CrisisTargetDefinition:
    """Governed description of one systemic-onset target."""

    name: str
    horizon_years: int
    role: str
    description: str


CRISIS_TARGET_HIERARCHY: tuple[CrisisTargetDefinition, ...] = (
    CrisisTargetDefinition(
        "crisis_hazard_1y",
        1,
        "champion_target",
        "Conditional probability that a systemic crisis starts in the next year.",
    ),
    CrisisTargetDefinition(
        "crisis_hazard_year_2",
        2,
        "horizon_component",
        "Conditional probability of onset in year two, if crisis-free through year one.",
    ),
    CrisisTargetDefinition(
        "crisis_hazard_year_3",
        3,
        "horizon_component",
        "Conditional probability of onset in year three, if crisis-free through year two.",
    ),
    CrisisTargetDefinition(
        "crisis_onset_in_years_2_3",
        3,
        "reported_increment",
        "Unconditional onset incidence in years two to three from the forecast origin.",
    ),
    CrisisTargetDefinition(
        "crisis_onset_within_3y",
        3,
        "reported_cumulative",
        "Cumulative systemic-crisis incidence within three years.",
    ),
)


def _normalise_events(events: pd.DataFrame | object) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        required = {"country_code", "crisis_start_year"}
        missing = required.difference(events.columns)
        if missing:
            raise ValueError(f"events missing required columns: {sorted(missing)}")
        frame = events.copy()
        if "crisis_event_id" not in frame:
            end = frame.get("crisis_end_year", frame["crisis_start_year"])
            frame["crisis_event_id"] = (
                frame["country_code"].astype(str).str.upper()
                + "-"
                + frame["crisis_start_year"].astype(int).astype(str)
                + "-"
                + pd.Series(end, index=frame.index).astype(int).astype(str)
            )
    else:
        frame = crisis_event_frame(events)
    if frame.empty:
        return pd.DataFrame(
            columns=["country_code", "crisis_start_year", "crisis_event_id"]
        )
    frame = frame[["country_code", "crisis_start_year", "crisis_event_id"]].copy()
    frame["country_code"] = frame["country_code"].astype(str).str.strip().str.upper()
    frame["crisis_start_year"] = pd.to_numeric(
        frame["crisis_start_year"], errors="raise"
    ).astype(int)
    return frame.drop_duplicates().reset_index(drop=True)


def build_crisis_target_hierarchy(
    panel: pd.DataFrame,
    events: pd.DataFrame | object | None = None,
    *,
    label_coverage_end_year: int | None = None,
    country_col: str = "country_code",
    origin_col: str = "forecast_origin_year",
) -> pd.DataFrame:
    """Add leakage-safe one-, two-, and three-year onset targets.

    Hazard targets for years two and three are nullable: a country that has
    already entered crisis is no longer at risk in later intervals.  When a
    label-coverage end year is supplied, an unresolved horizon is also marked
    ``NA`` rather than being silently treated as a tranquil negative.

    ``events`` should normally be the official episode table (or a
    ``CrisisLabels`` instance).  For compatibility with an existing 1--3 year
    crisis panel it may be omitted when exact ``years_to_crisis`` and
    ``crisis_event_id`` columns are already present.
    """

    required = {country_col, origin_col}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")

    result = panel.copy()
    result[country_col] = result[country_col].astype(str).str.strip().str.upper()
    origins = pd.to_numeric(result[origin_col], errors="raise").astype(int)
    result[origin_col] = origins

    if events is None:
        required_existing = {"years_to_crisis", "crisis_event_id"}
        missing_existing = required_existing.difference(result.columns)
        if missing_existing:
            raise ValueError(
                "events are required unless panel contains years_to_crisis and "
                "crisis_event_id"
            )
        years_to_onset = pd.to_numeric(
            result["years_to_crisis"], errors="coerce"
        ).astype("Int64")
        event_ids = result["crisis_event_id"].astype("string")
    else:
        event_frame = _normalise_events(events)
        row_key = pd.Series(np.arange(len(result)), index=result.index, dtype="int64")
        left = pd.DataFrame(
            {
                "_hazard_row_id": row_key.to_numpy(),
                "country_code": result[country_col].to_numpy(),
                "forecast_origin_year": origins.to_numpy(),
            }
        )
        candidates = left.merge(event_frame, on="country_code", how="left")
        candidates["_years_to_onset"] = (
            candidates["crisis_start_year"] - candidates["forecast_origin_year"]
        )
        candidates = candidates[
            candidates["_years_to_onset"].between(1, 3, inclusive="both")
        ].sort_values(
            ["_hazard_row_id", "_years_to_onset", "crisis_event_id"]
        )
        nearest = candidates.drop_duplicates("_hazard_row_id", keep="first").set_index(
            "_hazard_row_id"
        )
        years_to_onset = pd.Series(
            row_key.map(nearest["_years_to_onset"]), index=result.index, dtype="Int64"
        )
        event_ids = pd.Series(
            row_key.map(nearest["crisis_event_id"]), index=result.index, dtype="string"
        )

    years_to_onset = years_to_onset.where(years_to_onset.between(1, 3))
    event_ids = event_ids.where(years_to_onset.notna())
    result["hazard_event_id"] = event_ids
    result["years_to_onset"] = years_to_onset

    def observed_through(horizon: int) -> pd.Series:
        if label_coverage_end_year is None:
            return pd.Series(True, index=result.index)
        return origins.add(horizon).le(int(label_coverage_end_year))

    for horizon in (1, 2, 3):
        positive = years_to_onset.eq(horizon).fillna(False)
        earlier_event = years_to_onset.lt(horizon).fillna(False)
        observed = observed_through(horizon)
        values = pd.Series(pd.NA, index=result.index, dtype="Int8")
        values.loc[positive] = 1
        values.loc[~positive & ~earlier_event & observed] = 0
        result[f"crisis_hazard_year_{horizon}"] = values
        result[f"at_risk_year_{horizon}"] = (~earlier_event & (observed | positive)).astype(
            bool
        )

        cumulative_positive = years_to_onset.le(horizon).fillna(False)
        cumulative = pd.Series(pd.NA, index=result.index, dtype="Int8")
        cumulative.loc[cumulative_positive] = 1
        cumulative.loc[~cumulative_positive & observed] = 0
        result[f"crisis_onset_within_{horizon}y"] = cumulative

    result["crisis_hazard_1y"] = result["crisis_hazard_year_1"]
    medium_positive = years_to_onset.between(2, 3, inclusive="both").fillna(False)
    medium_observed = observed_through(3) | years_to_onset.notna()
    medium = pd.Series(pd.NA, index=result.index, dtype="Int8")
    medium.loc[medium_positive] = 1
    medium.loc[~medium_positive & medium_observed] = 0
    result["crisis_onset_in_years_2_3"] = medium
    return result


def cumulative_incidence_from_annual_hazards(
    annual_hazards: Sequence[Sequence[float]] | np.ndarray,
    *,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Convert three conditional annual hazards into reported probabilities.

    The years-two-to-three probability is incremental and unconditional from
    the current forecast origin: survival through year one is included.
    """

    hazards = np.asarray(annual_hazards, dtype=float)
    if hazards.ndim == 1:
        hazards = hazards.reshape(1, -1)
    if hazards.ndim != 2 or hazards.shape[1] != 3:
        raise ValueError("annual_hazards must have exactly three columns")
    if not np.isfinite(hazards).all() or ((hazards < 0) | (hazards > 1)).any():
        raise ValueError("annual hazards must be finite probabilities in [0, 1]")

    survival = np.cumprod(1.0 - hazards, axis=1)
    within = 1.0 - survival
    years_2_3 = (1.0 - hazards[:, 0]) * (
        1.0 - (1.0 - hazards[:, 1]) * (1.0 - hazards[:, 2])
    )
    return pd.DataFrame(
        {
            "annual_hazard_1y": hazards[:, 0],
            "annual_hazard_year_2": hazards[:, 1],
            "annual_hazard_year_3": hazards[:, 2],
            "probability_1y": within[:, 0],
            "probability_within_2y": within[:, 1],
            "probability_years_2_3": years_2_3,
            "probability_within_3y": within[:, 2],
        },
        index=index,
    )


def event_balanced_sample_weights(
    frame: pd.DataFrame,
    *,
    target_col: str = "crisis_onset_within_3y",
    event_id_col: str = "hazard_event_id",
) -> pd.Series:
    """Give every positive episode equal total influence without class drift.

    Negative rows retain weight one.  Positive weights are scaled so their
    *mean* is also one, while the sum assigned to each distinct episode is
    identical.  This composes cleanly with an estimator's class weighting.
    Nullable/censored targets receive ``NaN`` and must be excluded from fitting.
    """

    missing = {target_col, event_id_col}.difference(frame.columns)
    if missing:
        raise ValueError(f"frame missing required columns: {sorted(missing)}")
    target = pd.to_numeric(frame[target_col], errors="coerce")
    invalid = target.dropna()[~target.dropna().isin([0, 1])]
    if not invalid.empty:
        raise ValueError(f"{target_col} must contain only 0, 1, or missing")

    weights = pd.Series(np.nan, index=frame.index, dtype=float, name="sample_weight")
    observed = target.notna()
    weights.loc[observed] = 1.0
    positive = target.eq(1)
    if not positive.any():
        return weights
    event_ids = frame.loc[positive, event_id_col].astype("string")
    if event_ids.isna().any() or event_ids.str.strip().eq("").any():
        raise ValueError("Every positive row requires a non-empty crisis event id")

    counts = event_ids.value_counts()
    event_total = float(positive.sum()) / float(len(counts))
    weights.loc[positive] = event_ids.map(event_total / counts).astype(float)
    return weights


def _as_numeric_matrix(
    X: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            names = X.columns.astype(str).to_numpy(dtype=object)
            selected = X
        else:
            missing = sorted(set(feature_names).difference(X.columns))
            if missing:
                raise ValueError(f"X missing fitted features: {missing}")
            names = np.asarray(feature_names, dtype=object)
            selected = X[list(feature_names)]
        matrix = selected.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    else:
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        names = (
            np.asarray(feature_names, dtype=object)
            if feature_names is not None
            else np.asarray([f"x{i}" for i in range(matrix.shape[1])], dtype=object)
        )
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("X must be a two-dimensional feature matrix")
    if np.isinf(matrix).any():
        raise ValueError("X cannot contain infinite values")
    return matrix, names


class RegularizedDiscreteTimeHazardModel(ClassifierMixin, BaseEstimator):
    """Median-imputed, standardised L2 hazard model with logit or cloglog link.

    The fitted artifact is JSON-safe: coefficients, medians, scales, feature
    names, link and caller metadata can be exported with :meth:`to_dict` or
    :meth:`to_json`.  This avoids a pickle dependency in production serving.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        link: str = "cloglog",
        C: float = 0.25,
        class_weight: str | dict | None = "balanced",
        fit_intercept: bool = True,
        standardize: bool = True,
        nonnegative_coefficients: bool = False,
        max_iter: int = 2_000,
        tol: float = 1e-8,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.link = link
        self.C = C
        self.class_weight = class_weight
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.nonnegative_coefficients = nonnegative_coefficients
        self.max_iter = max_iter
        self.tol = tol
        self.metadata = metadata

    def _validate_parameters(self) -> None:
        if self.link not in {"logit", "cloglog"}:
            raise ValueError("link must be 'logit' or 'cloglog'")
        if self.C <= 0:
            raise ValueError("C must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if self.tol <= 0:
            raise ValueError("tol must be positive")

    def _prepare_fit_X(self, X) -> np.ndarray:
        raw, names = _as_numeric_matrix(X)
        medians = np.nanmedian(raw, axis=0)
        if np.isnan(medians).any():
            missing = names[np.isnan(medians)].tolist()
            raise ValueError(f"Training features are entirely missing: {missing}")
        filled = np.where(np.isnan(raw), medians, raw)
        if self.standardize:
            scales = np.std(filled, axis=0)
            scales = np.where(scales > 1e-12, scales, 1.0)
        else:
            scales = np.ones(raw.shape[1], dtype=float)
        self.feature_names_in_ = names
        self.feature_medians_ = medians.astype(float)
        self.feature_scales_ = scales.astype(float)
        return (filled - self.feature_medians_) / self.feature_scales_

    def _prepare_predict_X(self, X) -> np.ndarray:
        check_is_fitted(
            self,
            ["coef_", "intercept_", "feature_medians_", "feature_scales_"],
        )
        raw, _ = _as_numeric_matrix(X, self.feature_names_in_)
        if raw.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {raw.shape[1]} features; expected {self.n_features_in_}"
            )
        filled = np.where(np.isnan(raw), self.feature_medians_, raw)
        return (filled - self.feature_medians_) / self.feature_scales_

    def _inverse_link(self, linear: np.ndarray) -> np.ndarray:
        if self.link == "logit":
            return expit(linear)
        eta = np.clip(linear, -30.0, 30.0)
        return -np.expm1(-np.exp(eta))

    def fit(self, X, y, sample_weight=None):
        self._validate_parameters()
        transformed = self._prepare_fit_X(X)
        y_array = np.asarray(y)
        if y_array.ndim != 1 or len(y_array) != len(transformed):
            raise ValueError("y must have one value per row")
        if pd.isna(y_array).any():
            raise ValueError("y cannot contain missing values")
        classes = np.unique(y_array)
        if len(classes) != 2:
            raise ValueError("Discrete-time hazard fitting requires two classes")
        self.classes_ = classes
        binary_y = (y_array == classes[1]).astype(float)

        weights = compute_sample_weight(self.class_weight, y_array).astype(float)
        if sample_weight is not None:
            supplied = np.asarray(sample_weight, dtype=float)
            if supplied.shape != (len(y_array),):
                raise ValueError("sample_weight must have one value per row")
            weights *= supplied
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("sample weights must be finite, non-negative, and non-zero")
        weight_total = float(weights.sum())

        n_features = transformed.shape[1]
        n_parameters = n_features + int(self.fit_intercept)
        initial = np.zeros(n_parameters, dtype=float)
        prevalence = np.clip(np.average(binary_y, weights=weights), 1e-6, 1 - 1e-6)
        if self.fit_intercept:
            if self.link == "logit":
                initial[0] = np.log(prevalence / (1.0 - prevalence))
            else:
                initial[0] = np.log(-np.log1p(-prevalence))

        def unpack(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            if self.fit_intercept:
                return float(parameters[0]), parameters[1:]
            return 0.0, parameters

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            intercept, coefficients = unpack(parameters)
            linear = intercept + transformed @ coefficients
            if self.link == "logit":
                losses = np.logaddexp(0.0, linear) - binary_y * linear
                derivatives = expit(linear) - binary_y
            else:
                eta = np.clip(linear, -30.0, 30.0)
                intensity = np.exp(eta)
                probability = np.clip(-np.expm1(-intensity), 1e-15, 1.0)
                losses = np.where(binary_y == 1, -np.log(probability), intensity)
                positive_gradient = -intensity * np.exp(-intensity) / probability
                derivatives = np.where(binary_y == 1, positive_gradient, intensity)
            data_loss = float(np.dot(weights, losses) / weight_total)
            # ``C`` follows the familiar penalised-likelihood convention:
            #
            #   sum(weighted loss) + ||beta||^2 / (2 C)
            #
            # We optimise the *mean* weighted loss for numerical stability, so
            # the penalty must be divided by the same total weight.  Omitting
            # this factor makes regularisation grow with the sample size and,
            # on the crisis panel, shrinks every coefficient almost to zero.
            regularization_scale = self.C * weight_total
            penalty = (
                0.5
                * float(np.dot(coefficients, coefficients))
                / regularization_scale
            )
            residual = weights * derivatives / weight_total
            coefficient_gradient = (
                transformed.T @ residual + coefficients / regularization_scale
            )
            gradient = (
                np.concatenate(([residual.sum()], coefficient_gradient))
                if self.fit_intercept
                else coefficient_gradient
            )
            return data_loss + penalty, gradient

        coefficient_bound = (0.0, None) if self.nonnegative_coefficients else (None, None)
        bounds = [coefficient_bound] * n_features
        if self.fit_intercept:
            bounds = [(None, None), *bounds]
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": int(self.max_iter), "ftol": float(self.tol)},
        )
        if not np.isfinite(result.x).all():
            raise RuntimeError("Hazard-model optimisation returned non-finite parameters")
        intercept, coefficients = unpack(result.x)
        self.coef_ = np.asarray(coefficients, dtype=float).reshape(1, -1)
        self.intercept_ = np.asarray([intercept], dtype=float)
        self.n_features_in_ = n_features
        self.n_iter_ = np.asarray([int(result.nit)], dtype=int)
        self.optimization_success_ = bool(result.success)
        self.optimization_message_ = str(result.message)
        return self

    def decision_function(self, X) -> np.ndarray:
        transformed = self._prepare_predict_X(X)
        return transformed @ self.coef_[0] + self.intercept_[0]

    def predict_hazard(self, X) -> np.ndarray:
        return np.clip(self._inverse_link(self.decision_function(X)), 1e-12, 1 - 1e-12)

    def predict_proba(self, X) -> np.ndarray:
        positive = self.predict_hazard(X)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X) -> np.ndarray:
        return self.classes_[(self.predict_hazard(X) >= 0.5).astype(int)]

    def coefficient_frame(self) -> pd.DataFrame:
        """Return both standardised and original-unit effects."""

        check_is_fitted(self, ["coef_", "feature_scales_"])
        standardised = self.coef_[0]
        raw = standardised / self.feature_scales_
        return pd.DataFrame(
            {
                "feature": self.feature_names_in_,
                "standardised_coefficient": standardised,
                "raw_unit_coefficient": raw,
                "multiplicative_effect_per_sd": np.exp(standardised),
                "link": self.link,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        check_is_fitted(
            self,
            ["coef_", "intercept_", "feature_medians_", "feature_scales_"],
        )
        return {
            "schema_version": self.SCHEMA_VERSION,
            "model_type": "regularized_discrete_time_hazard",
            "link": self.link,
            "C": float(self.C),
            "class_weight": _json_safe(self.class_weight),
            "fit_intercept": bool(self.fit_intercept),
            "standardize": bool(self.standardize),
            "nonnegative_coefficients": bool(self.nonnegative_coefficients),
            "max_iter": int(self.max_iter),
            "tol": float(self.tol),
            "feature_names": self.feature_names_in_.astype(str).tolist(),
            "feature_medians": self.feature_medians_.astype(float).tolist(),
            "feature_scales": self.feature_scales_.astype(float).tolist(),
            "coefficients": self.coef_[0].astype(float).tolist(),
            "intercept": float(self.intercept_[0]),
            "classes": [value.item() if hasattr(value, "item") else value for value in self.classes_],
            "metadata": _json_safe(dict(self.metadata or {})),
        }

    def to_json(self, **json_kwargs) -> str:
        return json.dumps(self.to_dict(), **json_kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegularizedDiscreteTimeHazardModel":
        if payload.get("model_type") != "regularized_discrete_time_hazard":
            raise ValueError("Unsupported hazard model artifact")
        if int(payload.get("schema_version", -1)) != cls.SCHEMA_VERSION:
            raise ValueError("Unsupported hazard model schema version")
        class_weight = payload.get("class_weight")
        if isinstance(class_weight, Mapping):
            class_weight = {
                int(key) if str(key).lstrip("-").isdigit() else key: value
                for key, value in class_weight.items()
            }
        model = cls(
            link=str(payload["link"]),
            C=float(payload["C"]),
            class_weight=class_weight,
            fit_intercept=bool(payload["fit_intercept"]),
            standardize=bool(payload["standardize"]),
            nonnegative_coefficients=bool(payload["nonnegative_coefficients"]),
            max_iter=int(payload["max_iter"]),
            tol=float(payload["tol"]),
            metadata=payload.get("metadata"),
        )
        model.feature_names_in_ = np.asarray(payload["feature_names"], dtype=object)
        model.feature_medians_ = np.asarray(payload["feature_medians"], dtype=float)
        model.feature_scales_ = np.asarray(payload["feature_scales"], dtype=float)
        model.coef_ = np.asarray(payload["coefficients"], dtype=float).reshape(1, -1)
        model.intercept_ = np.asarray([payload["intercept"]], dtype=float)
        model.classes_ = np.asarray(payload["classes"])
        model.n_features_in_ = len(model.feature_names_in_)
        expected = model.n_features_in_
        if not all(
            len(values) == expected
            for values in (model.feature_medians_, model.feature_scales_, model.coef_[0])
        ):
            raise ValueError("Hazard model artifact has inconsistent feature lengths")
        return model

    @classmethod
    def from_json(cls, payload: str) -> "RegularizedDiscreteTimeHazardModel":
        return cls.from_dict(json.loads(payload))


def regularized_logit_hazard(**kwargs) -> RegularizedDiscreteTimeHazardModel:
    """Build the transparent regularised-logit champion candidate."""

    return RegularizedDiscreteTimeHazardModel(link="logit", **kwargs)


def regularized_cloglog_hazard(**kwargs) -> RegularizedDiscreteTimeHazardModel:
    """Build the transparent complementary-log-log champion candidate."""

    return RegularizedDiscreteTimeHazardModel(link="cloglog", **kwargs)


@dataclass(frozen=True)
class ExpertRoutingConfig:
    """Availability contract for long-history and modern-full experts."""

    historical_features: tuple[str, ...] = HISTORICAL_CORE_FEATURES
    modern_incremental_features: tuple[str, ...] = MODERN_INCREMENTAL_FEATURES
    # The historical contract is intentionally lower than the modern one:
    # pre-2000 rows have no direct banking block and only partial external
    # coverage.  Forty percent still requires evidence across several core
    # variables while allowing the long-history expert to do its intended job.
    historical_min_coverage: float = 0.40
    modern_min_coverage: float = 0.50

    def __post_init__(self) -> None:
        if not self.historical_features:
            raise ValueError("historical_features cannot be empty")
        if not self.modern_incremental_features:
            raise ValueError("modern_incremental_features cannot be empty")
        if set(self.historical_features).intersection(self.modern_incremental_features):
            raise ValueError("historical and modern incremental features must not overlap")
        if not 0 <= self.historical_min_coverage <= 1:
            raise ValueError("historical_min_coverage must be between zero and one")
        if not 0 <= self.modern_min_coverage <= 1:
            raise ValueError("modern_min_coverage must be between zero and one")

    @property
    def modern_full_features(self) -> tuple[str, ...]:
        return (*self.historical_features, *self.modern_incremental_features)


def _availability_matrix(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    available: dict[str, pd.Series] = {}
    for feature in features:
        flag = f"{feature}__available"
        if flag in frame:
            available[feature] = frame[flag].fillna(False).astype(bool)
        elif feature in frame:
            available[feature] = pd.to_numeric(frame[feature], errors="coerce").notna()
        else:
            available[feature] = pd.Series(False, index=frame.index)
    return pd.DataFrame(available, index=frame.index)


class HazardExpertRouter:
    """Route and score rows without conflating missing data with low risk."""

    def __init__(self, config: ExpertRoutingConfig | None = None) -> None:
        self.config = config or ExpertRoutingConfig()

    def route(self, frame: pd.DataFrame) -> pd.DataFrame:
        historical = _availability_matrix(frame, self.config.historical_features)
        modern = _availability_matrix(frame, self.config.modern_incremental_features)
        historical_coverage = historical.mean(axis=1)
        modern_coverage = modern.mean(axis=1)
        historical_eligible = historical_coverage.ge(
            self.config.historical_min_coverage
        )
        modern_eligible = historical_eligible & modern_coverage.ge(
            self.config.modern_min_coverage
        )
        selected = np.select(
            [modern_eligible, historical_eligible],
            ["modern_full", "historical_core"],
            default="insufficient_evidence",
        )
        evidence_score = np.where(
            modern_eligible,
            0.5 * historical_coverage + 0.5 * modern_coverage,
            np.where(historical_eligible, historical_coverage, historical_coverage),
        )
        confidence = np.select(
            [
                modern_eligible & pd.Series(evidence_score, index=frame.index).ge(0.80),
                modern_eligible,
                historical_eligible & historical_coverage.ge(0.80),
                historical_eligible,
            ],
            ["high", "moderate", "moderate", "limited"],
            default="insufficient",
        )
        reason = np.select(
            [modern_eligible, historical_eligible],
            [
                "core and modern banking evidence meet coverage contract",
                "modern banking evidence below coverage contract; using historical core",
            ],
            default="historical core evidence below coverage contract",
        )
        return pd.DataFrame(
            {
                "selected_expert": selected,
                "evidence_confidence": confidence,
                "evidence_coverage_score": evidence_score,
                "historical_coverage": historical_coverage,
                "modern_incremental_coverage": modern_coverage,
                "routing_reason": reason,
            },
            index=frame.index,
        )

    @staticmethod
    def _models_by_horizon(model_or_models) -> dict[int, Any]:
        if isinstance(model_or_models, Mapping):
            models = {int(key): value for key, value in model_or_models.items()}
            missing = {1, 2, 3}.difference(models)
            if missing:
                raise ValueError(f"Missing horizon models: {sorted(missing)}")
            return {horizon: models[horizon] for horizon in (1, 2, 3)}
        return {horizon: model_or_models for horizon in (1, 2, 3)}

    @staticmethod
    def _predict_one(model, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_hazard"):
            values = model.predict_hazard(X)
        elif hasattr(model, "predict_proba"):
            values = model.predict_proba(X)[:, 1]
        else:
            raise TypeError("Expert model must expose predict_hazard or predict_proba")
        values = np.asarray(values, dtype=float)
        if values.shape != (len(X),):
            raise ValueError("Expert model returned an unexpected probability shape")
        return values

    def predict_current(
        self,
        frame: pd.DataFrame,
        *,
        historical_models,
        modern_models,
    ) -> pd.DataFrame:
        """Score a current cross-section with three conditional horizon models.

        ``historical_models`` and ``modern_models`` may each be either one model
        reused across all horizons or a mapping with integer keys 1, 2, and 3.
        The output always reports this distinction in ``horizon_model_basis``.
        """

        routing = self.route(frame)
        hazards = np.full((len(frame), 3), np.nan, dtype=float)
        historical_by_horizon = self._models_by_horizon(historical_models)
        modern_by_horizon = self._models_by_horizon(modern_models)

        for expert_name, models, features in (
            (
                "historical_core",
                historical_by_horizon,
                self.config.historical_features,
            ),
            ("modern_full", modern_by_horizon, self.config.modern_full_features),
        ):
            mask = routing["selected_expert"].eq(expert_name)
            if not mask.any():
                continue
            # ``reindex`` deliberately materialises unavailable contract
            # features as NaN.  The expert's fitted medians can then handle
            # partial-but-eligible rows without a brittle KeyError.
            X = frame.reindex(columns=list(features)).loc[mask]
            row_positions = np.flatnonzero(mask.to_numpy())
            for horizon in (1, 2, 3):
                hazards[row_positions, horizon - 1] = self._predict_one(
                    models[horizon], X
                )

        probability_columns = [
            "annual_hazard_1y",
            "annual_hazard_year_2",
            "annual_hazard_year_3",
            "probability_1y",
            "probability_within_2y",
            "probability_years_2_3",
            "probability_within_3y",
        ]
        probability_output = pd.DataFrame(
            np.nan, index=frame.index, columns=probability_columns
        )
        scored = np.isfinite(hazards).all(axis=1)
        if scored.any():
            scored_positions = np.flatnonzero(scored)
            derived = cumulative_incidence_from_annual_hazards(
                hazards[scored], index=frame.index[scored_positions]
            )
            probability_output.loc[derived.index, derived.columns] = derived

        separate_horizons = isinstance(historical_models, Mapping) and isinstance(
            modern_models, Mapping
        )
        output = routing.join(probability_output)
        output["horizon_model_basis"] = (
            "horizon_specific" if separate_horizons else "same_model_flat_covariates"
        )
        return output


__all__ = [
    "CRISIS_TARGET_HIERARCHY",
    "HISTORICAL_CORE_FEATURES",
    "MODERN_FULL_FEATURES",
    "MODERN_INCREMENTAL_FEATURES",
    "CrisisTargetDefinition",
    "ExpertRoutingConfig",
    "HazardExpertRouter",
    "RegularizedDiscreteTimeHazardModel",
    "build_crisis_target_hierarchy",
    "cumulative_incidence_from_annual_hazards",
    "event_balanced_sample_weights",
    "regularized_cloglog_hazard",
    "regularized_logit_hazard",
]
