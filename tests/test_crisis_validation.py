import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

from src.crisis_validation import (
    ValidationConfig,
    bootstrap_confidence_intervals,
    evaluate_country_grouped,
    evaluate_forward_temporal,
    evaluate_outer_split,
    select_review_threshold,
)


def _panel(seed: int = 17):
    rng = np.random.default_rng(seed)
    countries = np.repeat([f"C{i:02d}" for i in range(18)], 10)
    origins = np.tile(np.arange(2000, 2010), 18)
    country_number = np.repeat(np.arange(18), 10)
    x1 = rng.normal(size=len(countries))
    x2 = rng.normal(size=len(countries))
    category = np.where(country_number % 3 == 0, "A", "B")
    # Every country has at least one event, including in early periods.  The
    # feature signal is deliberately imperfect so all threshold metrics matter.
    deterministic_event = ((country_number + origins) % 8 == 0).astype(int)
    latent_event = (1.15 * x1 - 0.65 * x2 + rng.normal(scale=0.9, size=len(x1)) > 1.5)
    y = np.maximum(deterministic_event, latent_event.astype(int))
    X = pd.DataFrame(
        {
            "credit_gap": x1,
            "debt_service": x2,
            "income_group": category,
        },
        index=np.arange(10_000, 10_000 + len(countries)),
    )
    event_id = np.where(
        y == 1,
        np.char.add("event-", ((country_number * 10 + origins) // 2).astype(str)),
        None,
    )
    metadata = pd.DataFrame(
        {"country": countries, "origin": origins, "event_id": event_id}
    )
    return X, y, countries, origins, metadata


def _estimator():
    return LogisticRegression(
        solver="liblinear", C=0.3, class_weight="balanced", random_state=31
    )


def _config(**overrides):
    values = dict(
        outer_splits=3,
        inner_splits=3,
        calibration="sigmoid",
        recall_floor=0.60,
        random_state=9,
        bootstrap_iterations=0,
        temporal_outer_splits=2,
        temporal_inner_splits=2,
        temporal_min_train_periods=7,
    )
    values.update(overrides)
    return ValidationConfig(**values)


def test_review_threshold_maximises_precision_with_recall_floor():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    probabilities = np.array([0.90, 0.80, 0.40, 0.20, 0.70, 0.35, 0.10, 0.05])
    result = select_review_threshold(y, probabilities, recall_floor=0.75)

    assert result["recall"] >= 0.75
    assert result["threshold"] == 0.40
    assert result["precision"] == 0.75


def test_outer_test_labels_cannot_change_threshold_or_probabilities():
    """The strongest leakage invariant: mutate every untouched test label.

    The fitted preprocessor, estimator, calibrator, inner predictions, selected
    threshold, outer probabilities, and decisions must remain byte-for-byte the
    same.  Only metrics that compare those frozen outputs to test labels may move.
    """

    X, y, groups, times, metadata = _panel()
    train = np.flatnonzero(np.isin(groups, [f"C{i:02d}" for i in range(14)]))
    test = np.flatnonzero(~np.isin(np.arange(len(y)), train))
    config = _config()

    original = evaluate_outer_split(
        _estimator,
        X,
        y,
        groups,
        train,
        test,
        times=times,
        metadata=metadata,
        config=config,
    )
    mutated_y = y.copy()
    mutated_y[test] = 1 - mutated_y[test]
    mutated = evaluate_outer_split(
        _estimator,
        X,
        mutated_y,
        groups,
        train,
        test,
        times=times,
        metadata=metadata,
        config=config,
    )

    assert original.fold_details[0]["threshold"] == mutated.fold_details[0]["threshold"]
    np.testing.assert_array_equal(original.ledger["proba"], mutated.ledger["proba"])
    np.testing.assert_array_equal(original.ledger["pred"], mutated.ledger["pred"])
    np.testing.assert_array_equal(
        original.tuning_ledger["raw_proba"], mutated.tuning_ledger["raw_proba"]
    )
    np.testing.assert_array_equal(
        original.tuning_ledger["proba"], mutated.tuning_ledger["proba"]
    )
    assert original.summary["recall"] != mutated.summary["recall"]


def test_outer_test_features_cannot_change_tuning_or_threshold():
    """Preprocessing and calibration fit inside training folds, never on test X."""

    X, y, groups, times, metadata = _panel()
    train = np.flatnonzero(np.isin(groups, [f"C{i:02d}" for i in range(14)]))
    test = np.setdiff1d(np.arange(len(y)), train)
    config = _config(calibration="isotonic")

    original = evaluate_outer_split(
        _estimator,
        X,
        y,
        groups,
        train,
        test,
        times=times,
        metadata=metadata,
        config=config,
    )
    mutated_X = X.copy()
    mutated_X.iloc[test, mutated_X.columns.get_loc("credit_gap")] = 1_000_000
    mutated_X.iloc[test, mutated_X.columns.get_loc("debt_service")] = -1_000_000
    mutated_X.iloc[test, mutated_X.columns.get_loc("income_group")] = "UNSEEN"
    mutated = evaluate_outer_split(
        _estimator,
        mutated_X,
        y,
        groups,
        train,
        test,
        times=times,
        metadata=metadata,
        config=config,
    )

    assert original.fold_details[0]["threshold"] == mutated.fold_details[0]["threshold"]
    assert_frame_equal(
        original.tuning_ledger.reset_index(drop=True),
        mutated.tuning_ledger.reset_index(drop=True),
        check_exact=True,
    )


class _RecordingTransformer(BaseEstimator, TransformerMixin):
    fit_indices: list[set[int]] = []

    def fit(self, X, y=None):
        type(self).fit_indices.append(set(map(int, X.index)))
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        return X[self.columns_].to_numpy(dtype=float)


def test_custom_preprocessor_is_fitted_only_on_fold_training_rows():
    X, y, groups, times, metadata = _panel()
    X = X[["credit_gap", "debt_service"]]
    train = np.flatnonzero(np.isin(groups, [f"C{i:02d}" for i in range(14)]))
    test = np.setdiff1d(np.arange(len(y)), train)
    _RecordingTransformer.fit_indices = []

    result = evaluate_outer_split(
        _estimator,
        X,
        y,
        groups,
        train,
        test,
        times=times,
        metadata=metadata,
        config=_config(calibration="none"),
        preprocessor_factory=lambda _: _RecordingTransformer(),
    )

    # Data are internally reset to positional indices.  The final fit must equal
    # outer training exactly; every earlier call is a strict inner-training subset.
    expected_train = set(train.tolist())
    expected_test = set(test.tolist())
    assert _RecordingTransformer.fit_indices[-1] == expected_train
    assert all(indices.isdisjoint(expected_test) for indices in _RecordingTransformer.fit_indices)
    assert all(indices <= expected_train for indices in _RecordingTransformer.fit_indices)
    assert len(result.tuning_ledger) > 0


def test_country_grouped_outer_folds_are_disjoint_and_complete():
    X, y, groups, times, metadata = _panel()
    result = evaluate_country_grouped(
        _estimator,
        X,
        y,
        groups,
        times=times,
        metadata=metadata,
        config=_config(),
    )

    for detail in result.fold_details:
        assert set(detail["train_groups"]).isdisjoint(detail["test_groups"])
    assert len(result.ledger) == len(X)
    assert result.ledger["source_index"].is_unique
    assert {
        "country",
        "origin",
        "event_id",
        "y",
        "proba",
        "threshold",
        "pred",
    } <= set(result.ledger.columns)
    for metric in (
        "roc_auc",
        "average_precision",
        "pr_auc",
        "brier",
        "log_loss",
        "precision",
        "recall",
        "f1",
        "alert_burden",
        "confusion",
    ):
        assert metric in result.summary


def test_forward_evaluation_never_trains_on_present_or_future_origins():
    X, y, groups, times, metadata = _panel()
    result = evaluate_forward_temporal(
        _estimator,
        X,
        y,
        groups,
        times,
        metadata=metadata,
        config=_config(calibration="none"),
    )

    for detail in result.fold_details:
        assert detail["train_max_time"] < detail["test_min_time"]
        for inner in detail["inner_folds"]:
            assert inner["train_max_time"] < inner["valid_min_time"]
    assert result.ledger["origin"].min() >= 2007


def test_cluster_bootstrap_is_deterministic_for_country_and_event():
    X, y, groups, times, metadata = _panel()
    result = evaluate_country_grouped(
        _estimator,
        X,
        y,
        groups,
        times=times,
        metadata=metadata,
        config=_config(),
    )
    first = bootstrap_confidence_intervals(
        result.ledger, iterations=30, confidence_level=0.90, random_state=123
    )
    second = bootstrap_confidence_intervals(
        result.ledger, iterations=30, confidence_level=0.90, random_state=123
    )

    assert_frame_equal(first, second, check_exact=True)
    assert set(first["cluster"]) == {"country", "event_id"}
    assert set(first["metric"]) >= {
        "roc_auc",
        "average_precision",
        "brier",
        "precision",
        "recall",
    }
    assert (first["valid_draws"] > 0).all()
