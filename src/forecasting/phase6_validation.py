"""Rare-event-safe time-ordered validation for the Phase 6 challenge.

Outer windows are pre-registered and strictly later than training observations.
A three-year gap ensures every training target window resolves before the outer
test begins. Probability calibration and review-threshold selection use only
the outer training sample, with country-grouped inner folds because annual
rare-event blocks can legitimately contain no positive events.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.crisis_validation import (
    ValidationConfig,
    ValidationResult,
    bootstrap_confidence_intervals,
    classification_metrics,
    evaluate_outer_split,
)

REGISTERED_WINDOWS = (
    (1995, 2000),
    (2001, 2007),
    (2008, 2014),
    (2015, 2021),
)
HORIZON_END = 3


def evaluate_registered_windows(
    estimator_factory,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    times: pd.Series,
    *,
    event_ids: pd.Series,
    config: ValidationConfig,
) -> ValidationResult:
    """Evaluate predeclared later-time blocks with training-only inner tuning."""
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True).astype(int)
    groups = pd.Series(groups).reset_index(drop=True).astype(str)
    times = pd.Series(times).reset_index(drop=True).astype(int)
    event_ids = pd.Series(event_ids).reset_index(drop=True).astype("string")
    if not (len(X) == len(y) == len(groups) == len(times) == len(event_ids)):
        raise ValueError("Phase 6 validation inputs must have equal row counts")

    metadata = pd.DataFrame({
        "country": groups,
        "origin": times,
        "event_id": event_ids,
    })
    ledgers, tuning_ledgers, fold_metrics, details = [], [], [], []
    skipped = []

    for fold_number, (start, end) in enumerate(REGISTERED_WINDOWS):
        train = np.flatnonzero((times + HORIZON_END < start).to_numpy())
        test = np.flatnonzero(times.between(start, end).to_numpy())
        if not len(train) or not len(test):
            skipped.append({
                "outer_fold": fold_number,
                "start": start,
                "end": end,
                "reason": "empty_train_or_test",
            })
            continue

        # Defensive event purge. With the registered three-year gap, a future
        # event should not already appear in training, but retain the invariant.
        test_events = set(event_ids.iloc[test].dropna().astype(str))
        if test_events:
            keep = ~event_ids.iloc[train].astype("string").isin(test_events).to_numpy()
            train = train[keep]
        if np.unique(y.iloc[train]).size < 2:
            skipped.append({
                "outer_fold": fold_number,
                "start": start,
                "end": end,
                "reason": "training_sample_has_one_class",
                "train_rows": len(train),
                "test_rows": len(test),
            })
            continue

        result = evaluate_outer_split(
            estimator_factory,
            X,
            y,
            groups,
            train,
            test,
            times=times,
            metadata=metadata,
            config=config,
            inner_strategy="grouped",
        )
        ledger = result.ledger.copy()
        tuning = result.tuning_ledger.copy()
        ledger["outer_fold"] = fold_number
        tuning["outer_fold"] = fold_number
        ledger["design"] = "registered_forward_country_grouped_inner"
        tuning["design"] = "registered_forward_country_grouped_inner"
        ledgers.append(ledger)
        tuning_ledgers.append(tuning)

        metric = result.fold_metrics.copy()
        metric["outer_fold"] = fold_number
        metric["registered_start"] = start
        metric["registered_end"] = end
        metric["train_max_origin"] = int(times.iloc[train].max())
        metric["test_min_origin"] = int(times.iloc[test].min())
        metric["outcome_gap_years"] = HORIZON_END
        fold_metrics.append(metric)

        detail = dict(result.fold_details[0])
        detail.update({
            "outer_fold": fold_number,
            "registered_start": start,
            "registered_end": end,
            "train_max_origin": int(times.iloc[train].max()),
            "test_min_origin": int(times.iloc[test].min()),
            "outcome_gap_years": HORIZON_END,
            "inner_strategy": "country_grouped_within_earlier_training_sample",
        })
        details.append(detail)

    if len(ledgers) < 2:
        raise ValueError(
            "Phase 6 needs at least two usable registered later-time windows; "
            f"skipped={skipped}"
        )

    ledger = pd.concat(ledgers, ignore_index=True)
    tuning = pd.concat(tuning_ledgers, ignore_index=True)
    metrics = pd.concat(fold_metrics, ignore_index=True)
    summary = classification_metrics(ledger.y, ledger.proba, ledger.pred)
    summary.update({
        "design": "registered_forward_country_grouped_inner",
        "folds": int(ledger.outer_fold.nunique()),
        "countries": int(ledger.country.nunique()),
        "threshold_policy": "review",
        "outer_time_ordered": True,
        "outcome_gap_years": HORIZON_END,
        "inner_tuning": "country_grouped_within_earlier_training_sample",
        "registered_windows": [list(window) for window in REGISTERED_WINDOWS],
        "skipped_windows": skipped,
        "note": (
            "Every outer test block is later than its training data. Inner "
            "calibration/threshold folds are country-grouped and use only the "
            "outer training sample to avoid one-class annual rare-event blocks."
        ),
    })
    cis = bootstrap_confidence_intervals(
        ledger,
        iterations=config.bootstrap_iterations,
        confidence_level=config.confidence_level,
        random_state=config.random_state,
    )
    return ValidationResult(
        design="registered_forward_country_grouped_inner",
        summary=summary,
        fold_metrics=metrics,
        ledger=ledger,
        tuning_ledger=tuning,
        bootstrap_cis=cis,
        fold_details=details,
    )


def validate_candidates_registered(replacement, estimator_factory, config) -> dict:
    """Run state, observed, and combined candidates for all three targets."""
    panel = replacement.panel
    feature_sets = {
        "state": replacement.state_features,
        "observed": replacement.observed_features,
        "combined": replacement.combined_features,
    }
    targets = {
        "banking": ("banking_target", "banking_event_id"),
        "sovereign": ("sovereign_target", "sovereign_event_id"),
        "either": ("either_target", "either_event_id"),
    }
    output = {}
    for target_name, (target_column, event_column) in targets.items():
        output[target_name] = {}
        for model_name, columns in feature_sets.items():
            output[target_name][model_name] = evaluate_registered_windows(
                estimator_factory(),
                panel[columns],
                panel[target_column],
                panel.entity_code,
                panel.forecast_origin_year,
                event_ids=panel[event_column],
                config=config,
            )
    return output
