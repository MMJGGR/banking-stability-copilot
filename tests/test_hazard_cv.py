import json

import numpy as np
import pandas as pd

from src.hazard_cv import (
    HazardCVResult,
    HazardCandidateSpec,
    _cross_fitted_sigmoid,
    evaluate_hazard_candidate,
    make_expanding_outcome_folds,
    select_hazard_candidate,
)


def test_outcome_year_folds_embargo_unobservable_and_future_labels():
    frame = pd.DataFrame(
        {
            "forecast_origin_year": range(2000, 2010),
            "crisis_hazard_year_2": [0, 1] * 5,
            "hazard_event_id": [pd.NA, "E1"] * 5,
        }
    )

    folds = make_expanding_outcome_folds(
        frame,
        horizon=2,
        locked_test_start_year=2011,
        validation_blocks=[(2005, 2007), (2008, 2010)],
        target_col="crisis_hazard_year_2",
    )

    label_year = frame["forecast_origin_year"] + 2
    for fold in folds:
        train_years = label_year.iloc[list(fold.train_positions)]
        validation_years = label_year.iloc[list(fold.validation_positions)]
        assert train_years.max() < validation_years.min()
        assert validation_years.between(
            fold.validation_start_year, fold.validation_end_year
        ).all()
        assert (train_years < 2011).all()
        assert (validation_years < 2011).all()
    used = {
        position
        for fold in folds
        for position in (*fold.train_positions, *fold.validation_positions)
    }
    assert not any(label_year.iloc[position] >= 2011 for position in used)


def test_event_ids_are_purged_from_training_across_fold_boundary():
    frame = pd.DataFrame(
        {
            "forecast_origin_year": [2000, 2001, 2002, 2003],
            "crisis_hazard_year_1": [0, 0, 1, "LOCKED"],
            "hazard_event_id": [pd.NA, "EPISODE-A", "EPISODE-A", "LOCKED"],
        }
    )

    fold = make_expanding_outcome_folds(
        frame,
        horizon=1,
        locked_test_start_year=2004,
        validation_blocks=[(2003, 2003)],
    )[0]

    assert fold.validation_positions == (2,)
    assert fold.train_positions == (0,)
    assert fold.purged_train_rows == 1
    assert fold.purged_event_ids == ("EPISODE-A",)
    train_events = set(
        frame.iloc[list(fold.train_positions)]["hazard_event_id"].dropna().astype(str)
    )
    validation_events = set(
        frame.iloc[list(fold.validation_positions)]["hazard_event_id"]
        .dropna()
        .astype(str)
    )
    assert train_events.isdisjoint(validation_events)


def test_candidate_evaluation_does_not_read_locked_target_values():
    origins = np.arange(2000, 2012)
    label_year = origins + 1
    development = label_year < 2010
    target = np.where(development, np.arange(len(origins)) % 2, "DO_NOT_READ")
    event_id = np.asarray(
        [f"EVENT-{index}" if index % 2 else pd.NA for index in range(len(origins))],
        dtype=object,
    )
    event_id[~development] = "DO_NOT_READ"
    feature = np.asarray(np.linspace(-2.0, 2.0, len(origins)), dtype=object)
    feature[~development] = "DO_NOT_READ"
    frame = pd.DataFrame(
        {
            "forecast_origin_year": origins,
            "crisis_hazard_year_1": target,
            "hazard_event_id": event_id,
            "signal": feature,
        }
    )

    result = evaluate_hazard_candidate(
        frame,
        HazardCandidateSpec(name="locked-safe", feature_names=("signal",), C=0.5),
        horizon=1,
        locked_test_start_year=2010,
        validation_blocks=[(2005, 2006), (2007, 2009)],
    )

    assert result.excluded_locked_or_later_rows == int((~development).sum())
    assert result.ledger["label_available_year"].lt(2010).all()
    assert set(result.ledger["source_position"]).isdisjoint(
        set(np.flatnonzero(~development))
    )


def test_sigmoid_calibration_is_forward_only_with_raw_fallbacks():
    raw = np.asarray(
        [0.10, 0.20, 0.80, 0.30, 0.15, 0.25, 0.75, 0.35, 0.20, 0.40, 0.60, 0.80]
    )
    target = np.asarray([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    fold_ids = np.repeat([0, 1, 2], 4)

    calibrated, audit = _cross_fitted_sigmoid(
        raw,
        target,
        fold_ids,
        minimum_positives=2,
        random_state=7,
    )

    # Fold 0 has no history and fold 1 has only one earlier positive, so both
    # retain their raw probabilities. Fold 2 can use folds 0 and 1 only.
    np.testing.assert_allclose(calibrated[:8], raw[:8])
    assert not np.allclose(calibrated[8:], raw[8:])
    assert audit["fallback_folds"] == [0, 1]
    assert audit["calibrated_folds"] == [2]
    fold_two = next(
        item for item in audit["cross_fitted_models"] if item["held_out_fold"] == 2
    )
    assert fold_two["training_folds"] == [0, 1]
    assert 2 not in fold_two["training_folds"]
    assert fold_two["training_rows"] == 8
    # The all-development artifact is available for a later refit, but it did
    # not rewrite the raw fallback probabilities reported for folds 0 and 1.
    assert audit["final_model"]["training_rows"] == len(target)
    assert "not applied to reported OOF metrics" in audit["final_model_usage"]


def _result(
    candidate: HazardCandidateSpec,
    *,
    average_precision: float,
    include_nonfinite: bool = False,
) -> HazardCVResult:
    return HazardCVResult(
        candidate=candidate,
        horizon=1,
        locked_test_start_year=2014,
        development_rows=10,
        excluded_locked_or_later_rows=2,
        pooled_metrics={
            "average_precision": average_precision,
            "roc_auc": np.nan if include_nonfinite else 0.60,
        },
        raw_pooled_metrics={"average_precision": average_precision},
        per_fold_metrics=[],
        stability={"average_precision": {"mean": np.inf if include_nonfinite else 0.5}},
        calibration={"final_model": None},
        fold_details=[],
        ledger=pd.DataFrame(
            {
                "source_position": [1],
                "probability": [np.nan if include_nonfinite else 0.25],
                "timestamp": [pd.Timestamp("2026-06-30")],
            }
        ),
    )


def test_result_serialization_is_strict_json_safe():
    result = _result(
        HazardCandidateSpec(name="json", feature_names=("x",)),
        average_precision=0.25,
        include_nonfinite=True,
    )

    text = result.to_json(include_ledger=True, sort_keys=True)
    payload = json.loads(text)

    assert "NaN" not in text
    assert "Infinity" not in text
    assert payload["pooled_metrics"]["roc_auc"] is None
    assert payload["stability"]["average_precision"]["mean"] is None
    assert payload["ledger"][0]["probability"] is None
    assert payload["ledger"][0]["timestamp"] == "2026-06-30T00:00:00"


def test_candidate_selection_prefers_simpler_candidate_within_tolerance():
    simple = _result(
        HazardCandidateSpec(
            name="simple", feature_names=("x",), complexity_rank=0
        ),
        average_precision=0.3000,
    )
    complex_result = _result(
        HazardCandidateSpec(
            name="complex", feature_names=("x", "z"), complexity_rank=1
        ),
        average_precision=0.3005,
    )

    selected = select_hazard_candidate(
        [complex_result, simple], metric="average_precision", tolerance=0.001
    )

    assert selected is simple
