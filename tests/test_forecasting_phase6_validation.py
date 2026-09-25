import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.crisis_validation import ValidationConfig
from src.forecasting.phase6_validation import evaluate_registered_windows


def test_registered_windows_are_later_and_outcome_resolved():
    rows = []
    for country_index in range(30):
        country = f"C{country_index:02d}"
        for year in range(1981, 2022):
            latent = np.sin((year - 1981) / 4) + country_index / 30
            target = int((country_index + 3 * year) % 19 == 0)
            rows.append({
                "country": country,
                "origin": year,
                "x1": latent,
                "x2": country_index % 5,
                "target": target,
                "event_id": f"{country}-{year}" if target else pd.NA,
            })
    frame = pd.DataFrame(rows)
    config = ValidationConfig(
        outer_splits=4,
        inner_splits=4,
        calibration="sigmoid",
        recall_floor=0.50,
        random_state=17,
        bootstrap_iterations=0,
        temporal_outer_splits=4,
        temporal_inner_splits=3,
        temporal_min_train_periods=12,
        temporal_gap_periods=0,
        purge_overlapping_events=True,
    )
    result = evaluate_registered_windows(
        lambda: LogisticRegression(
            C=0.1, solver="liblinear", max_iter=2000, random_state=17
        ),
        frame[["x1", "x2"]],
        frame.target,
        frame.country,
        frame.origin,
        event_ids=frame.event_id,
        config=config,
    )
    assert result.summary["outer_time_ordered"] is True
    assert result.summary["outcome_gap_years"] == 3
    assert result.ledger.outer_fold.nunique() >= 2
    for detail in result.fold_details:
        assert detail["train_max_origin"] + 3 < detail["test_min_origin"]
        assert detail["inner_strategy"] == (
            "country_grouped_within_earlier_training_sample"
        )
