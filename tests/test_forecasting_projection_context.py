import pandas as pd
import pytest

from src.forecasting.projection_context import (
    ProjectionContextError,
    assert_projection_use_allowed,
    split_weo_current_vintage,
)


def sample():
    return pd.DataFrame(
        {
            "COUNTRY": ["KEN", "KEN", "KEN", "KEN", None],
            "INDICATOR": ["NGDP_RPCH", "NGDP_RPCH", "NGDP_RPCH", "PCPIPCH", "NGDP_RPCH"],
            "FREQUENCY": ["A", "A", "A", "A", "A"],
            "TIME_PERIOD": [2025, 2026, 2031, 2027, None],
            "OBS_VALUE": [5.0, 5.2, 5.5, 4.1, None],
            "UNIT": ["PT", "PT", "PT", "PT", "PT"],
        }
    )


def test_future_weo_rows_are_separate_scenario_lane():
    split = split_weo_current_vintage(
        sample(),
        cutoff="2026-09-16",
        retrieved_at="2026-09-16T18:00:00Z",
        source_version="IMF.RES:WEO(9.0.0)",
    )
    assert split.historical_or_unverified.TIME_PERIOD.tolist() == [2025]
    assert split.provider_projections.TIME_PERIOD.tolist() == [2026, 2031, 2027]
    assert split.provider_projections.data_role.eq("provider_projection").all()
    assert split.provider_projections.model_admission.eq(
        "scenario_only_not_measurement_or_target"
    ).all()
    assert split.summary["metadata_only_rows"] == 1
    assert split.summary["projection_year_counts"] == {"2026": 1, "2027": 1, "2031": 1}


def test_provider_projections_fail_closed_for_training_and_targets():
    for usage in [
        "measurement_state_fit",
        "transition_target",
        "transition_calibration",
        "historical_backtest_input",
    ]:
        with pytest.raises(ProjectionContextError):
            assert_projection_use_allowed(usage)
    assert_projection_use_allowed("scenario_context")
    assert_projection_use_allowed("provider_benchmark")


def test_split_is_row_order_invariant_in_counts_and_ids():
    a = split_weo_current_vintage(
        sample(), cutoff="2026-09-16", retrieved_at="2026-09-16"
    )
    b = split_weo_current_vintage(
        sample().sample(frac=1, random_state=4),
        cutoff="2026-09-16",
        retrieved_at="2026-09-16",
    )
    assert a.summary == b.summary
    assert set(a.provider_projections.projection_row_id) == set(
        b.provider_projections.projection_row_id
    )
