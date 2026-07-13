import pandas as pd
import pytest

from src.dashboard.calculated_series import (
    compute_cross_sectional_share,
    compute_formula,
    compute_ratio,
    compute_temporal_change,
    normalize_observation_frame,
)


def _frame():
    return pd.DataFrame(
        {
            "country_code": ["KEN", "KEN", "KEN", "MOZ", "MOZ", "MOZ"],
            "indicator_code": ["A", "A", "B", "A", "A", "B"],
            "period": [
                "2024-12-31",
                "2025-12-31",
                "2025-12-31",
                "2025-12-31",
                "2026-12-31",
                "2025-12-31",
            ],
            "frequency": ["A", "A", "A", "A", "A", "A"],
            "value": [100.0, 120.0, 60.0, 80.0, 100.0, 40.0],
        }
    )


def test_normalize_observation_frame_aligns_one_indicator():
    normalized = normalize_observation_frame(
        _frame(),
        "A",
        "indicator_code",
        "Indicator A",
    )

    assert set(normalized["indicator_label"]) == {"Indicator A"}
    assert set(normalized["country_code"]) == {"KEN", "MOZ"}
    assert normalized["date"].dtype.kind == "M"


def test_compute_ratio_aligns_country_date_frequency_only():
    numerator = normalize_observation_frame(_frame(), "A", "indicator_code", "A")
    denominator = normalize_observation_frame(_frame(), "B", "indicator_code", "B")

    ratio = compute_ratio(numerator, denominator, scale=100)

    assert set(ratio["country_code"]) == {"KEN", "MOZ"}
    ken = ratio.loc[ratio["country_code"] == "KEN"].iloc[0]
    moz = ratio.loc[ratio["country_code"] == "MOZ"].iloc[0]
    assert ken["value"] == pytest.approx(200.0)
    assert moz["value"] == pytest.approx(200.0)
    assert ratio["date"].nunique() == 1


def test_compute_ratio_drops_zero_denominator():
    numerator = pd.DataFrame(
        {
            "country_code": ["KEN"],
            "date": [pd.Timestamp("2025-12-31")],
            "frequency": ["A"],
            "value": [10.0],
        }
    )
    denominator = numerator.copy()
    denominator["value"] = 0.0

    ratio = compute_ratio(numerator, denominator)

    assert ratio.empty


def test_compute_formula_supports_difference_divided_by_third_operand():
    frame = pd.DataFrame(
        {
            "country_code": ["KEN", "KEN", "KEN"],
            "indicator_code": ["NPL", "PROV", "CAP"],
            "period": ["2025-12-31", "2025-12-31", "2025-12-31"],
            "frequency": ["A", "A", "A"],
            "value": [120.0, 45.0, 300.0],
        }
    )
    npl = normalize_observation_frame(frame, "NPL", "indicator_code", "NPL")
    provisions = normalize_observation_frame(frame, "PROV", "indicator_code", "Provisions")
    capital = normalize_observation_frame(frame, "CAP", "indicator_code", "Capital")

    result = compute_formula(npl, provisions, "a_minus_b_div_c", capital, scale=100)

    assert result["value"].iloc[0] == pytest.approx(25.0)
    assert result["a_value"].iloc[0] == pytest.approx(120.0)
    assert result["b_value"].iloc[0] == pytest.approx(45.0)
    assert result["c_value"].iloc[0] == pytest.approx(300.0)


def test_compute_formula_drops_zero_third_operand():
    base = pd.DataFrame(
        {
            "country_code": ["KEN"],
            "date": [pd.Timestamp("2025-12-31")],
            "frequency": ["A"],
            "value": [10.0],
        }
    )
    zero = base.copy()
    zero["value"] = 0.0

    result = compute_formula(base, base, "a_minus_b_div_c", zero)

    assert result.empty


def test_compute_cross_sectional_share_sums_to_100_by_period():
    data = normalize_observation_frame(_frame(), "A", "indicator_code", "A")
    data = data.loc[data["date"] == pd.Timestamp("2025-12-31")]

    shares = compute_cross_sectional_share(data)

    assert shares["value"].sum() == pytest.approx(100.0)
    assert shares.loc[shares["country_code"] == "KEN", "value"].iloc[0] == pytest.approx(60.0)


def test_compute_temporal_change_modes():
    data = normalize_observation_frame(_frame(), "A", "indicator_code", "A")
    ken = data.loc[data["country_code"] == "KEN"]

    period = compute_temporal_change(ken, "period_pct")
    base = compute_temporal_change(ken, "base_pct")
    index = compute_temporal_change(ken, "index_100")

    assert period["value"].iloc[0] == pytest.approx(20.0)
    assert base["value"].iloc[-1] == pytest.approx(20.0)
    assert index["value"].iloc[0] == pytest.approx(100.0)
    assert index["value"].iloc[-1] == pytest.approx(120.0)


def test_compute_temporal_change_rejects_unknown_mode():
    data = normalize_observation_frame(_frame(), "A", "indicator_code", "A")

    with pytest.raises(ValueError):
        compute_temporal_change(data, "bad_mode")
