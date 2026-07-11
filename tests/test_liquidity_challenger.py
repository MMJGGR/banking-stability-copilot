import pandas as pd

from src.scripts.build_liquidity_challenger import _compare, _tier


def test_tier_bins_five_tier_scale():
    scores = pd.Series([1.0, 2.0, 2.1, 4.0, 6.0, 8.0, 9.5])
    assert list(_tier(scores)) == [1, 1, 2, 2, 3, 4, 5]


def test_compare_reports_deltas_and_tier_changes():
    baseline = pd.DataFrame(
        {"country_code": ["AAA", "BBB", "CCC"], "risk_score": [5.0, 3.0, 8.0]}
    )
    challenger = pd.DataFrame(
        {"country_code": ["AAA", "BBB", "CCC"], "risk_score": [5.2, 5.0, 8.0]}
    )

    result = _compare(baseline, challenger, "unit")
    assert result["countries_compared"] == 3
    # deltas: 0.2, 2.0, 0.0 -> mean 0.733, one country moves >= 1 point
    assert result["mean_absolute_score_change"] == 0.733
    assert result["countries_moving_at_least_one_point"] == 1
    # BBB moves tier 2 (3.0) -> tier 3 (5.0); others stay
    assert result["risk_tier_changes"] == 1
    # largest mover is BBB
    assert result["largest_movements"][0]["country_code"] == "BBB"
    assert result["largest_movements"][0]["delta"] == 2.0


def test_compare_drops_countries_missing_from_either_side():
    baseline = pd.DataFrame({"country_code": ["AAA", "BBB"], "risk_score": [5.0, 3.0]})
    challenger = pd.DataFrame({"country_code": ["AAA"], "risk_score": [5.0]})
    result = _compare(baseline, challenger, "unit")
    assert result["countries_compared"] == 1
