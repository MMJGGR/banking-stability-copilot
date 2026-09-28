from datetime import date

import pandas as pd

from src.monitor import compute_signals, new_releases


def _monthly(country, values, end="2026-08-31"):
    periods = pd.date_range(end=end, periods=len(values), freq="ME")
    return pd.DataFrame({"country_code": country, "period": periods, "value": values})


def test_signals_flag_stress_and_ignore_calm():
    imf = {
        "reserves_usd": pd.concat([_monthly("ZMB", [100, 95, 90, 80]), _monthly("KEN", [100, 101, 102, 103])]),
        "imf_credit_usd": _monthly("ZMB", [10, 10, 10, 20]),
        "deposits_lcy": _monthly("ZMB", [100, 99, 98, 90]),
        "npl_ratio": pd.DataFrame({
            "country_code": "ZMB",
            "period": pd.to_datetime(["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"]),
            "value": [5.0, 5.5, 6.0, 7.0, 9.5],
        }),
    }
    fx = pd.DataFrame({
        "date": pd.to_datetime(["2026-05-01", "2026-08-31"]),
        "ZMW": [20.0, 25.0],
        "KES": [129.0, 130.0],
    })
    signals = compute_signals(imf, fx, date(2026, 9, 28)).set_index(["country_code", "signal"])

    flagged = set(signals.index[signals.flag])
    assert {
        ("ZMB", "reserves_change_3m"),
        ("ZMB", "imf_credit_change_3m"),
        ("ZMB", "deposits_change_3m"),
        ("ZMB", "npl_change_4q"),
        ("ZMB", "fx_depreciation_90d"),
    } <= flagged
    assert not any(country == "KEN" for country, _ in flagged)
    assert signals.loc[("ZMB", "npl_change_4q"), "value"] == 4.5


def test_new_releases_only_after_a_baseline_exists():
    imf = {"npl_ratio": _monthly("GHA", [10.0, 11.0])}
    assert new_releases(imf, {}) == {}
    previous = {"latest_periods": {"npl_ratio": {"GHA": "2026-07-31"}}}
    assert new_releases(imf, previous) == {"npl_ratio": ["GHA"]}
