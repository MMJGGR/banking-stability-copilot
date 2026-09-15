import pandas as pd
import pytest
from src.scripts.review_data_candidate import compare_frames, country_coverage


def test_feature_values_and_missingness():
    old = pd.DataFrame({"country_code": ["KEN", "COD"], "capital": [10., None]})
    new = pd.DataFrame({"country_code": ["KEN", "COD"], "capital": [12., 8.]})
    changes = compare_frames(old, new)
    assert len(changes) == 2
    assert changes.loc[changes.country_code.eq("KEN"), "numeric_change"].iloc[0] == 2


def test_equal_numeric_types_and_missing_values_are_unchanged():
    old = pd.DataFrame({"country_code": ["KEN"], "capital": [10], "missing": [None]})
    new = pd.DataFrame({"country_code": ["KEN"], "capital": [10.0], "missing": [float("nan")]})
    assert compare_frames(old, new).empty


def test_added_and_removed_countries_are_visible():
    old = pd.DataFrame({"country_code": ["KEN"], "capital": [10.]})
    new = pd.DataFrame({"country_code": ["COD"], "capital": [8.]})
    changes = compare_frames(old, new)
    assert set(changes["_merge"].astype(str)) == {"left_only", "right_only"}


def test_duplicate_countries_are_rejected():
    frame = pd.DataFrame({"country_code": ["KEN", "KEN"], "capital": [10., 10.]})
    with pytest.raises(RuntimeError, match="duplicate"):
        compare_frames(frame, frame)


def test_source_dates_are_country_specific_and_cutoff_safe(tmp_path, monkeypatch):
    frame = pd.DataFrame({"country_code": ["KEN", "KEN", "COD"], "indicator_code": ["CAP"] * 3,
                  "period": ["2026-05-31", "2026-10-31", "2026-07-31"], "value": [10., 20., 8.]})
    monkeypatch.setattr(pd, "read_parquet", lambda path: frame.copy())
    result = country_coverage(tmp_path, "FSIC", "2026-09-15").set_index("country_code")
    assert result.loc["KEN", "latest_observation"] == "2026-05-31"
    assert result.loc["COD", "latest_observation"] == "2026-07-31"
