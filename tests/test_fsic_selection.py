import numpy as np
import pandas as pd
import pytest

from src.fsic_selection import FSICSelectionError, select_fsic_features
from src.feature_engineering import CrisisFeatureEngineer

T1 = "Tier 1 capital to risk-weighted assets, (Core FSI), Percent"
CET1 = "Common Equity Tier 1 capital to risk-weighted assets, (Core FSI), Percent"
CAR = "Regulatory capital to risk-weighted assets, (Core FSI), Percent"


def observation(code="FSI626_CFSI_PT", name=T1, value=10.5859364868, **kwargs):
    row = dict(country_code="VNM", indicator_code=code, indicator_name=name,
               value=value, period=pd.Timestamp("2025-12-31"), unit="PT",
               frequency="Q", observation_status="actual", dataset="FSIC")
    row.update(kwargs)
    return row


def sample():
    return pd.DataFrame([
        observation(), observation(frequency="A"),
        observation("FSI15_CFSI_PT", CET1, 0),
        observation("FSI15_CFSI_PT", CET1, 0, frequency="A"),
        observation("FSI688_CFSI_PT", CAR, 12.061807),
        observation(value=9, period=pd.Timestamp("2024-12-31")),
    ])


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 2026])
def test_actual_extractor_is_invariant_and_distinguishes_cet1(tmp_path, seed):
    eng = CrisisFeatureEngineer(output_dir=str(tmp_path))
    frame = sample()
    expected = eng.extract_fsic_features(frame, as_of_date="2026-09-15").sort_index(axis=1)
    reversed_result = eng.extract_fsic_features(frame.iloc[::-1], as_of_date="2026-09-15").sort_index(axis=1)
    shuffled = eng.extract_fsic_features(frame.sample(frac=1, random_state=seed), as_of_date="2026-09-15").sort_index(axis=1)
    pd.testing.assert_frame_equal(expected, reversed_result, check_exact=True)
    pd.testing.assert_frame_equal(expected, shuffled, check_exact=True)
    assert expected.loc[0, "tier1_capital"] == 10.5859364868
    assert expected.loc[0, "capital_quality"] == pytest.approx(10.5859364868 / 12.061807 * 100)


@pytest.mark.parametrize("frequency", ["Q", "A", "M"])
def test_conflicting_latest_values_fail_closed_in_every_order(frequency):
    frame = pd.DataFrame([observation(), observation(value=11, frequency=frequency)])
    for ordered in [frame, frame.iloc[::-1]]:
        with pytest.raises(FSICSelectionError, match="conflicting latest values"):
            select_fsic_features(ordered)


def test_identical_duplicates_have_explicit_audit():
    audit = []
    result = select_fsic_features(pd.DataFrame([observation(), observation(frequency="A")]), audit=audit)
    assert result.loc[0, "tier1_capital"] == 10.5859364868
    assert audit[0]["indicator_code"] == "FSI626_CFSI_PT"
    assert audit[0]["rows_coalesced"] == 2
    assert audit[0]["frequencies"] == ["A", "Q"]


@pytest.mark.parametrize("unit", ["XDC", "USD", "BP", ""])
def test_wrong_or_missing_units_rejected(unit):
    with pytest.raises(FSICSelectionError, match="units"):
        select_fsic_features(pd.DataFrame([observation(unit=unit)]))


def test_ambiguous_economic_dimension_rejected_even_when_values_equal():
    frame = pd.DataFrame([observation(sector="DT"), observation(sector="OFC")])
    with pytest.raises(FSICSelectionError, match="ambiguous dimension sector"):
        select_fsic_features(frame)


def test_unknown_code_is_not_accepted_by_similar_name():
    with pytest.raises(FSICSelectionError, match="unrecognized coded series"):
        select_fsic_features(pd.DataFrame([observation(code="NEW_UNREVIEWED_PT")]))


def test_name_cannot_contradict_canonical_code():
    with pytest.raises(FSICSelectionError, match="label mismatch"):
        select_fsic_features(pd.DataFrame([observation(name=CET1)]))


def test_cet1_alone_does_not_populate_tier1():
    result = select_fsic_features(pd.DataFrame([observation("FSI15_CFSI_PT", CET1, 0)]))
    assert result.empty
    assert "tier1_capital" not in result


def test_legacy_name_only_is_anchored():
    frame = pd.DataFrame([observation(), observation("FSI15_CFSI_PT", CET1, 0)]).drop(columns="indicator_code")
    assert select_fsic_features(frame).loc[0, "tier1_capital"] == 10.5859364868


def test_missing_latest_is_not_backfilled_without_authority():
    frame = pd.DataFrame([observation(value=np.nan), observation(value=9, period=pd.Timestamp("2024-12-31"))])
    result = select_fsic_features(frame)
    assert np.isnan(result.loc[0, "tier1_capital"])
    assert result.loc[0, "tier1_capital_year"] == 2025


def test_future_and_projected_data_are_excluded(tmp_path):
    frame = pd.DataFrame([observation(), observation(value=99, period=pd.Timestamp("2027-12-31")),
                          observation(value=88, period=pd.Timestamp("2026-06-30"), observation_status="projection")])
    result = CrisisFeatureEngineer(output_dir=str(tmp_path)).extract_fsic_features(frame, as_of_date="2026-09-15")
    assert result.loc[0, "tier1_capital"] == 10.5859364868


def test_source_frame_is_not_mutated_and_codes_are_trimmed():
    frame = pd.DataFrame([observation(code=" FSI626_CFSI_PT ")])
    original = frame.copy(deep=True)
    assert select_fsic_features(frame).loc[0, "tier1_capital"] == 10.5859364868
    pd.testing.assert_frame_equal(frame, original)


def test_infinite_value_fails_closed():
    with pytest.raises(FSICSelectionError, match="non-finite"):
        select_fsic_features(pd.DataFrame([observation(value=np.inf)]))
