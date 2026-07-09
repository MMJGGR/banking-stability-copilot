import pandas as pd
import pytest

from src.data_loader import (
    FSIBSISLoader,
    IMFDataLoader,
    is_time_period_column,
    parse_period_label,
)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("2025", "2025-12-31"),
        ("2025-Q1", "2025-03-31"),
        ("Q2 2025", "2025-06-30"),
        ("2025-M11", "2025-11-30"),
        ("M02 2025", "2025-02-28"),
        ("2025-09", "2025-09-30"),
    ],
)
def test_parse_period_label(label, expected):
    assert parse_period_label(label) == pd.Timestamp(expected)
    assert is_time_period_column(label)


def test_unrecognized_period_is_nat():
    assert pd.isna(parse_period_label("not-a-period"))
    assert not is_time_period_column("not-a-period")


def test_vectorized_period_parser_handles_imf_month_format():
    loader = IMFDataLoader()
    parsed = loader._vectorized_parse_periods(
        pd.Series(["2025-Q3", "2025-M10", "2025-M11"])
    )

    assert parsed.tolist() == [
        pd.Timestamp("2025-09-30"),
        pd.Timestamp("2025-10-31"),
        pd.Timestamp("2025-11-30"),
    ]


def test_weo_loader_preserves_observation_status(tmp_path):
    source = tmp_path / "weo.csv"
    pd.DataFrame(
        [
            {
                "DATASET": "IMF.RES:WEO(9.0.0)",
                "SERIES_CODE": "TST.NGDP_RPCH.A",
                "COUNTRY": "Testland",
                "INDICATOR": "GDP growth",
                "FREQUENCY": "Annual",
                "UNIT": "Percent",
                "LATEST_ACTUAL_ANNUAL_DATA": 2024,
                "2024": 2.0,
                "2025": 2.5,
                "2026": 3.0,
            }
        ]
    ).to_csv(source, index=False)

    loaded = IMFDataLoader()._load_and_melt(str(source), "WEO")
    statuses = loaded.set_index(loaded["period"].dt.year)["observation_status"]

    assert statuses.loc[2024] == "actual"
    assert statuses.loc[2025] == "estimate"
    assert statuses.loc[2026] == "projection"


def test_fsibsis_prefers_latest_quarterly_observation():
    loader = FSIBSISLoader()
    loader.period_cols = ["2024", "2025-Q1", "2025-Q2"]
    loader.year_cols = ["2024"]
    loader.bank_data = pd.DataFrame(
        [
            {
                "country_code": "TST",
                "INDICATOR": "Total assets, Domestic currency",
                "2024": 100.0,
                "2025-Q1": 110.0,
                "2025-Q2": 120.0,
            }
        ]
    )

    value, period = loader.get_indicator_data("TST", "total_assets")

    assert value == 120.0
    assert period == "2025-Q2"
