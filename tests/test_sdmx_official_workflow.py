import pandas as pd

from src.sources.sdmx import _filter_csv_by_period
from src.sources.sdmx_normalize import _fsibsis_indicator_name, _fsic_indicator_name


def test_filter_csv_by_period_is_client_side(tmp_path):
    source = tmp_path / "raw.csv"
    output = tmp_path / "filtered.csv"
    pd.DataFrame(
        {
            "TIME_PERIOD": ["2024", "2025", "2026-Q1", "2026-M07"],
            "OBS_VALUE": [1, 2, 3, 4],
        }
    ).to_csv(source, index=False)

    _filter_csv_by_period(source, output, start_period="2025", end_period="2026-Q1")

    result = pd.read_csv(output)
    assert result["TIME_PERIOD"].tolist() == ["2025", "2026-Q1"]


def test_fsic_indicator_name_reconstructs_feature_pattern(monkeypatch):
    def fake_codelist(agency, codelist_id):
        if codelist_id == "CL_FSI":
            return {"FSI20": "Regulatory capital to risk-weighted assets"}
        if codelist_id == "CL_FSI_STO":
            return {}
        return {}

    monkeypatch.setattr("src.sources.sdmx_normalize.fetch_codelist", fake_codelist)
    df = pd.DataFrame(
        {
            "FSI": ["FSI20"],
            "FSI_STO": [pd.NA],
            "FSI_COMPONENT": ["CFSI"],
            "UNIT": ["PT"],
            "INDICATOR": ["FSI20_CFSI_PT"],
        }
    )

    assert _fsic_indicator_name(df).iloc[0] == (
        "Regulatory capital to risk-weighted assets, (Core FSI), Percent"
    )


def test_fsibsis_indicator_name_reconstructs_balance_sheet_pattern(monkeypatch):
    def fake_codelist(agency, codelist_id):
        return {
            "NINTBKF4": "Noninterbank loans",
            "TA": "Total Assets",
        }

    monkeypatch.setattr("src.sources.sdmx_normalize.fetch_codelist", fake_codelist)
    df = pd.DataFrame({"INDICATOR": ["NINTBKF4_S13_A_XDC", "TA_A_XDC"]})

    assert _fsibsis_indicator_name(df).tolist() == [
        "Noninterbank loans, General government, Assets, Domestic currency",
        "Total Assets, Assets, Domestic currency",
    ]
