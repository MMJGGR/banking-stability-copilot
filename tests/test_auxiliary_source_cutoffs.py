import pandas as pd

from src.data_loader import FSIBSISLoader, WGILoader


def test_wgi_latest_scores_respects_as_of_year():
    loader = WGILoader.__new__(WGILoader)
    loader.data = pd.DataFrame(
        {
            "country_code": ["AAA", "AAA", "BBB"],
            "year": [2024, 2025, 2025],
            "govt_effectiveness": [40.0, 55.0, 70.0],
        }
    )

    latest = loader.get_latest_scores(as_of_date="2024-12-31")

    assert latest.loc[latest["country_code"] == "AAA", "govt_effectiveness"].item() == 40.0
    assert "BBB" not in latest["country_code"].values


def test_fsibsis_extract_features_respects_as_of_period():
    loader = FSIBSISLoader.__new__(FSIBSISLoader)
    loader.period_cols = ["2025-M06", "2025-M11"]
    loader.year_cols = []
    loader.bank_data = pd.DataFrame(
        {
            "country_code": ["AAA", "AAA", "AAA"],
            "INDICATOR": [
                "Interest income, Domestic currency",
                "Interest expenses, Domestic currency",
                "Total assets, Assets, Domestic currency",
            ],
            "2025-M06": [10.0, 4.0, 100.0],
            "2025-M11": [30.0, 5.0, 100.0],
        }
    )

    features = loader.extract_features(as_of_date="2025-06-30")

    row = features.set_index("country_code").loc["AAA"]
    assert row["net_interest_margin"] == 6.0
    assert row["net_interest_margin_year"] == "2025-M06"
