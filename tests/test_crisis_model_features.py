import numpy as np
import pandas as pd

from src.crisis_model_features import (
    FEATURE_RISK_DIRECTIONS,
    derive_crisis_model_features,
    model_matrix,
)


class Labels:
    crises = {"AAA": [(2001, 2002)]}


def _row(year, **values):
    row = {
        "country_code": "AAA",
        "forecast_origin_year": year,
        "crisis_target": 0,
    }
    for feature, value in values.items():
        row[feature] = value
        row[f"{feature}__available"] = pd.notna(value)
    return row


def test_calendar_lags_do_not_bridge_excluded_years():
    panel = pd.DataFrame(
        [
            _row(2000, inflation=1.0, gdp_growth=2.0),
            _row(2002, inflation=100.0, gdp_growth=3.0),
            _row(2003, inflation=4.0, gdp_growth=4.0),
        ]
    )
    features = derive_crisis_model_features(panel, Labels()).set_index(
        "forecast_origin_year"
    )
    assert features.loc[2003, "inflation_change_3y"] == 3.0
    assert np.isnan(features.loc[2002, "inflation_change_3y"])


def test_risk_orientation_respects_every_declared_direction():
    frame = pd.DataFrame(
        {
            "gdp_growth_3y_avg": [2.0],
            "inflation": [5.0],
        }
    )
    matrix = model_matrix(frame, frame.columns)
    assert matrix.loc[0, "gdp_growth_3y_avg"] == -2.0
    assert matrix.loc[0, "inflation"] == 5.0
    assert set(frame.columns) <= set(FEATURE_RISK_DIRECTIONS)
