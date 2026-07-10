import numpy as np
import pandas as pd

from src.country_names import country_name_from_code, fill_missing_country_names


def test_resolves_iso_codes():
    assert country_name_from_code("USA") == "United States"
    assert country_name_from_code("FIN") == "Finland"
    assert country_name_from_code(" gbr ") == "United Kingdom"


def test_resolves_imf_codes_without_iso_entry():
    assert country_name_from_code("KOS") == "Kosovo"
    assert country_name_from_code("WBG") == "West Bank and Gaza"


def test_unknown_code_behaviour():
    assert country_name_from_code("ZZZ") == ""
    assert country_name_from_code("ZZZ", fallback_to_code=True) == "ZZZ"
    assert country_name_from_code(None) == ""
    assert country_name_from_code("") == ""


def test_fill_missing_country_names_preserves_existing():
    df = pd.DataFrame(
        {
            "country_code": ["USA", "FIN", "KOS", "ZZZ"],
            "country_name": ["Custom States", "", np.nan, ""],
        }
    )
    fill_missing_country_names(df, fallback_to_code=True)
    assert df["country_name"].tolist() == [
        "Custom States",
        "Finland",
        "Kosovo",
        "ZZZ",
    ]


def test_fill_missing_country_names_creates_column():
    df = pd.DataFrame({"country_code": ["DNK"]})
    fill_missing_country_names(df)
    assert df["country_name"].tolist() == ["Denmark"]
