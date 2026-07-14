import pandas as pd

import src.liquidity_features as liquidity_features


def test_explicit_government_frame_is_used_for_candidate_assembly(monkeypatch):
    government = pd.DataFrame(
        {
            "country_code": ["KEN"],
            "govt_interest_to_revenue": [25.0],
            "govt_debt_to_revenue": [350.0],
            "govt_revenue_gdp": [20.0],
        }
    )
    monkeypatch.setattr(
        liquidity_features,
        "_government_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cached government data must not be reloaded")
        ),
    )
    monkeypatch.setattr(
        liquidity_features,
        "_external_features",
        lambda **_kwargs: pd.DataFrame(columns=["country_code"]),
    )

    assembled = liquidity_features.assemble_liquidity_features(
        as_of_date="2026-06-30",
        government_features=government,
    )

    assert assembled.to_dict("records") == [
        {
            "country_code": "KEN",
            "govt_interest_to_revenue": 25.0,
            "govt_debt_to_revenue": 350.0,
        }
    ]
