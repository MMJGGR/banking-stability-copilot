import pandas as pd

from src.utils import find_peers


def test_peer_selection_uses_scale_and_income_not_only_pillars():
    scores = pd.DataFrame(
        [
            {
                "country_code": "USA",
                "country_name": "United States",
                "risk_score": 6.9,
                "economic_pillar": 5.7,
                "industry_pillar": 2.2,
                "data_coverage": 0.95,
            },
            {
                "country_code": "SML",
                "country_name": "Small Similar Pillars",
                "risk_score": 6.9,
                "economic_pillar": 5.7,
                "industry_pillar": 2.2,
                "data_coverage": 0.95,
            },
            {
                "country_code": "GBR",
                "country_name": "United Kingdom",
                "risk_score": 6.2,
                "economic_pillar": 6.0,
                "industry_pillar": 2.0,
                "data_coverage": 0.95,
            },
            {
                "country_code": "FRA",
                "country_name": "France",
                "risk_score": 6.1,
                "economic_pillar": 6.1,
                "industry_pillar": 2.1,
                "data_coverage": 0.95,
            },
            {
                "country_code": "DEU",
                "country_name": "Germany",
                "risk_score": 5.5,
                "economic_pillar": 6.5,
                "industry_pillar": 1.9,
                "data_coverage": 0.95,
            },
            {
                "country_code": "JPN",
                "country_name": "Japan",
                "risk_score": 5.8,
                "economic_pillar": 6.3,
                "industry_pillar": 2.4,
                "data_coverage": 0.95,
            },
        ]
    )
    features = pd.DataFrame(
        [
            {"country_code": "USA", "nominal_gdp": 30_000_000, "gdp_per_capita": 90_000},
            {"country_code": "SML", "nominal_gdp": 2_000, "gdp_per_capita": 5_000},
            {"country_code": "GBR", "nominal_gdp": 4_000_000, "gdp_per_capita": 58_000},
            {"country_code": "FRA", "nominal_gdp": 3_400_000, "gdp_per_capita": 49_000},
            {"country_code": "DEU", "nominal_gdp": 5_000_000, "gdp_per_capita": 60_000},
            {"country_code": "JPN", "nominal_gdp": 4_200_000, "gdp_per_capita": 34_000},
        ]
    )

    peers = find_peers("USA", scores, n_peers=3, feature_values=features)

    assert "SML" not in peers["country_code"].tolist()
    assert set(peers["country_code"]).issubset({"GBR", "FRA", "DEU", "JPN"})
    assert "peer_basis" in peers.columns
