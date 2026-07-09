from datetime import datetime, timezone

import pandas as pd

from train_model import BankingRiskModel


def test_model_metadata_round_trip(tmp_path):
    model = BankingRiskModel()
    model.trained = True
    model.training_date = datetime.now(timezone.utc).isoformat()
    model.countries_trained = 1
    model.country_scores = pd.DataFrame(
        [{"country_code": "TST", "risk_score": 5.0}]
    )
    model.feature_values = pd.DataFrame(
        [{"country_code": "TST", "gdp_growth": 2.0}]
    )
    model.pca_info = {
        "training_date": model.training_date,
        "economic_loadings": {},
        "industry_loadings": {},
    }

    artifact_path = tmp_path / "risk_model.pkl"
    model.save(str(artifact_path))
    loaded = BankingRiskModel.load(str(artifact_path))

    assert loaded.trained
    assert loaded.training_date == model.training_date
    assert loaded.pca_info["training_date"] == model.training_date
    assert loaded.get_score("TST")["risk_score"] == 5.0
