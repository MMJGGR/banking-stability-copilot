import pickle

import numpy as np
import pandas as pd
import pytest

from src.pillar_pipeline import PillarInferencePipeline
from src.scripts.audit_model_policy import build_policy_audit
from train_model import BankingRiskModel


def _feature_matrix(rows=20):
    rng = np.random.default_rng(19)
    frame = pd.DataFrame(
        {
            "country_code": [f"C{i:02d}" for i in range(rows)],
            "gdp_growth": rng.normal(3, 2, rows),
            "inflation": rng.normal(5, 4, rows),
            "gdp_per_capita": rng.lognormal(9, 1, rows),
            "capital_adequacy": rng.normal(15, 3, rows),
            "npl_ratio": rng.lognormal(1.5, 0.5, rows),
            "loan_concentration": -rng.uniform(5, 80, rows),
        }
    )
    frame.loc[::4, "inflation"] = np.nan
    frame.loc[::5, "npl_ratio"] = np.nan
    return frame


def test_pillar_pipeline_is_batch_invariant_and_pickleable():
    features = _feature_matrix()
    anchor = features.set_index("country_code")["gdp_per_capita"]
    pipeline = PillarInferencePipeline().fit(features, anchor)

    target = features.iloc[[0]].copy()
    target_score = pipeline.transform(target).iloc[0]["risk_score"]
    mixed_score = pipeline.transform(
        pd.concat([target, features.iloc[[1]]], ignore_index=True)
    )
    mixed_target_score = mixed_score.loc[
        mixed_score["country_code"] == target.iloc[0]["country_code"],
        "risk_score",
    ].iloc[0]
    assert mixed_target_score == pytest.approx(target_score)

    restored = pickle.loads(pickle.dumps(pipeline))
    restored_score = restored.transform(target).iloc[0]["risk_score"]
    assert restored_score == pytest.approx(target_score)

    without_one_source_feature = target.drop(columns=["npl_ratio"])
    fallback_score = restored.transform(without_one_source_feature)
    assert len(fallback_score) == 1
    assert fallback_score.iloc[0]["industry_coverage"] < 1


def test_banking_model_persists_pipeline_as_sidecar(tmp_path):
    features = _feature_matrix()
    model = BankingRiskModel()
    model.trained = True
    model.training_date = "2026-07-09T00:00:00"
    model.countries_trained = len(features)
    model.country_scores = pd.DataFrame(
        {"country_code": features["country_code"], "risk_score": 5.0}
    )
    model.pca_info = {"snapshot_date": "2025-12-31"}
    model.pillar_pipeline = PillarInferencePipeline().fit(
        features,
        features.set_index("country_code")["gdp_per_capita"],
    )

    artifact_path = tmp_path / "risk_model.pkl"
    model.save(str(artifact_path))

    sidecar = tmp_path / "risk_model_inference_pipeline.pkl"
    assert sidecar.exists()
    loaded = BankingRiskModel.load(str(artifact_path))
    assert loaded.pillar_pipeline is not None
    assert loaded.pillar_pipeline.fitted_


def test_policy_audit_reports_required_scenarios():
    audit = build_policy_audit(_feature_matrix())

    assert audit["baseline"]["countries"] == 20
    assert set(audit["scenarios"]) == {
        "no_confidence_regression",
        "no_risk_floors",
        "no_gdp_pca_input",
        "no_gdp_orientation",
    }
    assert audit["crisis_labels"]["source_coverage_end_year"] == 2025
