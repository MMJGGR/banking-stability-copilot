import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pillar_pipeline import (
    CRITICAL_FEATURES,
    FEATURE_RISK_DIRECTIONS,
    PillarInferencePipeline,
)

CHALLENGER_BUNDLE = Path("artifacts/snapshots/2026-06-30-challenger-directional")


def _feature_matrix(rows=30, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "country_code": [f"C{i:02d}" for i in range(rows)],
            "gdp_growth": rng.normal(3, 2, rows),
            "inflation": rng.normal(5, 4, rows),
            "gdp_per_capita": rng.lognormal(9, 1, rows),
            "capital_adequacy": rng.normal(15, 3, rows),
            "npl_ratio": rng.lognormal(1.5, 0.5, rows),
            "liquid_assets_st_liab": rng.normal(40, 10, rows),
            "loan_concentration": -rng.uniform(5, 80, rows),
        }
    )


def test_every_pillar_feature_declares_a_risk_direction():
    from src.pillar_pipeline import ECONOMIC_FEATURES, INDUSTRY_FEATURES

    for feature in ECONOMIC_FEATURES + INDUSTRY_FEATURES:
        assert feature in FEATURE_RISK_DIRECTIONS, feature
        assert FEATURE_RISK_DIRECTIONS[feature] in (1.0, -1.0)


def test_higher_npl_ratio_cannot_lower_the_risk_score():
    features = _feature_matrix()
    anchor = features.set_index("country_code")["gdp_per_capita"]
    pipeline = PillarInferencePipeline(apply_risk_floors=False).fit(
        features, anchor
    )

    target = features.iloc[[0]].copy()
    low = target.copy()
    low["npl_ratio"] = features["npl_ratio"].quantile(0.1)
    high = target.copy()
    high["npl_ratio"] = features["npl_ratio"].quantile(0.9)
    low_score = pipeline.transform(low).iloc[0]["risk_score"]
    high_score = pipeline.transform(high).iloc[0]["risk_score"]
    assert high_score >= low_score


def test_higher_capital_adequacy_cannot_raise_the_risk_score():
    features = _feature_matrix()
    anchor = features.set_index("country_code")["gdp_per_capita"]
    pipeline = PillarInferencePipeline(apply_risk_floors=False).fit(
        features, anchor
    )

    target = features.iloc[[0]].copy()
    weak = target.copy()
    weak["capital_adequacy"] = features["capital_adequacy"].quantile(0.1)
    strong = target.copy()
    strong["capital_adequacy"] = features["capital_adequacy"].quantile(0.9)
    weak_score = pipeline.transform(weak).iloc[0]["risk_score"]
    strong_score = pipeline.transform(strong).iloc[0]["risk_score"]
    assert strong_score <= weak_score


def test_missing_critical_fields_incur_a_penalty():
    features = _feature_matrix()
    anchor = features.set_index("country_code")["gdp_per_capita"]
    pipeline = PillarInferencePipeline().fit(features, anchor)

    observed = features.iloc[[0]].copy()
    sparse = observed.copy()
    for column in ("npl_ratio", "capital_adequacy", "liquid_assets_st_liab"):
        sparse[column] = np.nan

    observed_row = pipeline.transform(observed).iloc[0]
    sparse_row = pipeline.transform(sparse).iloc[0]
    assert observed_row["critical_penalty"] == 0
    assert sparse_row["critical_penalty"] > 0
    assert sparse_row["critical_missing_share"] > 0


def test_signed_loadings_match_declared_directions():
    features = _feature_matrix()
    anchor = features.set_index("country_code")["gdp_per_capita"]
    pipeline = PillarInferencePipeline().fit(features, anchor)
    loadings = pipeline.loadings()
    for pillar in ("economic_loadings", "industry_loadings"):
        for feature, loading in loadings[pillar].items():
            expected = FEATURE_RISK_DIRECTIONS[feature]
            assert np.sign(loading) == expected, (feature, loading)


@pytest.mark.skipif(
    not (CHALLENGER_BUNDLE / "risk_model.pkl").exists(),
    reason="challenger bundle not present",
)
def test_challenger_acceptance_kenya_mozambique_and_no_derisking():
    """Backlog item 1 acceptance case: the challenger must not score Kenya
    riskier than Mozambique, and the classifier overlay must never de-risk."""
    with (CHALLENGER_BUNDLE / "risk_model.pkl").open("rb") as handle:
        scores = pickle.load(handle)["country_scores"]
    indexed = scores.set_index("country_code")

    assert indexed.loc["MOZ", "risk_score"] >= indexed.loc["KEN", "risk_score"]
    assert (scores["crisis_uplift"].fillna(0) >= 0).all()
    assert (scores["critical_penalty"].fillna(0) >= 0).all()
    assert scores["risk_score"].between(1, 10).all()


@pytest.mark.skipif(
    not (CHALLENGER_BUNDLE / "inference_pipeline.pkl").exists(),
    reason="challenger bundle not present",
)
def test_challenger_pipeline_loadings_are_directionally_constrained():
    with (CHALLENGER_BUNDLE / "inference_pipeline.pkl").open("rb") as handle:
        pipeline = pickle.load(handle)["pillar_pipeline"]
    loadings = pipeline.loadings()
    for pillar in ("economic_loadings", "industry_loadings"):
        assert loadings[pillar]
        for feature, loading in loadings[pillar].items():
            assert np.sign(loading) == FEATURE_RISK_DIRECTIONS[feature]
