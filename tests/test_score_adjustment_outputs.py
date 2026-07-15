import numpy as np
import pandas as pd
import pytest

from src.pillar_pipeline import PillarInferencePipeline


def _feature_matrix(rows: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(41)
    return pd.DataFrame(
        {
            "country_code": [f"C{i:02d}" for i in range(rows)],
            "gdp_growth": rng.normal(3, 2, rows),
            "inflation": rng.normal(5, 3, rows),
            "gdp_per_capita": rng.lognormal(9, 0.8, rows),
            "capital_adequacy": rng.normal(15, 2, rows),
            "npl_ratio": rng.lognormal(1.4, 0.4, rows),
            "liquid_assets_st_liab": rng.normal(40, 8, rows),
            "loan_concentration": -rng.uniform(5, 70, rows),
        }
    )


def _fit_pipeline(**kwargs) -> tuple[PillarInferencePipeline, pd.DataFrame]:
    features = _feature_matrix()
    anchor = features.set_index("country_code")["gdp_per_capita"]
    return PillarInferencePipeline(**kwargs).fit(features, anchor), features


def test_penalty_only_country_is_not_mislabelled_as_floor_applied():
    pipeline, features = _fit_pipeline()
    safest_code = (
        pipeline.transform(features)
        .sort_values("pillar_risk_score")
        .iloc[0]["country_code"]
    )
    probe = features.loc[features["country_code"] == safest_code].copy()
    probe["npl_ratio"] = np.nan

    row = pipeline.transform(probe).iloc[0]

    assert row["critical_missing_fields"] == ("npl_ratio",)
    assert row["critical_penalty"] > 0
    assert row["risk_floor_value"] == pytest.approx(1.0)
    assert row["risk_floor_delta"] == pytest.approx(0.0)
    assert not bool(row["risk_floor_applied"])


def test_floor_and_missingness_outputs_reconcile_to_structural_score():
    # A low median makes the deliberately sparse row remain below the six-point
    # policy floor before the floor is applied, giving this test both a true
    # floor case and a penalty-only case in one batch.
    pipeline, features = _fit_pipeline(median_risk=1.0)
    complete = features.iloc[[0]].copy()

    penalty_only = features.iloc[[1]].copy()
    penalty_only["npl_ratio"] = np.nan

    sparse_floor = features.iloc[[2]].copy()
    keep = {"country_code", "gdp_growth", "capital_adequacy"}
    for column in sparse_floor.columns:
        if column not in keep:
            sparse_floor[column] = np.nan

    scores = pipeline.transform(
        pd.concat([complete, penalty_only, sparse_floor], ignore_index=True)
    ).set_index("country_code")

    np.testing.assert_allclose(
        scores["pillar_risk_score"] + scores["confidence_adjustment"],
        scores["confidence_adjusted_risk_score"],
    )
    np.testing.assert_allclose(
        scores["confidence_adjusted_risk_score"]
        + scores["risk_floor_delta"],
        scores["score_after_risk_floor"],
    )
    np.testing.assert_allclose(
        scores["score_after_risk_floor"]
        + scores["critical_penalty_applied"],
        scores["pre_round_structural_risk_score"],
    )
    np.testing.assert_allclose(
        scores["pre_round_structural_risk_score"].round(1),
        scores["structural_risk_score"],
    )
    np.testing.assert_allclose(
        scores["structural_risk_score"],
        scores["risk_score"],
    )

    assert (
        scores["risk_floor_applied"]
        == (scores["risk_floor_delta"] > 1e-12)
    ).all()
    assert (
        scores["critical_penalty_applied"]
        <= scores["critical_penalty"] + 1e-12
    ).all()

    sparse_row = scores.loc[sparse_floor.iloc[0]["country_code"]]
    assert bool(sparse_row["risk_floor_applied"])
    assert sparse_row["risk_floor_value"] == pytest.approx(6.0)
    assert sparse_row["risk_floor_delta"] > 0
    assert sparse_row["critical_missing_fields"] == (
        "npl_ratio",
        "liquid_assets_st_liab",
        "loan_concentration",
    )
    assert sparse_row["critical_missing_share"] == pytest.approx(0.75)
