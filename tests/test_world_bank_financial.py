import pandas as pd

from src.world_bank_financial import (
    build_world_bank_financial_features,
    world_bank_feature_observations,
)


def test_world_bank_feature_builder_is_strictly_backward_looking():
    rows = []
    for year, value in zip(range(2000, 2012), range(40, 52)):
        rows.append(
            {
                "country_code": "AAA",
                "feature": "bank_credit_gdp",
                "year": year,
                "value": float(value),
            }
        )
    observations = pd.DataFrame(rows)

    baseline = build_world_bank_financial_features(observations)
    changed = observations.copy()
    changed.loc[changed["year"] == 2011, "value"] = 9999.0
    revised = build_world_bank_financial_features(changed)

    prior = baseline[baseline["year"] <= 2010].reset_index(drop=True)
    revised_prior = revised[revised["year"] <= 2010].reset_index(drop=True)
    pd.testing.assert_frame_equal(prior, revised_prior)
    row_2010 = baseline[baseline["year"] == 2010].iloc[0]
    assert row_2010["bank_credit_gdp_change_3y"] == 3.0
    assert pd.notna(row_2010["bank_credit_gdp_gap_10y"])


def test_bank_credit_falls_back_to_broader_private_credit():
    observations = pd.DataFrame(
        [
            {
                "country_code": "AAA",
                "feature": "private_credit_gdp_broad",
                "year": 2000,
                "value": 55.0,
            }
        ]
    )
    features = build_world_bank_financial_features(observations)
    assert features.loc[0, "bank_credit_gdp"] == 55.0


def test_feature_observations_mark_derived_values_as_not_direct():
    observations = pd.DataFrame(
        [
            {
                "country_code": "AAA",
                "feature": "bank_credit_gdp",
                "year": year,
                "value": float(value),
            }
            for year, value in zip(range(2000, 2012), range(40, 52))
        ]
    )
    long = world_bank_feature_observations(observations)
    raw = long[long["indicator_code"] == "bank_credit_gdp"]
    change = long[long["indicator_code"] == "bank_credit_gdp_change_3y"]
    assert raw["is_direct"].all()
    assert not change["is_direct"].any()


def test_commodity_derived_values_are_not_marked_as_direct():
    observations = pd.DataFrame(
        [
            {
                "country_code": "AAA",
                "feature": feature,
                "year": year,
                "value": value,
            }
            for year in range(2000, 2005)
            for feature, value in (
                ("fuel_exports_share", 30.0),
                ("terms_of_trade_index", 100.0 - 5.0 * (year - 2000)),
            )
        ]
    )
    long = world_bank_feature_observations(observations)
    derived = long[long["indicator_code"].isin(
        {
            "commodity_export_concentration",
            "terms_of_trade_deterioration_3y",
            "commodity_shock_exposure",
        }
    )]
    assert not derived.empty
    assert not derived["is_direct"].any()
