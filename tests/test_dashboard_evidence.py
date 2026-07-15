from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dashboard.evidence import (
    ACTIVE_MODEL_ROLE,
    FEATURE_EVIDENCE_METADATA,
    STATUS_IMPUTED,
    STATUS_REPORTED_DERIVED,
    STATUS_UNAVAILABLE,
    build_active_feature_registry,
    build_active_input_inventory,
    feature_metadata,
)


CURRENT_ACTIVE_FEATURES = {
    "gdp_growth",
    "inflation",
    "current_account_gdp",
    "gdp_per_capita",
    "govt_debt_gdp",
    "fiscal_balance_gdp",
    "unemployment",
    "credit_to_gdp_relative",
    "sovereign_liability_to_reserves",
    "inflation_differential_3yr",
    "interest_cost_gdp",
    "interest_cost_trend_3yr",
    "credit_growth_3yr",
    "m2_to_reserves",
    "ca_deficit_severity",
    "tot_deterioration_3yr",
    "voice_accountability",
    "political_stability",
    "govt_interest_to_revenue",
    "govt_debt_to_revenue",
    "net_iip_gdp",
    "external_liabilities_gdp",
    "reserves_to_goods_services_imports",
    "gross_external_financing_need_proxy_gdp",
    "investment_income_debits_to_cxr",
    "govt_revenue_gdp",
    "govt_interest_to_revenue_change_3y",
    "govt_debt_to_revenue_change_3y",
    "govt_revenue_gdp_change_3y",
    "capital_adequacy",
    "npl_ratio",
    "roa",
    "liquid_assets_st_liab",
    "liquid_assets_total",
    "customer_deposits_loans",
    "npl_provisions",
    "loan_concentration",
    "sovereign_exposure_ratio",
    "bank_liability_to_nfa",
    "years_since_banking_crisis",
}


def test_metadata_catalog_covers_all_40_current_active_features():
    assert len(CURRENT_ACTIVE_FEATURES) == 40
    assert CURRENT_ACTIVE_FEATURES <= FEATURE_EVIDENCE_METADATA.keys()

    for feature in CURRENT_ACTIVE_FEATURES:
        metadata = feature_metadata(feature)
        assert metadata.label
        assert metadata.unit != "Source-defined"
        assert metadata.source_family != "Unmapped source family"


def test_loading_maps_alone_determine_membership_and_pillar_role():
    registry = build_active_feature_registry(
        {
            # ROA is normally a banking-system metric. Deliberately place it in
            # the economic map to prove metadata does not assign the pillar.
            "economic_loadings": {"roa": 0.25},
            "industry_loadings": {"future_signal": "0.75"},
        }
    ).set_index("feature")

    assert registry.index.tolist() == ["roa", "future_signal"]
    assert registry.loc["roa", "pillar"] == "economic"
    assert registry.loc["roa", "pillar_label"] == "Operating environment"
    assert registry.loc["future_signal", "pillar"] == "industry"
    assert registry.loc["future_signal", "pillar_label"] == "Banking system"
    assert registry.loc["future_signal", "label"] == "Future Signal"
    assert registry.loc["future_signal", "unit"] == "Source-defined"
    assert (
        registry.loc["future_signal", "source_family"]
        == "Unmapped source family"
    )
    assert (registry["model_role"] == ACTIVE_MODEL_ROLE).all()
    assert registry.loc["future_signal", "loading"] == pytest.approx(0.75)


def test_inventory_distinguishes_direct_imputed_and_unavailable_inputs():
    pca_info = {
        "economic_loadings": {"gdp_growth": 0.5, "inflation": -0.2},
        "industry_loadings": {"npl_ratio": 0.3},
    }
    raw = pd.DataFrame(
        {
            "country_code": ["usa"],
            "gdp_growth": [3.0],
            "gdp_growth_year": [2025],
            "inflation": [np.nan],
            "inflation_year": [2024],
            "npl_ratio": [np.nan],
        }
    )
    imputed = pd.DataFrame(
        {
            "country_code": ["USA"],
            "gdp_growth": [99.0],
            "inflation": [4.5],
            "npl_ratio": [np.nan],
        }
    )

    inventory = build_active_input_inventory(
        "UsA", raw, pca_info, imputed_features=imputed
    )
    rows = inventory.rows.set_index("feature")

    assert inventory.country_code == "USA"
    assert inventory.coverage.numerator == 1
    assert inventory.coverage.denominator == 3
    assert inventory.coverage.ratio == pytest.approx(1 / 3)

    assert rows.loc["gdp_growth", "status"] == STATUS_REPORTED_DERIVED
    assert rows.loc["gdp_growth", "value"] == pytest.approx(3.0)
    assert rows.loc["gdp_growth", "imputed_value"] == pytest.approx(99.0)
    assert rows.loc["gdp_growth", "period"] == 2025
    assert bool(rows.loc["gdp_growth", "is_direct"]) is True

    assert rows.loc["inflation", "status"] == STATUS_IMPUTED
    assert rows.loc["inflation", "value"] == pytest.approx(4.5)
    assert rows.loc["inflation", "period"] == 2024
    assert bool(rows.loc["inflation", "is_direct"]) is False

    assert rows.loc["npl_ratio", "status"] == STATUS_UNAVAILABLE
    assert np.isnan(rows.loc["npl_ratio", "value"])
    assert rows.loc["npl_ratio", "period"] is None
    assert bool(rows.loc["npl_ratio", "is_direct"]) is False


def test_contextual_columns_are_not_misrepresented_as_active_inputs():
    raw = pd.DataFrame(
        {
            "country_code": ["USA"],
            "gdp_growth": [2.5],
            "roe": [12.0],
            "credit_rating": ["AA+"],
        }
    )
    inventory = build_active_input_inventory(
        "USA",
        raw,
        {
            "economic_loadings": {"gdp_growth": 1.0},
            "industry_loadings": {},
        },
    )

    assert inventory.rows["feature"].tolist() == ["gdp_growth"]
    assert "roe" not in inventory.rows["feature"].tolist()
    assert "credit_rating" not in inventory.rows["feature"].tolist()


def test_missing_country_returns_complete_unavailable_active_universe():
    inventory = build_active_input_inventory(
        "KEN",
        pd.DataFrame({"country_code": ["USA"], "gdp_growth": [2.0]}),
        {
            "economic_loadings": {"gdp_growth": 0.5},
            "industry_loadings": {"npl_ratio": 0.5},
        },
    )

    assert inventory.rows["feature"].tolist() == ["gdp_growth", "npl_ratio"]
    assert inventory.rows["status"].tolist() == [
        STATUS_UNAVAILABLE,
        STATUS_UNAVAILABLE,
    ]
    assert inventory.coverage.numerator == 0
    assert inventory.coverage.denominator == 2
    assert inventory.coverage.ratio == 0.0


def test_index_based_feature_frames_and_imputed_periods_are_supported():
    raw = pd.DataFrame({"gdp_growth": [np.nan]}, index=pd.Index(["ken"]))
    imputed = pd.DataFrame(
        {"gdp_growth": [4.0], "gdp_growth_year": ["2023Q4"]},
        index=pd.Index(["KEN"]),
    )

    inventory = build_active_input_inventory(
        "KEN",
        raw,
        {
            "economic_loadings": {"gdp_growth": 1.0},
            "industry_loadings": {},
        },
        imputed_features=imputed,
    )

    row = inventory.rows.iloc[0]
    assert row["status"] == STATUS_IMPUTED
    assert row["value"] == pytest.approx(4.0)
    assert row["period"] == "2023Q4"


def test_empty_loading_maps_have_explicit_zero_coverage_semantics():
    inventory = build_active_input_inventory(
        "USA",
        pd.DataFrame(),
        {"economic_loadings": {}, "industry_loadings": {}},
    )

    assert inventory.rows.empty
    assert inventory.coverage.numerator == 0
    assert inventory.coverage.denominator == 0
    assert inventory.coverage.ratio is None


def test_registry_rejects_duplicate_cross_pillar_membership():
    with pytest.raises(ValueError, match="both pillar loading maps"):
        build_active_feature_registry(
            {
                "economic_loadings": {"gdp_growth": 1.0},
                "industry_loadings": {"gdp_growth": 0.5},
            }
        )


def test_duplicate_country_rows_are_rejected():
    raw = pd.DataFrame(
        {"country_code": ["USA", "usa"], "gdp_growth": [2.0, 3.0]}
    )

    with pytest.raises(ValueError, match="duplicate rows for USA"):
        build_active_input_inventory(
            "USA",
            raw,
            {
                "economic_loadings": {"gdp_growth": 1.0},
                "industry_loadings": {},
            },
        )


@pytest.mark.parametrize(
    ("pca_info", "message"),
    [
        ([], "pca_info must be a mapping"),
        (
            {"economic_loadings": [], "industry_loadings": {}},
            "economic_loadings",
        ),
    ],
)
def test_registry_rejects_invalid_loading_map_shapes(pca_info, message):
    with pytest.raises(TypeError, match=message):
        build_active_feature_registry(pca_info)


def test_inventory_rejects_non_dataframe_feature_frames():
    pca_info = {
        "economic_loadings": {"gdp_growth": 1.0},
        "industry_loadings": {},
    }

    with pytest.raises(TypeError, match="model_features"):
        build_active_input_inventory("USA", {"gdp_growth": 2.0}, pca_info)
    with pytest.raises(TypeError, match="imputed_features"):
        build_active_input_inventory(
            "USA", pd.DataFrame(), pca_info, imputed_features={}
        )
