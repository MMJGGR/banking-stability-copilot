import pandas as pd
import pytest

import src.government_liquidity as government_liquidity
from src.government_liquidity import (
    build_government_liquidity_features,
    latest_fiscal_trend_features,
    latest_fiscal_matrix,
    load_weo_fiscal_observations,
    refresh_government_liquidity_outputs,
)


def _weo_row(country, code, year, value, status="actual"):
    return {
        "country_code": country,
        "indicator_code": code,
        "period": pd.Timestamp(f"{year}-12-31"),
        "value": value,
        "observation_status": status,
        "dataset": "WEO",
    }


def test_load_weo_fiscal_observations_respects_cutoff_and_status():
    weo = pd.DataFrame(
        [
            _weo_row("KEN", "GGR_NGDP", 2025, 17.0, "actual"),
            _weo_row("KEN", "GGR_NGDP", 2026, 18.0, "projection"),  # excluded by status
            _weo_row("KEN", "GGR_NGDP", 2028, 19.0, "projection"),  # excluded by cutoff+status
            _weo_row("KEN", "PCPIPCH", 2025, 5.0, "actual"),  # non-fiscal, excluded
            _weo_row("G20", "GGR_NGDP", 2025, 20.0, "actual"),  # not ISO3, excluded
        ]
    )
    obs = load_weo_fiscal_observations(weo, as_of_date="2026-06-30")

    assert set(obs["country_code"]) == {"KEN"}
    assert set(obs["feature_key"]) == {"revenue_gdp"}
    # Only the 2025 actual survives the cutoff + status filter.
    assert len(obs) == 1
    assert obs.iloc[0]["value"] == pytest.approx(17.0)
    assert obs.iloc[0]["source"] == "WEO"


def test_load_weo_fiscal_observations_can_include_projections():
    weo = pd.DataFrame(
        [
            _weo_row("KEN", "GGR_NGDP", 2025, 17.0, "actual"),
            _weo_row("KEN", "GGR_NGDP", 2026, 18.0, "projection"),
        ]
    )
    obs = load_weo_fiscal_observations(
        weo, as_of_date="2026-12-31", include_projections=True
    )
    assert len(obs) == 2


def test_latest_fiscal_matrix_takes_latest_period():
    obs = pd.DataFrame(
        [
            {"country_code": "KEN", "feature_key": "revenue_gdp", "period": pd.Timestamp("2024-12-31"), "value": 16.0},
            {"country_code": "KEN", "feature_key": "revenue_gdp", "period": pd.Timestamp("2025-12-31"), "value": 17.5},
        ]
    )
    matrix = latest_fiscal_matrix(obs)
    row = matrix.set_index("country_code").loc["KEN"]
    assert row["revenue_gdp"] == pytest.approx(17.5)


def test_build_government_liquidity_features_computes_affordability_ratios():
    observations = pd.DataFrame(
        [
            ("KEN", "gross_debt_gdp", 70.0),
            ("KEN", "fiscal_balance_gdp", -6.0),
            ("KEN", "primary_balance_gdp", -1.0),
            ("KEN", "revenue_gdp", 20.0),
            ("KEN", "expenditure_gdp", 26.0),
        ],
        columns=["country_code", "feature_key", "value"],
    )
    observations["period"] = pd.Timestamp("2025-12-31")
    model_features = pd.DataFrame(
        {"country_code": ["KEN", "MOZ"], "nominal_gdp": [100.0, 200.0]}
    )

    features, report = build_government_liquidity_features(observations, model_features)
    ken = features.set_index("country_code").loc["KEN"]

    # interest = primary - overall = -1 - (-6) = 5 % of GDP
    assert ken["govt_interest_gdp"] == pytest.approx(5.0)
    # interest / revenue = 5 / 20 * 100 = 25 %
    assert ken["govt_interest_to_revenue"] == pytest.approx(25.0)
    # debt / revenue = 70 / 20 * 100 = 350 %
    assert ken["govt_debt_to_revenue"] == pytest.approx(350.0)
    # overall deficit floored at zero = 6
    assert ken["govt_overall_deficit_gdp"] == pytest.approx(6.0)
    # primary deficit floored at zero = 1
    assert ken["govt_primary_deficit_gdp"] == pytest.approx(1.0)

    # MOZ has no observations -> ratios missing, deficits floored to 0 via NaN<0 False
    moz = features.set_index("country_code").loc["MOZ"]
    assert pd.isna(moz["govt_interest_to_revenue"])

    assert report["feature_coverage"]["govt_interest_to_revenue"]["countries"] == 1
    assert report["feature_coverage"]["govt_interest_to_revenue"]["pct_model_countries"] == 50.0


def test_surplus_country_has_zero_floored_deficits():
    observations = pd.DataFrame(
        [
            ("NOR", "fiscal_balance_gdp", 8.0),
            ("NOR", "primary_balance_gdp", 7.0),
            ("NOR", "revenue_gdp", 55.0),
            ("NOR", "gross_debt_gdp", 40.0),
        ],
        columns=["country_code", "feature_key", "value"],
    )
    observations["period"] = pd.Timestamp("2025-12-31")
    model_features = pd.DataFrame({"country_code": ["NOR"], "nominal_gdp": [500.0]})

    features, _ = build_government_liquidity_features(observations, model_features)
    nor = features.set_index("country_code").loc["NOR"]
    assert nor["govt_overall_deficit_gdp"] == pytest.approx(0.0)
    assert nor["govt_primary_deficit_gdp"] == pytest.approx(0.0)
    # interest = 7 - 8 = -1 (implied net interest income); ratio still computed
    assert nor["govt_interest_gdp"] == pytest.approx(-1.0)


def test_latest_fiscal_trend_features_compute_three_year_changes():
    rows = []
    for year, debt, balance, primary, revenue in [
        (2021, 60.0, -4.0, -1.0, 20.0),
        (2022, 62.0, -4.5, -1.5, 21.0),
        (2023, 65.0, -5.0, -2.0, 22.0),
        (2024, 68.0, -5.5, -2.5, 22.5),
        (2025, 72.0, -7.0, -3.0, 24.0),
    ]:
        rows.extend(
            [
                ("KEN", "gross_debt_gdp", year, debt),
                ("KEN", "fiscal_balance_gdp", year, balance),
                ("KEN", "primary_balance_gdp", year, primary),
                ("KEN", "revenue_gdp", year, revenue),
            ]
        )
    observations = pd.DataFrame(
        rows,
        columns=["country_code", "feature_key", "year", "value"],
    )
    observations["period"] = pd.to_datetime(observations["year"].astype(str) + "-12-31")

    trends = latest_fiscal_trend_features(observations).set_index("country_code")
    ken = trends.loc["KEN"]

    # Latest 2025 values are compared with the nearest observation at least
    # three years earlier: 2022.
    interest_2025 = (-3.0) - (-7.0)
    interest_2022 = (-1.5) - (-4.5)
    assert ken["govt_interest_to_revenue_change_3y"] == pytest.approx(
        interest_2025 / 24.0 * 100 - interest_2022 / 21.0 * 100
    )
    assert ken["govt_debt_to_revenue_change_3y"] == pytest.approx(
        72.0 / 24.0 * 100 - 62.0 / 21.0 * 100
    )
    assert ken["govt_primary_deficit_gdp_change_3y"] == pytest.approx(3.0 - 1.5)
    assert ken["govt_revenue_gdp_change_3y"] == pytest.approx(24.0 - 21.0)


def test_refresh_outputs_use_passed_weo_snapshot_and_package_references(tmp_path):
    weo = pd.DataFrame(
        [
            _weo_row("KEN", "GGXWDG_NGDP", 2024, 60.0),
            _weo_row("KEN", "GGXWDG_NGDP", 2025, 70.0),
            _weo_row("KEN", "GGXCNL_NGDP", 2025, -6.0),
            _weo_row("KEN", "GGXONLB_NGDP", 2025, -1.0),
            _weo_row("KEN", "GGR_NGDP", 2025, 20.0),
        ]
    )
    model_features = pd.DataFrame({"country_code": ["KEN"]})
    cache_dir = tmp_path / "cache" / "government"
    artifact_dir = tmp_path / "artifacts"
    reference_dir = tmp_path / "data" / "reference"

    observations, features, report = refresh_government_liquidity_outputs(
        weo_df=weo,
        as_of_date="2025-12-31",
        model_countries=["KEN"],
        model_features=model_features,
        observations_path=cache_dir / "government_liquidity_observations.parquet",
        features_path=cache_dir / "government_liquidity_features.parquet",
        report_path=artifact_dir / "government_liquidity_features_report.json",
        reference_dir=reference_dir,
    )

    assert observations["period"].max() == pd.Timestamp("2025-12-31")
    assert features.set_index("country_code").loc["KEN", "govt_debt_to_revenue"] == pytest.approx(350.0)
    assert report["observation_countries"] == 1
    assert (cache_dir / "government_liquidity_observations.parquet").stat().st_size > 0
    assert (cache_dir / "government_liquidity_features.parquet").stat().st_size > 0
    assert (artifact_dir / "government_liquidity_features_report.json").stat().st_size > 0
    assert (reference_dir / "government_liquidity_observations.parquet").stat().st_size > 0
    assert (reference_dir / "government_liquidity_features.parquet").stat().st_size > 0
    assert (reference_dir / "government_liquidity_features_report.json").stat().st_size > 0

    packaged = pd.read_parquet(reference_dir / "government_liquidity_features.parquet")
    assert packaged.columns.tolist() == features.columns.tolist()
    assert packaged["country_code"].tolist() == features["country_code"].tolist()
    assert packaged.set_index("country_code").loc[
        "KEN", "govt_debt_to_revenue"
    ] == pytest.approx(features.set_index("country_code").loc["KEN", "govt_debt_to_revenue"])
    assert b"\r\n" not in (
        artifact_dir / "government_liquidity_features_report.json"
    ).read_bytes()
    assert b"\r\n" not in (
        reference_dir / "government_liquidity_features_report.json"
    ).read_bytes()


def test_explicit_candidate_universe_is_authoritative(monkeypatch):
    observations = pd.DataFrame(
        [
            ("KEN", "gross_debt_gdp", 70.0),
            ("KEN", "revenue_gdp", 20.0),
            ("UGA", "gross_debt_gdp", 50.0),
            ("UGA", "revenue_gdp", 25.0),
        ],
        columns=["country_code", "feature_key", "value"],
    )
    observations["period"] = pd.Timestamp("2025-12-31")
    monkeypatch.setattr(
        government_liquidity,
        "load_model_artifact",
        lambda: (_ for _ in ()).throw(AssertionError("active model must not be loaded")),
    )

    features, report = build_government_liquidity_features(
        observations,
        model_features=pd.DataFrame({"country_code": ["KEN"]}),
    )

    assert features["country_code"].tolist() == ["KEN"]
    assert report["model_countries"] == 1
    assert report["feature_coverage"]["govt_debt_to_revenue"] == {
        "countries": 1,
        "pct_model_countries": 100.0,
    }


def test_final_package_filters_exact_observations_to_candidate_universe(tmp_path):
    observations = pd.DataFrame(
        [
            {
                "source": "WEO",
                "feature_key": "revenue_gdp",
                "feature_label": "Revenue",
                "quality": "observed",
                "dataset_version": "WEO-2026",
                "country_code": country,
                "period_label": "2025",
                "period": pd.Timestamp("2025-12-31"),
                "observation_status": "actual",
                "value": value,
            }
            for country, value in (("KEN", 20.0), ("UGA", 25.0))
        ]
    )
    reference_dir = tmp_path / "reference"

    packaged_observations, features, report = refresh_government_liquidity_outputs(
        fiscal_observations=observations,
        model_countries=["KEN"],
        model_features=pd.DataFrame({"country_code": ["KEN"]}),
        observations_path=tmp_path / "cache" / "observations.parquet",
        features_path=tmp_path / "cache" / "features.parquet",
        report_path=tmp_path / "artifacts" / "report.json",
        reference_dir=reference_dir,
    )

    assert packaged_observations["country_code"].unique().tolist() == ["KEN"]
    assert features["country_code"].tolist() == ["KEN"]
    assert report["model_countries"] == 1
    assert report["observation_countries"] == 1
    reference_observations = pd.read_parquet(
        reference_dir / "government_liquidity_observations.parquet"
    )
    assert reference_observations["country_code"].unique().tolist() == ["KEN"]
