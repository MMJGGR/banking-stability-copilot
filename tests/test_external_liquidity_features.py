import pandas as pd
import pytest

from src.external_liquidity import (
    ExternalSeriesSpec,
    WorldBankSeriesSpec,
    build_external_liquidity_features,
    normalize_feature_csv,
    normalize_world_bank_records,
    reer_appreciation_gap,
)


def test_normalize_feature_csv_keeps_iso3_observations(tmp_path):
    csv = (
        "STRUCTURE_ID,COUNTRY,TIME_PERIOD,OBS_VALUE\n"
        "IMF.STA:BOP(21.0.0),KEN,2025,100\n"
        "IMF.STA:BOP(21.0.0),G001,2025,200\n"
        "IMF.STA:BOP(21.0.0),USA,bad,300\n"
    ).encode()
    spec = ExternalSeriesSpec(
        "current_account_receipts_usd",
        "BOP",
        "IMF.STA",
        "BOP",
        "{countries}.CD_T.TCDCA.USD.A",
        "Current account receipts",
    )
    normalized = normalize_feature_csv(csv, spec)
    assert len(normalized) == 1
    assert normalized.iloc[0]["country_code"] == "KEN"
    assert normalized.iloc[0]["feature_key"] == "current_account_receipts_usd"
    assert normalized.iloc[0]["period"].year == 2025


def test_build_external_liquidity_features_computes_core_ratios():
    observations = pd.DataFrame(
        [
            ("KEN", "current_account_receipts_usd", "2025", 50.0),
            ("KEN", "current_account_payments_usd", "2025", 80.0),
            ("KEN", "current_account_balance_usd", "2025", -30.0),
            ("KEN", "goods_services_imports_usd", "2025", 40.0),
            ("KEN", "reserve_assets_usd", "2025", 20.0),
            ("KEN", "portfolio_liability_flows_usd", "2025", 10.0),
            ("KEN", "portfolio_investment_income_debits_usd", "2025", 5.0),
            ("KEN", "direct_investment_income_debits_usd", "2025", 3.0),
            ("KEN", "other_investment_income_debits_usd", "2025", 2.0),
            ("KEN", "net_iip_usd", "2025", -100.0),
            ("KEN", "external_liabilities_usd", "2025", 200.0),
            ("KEN", "portfolio_liabilities_usd", "2025", 70.0),
            ("KEN", "wb_total_external_debt_service_usd", "2025", 12.0),
            ("KEN", "wb_ppg_external_debt_service_usd", "2025", 8.0),
            ("KEN", "wb_government_revenue_ex_grants_gdp_pct", "2025", 20.0),
            ("KEN", "wb_government_interest_payments_revenue_pct", "2025", 15.0),
        ],
        columns=["country_code", "feature_key", "period_label", "value"],
    )
    observations["source"] = "BOP"
    observations["feature_label"] = observations["feature_key"]
    observations["quality"] = "observed"
    observations["dataset_version"] = "test"
    observations["period"] = pd.to_datetime(observations["period_label"] + "-12-31")
    model_features = pd.DataFrame(
        {
            "country_code": ["KEN", "MOZ"],
            "nominal_gdp": [100.0, 200.0],
            "fiscal_balance_gdp": [-4.0, 1.0],
        }
    )

    features, report = build_external_liquidity_features(observations, model_features)
    ken = features.set_index("country_code").loc["KEN"]

    assert ken["current_account_receipts_gdp"] == pytest.approx(50.0)
    assert ken["reserves_to_current_account_payments"] == pytest.approx(25.0)
    assert ken["investment_income_debits_to_cxr"] == pytest.approx(20.0)
    assert ken["gross_external_financing_need_proxy_gdp"] == pytest.approx(40.0)
    assert ken["wb_total_external_debt_service_gdp"] == pytest.approx(12.0)
    assert ken["wb_total_external_debt_service_revenue_proxy"] == pytest.approx(60.0)
    assert ken["wb_public_financing_need_ext_debt_service_proxy_gdp"] == pytest.approx(12.0)
    assert ken["wb_government_interest_payments_revenue_pct"] == pytest.approx(15.0)
    assert report["feature_coverage"]["current_account_receipts_gdp"]["countries"] == 1
    assert report["feature_coverage"]["current_account_receipts_gdp"]["pct_model_countries"] == 50.0


def test_build_external_liquidity_features_computes_market_and_fdi_features():
    observations = pd.DataFrame(
        [
            ("KEN", "fdi_liability_flows_usd", "2025", 30.0),
            ("KEN", "portfolio_liability_flows_usd", "2025", 10.0),
            ("KEN", "fdi_net_flows_usd", "2025", 25.0),
            ("KEN", "wb_terms_of_trade_index", "2025", 95.0),
            ("KEN", "wb_fuel_exports_pct", "2025", 5.0),
            ("KEN", "wb_ores_metals_exports_pct", "2025", 10.0),
            ("KEN", "wb_agri_raw_exports_pct", "2025", 20.0),
            ("KEN", "wb_food_exports_pct", "2025", 30.0),
            ("KEN", "wb_reer_index", "2025", 110.0),
        ],
        columns=["country_code", "feature_key", "period_label", "value"],
    )
    observations["source"] = "BOP"
    observations["feature_label"] = observations["feature_key"]
    observations["quality"] = "observed"
    observations["dataset_version"] = "test"
    observations["period"] = pd.to_datetime(observations["period_label"] + "-12-31")
    model_features = pd.DataFrame(
        {"country_code": ["KEN"], "nominal_gdp": [100.0]}
    )

    features, _ = build_external_liquidity_features(observations, model_features)
    ken = features.set_index("country_code").loc["KEN"]

    assert ken["fdi_liability_flows_gdp"] == pytest.approx(30.0)
    # stable financing share = |FDI| / (|FDI| + |portfolio|) = 30 / 40 * 100 = 75
    assert ken["stable_financing_share"] == pytest.approx(75.0)
    assert ken["terms_of_trade_index"] == pytest.approx(95.0)
    # commodity share = 5 + 10 + 20 + 30 = 65
    assert ken["commodity_export_share_pct"] == pytest.approx(65.0)
    assert ken["reer_index"] == pytest.approx(110.0)


def test_commodity_export_share_is_capped_and_nan_when_all_missing():
    observations = pd.DataFrame(
        [
            ("KEN", "wb_fuel_exports_pct", "2025", 80.0),
            ("KEN", "wb_ores_metals_exports_pct", "2025", 40.0),
            ("MOZ", "wb_reer_index", "2025", 100.0),  # no commodity components
        ],
        columns=["country_code", "feature_key", "period_label", "value"],
    )
    observations["source"] = "WB_WDI_IDS"
    observations["feature_label"] = observations["feature_key"]
    observations["quality"] = "observed"
    observations["dataset_version"] = "test"
    observations["period"] = pd.to_datetime(observations["period_label"] + "-12-31")
    model_features = pd.DataFrame(
        {"country_code": ["KEN", "MOZ"], "nominal_gdp": [100.0, 200.0]}
    )

    features, _ = build_external_liquidity_features(observations, model_features)
    indexed = features.set_index("country_code")
    # 80 + 40 = 120, capped at 100
    assert indexed.loc["KEN"]["commodity_export_share_pct"] == pytest.approx(100.0)
    # MOZ has no commodity components at all -> NaN, not 0
    assert pd.isna(indexed.loc["MOZ"]["commodity_export_share_pct"])


def test_reer_appreciation_gap_measures_latest_vs_trailing_mean():
    observations = pd.DataFrame(
        [
            ("KEN", "wb_reer_index", 2021, 100.0),
            ("KEN", "wb_reer_index", 2022, 100.0),
            ("KEN", "wb_reer_index", 2023, 100.0),
            ("KEN", "wb_reer_index", 2024, 100.0),
            ("KEN", "wb_reer_index", 2025, 110.0),  # latest, +10% vs mean of prior 100
            ("SGL", "wb_reer_index", 2025, 90.0),   # single obs -> skipped
        ],
        columns=["country_code", "feature_key", "period_label", "value"],
    )
    observations["period"] = pd.to_datetime(observations["period_label"].astype(str) + "-12-31")

    gap = reer_appreciation_gap(observations).set_index("country_code")
    assert gap.loc["KEN"]["reer_appreciation_5y_pct"] == pytest.approx(10.0)
    assert "SGL" not in gap.index


def test_normalize_world_bank_records_keeps_model_country_rows():
    records = [
        {
            "countryiso3code": "KEN",
            "date": "2024",
            "value": 12.5,
        },
        {
            "countryiso3code": "GHA",
            "date": "bad",
            "value": 4.0,
        },
        {
            "countryiso3code": "",
            "date": "2024",
            "value": 8.0,
        },
    ]
    spec = WorldBankSeriesSpec(
        "wb_total_external_debt_service_gni_pct",
        "DT.TDS.DECT.GN.ZS",
        "Total debt service, percent of GNI",
    )

    normalized = normalize_world_bank_records(records, spec)

    assert len(normalized) == 1
    assert normalized.iloc[0]["source"] == "WB_WDI_IDS"
    assert normalized.iloc[0]["feature_key"] == "wb_total_external_debt_service_gni_pct"
    assert normalized.iloc[0]["country_code"] == "KEN"
    assert normalized.iloc[0]["period"].year == 2024
