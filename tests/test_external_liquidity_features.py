import pandas as pd
import pytest

from src.external_liquidity import (
    ExternalSeriesSpec,
    WorldBankSeriesSpec,
    build_external_liquidity_features,
    normalize_feature_csv,
    normalize_world_bank_records,
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
