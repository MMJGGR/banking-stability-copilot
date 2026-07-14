import json
import pickle

import numpy as np
import pandas as pd

from src.scripts.build_hierarchical_risk_snapshot import (
    BuildConfig,
    build_hierarchical_risk_snapshot,
    merge_candidate_features,
    write_json_artifacts,
)


def _annual_panel(*, include_events=True):
    countries = [f"Z{letter}A" for letter in "ABCDEFGHIJKL"]
    event_years = [2004, 2009, 2015, 2021] if include_events else []
    rows = []
    for country_position, country in enumerate(countries):
        country_shift = country_position / 20.0
        for year in range(1998, 2023):
            future = [event for event in event_years if 1 <= event - year <= 3]
            next_event = min(future) if future else None
            distance = next_event - year if next_event else None
            stress = float(4 - distance) if distance else 0.0
            rows.append(
                {
                    "country_code": country,
                    "forecast_origin_year": year,
                    "feature_cutoff_year": year - 1,
                    "crisis_target": int(next_event is not None),
                    "crisis_event_id": (
                        f"{country}-{next_event}-{next_event}" if next_event else None
                    ),
                    "crisis_start_year": float(next_event) if next_event else np.nan,
                    "crisis_end_year": float(next_event) if next_event else np.nan,
                    "years_to_crisis": distance if distance else pd.NA,
                    "gdp_growth": 4.0 - 2.0 * stress + country_shift,
                    "inflation": 2.0 + 3.0 * stress + country_shift,
                    "current_account_gdp": 1.0 - 2.0 * stress,
                    "govt_debt_gdp": 35.0 + 12.0 * stress + country_position,
                    "fiscal_balance_gdp": 1.0 - 2.0 * stress,
                    "primary_balance_gdp": 2.0 - 1.5 * stress,
                    "unemployment": 5.0 + 2.0 * stress,
                    "govt_revenue_gdp": 25.0 - stress,
                    "capital_adequacy": 18.0 - 2.0 * stress,
                    "npl_ratio": 2.0 + 2.5 * stress,
                    "roe": 14.0 - 2.5 * stress,
                    "liquid_assets_total": 30.0 - 3.0 * stress,
                    "bank_credit_gdp_change_3y": 2.0 + 4.0 * stress,
                    "bank_credit_gdp_gap_10y": -1.0 + 3.0 * stress,
                    "bank_credit_to_deposits": 70.0 + 8.0 * stress,
                    "bank_zscore": 18.0 - 2.0 * stress,
                    "wb_bank_npl_ratio": 2.0 + 2.5 * stress,
                    "wb_bank_capital_assets": 12.0 - stress,
                    "wb_bank_liquid_reserves_assets": 25.0 - 2.0 * stress,
                    "broad_money_to_reserves": 120.0 + 20.0 * stress,
                    "reserves_months_imports": 8.0 - stress,
                    "lending_interest_rate_change_3y": stress,
                    "real_interest_rate": 2.0 + stress,
                    "commodity_export_concentration": 20.0 + country_position,
                    "natural_resource_rents_gdp": 5.0 + country_shift,
                    "terms_of_trade_deterioration_3y": stress,
                    "commodity_shock_exposure": stress,
                }
            )
    frame = pd.DataFrame(rows)
    frame["years_to_crisis"] = frame["years_to_crisis"].astype("Int64")
    return frame


def _risk_artifact(panel):
    countries = sorted(panel["country_code"].unique())
    scores = pd.DataFrame(
        {
            "country_code": countries,
            "country_name": [f"Country {code}" for code in countries],
            "risk_score": np.linspace(2.0, 8.0, len(countries)),
            "economic_pillar": np.linspace(8.0, 3.0, len(countries)),
            "industry_pillar": np.linspace(7.0, 4.0, len(countries)),
        }
    )
    values = pd.DataFrame(
        {
            "country_code": countries,
            "gdp_growth_3yr_avg": np.linspace(1.0, 4.0, len(countries)),
            "inflation": np.linspace(2.0, 10.0, len(countries)),
            "inflation_acceleration": np.linspace(-1.0, 3.0, len(countries)),
            "ca_deficit_severity": np.linspace(0.0, 8.0, len(countries)),
            "ca_deterioration_3yr": np.linspace(2.0, -4.0, len(countries)),
            "govt_debt_gdp": np.linspace(30.0, 100.0, len(countries)),
            "debt_buildup_3yr": np.linspace(-5.0, 20.0, len(countries)),
            "fiscal_balance_gdp": np.linspace(3.0, -8.0, len(countries)),
            "govt_interest_to_revenue": np.linspace(5.0, 35.0, len(countries)),
            "govt_interest_to_revenue_change_3y": np.linspace(-2.0, 10.0, len(countries)),
            "unemployment": np.linspace(3.0, 15.0, len(countries)),
            "years_since_banking_crisis": np.linspace(20.0, 1.0, len(countries)),
            "credit_growth_3yr": np.linspace(2.0, 15.0, len(countries)),
            "m2_to_reserves": np.linspace(100.0, 500.0, len(countries)),
            "reserves_to_goods_services_imports": np.linspace(80.0, 20.0, len(countries)),
            "npl_ratio": np.linspace(1.0, 16.0, len(countries)),
            "capital_adequacy": np.linspace(22.0, 8.0, len(countries)),
            "liquid_assets_total": np.linspace(40.0, 10.0, len(countries)),
            "gdp_growth": np.linspace(5.0, -2.0, len(countries)),
            "current_account_gdp": np.linspace(5.0, -12.0, len(countries)),
            "govt_revenue_gdp": np.linspace(35.0, 12.0, len(countries)),
            "govt_debt_to_revenue": np.linspace(100.0, 600.0, len(countries)),
            "net_iip_gdp": np.linspace(20.0, -120.0, len(countries)),
            "external_liabilities_gdp": np.linspace(40.0, 200.0, len(countries)),
        }
    )
    return {
        "country_scores": scores,
        "feature_values": values,
        "trained": True,
        "training_date": "2026-06-30",
        "countries_trained": len(countries),
    }


def _write_inputs(tmp_path, *, include_events=True):
    panel = _annual_panel(include_events=include_events)
    panel_path = tmp_path / "annual.parquet"
    panel.to_parquet(panel_path, index=False)
    risk_path = tmp_path / "risk.pkl"
    with risk_path.open("wb") as target:
        pickle.dump(_risk_artifact(panel), target)
    countries = sorted(panel["country_code"].unique())
    external = pd.DataFrame(
        {
            "country_code": countries,
            # Must not overwrite the production field.
            "net_iip_gdp": [999.0] * len(countries),
            "portfolio_liabilities_gdp": np.linspace(5.0, 45.0, len(countries)),
            "gross_external_financing_need_proxy_gdp": np.linspace(
                2.0, 25.0, len(countries)
            ),
        }
    )
    external_path = tmp_path / "external.parquet"
    external.to_parquet(external_path, index=False)
    government = pd.DataFrame(
        {
            "country_code": countries,
            "govt_gross_debt_gdp": [777.0] * len(countries),
            "govt_primary_deficit_gdp": np.linspace(0.0, 8.0, len(countries)),
        }
    )
    government_path = tmp_path / "government.parquet"
    government.to_parquet(government_path, index=False)
    return panel_path, risk_path, external_path, government_path


def test_candidate_merge_preserves_production_and_uses_canonical_aliases():
    production = pd.DataFrame(
        {
            "country_code": ["AAA", "BBB"],
            "govt_debt_gdp": [50.0, np.nan],
        }
    )
    staged = pd.DataFrame(
        {
            "country_code": ["AAA", "BBB"],
            "govt_gross_debt_gdp": [500.0, 70.0],
            "government_liquidity_feature_count": [1, 1],
        }
    )

    merged, report = merge_candidate_features(
        production,
        staged,
        source_name="government",
        aliases={"govt_gross_debt_gdp": "govt_debt_gdp"},
    )

    assert merged["govt_debt_gdp"].tolist() == [50.0, 70.0]
    assert "govt_gross_debt_gdp" not in merged
    assert not any(column.endswith("_x") or column.endswith("_y") for column in merged)
    assert report["preserved_production_values"]["govt_debt_gdp"] == 1
    assert report["filled_values"]["govt_debt_gdp"] == 1
    assert report["ignored_columns"] == ["government_liquidity_feature_count"]


def test_builder_creates_compact_governed_snapshot_and_forward_validation(tmp_path):
    panel_path, risk_path, external_path, government_path = _write_inputs(tmp_path)
    config = BuildConfig(
        as_of_date="2026-06-30",
        validation_start_year=2012,
        test_start_year=2018,
        modern_start_year=2000,
    )

    snapshot, validation = build_hierarchical_risk_snapshot(
        annual_panel_path=panel_path,
        risk_model_path=risk_path,
        external_candidates_path=external_path,
        government_candidates_path=government_path,
        config=config,
    )

    assert snapshot["schema_version"] == 1
    assert snapshot["as_of_date"] == "2026-06-30"
    # The synthetic sample deliberately fails the 50-country promotion gate.
    assert snapshot["model_status"] == "research_challenger"
    assert len(snapshot["countries"]) == 12
    record = snapshot["countries"][0]
    assert {
        "systemic_hazard_1y",
        "systemic_hazard_2_3y",
        "systemic_hazard_3y",
        "hazard_expert",
        "evidence_confidence",
        "alert_status",
        "dominant_mechanism",
        "mechanisms",
    }.issubset(record)
    assert len(record["mechanisms"]) >= 6
    assert 0 <= record["evidence_confidence"] <= 1
    assert "statistical confidence" in snapshot["confidence_semantics"]

    assert validation["design"].startswith("horizon-embargoed forward-time")
    assert validation["evaluation_period"]["training"] == {
        "origin_start": 1998,
        "label_cutoff_exclusive": 2012,
        "origin_end_by_horizon": {"1y": 2010, "2y": 2009, "3y": 2008},
    }
    assert validation["evaluation_period"]["threshold_validation"] == {
        "origin_start": 2012,
        "label_cutoff_exclusive": 2018,
        "origin_end_by_horizon": {"1y": 2016, "2_3y": 2014, "3y": 2014},
    }
    for horizon in ("1y", "2_3y", "3y"):
        operating_point = next(
            value
            for value in validation["metrics"][horizon].values()
            if isinstance(value, dict) and "confusion_matrix" in value
        )
        assert "unique_event_recall" in operating_point
        assert "false_alerts_per_100_country_years" in operating_point
    assert not validation["promotion_gates"]["passed"]
    assert validation["fit_diagnostics"]["final_experts"]["historical_core"][
        "complete"
    ]
    assert validation["fit_diagnostics"]["final_experts"]["modern_full"][
        "complete"
    ]
    threshold_horizons = validation["fit_diagnostics"]["threshold_training_experts"][
        "historical_core"
    ]["horizons"]
    assert threshold_horizons["1"]["training_year_end"] <= 2010
    assert threshold_horizons["3"]["training_year_end"] <= 2008
    forward_horizons = validation["fit_diagnostics"]["forward_test_experts"][
        "historical_core"
    ]["horizons"]
    assert forward_horizons["1"]["training_year_end"] <= 2016
    assert forward_horizons["3"]["training_year_end"] <= 2014

    snapshot_output = tmp_path / "snapshot.json"
    validation_output = tmp_path / "validation.json"
    write_json_artifacts(
        snapshot,
        validation,
        snapshot_output=snapshot_output,
        validation_output=validation_output,
    )
    first = snapshot_output.read_text(encoding="utf-8")
    write_json_artifacts(
        snapshot,
        validation,
        snapshot_output=snapshot_output,
        validation_output=validation_output,
    )
    assert snapshot_output.read_text(encoding="utf-8") == first
    assert "NaN" not in first
    assert str(tmp_path) not in first
    assert json.loads(first)["countries"] == snapshot["countries"]


def test_incompatible_tranquil_panel_reports_research_and_null_hazards(tmp_path):
    panel_path, risk_path, external_path, government_path = _write_inputs(
        tmp_path, include_events=False
    )
    snapshot, validation = build_hierarchical_risk_snapshot(
        annual_panel_path=panel_path,
        risk_model_path=risk_path,
        external_candidates_path=external_path,
        government_candidates_path=government_path,
        config=BuildConfig(
            as_of_date="2026-06-30",
            validation_start_year=2012,
            test_start_year=2018,
            modern_start_year=2000,
        ),
    )

    assert snapshot["model_status"] == "research_challenger"
    assert snapshot["countries"][0]["systemic_hazard_1y"] is None
    assert snapshot["countries"][0]["alert_status"] == "insufficient_evidence"
    assert not validation["fit_diagnostics"]["final_experts"]["historical_core"][
        "complete"
    ]
    reason = validation["fit_diagnostics"]["final_experts"]["historical_core"][
        "horizons"
    ]["1"]["reason"]
    assert "fewer than two classes" in reason
