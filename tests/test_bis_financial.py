import json
from pathlib import Path
import zipfile

import pandas as pd

from src.bis_financial import (
    BisDownload,
    build_bis_financial_history,
    normalise_bis_credit_gap,
    normalise_bis_debt_service,
    normalise_bis_selected_property_prices,
    normalise_bis_total_credit,
    read_bis_bulk_csv,
    write_bis_financial_history,
)
from src.crisis_panel import BIS_FEATURE_SPECS, CrisisPanelConfig, build_crisis_panel


def _metadata(dataset: str, dataset_id: str) -> BisDownload:
    return BisDownload(
        dataset=dataset,
        dataset_id=dataset_id,
        path=f"{dataset}.zip",
        source_url=f"https://data.bis.org/{dataset}.zip",
        retrieved_at="2026-07-13T12:00:00+00:00",
        source_vintage="Wed, 08 Jul 2026 11:51:36 GMT",
        etag="fixture",
        bytes=100,
        sha256="0" * 64,
    )


def _total_credit_frame() -> pd.DataFrame:
    rows = []
    for country, lender, valuation, adjustment, period, value in (
        ("US: United States", "A: All sectors", "M: Market value", "A: Adjusted for breaks", "1999-Q4", 120.0),
        ("US: United States", "B: Banks, domestic", "M: Market value", "A: Adjusted for breaks", "1999-Q4", 70.0),
        ("US: United States", "A: All sectors", "N: Nominal value", "A: Adjusted for breaks", "1999-Q4", 999.0),
        ("US: United States", "A: All sectors", "M: Market value", "U: Unadjusted", "1999-Q4", 998.0),
        ("5R: Advanced economies", "A: All sectors", "M: Market value", "A: Adjusted for breaks", "1999-Q4", 150.0),
    ):
        rows.append(
            {
                "STRUCTURE_ID": "BIS:WS_TC(2.0): Total credit",
                "FREQ:Frequency": "Q: Quarterly",
                "BORROWERS_CTY:Borrowers' country": country,
                "TC_BORROWERS:Borrowing sector": "P: Private non-financial sector",
                "TC_LENDERS:Lending sector": lender,
                "VALUATION:Valuation method": valuation,
                "UNIT_TYPE:Unit type": "770: Percentage of GDP",
                "TC_ADJUST:Adjustment": adjustment,
                "TIME_PERIOD:Time period or range": period,
                "OBS_VALUE:Observation Value": value,
                "OBS_STATUS:Observation Status": "A: Normal value",
            }
        )
    return pd.DataFrame(rows)


def _credit_gap_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "STRUCTURE_ID": "BIS:WS_CREDIT_GAP(1.0): Credit-to-GDP gaps",
                "FREQ:Frequency": "Q: Quarterly",
                "BORROWERS_CTY:Borrowers' country": "US: United States",
                "TC_BORROWERS:Borrowing sector": "P: Private non-financial sector",
                "TC_LENDERS:Lending sector": "A: All sectors",
                "CG_DTYPE:Credit gap data type": data_type,
                "TIME_PERIOD:Time period or range": "1999-Q4",
                "OBS_VALUE:Observation Value": value,
                "OBS_STATUS:Observation Status": "A: Normal value",
            }
            for data_type, value in (
                ("A: Credit-to-GDP ratios (actual data)", 120.0),
                ("B: Credit-to-GDP trend (HP filter)", 110.0),
                ("C: Credit-to-GDP gaps (actual-trend)", 10.0),
            )
        ]
    )


def _debt_service_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "STRUCTURE_ID": "BIS:WS_DSR(1.0): Debt service ratios",
                "FREQ:Frequency": "Q: Quarterly",
                "BORROWERS_CTY:Borrowers' country": "US: United States",
                "DSR_BORROWERS:Borrowers": borrower,
                "TIME_PERIOD:Time period or range": "2000-Q4",
                "OBS_VALUE:Observation Value": value,
                "OBS_STATUS:Observation Status": "A: Normal value",
            }
            for borrower, value in (
                ("P: Private non-financial sector", 14.0),
                ("H: Households & NPISHs", 11.0),
                ("N: Non-financial corporations", 17.0),
            )
        ]
    )


def _property_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "STRUCTURE_ID": "BIS:WS_SPP(1.0): Selected residential property prices",
                "FREQ:Frequency": "Q: Quarterly",
                "REF_AREA:Reference area": country,
                "VALUE:Value": value_type,
                "UNIT_MEASURE:Unit of measure": unit,
                "TIME_PERIOD:Time period or range": "2001-Q4",
                "OBS_VALUE:Observation Value": value,
                "OBS_STATUS:Observation Status": "A: Normal value",
            }
            for country, value_type, unit, value in (
                ("US: United States", "R: Real", "771: Year-on-year changes, in per cent", 5.0),
                ("US: United States", "N: Nominal", "771: Year-on-year changes, in per cent", 8.0),
                ("US: United States", "R: Real", "628: Index, 2010 = 100", 91.0),
                ("US: United States", "N: Nominal", "628: Index, 2010 = 100", 105.0),
                ("XW: World", "R: Real", "771: Year-on-year changes, in per cent", 3.0),
            )
        ]
    )


def test_total_credit_selects_comparable_reported_series_and_drops_aggregates():
    result = normalise_bis_total_credit(
        _total_credit_frame(), metadata=_metadata("total_credit", "WS_TC")
    )
    assert set(result["indicator_code"]) == {
        "bis_private_credit_gdp",
        "bis_bank_credit_gdp",
    }
    assert set(result["value"]) == {120.0, 70.0}
    assert result["country_code"].eq("USA").all()
    assert result["period"].dt.strftime("%Y-%m-%d").eq("1999-12-31").all()
    assert result["is_direct"].all()
    assert result["source_vintage"].notna().all()


def test_credit_gap_name_is_reserved_for_official_bis_gap_series():
    result = normalise_bis_credit_gap(_credit_gap_frame())
    assert len(result) == 1
    assert result.loc[0, "indicator_code"] == "bis_private_credit_to_gdp_gap"
    assert result.loc[0, "value"] == 10.0
    assert result.loc[0, "source_dataset_id"] == "WS_CREDIT_GAP"
    assert result.loc[0, "source_series_key"] == "Q.US.P.A.C"
    assert bool(result.loc[0, "is_direct"])


def test_debt_service_keeps_total_and_sector_mechanisms_separate():
    result = normalise_bis_debt_service(_debt_service_frame())
    assert set(result["indicator_code"]) == {
        "bis_private_debt_service_ratio",
        "bis_household_debt_service_ratio",
        "bis_corporate_debt_service_ratio",
    }
    assert result["family"].eq("debt_service_pressure").all()


def test_selected_property_prices_normalises_four_direct_measures():
    result = normalise_bis_selected_property_prices(_property_frame())
    assert len(result) == 4
    assert set(result["indicator_code"]) == {
        "bis_real_house_price_growth_yoy",
        "bis_nominal_house_price_growth_yoy",
        "bis_real_house_price_index",
        "bis_nominal_house_price_index",
    }
    assert result["country_code"].eq("USA").all()


def test_combined_builder_filters_years_without_deriving_extra_gap_series():
    result = build_bis_financial_history(
        {
            "total_credit": _total_credit_frame(),
            "credit_gap": _credit_gap_frame(),
            "debt_service": _debt_service_frame(),
            "selected_property_prices": _property_frame(),
        },
        start_year=2000,
        end_year=2001,
    )
    assert "bis_private_credit_gdp" not in set(result["indicator_code"])
    gaps = result[result["indicator_code"].str.contains("gap")]
    assert gaps.empty  # The fixture's only official gap is in 1999.


def test_read_bulk_zip_uses_small_flat_csv_fixture(tmp_path: Path):
    archive_path = tmp_path / "WS_DSR_csv_flat.zip"
    csv_bytes = _debt_service_frame().to_csv(index=False).encode("utf-8")
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("WS_DSR_csv_flat.csv", csv_bytes)
    result = read_bis_bulk_csv(archive_path, "debt_service")
    assert len(result) == 3


def test_writer_persists_auditable_manifest(tmp_path: Path):
    observations = normalise_bis_credit_gap(_credit_gap_frame())
    output = tmp_path / "bis.parquet"
    manifest = tmp_path / "bis.json"
    write_bis_financial_history(
        observations,
        output_path=output,
        manifest_path=manifest,
        downloads=[_metadata("credit_gap", "WS_CREDIT_GAP")],
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert output.exists()
    assert payload["coverage"]["rows"] == 1
    assert payload["downloads"][0]["dataset_id"] == "WS_CREDIT_GAP"
    assert "official BIS WS_CREDIT_GAP" in payload["provenance_note"]


class _Labels:
    crises = {}
    SOURCE_COVERAGE_END_YEAR = 2005


def test_optional_bis_specs_use_existing_panel_cutoff_contract():
    bis = normalise_bis_credit_gap(_credit_gap_frame())
    spec = next(
        item
        for item in BIS_FEATURE_SPECS
        if item.name == "bis_private_credit_to_gdp_gap"
    )
    panel = build_crisis_panel(
        None,
        None,
        _Labels(),
        ["USA"],
        [spec],
        CrisisPanelConfig(
            start_year=2000,
            end_year=2000,
            feature_lag_years=1,
            horizon_end_years=1,
        ),
        additional_sources={"BIS": bis},
    )
    assert panel.loc[0, "bis_private_credit_to_gdp_gap"] == 10.0
    assert panel.loc[0, "bis_private_credit_to_gdp_gap__observation_year"] == 1999
    assert bool(panel.loc[0, "bis_private_credit_to_gdp_gap__direct"])
