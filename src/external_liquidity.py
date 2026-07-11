"""External-liquidity feature retrieval and construction.

This module intentionally uses a bounded, feature-oriented SDMX retrieval
strategy. The IMF SDMX 3.0 API ignores ``c[...]`` filters for the dataflow
URLs used here, so the only reliable way to avoid full-flow downloads is to
place countries, indicators, units, and frequency directly in the path key.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import io
import json

import pandas as pd
import requests

from src.config import BASE_DIR, CACHE_DIR
from src.model_store import load_model_artifact
from src.sources.sdmx import IMF_SDMX_BASE, SDMX_CSV_ACCEPT, WORLD_BANK_BASE, _parse_sdmx_period


EXTERNAL_FEATURE_OBSERVATIONS = Path(CACHE_DIR) / "external" / "external_feature_observations.parquet"
EXTERNAL_FEATURE_VALUES = Path(CACHE_DIR) / "external" / "external_liquidity_features.parquet"
EXTERNAL_FEATURE_REPORT = Path(BASE_DIR) / "artifacts" / "external_liquidity_features_report.json"


@dataclass(frozen=True)
class ExternalSeriesSpec:
    feature_key: str
    source: str
    agency: str
    dataflow_id: str
    key_template: str
    label: str
    quality: str = "observed"

    def key_for_countries(self, country_codes: list[str]) -> str:
        country_expr = "+".join(country_codes)
        return self.key_template.format(countries=country_expr)

    def data_url(self, country_codes: list[str]) -> str:
        return (
            f"{IMF_SDMX_BASE}/data/dataflow/"
            f"{self.agency}/{self.dataflow_id}/+/{self.key_for_countries(country_codes)}"
        )


@dataclass(frozen=True)
class WorldBankSeriesSpec:
    feature_key: str
    indicator_code: str
    label: str
    source_id: int | None = 2
    source: str = "WB_WDI_IDS"
    quality: str = "observed"

    def indicator_url(self) -> str:
        return f"{WORLD_BANK_BASE}/country/all/indicator/{self.indicator_code}"


EXTERNAL_SERIES_SPECS: tuple[ExternalSeriesSpec, ...] = (
    ExternalSeriesSpec(
        "current_account_balance_usd", "BOP", "IMF.STA", "BOP",
        "{countries}.NETCD_T.CAB.USD.A",
        "Current account balance, USD",
    ),
    ExternalSeriesSpec(
        "current_account_receipts_usd", "BOP", "IMF.STA", "BOP",
        "{countries}.CD_T.TCDCA.USD.A",
        "Total current account credit/receipts, USD",
    ),
    ExternalSeriesSpec(
        "current_account_payments_usd", "BOP", "IMF.STA", "BOP",
        "{countries}.DB_T.TDBCA.USD.A",
        "Total current account debit/payments, USD",
    ),
    ExternalSeriesSpec(
        "goods_services_exports_usd", "BOP", "IMF.STA", "BOP",
        "{countries}.CD_T.GS.USD.A",
        "Goods and services exports, USD",
    ),
    ExternalSeriesSpec(
        "goods_services_imports_usd", "BOP", "IMF.STA", "BOP",
        "{countries}.DB_T.GS.USD.A",
        "Goods and services imports, USD",
    ),
    ExternalSeriesSpec(
        "portfolio_liability_flows_usd", "BOP", "IMF.STA", "BOP",
        "{countries}.L_NIL_T.P_F.USD.A",
        "Portfolio investment liabilities, net incurrence, USD",
    ),
    ExternalSeriesSpec(
        "portfolio_net_flows_usd", "BOP", "IMF.STA", "BOP",
        "{countries}.NNAFANIL_T.P_F.USD.A",
        "Portfolio investment net flows, USD",
    ),
    # Direct-investment flows (functional category D_F) mirror the portfolio
    # specs above (P_F). FDI is the stable-financing counterpart to fickle
    # portfolio capital (backlog rank 19).
    ExternalSeriesSpec(
        "fdi_liability_flows_usd", "BOP", "IMF.STA", "BOP",
        "{countries}.L_NIL_T.D_F.USD.A",
        "Direct investment liabilities, net incurrence, USD",
    ),
    ExternalSeriesSpec(
        "fdi_net_flows_usd", "BOP", "IMF.STA", "BOP",
        "{countries}.NNAFANIL_T.D_F.USD.A",
        "Direct investment net flows, USD",
    ),
    ExternalSeriesSpec(
        "direct_investment_income_debits_usd", "BOP", "IMF.STA", "BOP",
        "{countries}.DB_T.D_F_D4P.USD.A",
        "Direct-investment income debits, USD",
    ),
    ExternalSeriesSpec(
        "portfolio_investment_income_debits_usd", "BOP", "IMF.STA", "BOP",
        "{countries}.DB_T.P_F_D4P.USD.A",
        "Portfolio-investment income debits, USD",
    ),
    ExternalSeriesSpec(
        "other_investment_income_debits_usd", "BOP", "IMF.STA", "BOP",
        "{countries}.DB_T.O_F_D4P.USD.A",
        "Other-investment income debits, USD",
    ),
    ExternalSeriesSpec(
        "reserve_assets_usd", "IIP", "IMF.STA", "IIP",
        "{countries}.A_P.R.USD.A",
        "Reserve assets, USD",
    ),
    ExternalSeriesSpec(
        "net_iip_usd", "IIP", "IMF.STA", "IIP",
        "{countries}.NETAL_P.NIIP.USD.A",
        "Net international investment position, USD",
    ),
    ExternalSeriesSpec(
        "external_assets_usd", "IIP", "IMF.STA", "IIP",
        "{countries}.A_P.IIP.USD.A",
        "External assets, USD",
    ),
    ExternalSeriesSpec(
        "external_liabilities_usd", "IIP", "IMF.STA", "IIP",
        "{countries}.L_P.IIP.USD.A",
        "External liabilities, USD",
    ),
    ExternalSeriesSpec(
        "portfolio_liabilities_usd", "IIP", "IMF.STA", "IIP",
        "{countries}.L_P.P_MV.USD.A",
        "Portfolio investment liabilities position, USD",
    ),
)


WORLD_BANK_SERIES_SPECS: tuple[WorldBankSeriesSpec, ...] = (
    # WB additions are deliberately limited to missing debt-service and
    # debt-affordability metrics. Current-account, reserves, and broad
    # external-liquidity measures already exist from IMF/WEO/MFS/BOP/IIP and
    # should not be duplicated with alternate WB versions.
    WorldBankSeriesSpec(
        "wb_total_external_debt_service_usd",
        "DT.TDS.DECT.CD",
        "Debt service on external debt, total, current USD",
    ),
    WorldBankSeriesSpec(
        "wb_total_external_debt_service_exports_pct",
        "DT.TDS.DECT.EX.ZS",
        "Total debt service, percent of exports of goods, services and primary income",
    ),
    WorldBankSeriesSpec(
        "wb_total_external_debt_service_gni_pct",
        "DT.TDS.DECT.GN.ZS",
        "Total debt service, percent of GNI",
    ),
    WorldBankSeriesSpec(
        "wb_ppg_external_debt_service_usd",
        "DT.TDS.DPPG.CD",
        "Public and publicly guaranteed external debt service, current USD",
    ),
    WorldBankSeriesSpec(
        "wb_ppg_external_debt_service_exports_pct",
        "DT.TDS.DPPG.XP.ZS",
        "Public and publicly guaranteed external debt service, percent of exports",
    ),
    WorldBankSeriesSpec(
        "wb_ppg_external_debt_service_gni_pct",
        "DT.TDS.DPPG.GN.ZS",
        "Public and publicly guaranteed external debt service, percent of GNI",
    ),
    WorldBankSeriesSpec(
        "wb_government_interest_payments_revenue_pct",
        "GC.XPN.INTP.RV.ZS",
        "Central government interest payments, percent of revenue",
    ),
    WorldBankSeriesSpec(
        "wb_government_revenue_ex_grants_gdp_pct",
        "GC.REV.XGRT.GD.ZS",
        "Central government revenue excluding grants, percent of GDP",
    ),
    # Market / external stress inputs (backlog ranks 20-21). Terms of trade and
    # merchandise export composition proxy commodity/export-concentration
    # exposure; the real effective exchange rate proxies valuation stress.
    WorldBankSeriesSpec(
        "wb_terms_of_trade_index",
        "TT.PRI.MRCH.XD.WD",
        "Net barter terms of trade index (2015 = 100)",
    ),
    WorldBankSeriesSpec(
        "wb_fuel_exports_pct",
        "TX.VAL.FUEL.ZS.UN",
        "Fuel exports, percent of merchandise exports",
    ),
    WorldBankSeriesSpec(
        "wb_ores_metals_exports_pct",
        "TX.VAL.MMTL.ZS.UN",
        "Ores and metals exports, percent of merchandise exports",
    ),
    WorldBankSeriesSpec(
        "wb_agri_raw_exports_pct",
        "TX.VAL.AGRI.ZS.UN",
        "Agricultural raw materials exports, percent of merchandise exports",
    ),
    WorldBankSeriesSpec(
        "wb_food_exports_pct",
        "TX.VAL.FOOD.ZS.UN",
        "Food exports, percent of merchandise exports",
    ),
    WorldBankSeriesSpec(
        "wb_reer_index",
        "PX.REX.REER",
        "Real effective exchange rate index (2010 = 100)",
    ),
)


def batched(values: Iterable[str], size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def normalize_feature_csv(content: bytes, spec: ExternalSeriesSpec) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(content), low_memory=False)
    required = {"COUNTRY", "TIME_PERIOD", "OBS_VALUE"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    values = pd.to_numeric(frame["OBS_VALUE"], errors="coerce")
    normalized = pd.DataFrame(
        {
            "source": spec.source,
            "feature_key": spec.feature_key,
            "feature_label": spec.label,
            "quality": spec.quality,
            "dataset_version": frame.get("STRUCTURE_ID"),
            "country_code": frame["COUNTRY"].astype(str).str.strip().str.upper(),
            "period_label": frame["TIME_PERIOD"].astype(str).str.strip(),
            "period": frame["TIME_PERIOD"].map(_parse_sdmx_period),
            "value": values,
        }
    )
    return normalized[
        normalized["value"].notna()
        & normalized["period"].notna()
        & normalized["country_code"].str.fullmatch(r"[A-Z]{3}")
    ].reset_index(drop=True)


def normalize_world_bank_records(
    records: list[dict],
    spec: WorldBankSeriesSpec,
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "source", "feature_key", "feature_label", "quality",
                "dataset_version", "country_code", "period_label", "period",
                "value",
            ]
        )
    frame = pd.DataFrame.from_records(records)
    values = pd.to_numeric(frame["value"], errors="coerce")
    normalized = pd.DataFrame(
        {
            "source": spec.source,
            "feature_key": spec.feature_key,
            "feature_label": spec.label,
            "quality": spec.quality,
            "dataset_version": f"WorldBank source={spec.source_id or 'default'} indicator={spec.indicator_code}",
            "country_code": frame["countryiso3code"].astype(str).str.strip().str.upper(),
            "period_label": frame["date"].astype(str).str.strip(),
            "period": frame["date"].map(_parse_sdmx_period),
            "value": values,
        }
    )
    return normalized[
        normalized["value"].notna()
        & normalized["period"].notna()
        & normalized["country_code"].str.fullmatch(r"[A-Z]{3}")
    ].reset_index(drop=True)


def fetch_world_bank_feature_observations(
    country_codes: list[str],
    start_period: str = "2005",
    specs: tuple[WorldBankSeriesSpec, ...] = WORLD_BANK_SERIES_SPECS,
    timeout: int = 60,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    country_codes = sorted({str(code).upper() for code in country_codes if pd.notna(code)})
    country_set = set(country_codes)
    start_ts = _parse_sdmx_period(start_period)
    end_year = datetime.now(timezone.utc).year
    start_year = start_ts.year if start_ts is not pd.NaT else 2005

    for spec in specs:
        page = 1
        records: list[dict] = []
        while True:
            params = {
                "format": "json",
                "per_page": 20000,
                "page": page,
                "date": f"{start_year}:{end_year}",
            }
            if spec.source_id is not None:
                params["source"] = spec.source_id
            try:
                response = requests.get(spec.indicator_url(), params=params, timeout=timeout)
                response.raise_for_status()
            except requests.RequestException:
                break
            payload = response.json()
            if not isinstance(payload, list) or len(payload) < 2:
                break
            meta, page_records = payload[0], payload[1] or []
            records.extend(
                record
                for record in page_records
                if str(record.get("countryiso3code", "")).strip().upper() in country_set
                and record.get("value") is not None
            )
            if page >= int(meta.get("pages", 1)):
                break
            page += 1
        normalized = normalize_world_bank_records(records, spec)
        if start_ts is not pd.NaT and not normalized.empty:
            normalized = normalized[normalized["period"] >= start_ts]
        if not normalized.empty:
            frames.append(normalized)

    if not frames:
        return pd.DataFrame(
            columns=[
                "source", "feature_key", "feature_label", "quality",
                "dataset_version", "country_code", "period_label", "period",
                "value",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def fetch_feature_observations(
    country_codes: list[str],
    start_period: str = "2005",
    specs: tuple[ExternalSeriesSpec, ...] = EXTERNAL_SERIES_SPECS,
    world_bank_specs: tuple[WorldBankSeriesSpec, ...] = WORLD_BANK_SERIES_SPECS,
    batch_size: int = 25,
    timeout: int = 120,
    include_world_bank: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    country_codes = sorted({str(code).upper() for code in country_codes if pd.notna(code)})
    start_ts = _parse_sdmx_period(start_period)
    for spec in specs:
        for batch in batched(country_codes, batch_size):
            response = requests.get(
                spec.data_url(batch),
                headers={"Accept": SDMX_CSV_ACCEPT},
                timeout=timeout,
            )
            response.raise_for_status()
            normalized = normalize_feature_csv(response.content, spec)
            if start_ts is not pd.NaT and not normalized.empty:
                normalized = normalized[normalized["period"] >= start_ts]
            if not normalized.empty:
                frames.append(normalized)
    if include_world_bank:
        wb_observations = fetch_world_bank_feature_observations(
            country_codes,
            start_period=start_period,
            specs=world_bank_specs,
            timeout=min(timeout, 60),
        )
        if not wb_observations.empty:
            frames.append(wb_observations)
    if not frames:
        return pd.DataFrame(
            columns=[
                "source", "feature_key", "feature_label", "quality",
                "dataset_version", "country_code", "period_label", "period",
                "value",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def latest_feature_matrix(observations: pd.DataFrame) -> pd.DataFrame:
    if observations is None or observations.empty:
        return pd.DataFrame(columns=["country_code"])
    latest = (
        observations.sort_values("period")
        .groupby(["country_code", "feature_key"], as_index=False)
        .last()
    )
    values = latest.pivot(index="country_code", columns="feature_key", values="value")
    periods = latest.pivot(index="country_code", columns="feature_key", values="period")
    periods = periods.rename(columns={col: f"{col}_period" for col in periods.columns})
    result = values.join(periods).reset_index()
    result.columns.name = None
    return result


def reer_appreciation_gap(
    observations: pd.DataFrame,
    feature_key: str = "wb_reer_index",
    window: int = 5,
) -> pd.DataFrame:
    """Latest REER vs its trailing mean, as a percentage gap, per country.

    A positive gap means the real effective exchange rate is above its recent
    average (real appreciation / competitiveness and valuation stress).
    """
    empty = pd.DataFrame(columns=["country_code", "reer_appreciation_5y_pct"])
    if observations is None or observations.empty:
        return empty
    reer = observations[observations["feature_key"] == feature_key].copy()
    if reer.empty:
        return empty
    reer = reer.sort_values("period")
    rows = []
    for country, group in reer.groupby("country_code"):
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        if len(values) < 2:
            continue
        latest = float(values.iloc[-1])
        prior = values.iloc[-(window + 1):-1]
        if prior.empty:
            continue
        baseline = float(prior.mean())
        if baseline == 0:
            continue
        rows.append({
            "country_code": country,
            "reer_appreciation_5y_pct": (latest / baseline - 1.0) * 100.0,
        })
    if not rows:
        return empty
    return pd.DataFrame(rows)


def safe_ratio(numerator: pd.Series, denominator: pd.Series, scale: float = 100.0) -> pd.Series:
    if numerator is None:
        return pd.Series(pd.NA, index=denominator.index, dtype="Float64")
    denominator = denominator.replace({0: pd.NA})
    return numerator / denominator * scale


def build_external_liquidity_features(
    observations: pd.DataFrame,
    model_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    model_country_set = None
    if model_features is None:
        model = load_model_artifact()
        model_features = model["feature_values"]
        model_country_set = set(
            model["country_scores"]["country_code"].dropna().astype(str).str.upper()
        )
    base_columns = ["country_code", "nominal_gdp"]
    if "fiscal_balance_gdp" in model_features.columns:
        base_columns.append("fiscal_balance_gdp")
    base = model_features[base_columns].copy()
    base["country_code"] = base["country_code"].astype(str).str.upper()
    if model_country_set is not None:
        base = base[base["country_code"].isin(model_country_set)]
    raw = latest_feature_matrix(observations)
    features = base.merge(raw, on="country_code", how="left")
    gdp = pd.to_numeric(features["nominal_gdp"], errors="coerce")

    def col(name: str) -> pd.Series:
        if name in features:
            return pd.to_numeric(features[name], errors="coerce")
        return pd.Series(pd.NA, index=features.index, dtype="Float64")

    derived = pd.DataFrame({"country_code": features["country_code"]})
    derived["current_account_receipts_gdp"] = safe_ratio(col("current_account_receipts_usd"), gdp)
    derived["current_account_payments_gdp"] = safe_ratio(col("current_account_payments_usd"), gdp)
    derived["current_account_balance_gdp_bop"] = safe_ratio(col("current_account_balance_usd"), gdp)
    derived["goods_services_exports_gdp"] = safe_ratio(col("goods_services_exports_usd"), gdp)
    derived["goods_services_imports_gdp"] = safe_ratio(col("goods_services_imports_usd"), gdp)
    derived["reserves_gdp_iip"] = safe_ratio(col("reserve_assets_usd"), gdp)
    derived["reserves_to_current_account_payments"] = safe_ratio(
        col("reserve_assets_usd"),
        col("current_account_payments_usd"),
    )
    derived["reserves_to_goods_services_imports"] = safe_ratio(
        col("reserve_assets_usd"),
        col("goods_services_imports_usd"),
    )
    derived["net_iip_gdp"] = safe_ratio(col("net_iip_usd"), gdp)
    derived["external_liabilities_gdp"] = safe_ratio(col("external_liabilities_usd"), gdp)
    derived["portfolio_liabilities_gdp"] = safe_ratio(col("portfolio_liabilities_usd"), gdp)
    derived["portfolio_liability_flows_gdp"] = safe_ratio(col("portfolio_liability_flows_usd"), gdp)
    derived["portfolio_net_flows_gdp"] = safe_ratio(col("portfolio_net_flows_usd"), gdp)

    # FDI flow stability (rank 19): FDI is the stable-financing counterpart to
    # fickle portfolio capital.
    fdi_liab_flows = col("fdi_liability_flows_usd")
    portfolio_liab_flows = col("portfolio_liability_flows_usd")
    derived["fdi_liability_flows_gdp"] = safe_ratio(fdi_liab_flows, gdp)
    derived["fdi_net_flows_gdp"] = safe_ratio(col("fdi_net_flows_usd"), gdp)
    # Share of gross inward financing that is FDI rather than portfolio capital.
    inward_financing = fdi_liab_flows.abs() + portfolio_liab_flows.abs()
    derived["stable_financing_share"] = safe_ratio(fdi_liab_flows.abs(), inward_financing)

    # Terms of trade and export-concentration / commodity dependence (rank 20).
    derived["terms_of_trade_index"] = col("wb_terms_of_trade_index")
    commodity_components = pd.concat(
        [
            col("wb_fuel_exports_pct"),
            col("wb_ores_metals_exports_pct"),
            col("wb_agri_raw_exports_pct"),
            col("wb_food_exports_pct"),
        ],
        axis=1,
    )
    commodity_share = commodity_components.sum(axis=1, min_count=1)
    derived["commodity_export_share_pct"] = commodity_share.clip(upper=100)

    # Real effective exchange-rate valuation stress (rank 21).
    derived["reer_index"] = col("wb_reer_index")
    reer_gap = reer_appreciation_gap(observations)
    if not reer_gap.empty:
        gap_by_country = reer_gap.set_index("country_code")["reer_appreciation_5y_pct"]
        derived["reer_appreciation_5y_pct"] = derived["country_code"].map(gap_by_country)
    else:
        derived["reer_appreciation_5y_pct"] = pd.Series(pd.NA, index=derived.index, dtype="Float64")
    derived["wb_total_external_debt_service_exports_pct"] = col(
        "wb_total_external_debt_service_exports_pct"
    )
    derived["wb_total_external_debt_service_gni_pct"] = col(
        "wb_total_external_debt_service_gni_pct"
    )
    derived["wb_ppg_external_debt_service_exports_pct"] = col(
        "wb_ppg_external_debt_service_exports_pct"
    )
    derived["wb_ppg_external_debt_service_gni_pct"] = col(
        "wb_ppg_external_debt_service_gni_pct"
    )
    derived["wb_total_external_debt_service_gdp"] = safe_ratio(
        col("wb_total_external_debt_service_usd"),
        gdp,
    )
    derived["wb_ppg_external_debt_service_gdp"] = safe_ratio(
        col("wb_ppg_external_debt_service_usd"),
        gdp,
    )
    revenue_usd = (
        col("wb_government_revenue_ex_grants_gdp_pct")
        / 100
        * gdp
    )
    derived["wb_total_external_debt_service_revenue_proxy"] = safe_ratio(
        col("wb_total_external_debt_service_usd"),
        revenue_usd,
    )
    derived["wb_ppg_external_debt_service_revenue_proxy"] = safe_ratio(
        col("wb_ppg_external_debt_service_usd"),
        revenue_usd,
    )
    derived["wb_government_interest_payments_revenue_pct"] = col(
        "wb_government_interest_payments_revenue_pct"
    )
    derived["wb_government_revenue_ex_grants_gdp_pct"] = col(
        "wb_government_revenue_ex_grants_gdp_pct"
    )

    investment_income_debits = (
        col("direct_investment_income_debits_usd").fillna(0)
        + col("portfolio_investment_income_debits_usd").fillna(0)
        + col("other_investment_income_debits_usd").fillna(0)
    )
    derived["investment_income_debits_to_cxr"] = safe_ratio(
        investment_income_debits,
        col("current_account_receipts_usd"),
    )
    current_account_deficit = -col("current_account_balance_usd")
    current_account_deficit = current_account_deficit.where(current_account_deficit > 0, 0)
    derived["gross_external_financing_need_proxy_gdp"] = safe_ratio(
        current_account_deficit + col("portfolio_liability_flows_usd").abs(),
        gdp,
    )
    fiscal_deficit = -pd.to_numeric(
        features.get("fiscal_balance_gdp", pd.Series(pd.NA, index=features.index)),
        errors="coerce",
    )
    fiscal_deficit = fiscal_deficit.where(fiscal_deficit > 0, 0)
    derived["wb_public_financing_need_ext_debt_service_proxy_gdp"] = (
        fiscal_deficit + derived["wb_ppg_external_debt_service_gdp"]
    )
    derived["external_liquidity_feature_count"] = (
        derived.drop(columns=["country_code"]).notna().sum(axis=1)
    )

    feature_cols = [c for c in derived.columns if c != "country_code"]
    model_country_count = int(base["country_code"].nunique())
    coverage_counts = {}
    for col in feature_cols:
        if col == "external_liquidity_feature_count":
            count = int((derived[col] > 0).sum())
        else:
            count = int(derived[col].notna().sum())
        coverage_counts[col] = count
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_countries": model_country_count,
        "observation_rows": int(len(observations)),
        "observation_countries": int(observations["country_code"].nunique()) if not observations.empty else 0,
        "feature_coverage": {
            col: {
                "countries": coverage_counts[col],
                "pct_model_countries": round(float(coverage_counts[col]) / model_country_count * 100, 1),
            }
            for col in feature_cols
        },
        "notes": [
            "Features are staged challenger inputs and are not wired into production scoring.",
            "gross_external_financing_need_proxy_gdp is a proxy: current-account deficit plus absolute portfolio-liability flow over GDP.",
            "WB additions intentionally avoid duplicating current-account, reserves, and broad external-liquidity metrics already covered by IMF/WEO/MFS/BOP/IIP.",
            "wb_public_financing_need_ext_debt_service_proxy_gdp is a fiscal/debt-service stress proxy: fiscal deficit plus public-and-publicly-guaranteed external debt service over GDP; it is not a classic gross financing need measure because debt service includes interest.",
            "wb_total_external_debt_service_revenue_proxy and wb_ppg_external_debt_service_revenue_proxy combine WB debt-service USD with WB revenue/GDP and model nominal GDP.",
            "investment_income_debits_to_cxr is a broad external-income-service proxy, not contractual debt service.",
            "stable_financing_share is FDI liability flows over gross inward financing (|FDI| + |portfolio| liability flows); it measures how much external financing is stable FDI rather than fickle portfolio capital.",
            "commodity_export_share_pct sums World Bank fuel, ores/metals, agricultural-raw, and food merchandise-export shares as an export-concentration / commodity-dependence proxy (capped at 100).",
            "terms_of_trade_index (WB, 2015=100) and reer_index (WB real effective exchange rate, 2010=100) are levels; reer_appreciation_5y_pct is the latest REER versus its trailing five-year mean (positive = real appreciation / valuation stress).",
            "Equity-price and property-price stress remain uncovered: no reliable public API series is wired yet (BIS/IMF/OECD/national coverage is a follow-up).",
            "QEDS/CPIS/CDIS and principal-only amortization remain separate source-family gaps pending usable current API coverage.",
        ],
    }
    return derived, report


def model_country_codes() -> list[str]:
    scores = load_model_artifact()["country_scores"]
    return sorted(scores["country_code"].dropna().astype(str).str.upper().unique().tolist())


def write_external_liquidity_outputs(
    observations: pd.DataFrame,
    features: pd.DataFrame,
    report: dict,
    observations_path: Path = EXTERNAL_FEATURE_OBSERVATIONS,
    features_path: Path = EXTERNAL_FEATURE_VALUES,
    report_path: Path = EXTERNAL_FEATURE_REPORT,
) -> None:
    observations_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    observations.to_parquet(observations_path, index=False)
    features.to_parquet(features_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
