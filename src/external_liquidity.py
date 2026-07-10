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
from src.sources.sdmx import IMF_SDMX_BASE, SDMX_CSV_ACCEPT, _parse_sdmx_period


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


def fetch_feature_observations(
    country_codes: list[str],
    start_period: str = "2005",
    specs: tuple[ExternalSeriesSpec, ...] = EXTERNAL_SERIES_SPECS,
    batch_size: int = 25,
    timeout: int = 120,
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
    base = model_features[["country_code", "nominal_gdp"]].copy()
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
            "investment_income_debits_to_cxr is a broad external-income-service proxy, not contractual debt service.",
            "QEDS/CPIS/CDIS remain separate source-family gaps pending usable current IMF API IDs or fallback providers.",
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
