"""World Bank financial-sector history used by the crisis early-warning model.

The World Bank API provides long annual series for private credit, deposits,
bank balance-sheet buffers, interest rates, and reserve adequacy.  These series
fill the pre-2007 history gap in IMF FSIC without pretending that imputed FSIC
observations were reported banking data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from src.config import BASE_DIR, CACHE_DIR


WORLD_BANK_API = "https://api.worldbank.org/v2"
DEFAULT_CACHE_PATH = Path(CACHE_DIR) / "WB_FINANCIAL_cache.parquet"


@dataclass(frozen=True)
class IndicatorSpec:
    feature: str
    label: str
    risk_direction: int
    family: str


INDICATORS: dict[str, IndicatorSpec] = {
    "FD.AST.PRVT.GD.ZS": IndicatorSpec(
        "bank_credit_gdp",
        "Domestic credit to private sector by banks (% GDP)",
        1,
        "credit_cycle",
    ),
    "FS.AST.PRVT.GD.ZS": IndicatorSpec(
        "private_credit_gdp_broad",
        "Domestic credit to private sector (% GDP)",
        1,
        "credit_cycle",
    ),
    "GFDD.SI.04": IndicatorSpec(
        "bank_credit_to_deposits",
        "Bank credit to bank deposits (%)",
        1,
        "funding",
    ),
    "GFDD.OI.02": IndicatorSpec(
        "bank_deposits_gdp",
        "Bank deposits (% GDP)",
        -1,
        "funding",
    ),
    "GFDD.DI.08": IndicatorSpec(
        "financial_system_deposits_gdp",
        "Financial-system deposits (% GDP)",
        -1,
        "funding",
    ),
    "GFDD.SI.01": IndicatorSpec(
        "bank_zscore",
        "Bank Z-score",
        -1,
        "bank_resilience",
    ),
    "FB.AST.NPER.ZS": IndicatorSpec(
        "wb_bank_npl_ratio",
        "Bank nonperforming loans to gross loans (%)",
        1,
        "bank_resilience",
    ),
    "FB.BNK.CAPA.ZS": IndicatorSpec(
        "wb_bank_capital_assets",
        "Bank capital to assets (%)",
        -1,
        "bank_resilience",
    ),
    "FD.RES.LIQU.AS.ZS": IndicatorSpec(
        "wb_bank_liquid_reserves_assets",
        "Bank liquid reserves to bank assets (%)",
        -1,
        "liquidity",
    ),
    "FM.LBL.BMNY.IR.ZS": IndicatorSpec(
        "broad_money_to_reserves",
        "Broad money to total reserves ratio",
        1,
        "external_liquidity",
    ),
    "FI.RES.TOTL.MO": IndicatorSpec(
        "reserves_months_imports",
        "Total reserves in months of imports",
        -1,
        "external_liquidity",
    ),
    "FR.INR.LEND": IndicatorSpec(
        "lending_interest_rate",
        "Lending interest rate (%)",
        1,
        "debt_service_pressure",
    ),
    "FR.INR.RINR": IndicatorSpec(
        "real_interest_rate",
        "Real interest rate (%)",
        1,
        "debt_service_pressure",
    ),
    "TX.VAL.FUEL.ZS.UN": IndicatorSpec(
        "fuel_exports_share",
        "Fuel exports (% merchandise exports)",
        1,
        "external_vulnerability",
    ),
    "TX.VAL.MMTL.ZS.UN": IndicatorSpec(
        "ores_metals_exports_share",
        "Ores and metals exports (% merchandise exports)",
        1,
        "external_vulnerability",
    ),
    "TX.VAL.AGRI.ZS.UN": IndicatorSpec(
        "agricultural_raw_exports_share",
        "Agricultural raw-material exports (% merchandise exports)",
        1,
        "external_vulnerability",
    ),
    "NY.GDP.TOTL.RT.ZS": IndicatorSpec(
        "natural_resource_rents_gdp",
        "Total natural-resource rents (% GDP)",
        1,
        "external_vulnerability",
    ),
    "NY.GDP.PETR.RT.ZS": IndicatorSpec(
        "oil_rents_gdp",
        "Oil rents (% GDP)",
        1,
        "external_vulnerability",
    ),
    "NY.GDP.MINR.RT.ZS": IndicatorSpec(
        "mineral_rents_gdp",
        "Mineral rents (% GDP)",
        1,
        "external_vulnerability",
    ),
    "TT.PRI.MRCH.XD.WD": IndicatorSpec(
        "terms_of_trade_index",
        "Net barter terms-of-trade index (2015=100)",
        -1,
        "external_vulnerability",
    ),
}


def _request_json(
    session: requests.Session,
    url: str,
    *,
    timeout: int = 90,
    attempts: int = 3,
):
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"World Bank request failed after {attempts} attempts: {url}") from last_error


def fetch_world_bank_financial_history(
    *,
    start_year: int = 1960,
    end_year: int | None = None,
    indicator_codes: Iterable[str] | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch normalized annual observations from official World Bank APIs."""

    end_year = end_year or datetime.now(timezone.utc).year
    codes = list(indicator_codes or INDICATORS)
    unknown = sorted(set(codes).difference(INDICATORS))
    if unknown:
        raise ValueError(f"Unknown World Bank financial indicators: {unknown}")

    own_session = session is None
    session = session or requests.Session()
    session.headers.update({"User-Agent": "BankEnv/2.0 (model data refresh)"})
    retrieved_at = datetime.now(timezone.utc).isoformat()
    frames: list[pd.DataFrame] = []

    try:
        for code in codes:
            url = (
                f"{WORLD_BANK_API}/country/all/indicator/{code}"
                f"?format=json&per_page=20000&date={start_year}:{end_year}"
            )
            payload = _request_json(session, url)
            if not isinstance(payload, list) or len(payload) < 2:
                raise RuntimeError(f"Unexpected World Bank response for {code}")
            metadata, records = payload[0], payload[1] or []
            spec = INDICATORS[code]
            normalized = []
            for record in records:
                value = record.get("value")
                country_code = str(record.get("countryiso3code") or "").upper()
                year_text = str(record.get("date") or "")
                if value is None or len(country_code) != 3 or not year_text.isdigit():
                    continue
                year = int(year_text)
                normalized.append(
                    {
                        "country_code": country_code,
                        "country_name": (record.get("country") or {}).get("value"),
                        "indicator_code": code,
                        "indicator_name": spec.label,
                        "feature": spec.feature,
                        "family": spec.family,
                        "risk_direction": spec.risk_direction,
                        "year": year,
                        "period": pd.Timestamp(year=year, month=12, day=31),
                        "value": float(value),
                        "source_id": str(metadata.get("sourceid") or ""),
                        "source_last_updated": metadata.get("lastupdated"),
                        "retrieved_at": retrieved_at,
                        "source_url": url,
                    }
                )
            frames.append(pd.DataFrame(normalized))
    finally:
        if own_session:
            session.close()

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(
        ["country_code", "indicator_code", "year"], keep="last"
    ).sort_values(["country_code", "indicator_code", "year"])
    return result.reset_index(drop=True)


def build_world_bank_financial_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Create strictly backward-looking annual financial-cycle features."""

    required = {"country_code", "feature", "year", "value"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"World Bank observations missing columns: {sorted(missing)}")

    panel = observations.pivot_table(
        index=["country_code", "year"],
        columns="feature",
        values="value",
        aggfunc="last",
    ).reset_index()
    panel.columns.name = None
    panel = panel.sort_values(["country_code", "year"]).reset_index(drop=True)

    # Prefer the banking-sector measure and use the broader private-credit
    # measure only when the bank-specific series is absent.
    if "bank_credit_gdp" not in panel:
        panel["bank_credit_gdp"] = np.nan
    if "private_credit_gdp_broad" in panel:
        panel["bank_credit_gdp"] = panel["bank_credit_gdp"].combine_first(
            panel["private_credit_gdp_broad"]
        )

    by_country = panel.groupby("country_code", group_keys=False)
    if "bank_credit_gdp" in panel:
        panel["bank_credit_gdp_change_3y"] = by_country["bank_credit_gdp"].diff(3)
        # A transparent, one-sided credit-cycle deviation. This is deliberately
        # not labelled as the BIS quarterly HP-filter credit gap.
        prior_trend = by_country["bank_credit_gdp"].transform(
            lambda values: values.shift(1).rolling(10, min_periods=5).median()
        )
        panel["bank_credit_gdp_gap_10y"] = panel["bank_credit_gdp"] - prior_trend

    for feature in ("lending_interest_rate", "real_interest_rate"):
        if feature in panel:
            panel[f"{feature}_change_3y"] = by_country[feature].diff(3)

    concentration_columns = [
        column
        for column in (
            "fuel_exports_share",
            "ores_metals_exports_share",
            "agricultural_raw_exports_share",
        )
        if column in panel
    ]
    if concentration_columns:
        panel["commodity_export_concentration"] = panel[
            concentration_columns
        ].max(axis=1, skipna=True)
    if "terms_of_trade_index" in panel:
        terms_change = by_country["terms_of_trade_index"].pct_change(
            periods=3, fill_method=None
        ) * 100
        panel["terms_of_trade_deterioration_3y"] = (-terms_change).clip(lower=0)
    if {
        "commodity_export_concentration",
        "terms_of_trade_deterioration_3y",
    }.issubset(panel.columns):
        panel["commodity_shock_exposure"] = (
            panel["commodity_export_concentration"]
            * panel["terms_of_trade_deterioration_3y"]
            / 100
        )

    return panel


def world_bank_feature_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Return raw and derived World Bank features in panel-builder long form."""

    features = build_world_bank_financial_features(observations)
    value_columns = [
        column
        for column in features.columns
        if column not in {"country_code", "year"}
    ]
    long = features.melt(
        id_vars=["country_code", "year"],
        value_vars=value_columns,
        var_name="indicator_code",
        value_name="value",
    ).dropna(subset=["value"])
    long["indicator_name"] = long["indicator_code"].map(
        {
            spec.feature: spec.label
            for spec in INDICATORS.values()
        }
    ).fillna(long["indicator_code"].str.replace("_", " ").str.title())
    long["period"] = pd.to_datetime(long["year"].astype(str), format="%Y")
    long["observation_status"] = "actual"
    derived_features = {
        "commodity_export_concentration",
        "terms_of_trade_deterioration_3y",
        "commodity_shock_exposure",
    }
    long["is_direct"] = ~(
        long["indicator_code"].str.endswith(("_change_3y", "_gap_10y"))
        | long["indicator_code"].isin(derived_features)
    )
    return long[
        [
            "country_code",
            "indicator_code",
            "indicator_name",
            "period",
            "value",
            "observation_status",
            "is_direct",
        ]
    ].reset_index(drop=True)


def write_world_bank_financial_history(
    observations: pd.DataFrame,
    *,
    reference_path: Path | None = None,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> tuple[Path | None, Path]:
    """Write the training cache atomically and an optional reference copy."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    observations.to_parquet(cache_tmp, index=False)
    cache_tmp.replace(cache_path)
    if reference_path is not None:
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_tmp = reference_path.with_suffix(reference_path.suffix + ".tmp")
        observations.to_parquet(reference_tmp, index=False)
        reference_tmp.replace(reference_path)
    return reference_path, cache_path
