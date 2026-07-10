"""Normalize official SDMX/World Bank pulls into app cache schemas."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from src.config import CACHE_DIR
from src.country_names import country_name_from_code
from src.data_loader import parse_period_label
from src.sources.sdmx import IMF_SDMX_BASE, SDMX_CSV_ACCEPT


UNIT_NAMES = {
    "PT": "Percent",
    "USD": "US dollar",
    "EUR": "Euro",
    "XDC": "Domestic currency",
}

COMPONENT_NAMES = {
    "CFSI": "Core FSI",
    "AFSI": "Additional FSI",
    "FSIN": "FSI Numerator",
    "FSID": "FSI Denominator",
}

TOKEN_NAMES = {
    "A": "Assets",
    "L": "Liabilities",
    "S13": "General government",
    "S1311MIXED": "Central Government",
    "S11": "Nonfinancial corporations",
    "S12R": "Other financial corporations",
    "S1": "Other sectors",
    "ODS": "Other domestic sectors",
    "NRES": "Nonresidents",
    "RESS": "Residents",
    "T": "",
    "Z": "",
}


def _name_from_code(code: dict) -> str:
    name = code.get("name") or code.get("names", {}).get("en")
    return str(name or code.get("id") or "")


@lru_cache(maxsize=None)
def fetch_codelist(agency: str, codelist_id: str) -> dict[str, str]:
    url = f"{IMF_SDMX_BASE}/structure/codelist/{agency}/{codelist_id}/+"
    response = requests.get(
        url,
        headers={"Accept": "application/vnd.sdmx.structure+json"},
        timeout=120,
    )
    if response.status_code == 204:
        return {}
    response.raise_for_status()
    codelists = response.json().get("data", {}).get("codelists", [])
    if not codelists:
        return {}
    return {
        code.get("id"): _name_from_code(code)
        for code in codelists[0].get("codes", [])
        if code.get("id")
    }


def _period_end_series(values: pd.Series) -> pd.Series:
    labels = values.astype(str).str.strip()
    labels = labels.str.replace(r"\.0$", "", regex=True)
    mapping = {
        label: parse_period_label(label)
        for label in labels.dropna().drop_duplicates()
    }
    return labels.map(mapping)


def _load_long_csv(path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype={"TIME_PERIOD": "string"},
        low_memory=False,
    )
    df = df[df.get("TIME_PERIOD").notna()].copy()
    df["value"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["period"] = _period_end_series(df["TIME_PERIOD"])
    df = df.dropna(subset=["period"])
    return df


def _append_unit(parts: list[str], unit_code) -> list[str]:
    unit = UNIT_NAMES.get(str(unit_code), str(unit_code) if pd.notna(unit_code) else "")
    if unit:
        parts.append(unit)
    return parts


def _fsic_indicator_name(df: pd.DataFrame) -> pd.Series:
    fsi_names = fetch_codelist("IMF.STA", "CL_FSI")
    fsi_stock_names = fetch_codelist("IMF.STA", "CL_FSI_STO")

    def build(row) -> str:
        base_code = row.get("FSI")
        base = fsi_names.get(base_code) or fsi_stock_names.get(base_code)
        if not base:
            base = str(row.get("INDICATOR") or "")
        component = COMPONENT_NAMES.get(str(row.get("FSI_COMPONENT")))
        stock = fsi_stock_names.get(str(row.get("FSI_STO")))
        parts = [base]
        if stock and stock != base:
            parts.append(stock)
        if component:
            parts.append(f"({component})")
        _append_unit(parts, row.get("UNIT"))
        return ", ".join(part for part in parts if part and part != "nan")

    return df.apply(build, axis=1)


def _longest_stock_prefix(indicator: str, stock_names: dict[str, str]) -> tuple[str, list[str]]:
    parts = str(indicator).split("_")
    for length in range(len(parts), 0, -1):
        prefix = "_".join(parts[:length])
        if prefix in stock_names:
            return prefix, parts[length:]
    return parts[0], parts[1:]


def _fsibsis_indicator_name(df: pd.DataFrame) -> pd.Series:
    stock_names = fetch_codelist("IMF.STA", "CL_FSI_STO")

    def build(indicator: str) -> str:
        base_code, tokens = _longest_stock_prefix(indicator, stock_names)
        parts = [stock_names.get(base_code, base_code)]
        for token in tokens:
            if token in UNIT_NAMES:
                parts.append(UNIT_NAMES[token])
            else:
                label = TOKEN_NAMES.get(token)
                if label:
                    parts.append(label)
        return ", ".join(part for part in parts if part)

    return df["INDICATOR"].astype(str).map(build)


def _indicator_names(agency: str, codelist_id: str, values: pd.Series) -> pd.Series:
    names = fetch_codelist(agency, codelist_id)
    return values.map(names).fillna("")


def normalize_imf_long_cache(source: str, csv_path, cache_dir=None) -> pd.DataFrame:
    """Convert official IMF long CSV to the parquet schema used by the app."""
    source = source.upper()
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if source == "WEO":
        usecols = [
            "STRUCTURE_ID", "COUNTRY", "INDICATOR", "FREQUENCY",
            "TIME_PERIOD", "OBS_VALUE", "UNIT", "COUNTRY_UPDATE_DATE",
        ]
        df = _load_long_csv(csv_path, usecols=usecols)
        df["indicator_name"] = _indicator_names(
            "IMF.RES", "CL_WEO_INDICATOR", df["INDICATOR"]
        )
        latest_actual = pd.to_datetime(
            df.get("COUNTRY_UPDATE_DATE"),
            errors="coerce",
        ).dt.year
        df["latest_actual_year"] = latest_actual.fillna(np.nan)
    elif source == "MFS":
        usecols = [
            "STRUCTURE_ID", "COUNTRY", "INDICATOR", "FREQUENCY",
            "TIME_PERIOD", "OBS_VALUE", "UNIT",
        ]
        df = _load_long_csv(csv_path, usecols=usecols)
        df["indicator_name"] = _indicator_names(
            "IMF.STA", "CL_MFS_DCS_INDICATOR", df["INDICATOR"]
        )
        df["latest_actual_year"] = np.nan
    elif source == "FSIC":
        usecols = [
            "STRUCTURE_ID", "COUNTRY", "SECTOR", "INDICATOR", "FREQUENCY",
            "TIME_PERIOD", "OBS_VALUE", "FSI", "FSI_STO",
            "FSI_COMPONENT", "UNIT",
        ]
        df = _load_long_csv(csv_path, usecols=usecols)
        df["indicator_name"] = _fsic_indicator_name(df)
        df["latest_actual_year"] = np.nan
    else:
        raise ValueError(f"Unsupported IMF SDMX source for cache normalization: {source}")

    unit_values = df["UNIT"].fillna("") if "UNIT" in df else ""
    country_codes = df["COUNTRY"].astype(str).str.upper().str[:3]
    name_map = {
        code: country_name_from_code(code) for code in country_codes.unique()
    }
    result = pd.DataFrame(
        {
            "country_code": country_codes,
            "country_name": country_codes.map(name_map),
            "indicator_code": df["INDICATOR"].astype(str),
            "indicator_name": df["indicator_name"].fillna(""),
            "frequency": df["FREQUENCY"].fillna(""),
            "unit": unit_values,
            "latest_actual_year": df["latest_actual_year"],
            "period_str": df["TIME_PERIOD"].astype(str),
            "value": df["value"],
            "period": df["period"],
            "dataset": source,
        }
    )

    result["observation_status"] = "unknown"
    if source == "WEO":
        year = result["period"].dt.year
        has_cutoff = result["latest_actual_year"].notna()
        result.loc[has_cutoff & (year <= result["latest_actual_year"]), "observation_status"] = "actual"
        result.loc[has_cutoff & (year == result["latest_actual_year"] + 1), "observation_status"] = "estimate"
        result.loc[has_cutoff & (year > result["latest_actual_year"] + 1), "observation_status"] = "projection"

    result = result.sort_values(["country_code", "indicator_code", "period"])
    output = cache_dir / f"{source}_cache.parquet"
    result.to_parquet(output, index=False)
    return result


def normalize_wgi_cache(csv_path, cache_dir=None) -> pd.DataFrame:
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, low_memory=False)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["country_code", "year", "feature_name", "value"])
    wide = (
        df.pivot_table(
            index=["country_code", "year"],
            columns="feature_name",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    output = cache_dir / "WGI_cache.parquet"
    wide.to_parquet(output, index=False)
    return wide


def normalize_fsibsis_cache(csv_path, cache_dir=None) -> pd.DataFrame:
    """Convert FSIBSIS SDMX long CSV to the wide balance-sheet cache shape."""
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "STRUCTURE_ID", "COUNTRY", "SECTOR", "INDICATOR", "FREQUENCY",
        "TIME_PERIOD", "OBS_VALUE",
    ]
    df = _load_long_csv(csv_path, usecols=usecols)
    df = df[df["SECTOR"].eq("S12CFSI")].copy()
    df["INDICATOR"] = _fsibsis_indicator_name(df)
    df["country_code"] = df["COUNTRY"].astype(str).str.upper().str[:3]
    wide = (
        df.pivot_table(
            index=["country_code", "COUNTRY", "SECTOR", "INDICATOR"],
            columns="TIME_PERIOD",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    period_cols = [
        col for col in wide.columns
        if pd.notna(parse_period_label(col))
    ]
    fixed_cols = ["country_code", "COUNTRY", "SECTOR", "INDICATOR"]
    wide = wide[fixed_cols + sorted(period_cols, key=parse_period_label)]
    wide["SECTOR"] = "Deposit takers"
    output = cache_dir / "FSIBSIS_cache.parquet"
    wide.to_parquet(output, index=False)
    return wide
