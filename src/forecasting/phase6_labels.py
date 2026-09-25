"""Phase 6 banking and sovereign event-ledger construction.

The banking label source is the pinned Laeven-Valencia 1970-2025 episode
artifact already governed by the repository. Sovereign distress can be read
from the public World Bank 2026 debt-distress reproducibility extract and the
Bank of Canada-Bank of England sovereign-default database.

This module never reads WEO provider projections as outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


COUNTRY_OVERRIDES = {
    "bolivia": "BOL",
    "brunei darussalam": "BRN",
    "cape verde": "CPV",
    "congo, dem. rep.": "COD",
    "democratic republic of congo": "COD",
    "congo, rep.": "COG",
    "republic of congo": "COG",
    "cote d'ivoire": "CIV",
    "côte d’ivoire": "CIV",
    "egypt, arab rep.": "EGY",
    "gambia, the": "GMB",
    "hong kong sar, china": "HKG",
    "iran, islamic rep.": "IRN",
    "korea, dem. people's rep.": "PRK",
    "korea, rep.": "KOR",
    "kyrgyz republic": "KGZ",
    "lao pdr": "LAO",
    "micronesia, fed. sts.": "FSM",
    "russian federation": "RUS",
    "slovak republic": "SVK",
    "syrian arab republic": "SYR",
    "taiwan, china": "TWN",
    "turkiye": "TUR",
    "türkiye": "TUR",
    "venezuela, rb": "VEN",
    "viet nam": "VNM",
    "west bank and gaza": "PSE",
    "yemen, rep.": "YEM",
}

CODE_CANDIDATES = (
    "country_code", "iso3", "iso_3", "iso", "wbcode", "code", "ccode",
)
COUNTRY_CANDIDATES = (
    "country", "country_name", "economy", "sovereign", "name",
)
YEAR_CANDIDATES = ("year", "date", "period", "time")


@dataclass(frozen=True)
class EventPolicy:
    horizon: int = 3
    cooldown_years: int = 3
    label_coverage_end_year: int | None = None

    def __post_init__(self):
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.cooldown_years < 0:
            raise ValueError("cooldown_years must be non-negative")


def _normalise_name(value) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _country_code(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if re.fullmatch(r"[A-Za-z]{3}", text):
        return text.upper()
    normalised = _normalise_name(text)
    if normalised in COUNTRY_OVERRIDES:
        return COUNTRY_OVERRIDES[normalised]
    try:
        import pycountry

        match = pycountry.countries.lookup(text)
        return str(match.alpha_3).upper()
    except Exception:
        return None


def _first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    mapping = {_normalise_name(column): column for column in frame.columns}
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    for normalised, original in mapping.items():
        if any(candidate in normalised for candidate in candidates):
            return original
    return None


def _read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".dta":
        return pd.read_stata(path, convert_categoricals=False)
    if suffix in {".csv", ".gz"}:
        return pd.read_csv(path, low_memory=False, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    raise ValueError(f"Unsupported label source: {path}")


def _binary_candidate(frame: pd.DataFrame) -> str | None:
    preferred = []
    for column in frame.columns:
        name = _normalise_name(column)
        if any(token in name for token in ("distress", "default", "arrears")):
            values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
            if len(values) and set(np.asarray(values, dtype=float)).issubset({0.0, 1.0}):
                preferred.append(column)
    return preferred[0] if preferred else None


def parse_world_bank_debt_distress(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Parse the public non-confidential debt-distress extract.

    The registered paper uses ``t12345`` as its primary one-to-five-year debt
    distress dependent variable. The adapter accepts that name first and then
    fails closed unless it finds an unambiguous binary distress/default field.
    """
    frame = _read_table(path)
    frame.columns = [str(column) for column in frame.columns]
    code_column = _first_column(frame, CODE_CANDIDATES)
    country_column = _first_column(frame, COUNTRY_CANDIDATES)
    year_column = _first_column(frame, YEAR_CANDIDATES)
    if year_column is None:
        raise ValueError("World Bank debt-distress extract has no year column")
    target_column = "t12345" if "t12345" in frame.columns else _binary_candidate(frame)
    if target_column is None:
        raise ValueError("World Bank debt-distress extract has no unambiguous binary target")

    if code_column is not None:
        codes = frame[code_column].map(_country_code)
    elif country_column is not None:
        codes = frame[country_column].map(_country_code)
    else:
        raise ValueError("World Bank debt-distress extract has no country identity")

    result = pd.DataFrame({
        "country_code": codes,
        "year": pd.to_numeric(frame[year_column], errors="coerce"),
        "distress_status": pd.to_numeric(frame[target_column], errors="coerce"),
    }).dropna()
    result["year"] = result.year.astype(int)
    result["distress_status"] = result.distress_status.gt(0).astype("int8")
    result = result.loc[result.country_code.str.fullmatch(r"[A-Z]{3}")].copy()
    result = result.groupby(["country_code", "year"], observed=True).distress_status.max().reset_index()
    report = {
        "source": "World Bank 2026 Predicting Debt Distress public extract",
        "target_column": target_column,
        "rows": len(result),
        "countries": int(result.country_code.nunique()),
        "first_year": int(result.year.min()),
        "last_year": int(result.year.max()),
        "positive_country_years": int(result.distress_status.sum()),
    }
    return result, report


def _coerce_long_default_table(frame: pd.DataFrame, source_name: str) -> pd.DataFrame | None:
    code_column = _first_column(frame, CODE_CANDIDATES)
    country_column = _first_column(frame, COUNTRY_CANDIDATES)
    year_column = _first_column(frame, YEAR_CANDIDATES)
    if year_column is None or (code_column is None and country_column is None):
        return None
    year = pd.to_numeric(frame[year_column], errors="coerce")
    if year.notna().sum() < 3:
        return None
    codes = (
        frame[code_column].map(_country_code)
        if code_column is not None
        else frame[country_column].map(_country_code)
    )
    binary = _binary_candidate(frame)
    numeric_candidates = []
    for column in frame.columns:
        if column in {code_column, country_column, year_column, binary}:
            continue
        name = _normalise_name(column)
        if any(token in name for token in ("default", "arrears", "total external", "total")):
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().sum() >= 3:
                numeric_candidates.append(column)
    if binary is None and not numeric_candidates:
        return None
    if binary is not None:
        status = pd.to_numeric(frame[binary], errors="coerce").fillna(0).gt(0)
        amount = status.astype(float)
        method = f"binary:{binary}"
    else:
        numeric = frame[numeric_candidates].apply(pd.to_numeric, errors="coerce")
        amount = numeric.clip(lower=0).sum(axis=1, min_count=1)
        status = amount.gt(0)
        method = "amount_sum:" + ",".join(map(str, numeric_candidates))
    result = pd.DataFrame({
        "country_code": codes,
        "year": year,
        "default_amount_usd": amount,
        "any_default_status": status.astype("int8"),
        "source_table": source_name,
        "parse_method": method,
    }).dropna(subset=["country_code", "year"])
    result["year"] = result.year.astype(int)
    return result


def _coerce_wide_default_table(frame: pd.DataFrame, source_name: str) -> pd.DataFrame | None:
    # Layout A: one country column and many year columns.
    code_column = _first_column(frame, CODE_CANDIDATES)
    country_column = _first_column(frame, COUNTRY_CANDIDATES)
    identity_column = code_column or country_column
    year_columns = [
        column for column in frame.columns
        if re.fullmatch(r"(?:19|20)\d{2}", str(column).strip())
    ]
    if identity_column is not None and len(year_columns) >= 3:
        long = frame[[identity_column, *year_columns]].melt(
            id_vars=[identity_column], var_name="year", value_name="default_amount_usd"
        )
        long["country_code"] = long[identity_column].map(_country_code)
        long["year"] = pd.to_numeric(long.year, errors="coerce")
        long["default_amount_usd"] = pd.to_numeric(long.default_amount_usd, errors="coerce")
        long = long.dropna(subset=["country_code", "year", "default_amount_usd"])
        long["year"] = long.year.astype(int)
        long["any_default_status"] = long.default_amount_usd.gt(0).astype("int8")
        long["source_table"] = source_name
        long["parse_method"] = "wide_country_rows"
        return long[[
            "country_code", "year", "default_amount_usd", "any_default_status",
            "source_table", "parse_method",
        ]]

    # Layout B: one year column and many country columns.
    year_column = _first_column(frame, YEAR_CANDIDATES)
    if year_column is None:
        return None
    numeric_year = pd.to_numeric(frame[year_column], errors="coerce")
    country_columns = [
        column for column in frame.columns
        if column != year_column and _country_code(column) is not None
    ]
    if numeric_year.notna().sum() < 3 or len(country_columns) < 3:
        return None
    long = frame[[year_column, *country_columns]].melt(
        id_vars=[year_column], var_name="country_name", value_name="default_amount_usd"
    )
    long["country_code"] = long.country_name.map(_country_code)
    long["year"] = pd.to_numeric(long[year_column], errors="coerce")
    long["default_amount_usd"] = pd.to_numeric(long.default_amount_usd, errors="coerce")
    long = long.dropna(subset=["country_code", "year", "default_amount_usd"])
    long["year"] = long.year.astype(int)
    long["any_default_status"] = long.default_amount_usd.gt(0).astype("int8")
    long["source_table"] = source_name
    long["parse_method"] = "wide_country_columns"
    return long[[
        "country_code", "year", "default_amount_usd", "any_default_status",
        "source_table", "parse_method",
    ]]


def parse_boc_boe_default_source(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Parse a BoC-BoE XLSX/DTA into country-year default amounts/status.

    The official workbook has changed layout across editions. The adapter
    therefore inspects every sheet and accepts tidy or wide layouts, then sums
    creditor-class amounts by country-year. It fails closed if no credible
    country-year table can be recovered.
    """
    path = Path(path)
    tables: dict[str, pd.DataFrame]
    if path.suffix.lower() in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(path)
        tables = {}
        for sheet in workbook.sheet_names:
            found = None
            for header in range(0, 16):
                candidate = pd.read_excel(path, sheet_name=sheet, header=header)
                candidate = candidate.dropna(how="all").dropna(axis=1, how="all")
                if candidate.empty:
                    continue
                long = _coerce_long_default_table(candidate, sheet)
                if long is None:
                    long = _coerce_wide_default_table(candidate, sheet)
                if long is not None and len(long):
                    found = long
                    break
            if found is not None:
                tables[sheet] = found
    else:
        frame = _read_table(path)
        long = _coerce_long_default_table(frame, path.name)
        if long is None:
            long = _coerce_wide_default_table(frame, path.name)
        tables = {path.name: long} if long is not None else {}

    if not tables:
        raise ValueError("No country-year sovereign-default table was recovered")
    combined = pd.concat(tables.values(), ignore_index=True)
    combined = combined.loc[combined.year.between(1960, 2100)].copy()
    combined["default_amount_usd"] = pd.to_numeric(
        combined.default_amount_usd, errors="coerce"
    ).fillna(0).clip(lower=0)
    result = combined.groupby(["country_code", "year"], observed=True).agg(
        default_amount_usd=("default_amount_usd", "sum"),
        any_default_status=("any_default_status", "max"),
        source_tables=("source_table", lambda values: "|".join(sorted(set(map(str, values))))),
        parse_methods=("parse_method", lambda values: "|".join(sorted(set(map(str, values))))),
    ).reset_index()
    result["any_default_status"] = (
        result.any_default_status.astype(bool) | result.default_amount_usd.gt(0)
    ).astype("int8")
    report = {
        "source": "Bank of Canada-Bank of England Sovereign Default Database",
        "source_file": path.name,
        "parsed_tables": sorted(tables),
        "rows": len(result),
        "countries": int(result.country_code.nunique()),
        "first_year": int(result.year.min()),
        "last_year": int(result.year.max()),
        "positive_country_years": int(result.any_default_status.sum()),
    }
    return result, report


def apply_materiality_policy(
    default_frame: pd.DataFrame,
    gdp_frame: pd.DataFrame | None = None,
    *,
    absolute_floor_usd: float = 100_000_000.0,
    gdp_share_floor: float = 0.005,
) -> pd.DataFrame:
    """Flag material default while retaining the any-positive sensitivity.

    Primary materiality is met when defaulted debt exceeds either USD100m or
    0.5% of contemporaneous GDP. Both thresholds are explicit and are exported
    for sensitivity review; any-positive status is never discarded.
    """
    frame = default_frame.copy()
    if gdp_frame is not None and len(gdp_frame):
        required = {"country_code", "year", "gdp_usd"}
        if not required <= set(gdp_frame):
            raise ValueError("GDP frame must contain country_code/year/gdp_usd")
        frame = frame.merge(
            gdp_frame[list(required)], on=["country_code", "year"], how="left",
            validate="one_to_one",
        )
    else:
        frame["gdp_usd"] = np.nan
    frame["default_share_gdp"] = frame.default_amount_usd / frame.gdp_usd.replace(0, np.nan)
    frame["material_default_status"] = (
        frame.default_amount_usd.ge(absolute_floor_usd)
        | frame.default_share_gdp.ge(gdp_share_floor)
    ).astype("int8")
    frame["absolute_floor_usd"] = absolute_floor_usd
    frame["gdp_share_floor"] = gdp_share_floor
    return frame


def status_to_episodes(
    status_frame: pd.DataFrame,
    *,
    status_column: str,
    maximum_internal_gap_years: int = 1,
    event_type: str,
) -> pd.DataFrame:
    """Convert positive country-years into episodes, bridging short gaps."""
    required = {"country_code", "year", status_column}
    if not required <= set(status_frame):
        raise ValueError(f"Missing status columns: {sorted(required-set(status_frame))}")
    records = []
    for country, group in status_frame.groupby("country_code", observed=True):
        years = sorted(
            int(year) for year in group.loc[group[status_column].astype(bool), "year"].unique()
        )
        if not years:
            continue
        start = previous = years[0]
        for year in years[1:]:
            if year - previous <= maximum_internal_gap_years + 1:
                previous = year
                continue
            records.append((country, start, previous))
            start = previous = year
        records.append((country, start, previous))
    result = pd.DataFrame(records, columns=["country_code", "start_year", "end_year"])
    if result.empty:
        return pd.DataFrame(columns=[
            "country_code", "start_year", "end_year", "event_type", "event_id",
        ])
    result["event_type"] = event_type
    result["event_id"] = (
        result.event_type + ":" + result.country_code + ":"
        + result.start_year.astype(str) + "-" + result.end_year.astype(str)
    )
    return result


def load_banking_episodes(path: str | Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    required = {"country_code", "start_year", "label_end_year", "classification"}
    if not required <= set(frame):
        raise ValueError("Incomplete banking-crisis episode source")
    systemic = frame.loc[
        frame.classification.astype(str).str.lower().eq("systemic")
    ].copy()
    result = systemic.rename(columns={"label_end_year": "end_year"})[
        ["country_code", "start_year", "end_year"]
    ]
    result["country_code"] = result.country_code.astype(str).str.upper()
    result["event_type"] = "banking_crisis"
    result["event_id"] = (
        result.event_type + ":" + result.country_code + ":"
        + result.start_year.astype(int).astype(str) + "-"
        + result.end_year.astype(int).astype(str)
    )
    report = {
        "source": "Laeven-Valencia Systemic Banking Crises 1970-2025",
        "episodes": len(result),
        "countries": int(result.country_code.nunique()),
        "first_start_year": int(result.start_year.min()),
        "last_start_year": int(result.start_year.max()),
        "borderline_included": False,
    }
    return result.reset_index(drop=True), report


def build_horizon_targets(
    base_rows: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    horizons=(1, 2, 3),
    cooldown_years: int = 3,
    coverage_end_year: int | None = None,
    target_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach onset targets and exclusion reasons to country-origin rows."""
    required = {"country_code", "forecast_origin_year"}
    if not required <= set(base_rows):
        raise ValueError("Base rows need country_code and forecast_origin_year")
    data = base_rows.copy()
    by_country = {
        country: group[["start_year", "end_year"]].to_numpy(dtype=int)
        for country, group in episodes.groupby("country_code", observed=True)
    }
    exclusion_records = []
    for horizon in horizons:
        target = np.zeros(len(data), dtype=np.int8)
        active = np.zeros(len(data), dtype=bool)
        cooldown = np.zeros(len(data), dtype=bool)
        censored = np.zeros(len(data), dtype=bool)
        next_event = np.full(len(data), np.nan)
        for index, row in enumerate(
            data[["country_code", "forecast_origin_year"]].itertuples(index=False)
        ):
            year = int(row.forecast_origin_year)
            periods = by_country.get(str(row.country_code))
            if periods is not None:
                starts, ends = periods[:, 0], periods[:, 1]
                active[index] = bool(np.any((starts <= year) & (year <= ends)))
                cooldown[index] = bool(np.any((ends < year) & (year <= ends + cooldown_years)))
                future = starts[(starts >= year + 1) & (starts <= year + horizon)]
                if len(future):
                    target[index] = 1
                    next_event[index] = float(np.min(future))
            if coverage_end_year is not None and year + horizon > coverage_end_year:
                censored[index] = True
        reason = np.select(
            [active, cooldown, censored],
            ["active_event", "post_event_cooldown", "right_censored"],
            default="",
        )
        data[f"{target_prefix}_{horizon}y"] = target
        data[f"{target_prefix}_{horizon}y_event_start"] = next_event
        data[f"{target_prefix}_{horizon}y_eligible"] = reason == ""
        exclusion_records.append(pd.DataFrame({
            "country_code": data.country_code,
            "forecast_origin_year": data.forecast_origin_year,
            "event_family": target_prefix,
            "horizon": horizon,
            "exclusion_reason": reason,
        }).loc[lambda frame: frame.exclusion_reason.ne("")])
    exclusions = pd.concat(exclusion_records, ignore_index=True) if exclusion_records else pd.DataFrame()
    return data, exclusions
