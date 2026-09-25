"""Governed BoC–BoE sovereign-default source ingestion for Phase 6.

The 2025 workbook is a stock-of-debt-in-default database, not an episode table.
This module converts positive conventional-default stocks into country-year
status and derives onset episodes. Domestic fiscal arrears remain a separate
sensitivity and are never silently merged into the primary target.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

EXPECTED_SOURCE_SHA256 = (
    "5ed273650bed7d5c50cfdd04a28d2515856ae8786be52b8916af20af91cd2500"
)
SOURCE_URL = (
    "https://www.bankofcanada.ca/wp-content/uploads/2025/10/"
    "BoC-BoE-Database-2025.xlsx"
)
SOURCE_COVERAGE_START = 1960
SOURCE_COVERAGE_END = 2024
PRIMARY_STOCK_THRESHOLD_USD_M = 0.5

# Current-country naming aliases not resolved consistently by pycountry.
# Historical predecessors are deliberately not mapped to modern successors.
COUNTRY_ALIASES = {
    "BOLIVIA": "BOL",
    "BOLIVIA PLURINATIONAL STATE OF": "BOL",
    "BOSNIA HERZEGOVINA": "BIH",
    "BOSNIA AND HERZEGOVINA": "BIH",
    "BRUNEI": "BRN",
    "CABO VERDE": "CPV",
    "CAPE VERDE": "CPV",
    "CHINA PEOPLES REPUBLIC OF": "CHN",
    "CONGO DEM REP": "COD",
    "CONGO DEMOCRATIC REPUBLIC OF THE": "COD",
    "DEM REP OF CONGO KINSHASA": "COD",
    "CONGO REP": "COG",
    "CONGO REPUBLIC OF": "COG",
    "REP OF CONGO BRAZZAVILLE": "COG",
    "COTE D IVOIRE": "CIV",
    "CZECH REPUBLIC": "CZE",
    "EGYPT ARAB REP": "EGY",
    "ESWATINI": "SWZ",
    "ESWATINI SWAZILAND": "SWZ",
    "GAMBIA THE": "GMB",
    "THE GAMBIA": "GMB",
    "HONG KONG SAR CHINA": "HKG",
    "IRAN ISLAMIC REP OF": "IRN",
    "KOREA DEM PEOPLE S REP": "PRK",
    "KOREA DEMOCRATIC PEOPLE S REPUBLIC OF NORTH": "PRK",
    "KOREA REP": "KOR",
    "KOSOVO": "XKX",
    "KYRGYZ REPUBLIC": "KGZ",
    "LAO PDR": "LAO",
    "MACAO SAR CHINA": "MAC",
    "MICRONESIA FED STATES OF": "FSM",
    "MOLDOVA": "MDA",
    "RUSSIA": "RUS",
    "SAO TOME AND PRINCIPE": "STP",
    "SINT MAARTEN": "SXM",
    "SLOVAK REPUBLIC": "SVK",
    "ST KITTS NEVIS": "KNA",
    "ST LUCIA": "LCA",
    "ST VINCENT AND THE GRENADINES": "VCT",
    "SYRIA": "SYR",
    "TAIWAN PROVINCE OF CHINA": "TWN",
    "TANZANIA": "TZA",
    "TIMOR LESTE": "TLS",
    "TRINIDAD TOBAGO": "TTO",
    "TURKEY": "TUR",
    "TURKIYE": "TUR",
    "VENEZUELA": "VEN",
    "VENEZUELA RB": "VEN",
    "VIET NAM": "VNM",
    "WEST BANK AND GAZA": "PSE",
    "YEMEN REP": "YEM",
}

# These workbook rows are aggregates or historical political entities. Mapping
# them into modern ISO countries would manufacture a label and can create
# duplicate country-years. They are retained in the audit but excluded from the
# modern-country label panel and from the mapping-rate denominator.
EXCLUDED_NONCURRENT_ENTITIES = {
    "CZECHOSLOVAKIA",
    "NETHERLANDS ANTILLES",
    "USSR RUSSIAN FEDERATION",
    "WORLD",
    "YUGOSLAVIA",
}

_HEADER_ALIASES = {
    "country_name": {"COUNTRY", "DEBT_COUNTRY"},
    "country_group": {"COUNTRY_GROUP", "DEBT_COUNTRY_GROUP"},
    "year": {"YEAR", "DEBT_YEAR"},
    "primary_default_stock_usd_m": {"TOTAL_2025", "DEBT_TOTAL_2025"},
    "domestic_arrears_usd_m": {
        "FISCAL_ARREARS_2025",
        "DEBT_FISCAL_ARREARS_2025",
    },
    "local_currency_default_usd_m": {"LC_DEBT_2025", "DEBT_LC_DEBT_2025"},
    "total_government_debt_usd_m": {
        "TOTAL_DEBT_2025",
        "DEBT_TOTAL_DEBT_2025",
    },
}


@dataclass(frozen=True)
class SovereignDefaultSource:
    status: pd.DataFrame
    episodes: pd.DataFrame
    domestic_arrears_status: pd.DataFrame
    audit: dict


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().upper()
    return re.sub(r"\s+", " ", text)


def country_name_to_iso3(value: object) -> str | None:
    text = _normalize_text(value)
    if not text or text in EXCLUDED_NONCURRENT_ENTITIES:
        return None
    if re.fullmatch(r"[A-Z]{3}", text):
        return text
    if text in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[text]
    try:
        import pycountry

        result = pycountry.countries.lookup(str(value).strip())
        return str(result.alpha_3)
    except (LookupError, AttributeError):
        return None


def _match_header(value: object, aliases: set[str]) -> bool:
    return _normalize_text(value).replace(" ", "_") in aliases


def find_data_header(raw: pd.DataFrame) -> tuple[int, dict[str, int]]:
    """Find the rectangular data header, ignoring the workbook metadata rows."""
    for row_number, row in raw.iterrows():
        mapping: dict[str, int] = {}
        for logical_name, aliases in _HEADER_ALIASES.items():
            matches = [
                index for index, value in enumerate(row.tolist())
                if _match_header(value, aliases)
            ]
            if matches:
                mapping[logical_name] = int(matches[0])
        if {
            "country_name",
            "year",
            "primary_default_stock_usd_m",
        } <= set(mapping):
            return int(row_number), mapping
    raise ValueError("Could not locate the sovereign-default rectangular data header")


def _derive_episodes(status: pd.DataFrame, status_column: str) -> pd.DataFrame:
    records: list[dict] = []
    for country, group in status.groupby("country_code", observed=True):
        ordered = group.sort_values("year")
        years = ordered.year.to_numpy(dtype=int)
        active = ordered[status_column].to_numpy(dtype=bool)
        start: int | None = None
        previous_year: int | None = None
        for year, is_active in zip(years, active):
            if is_active and (start is None or previous_year is None or year != previous_year + 1):
                if start is not None and previous_year is not None:
                    records.append({
                        "country_code": country,
                        "start_year": start,
                        "end_year": previous_year,
                    })
                start = int(year)
            if not is_active and start is not None:
                records.append({
                    "country_code": country,
                    "start_year": start,
                    "end_year": int(previous_year),
                })
                start = None
            previous_year = int(year)
        if start is not None and previous_year is not None:
            records.append({
                "country_code": country,
                "start_year": start,
                "end_year": previous_year,
            })
    result = pd.DataFrame(records)
    if result.empty:
        return pd.DataFrame(columns=["country_code", "start_year", "end_year"])
    result["event_id"] = (
        result.country_code.astype(str)
        + "-"
        + result.start_year.astype(str)
        + "-"
        + result.end_year.astype(str)
    )
    return result.sort_values(["country_code", "start_year"]).reset_index(drop=True)


def parse_sovereign_default_workbook(
    path: str | Path,
    *,
    expected_sha256: str = EXPECTED_SOURCE_SHA256,
    primary_threshold_usd_m: float = PRIMARY_STOCK_THRESHOLD_USD_M,
) -> SovereignDefaultSource:
    """Parse and validate the official 2025 workbook.

    Primary default status is based on TOTAL_2025 and therefore excludes the
    separately reported fiscal-arrears series. Local-currency defaults are part
    of TOTAL_2025. Fiscal arrears are exported as a sensitivity lane.
    """
    path = Path(path)
    actual_sha = sha256(path)
    if expected_sha256 and actual_sha != expected_sha256:
        raise ValueError(
            f"Unexpected sovereign source SHA256: {actual_sha}; expected {expected_sha256}"
        )
    raw = pd.read_excel(path, sheet_name="Debt_2025", header=None, engine="openpyxl")
    header_row, mapping = find_data_header(raw)
    selected = pd.DataFrame()
    for logical_name, column_number in mapping.items():
        selected[logical_name] = raw.iloc[header_row + 1 :, column_number].to_numpy()
    selected = selected.dropna(how="all")
    selected["year"] = pd.to_numeric(selected.year, errors="coerce")
    selected["primary_default_stock_usd_m"] = pd.to_numeric(
        selected.primary_default_stock_usd_m, errors="coerce"
    )
    for optional in (
        "domestic_arrears_usd_m",
        "local_currency_default_usd_m",
        "total_government_debt_usd_m",
    ):
        if optional not in selected:
            selected[optional] = np.nan
        selected[optional] = pd.to_numeric(selected[optional], errors="coerce")
    selected = selected.loc[
        selected.year.between(SOURCE_COVERAGE_START, SOURCE_COVERAGE_END)
        & selected.primary_default_stock_usd_m.notna()
    ].copy()
    selected["year"] = selected.year.astype(int)
    selected["normalized_country_name"] = selected.country_name.map(_normalize_text)
    selected["excluded_noncurrent_entity"] = selected.normalized_country_name.isin(
        EXCLUDED_NONCURRENT_ENTITIES
    )
    selected["country_code"] = selected.country_name.map(country_name_to_iso3)

    positive_rows = selected.primary_default_stock_usd_m.gt(primary_threshold_usd_m)
    positive_eligible = positive_rows & ~selected.excluded_noncurrent_entity
    positive_mapping_rate = float(
        selected.loc[positive_eligible, "country_code"].notna().mean()
    ) if positive_eligible.any() else 0.0
    if positive_mapping_rate < 0.97:
        unmapped = sorted(
            selected.loc[
                positive_eligible & selected.country_code.isna(), "country_name"
            ].dropna().astype(str).unique().tolist()
        )
        raise ValueError(
            "Sovereign positive-row country mapping below 97% after governed "
            "historical/aggregate exclusions; "
            f"rate={positive_mapping_rate:.3f}, unmapped={unmapped[:30]}"
        )

    excluded_noncurrent_rows = selected.loc[selected.excluded_noncurrent_entity].copy()
    unmapped_rows = selected.loc[
        ~selected.excluded_noncurrent_entity & selected.country_code.isna()
    ].copy()
    status = selected.loc[
        ~selected.excluded_noncurrent_entity & selected.country_code.notna()
    ].copy()
    status["country_code"] = status.country_code.astype(str)
    if status.duplicated(["country_code", "year"]).any():
        duplicates = status.loc[
            status.duplicated(["country_code", "year"], keep=False),
            ["country_code", "year", "country_name"],
        ]
        raise ValueError(
            "Duplicate sovereign country-year rows after governed mapping: "
            + duplicates.head(20).to_dict("records").__repr__()
        )
    status["sovereign_default_active"] = status.primary_default_stock_usd_m.gt(
        primary_threshold_usd_m
    )
    status["domestic_arrears_active"] = status.domestic_arrears_usd_m.fillna(0).gt(
        primary_threshold_usd_m
    )
    status["source_sha256"] = actual_sha
    status["source_version"] = "BoC-BoE Sovereign Default Database 2025"
    status["source_coverage_end_year"] = SOURCE_COVERAGE_END

    episodes = _derive_episodes(status, "sovereign_default_active")
    episodes["source_sha256"] = actual_sha
    episodes["event_definition"] = (
        "TOTAL_2025 stock above 0.5 US$ million; fiscal arrears excluded"
    )
    arrears_episodes = _derive_episodes(status, "domestic_arrears_active")
    arrears_episodes["source_sha256"] = actual_sha
    arrears_episodes["event_definition"] = "FISCAL_ARREARS_2025 stock above 0.5 US$ million"

    if status.year.min() != SOURCE_COVERAGE_START or status.year.max() != SOURCE_COVERAGE_END:
        raise ValueError(
            f"Unexpected sovereign source year range {status.year.min()}-{status.year.max()}"
        )
    if status.country_code.nunique() < 150 or len(episodes) < 100:
        raise ValueError(
            "Sovereign source failed coverage checks: "
            f"countries={status.country_code.nunique()}, episodes={len(episodes)}"
        )
    audit = {
        "source_url": SOURCE_URL,
        "source_sha256": actual_sha,
        "sheet": "Debt_2025",
        "header_row_zero_based": header_row,
        "rows_after_year_filter": int(len(selected)),
        "mapped_rows": int(status.shape[0]),
        "mapped_countries": int(status.country_code.nunique()),
        "unmapped_rows": int(len(unmapped_rows)),
        "unmapped_names": sorted(
            unmapped_rows.country_name.dropna().astype(str).unique().tolist()
        ),
        "excluded_noncurrent_rows": int(len(excluded_noncurrent_rows)),
        "excluded_noncurrent_names": sorted(
            excluded_noncurrent_rows.country_name.dropna().astype(str).unique().tolist()
        ),
        "excluded_noncurrent_policy": (
            "aggregates and historical political entities are audited but not "
            "mapped into modern country labels"
        ),
        "positive_row_mapping_rate_after_exclusions": positive_mapping_rate,
        "primary_default_country_years": int(status.sovereign_default_active.sum()),
        "primary_default_episodes": int(len(episodes)),
        "domestic_arrears_country_years": int(status.domestic_arrears_active.sum()),
        "domestic_arrears_episodes": int(len(arrears_episodes)),
        "coverage_start_year": int(status.year.min()),
        "coverage_end_year": int(status.year.max()),
        "primary_threshold_usd_m": float(primary_threshold_usd_m),
    }
    columns = [
        "country_code",
        "country_name",
        "country_group",
        "year",
        "primary_default_stock_usd_m",
        "local_currency_default_usd_m",
        "total_government_debt_usd_m",
        "sovereign_default_active",
        "source_version",
        "source_sha256",
        "source_coverage_end_year",
    ]
    return SovereignDefaultSource(
        status=status[[column for column in columns if column in status]].sort_values(
            ["country_code", "year"]
        ).reset_index(drop=True),
        episodes=episodes,
        domestic_arrears_status=arrears_episodes,
        audit=audit,
    )
