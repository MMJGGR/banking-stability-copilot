"""Auditable FSIC selection, independent of source row order.

Codes below were reconciled to the official normalized 2026-09-15 source
catalog. In particular FSI626 (Tier 1) is not FSI15 (Common Equity Tier 1).
Only identical-valued observations may be coalesced across frequencies at
one period end. Conflicting values or economic dimensions fail closed;
there is no arbitrary monthly/quarterly/annual or last-row precedence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# feature: (canonical code, anchored legacy/metadata label validation)
FSIC_SERIES = {
    "capital_adequacy": ("FSI688_CFSI_PT", r"^Regulatory capital to risk-weighted assets.*Core FSI"),
    "npl_ratio": ("AQ12_CFSI_PT", r"^Nonperforming loans to total gross loans.*Core FSI"),
    "roe": ("ROE_CFSI_PT", r"^Return on equity.*Core FSI"),
    "roa": ("ROA_CFSI_PT", r"^Return on assets.*Core FSI"),
    "liquid_assets_st_liab": ("FSI765_CFSI_PT", r"^Liquid assets to short.?term liabilities.*Core FSI"),
    "liquid_assets_total": ("FSI283_LIQATTA_PT", r"^Liquid assets to total assets.*Percent"),
    # No corresponding coded percent series exists in this source vintage.
    # Retain name-only legacy support, never invent a coded replacement.
    "deposit_to_total_assets": (None, r"^Deposits to total.*assets.*Percent"),
    "customer_deposits_loans": ("FSI55_AFSI_PT", r"^Customer deposits to total.*loans.*Percent"),
    "fx_loan_exposure": ("FSI131_AFSI_PT", r"^Foreign.currency.*loans to total.*loans.*Percent"),
    "tier1_capital": ("FSI626_CFSI_PT", r"^Tier 1 capital to risk-weighted assets.*Core FSI"),
    "npl_provisions": ("AQ14_CFSI_PT", r"^Provisions to nonperforming loans.*Percent"),
    "loan_concentration": ("AQ1_CFSI_PT", r"^Loan concentration.*Percent"),
    "real_estate_loans": ("FSI524_CFSI_PT", r"^Residential real estate loans to total gross loans.*Core FSI"),
}

# Additional columns are treated as source dimensions, not silently discarded.
# Labels and retrieval metadata do not define an economic observation.
NON_DIMENSION_COLUMNS = {
    "country_code", "country_name", "indicator_code", "indicator_name",
    "frequency", "unit", "latest_actual_year", "period_str", "value",
    "period", "dataset", "observation_status", "retrieved_at",
    "source_url", "_code", "_name",
}


class FSICSelectionError(ValueError):
    """The intended FSIC observation cannot be determined unambiguously."""


def select_fsic_features(frame: pd.DataFrame, *, audit: list | None = None) -> pd.DataFrame:
    """Select already-cutoff-filtered ratios without mutating the source.

    Name-only legacy frames are accepted using anchored measurement names.
    A coded frame must use the canonical code: an unknown coded series is
    never substituted just because its label resembles a requested measure.
    Equal numeric duplicates at the same date can be collapsed; differing
    latest values (including across frequencies/statuses) raise explicitly.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["country_code"])
    missing = {"country_code", "period", "value"} - set(frame.columns)
    if missing:
        raise FSICSelectionError(f"Missing FSIC fields: {sorted(missing)}")
    data = frame.copy()
    data["period"] = pd.to_datetime(data["period"], errors="raise")
    if data["period"].isna().any():
        raise FSICSelectionError("Missing FSIC observation period")
    data["_code"] = (data["indicator_code"].fillna("").astype(str).str.strip().str.upper()
                     if "indicator_code" in data else "")
    data["_name"] = (data["indicator_name"].fillna("").astype(str).str.strip()
                     if "indicator_name" in data else "")
    results = []
    for country, country_data in data.groupby("country_code", sort=True):
        result = {"country_code": country}
        for feature, (code, pattern) in FSIC_SERIES.items():
            named = country_data["_name"].str.contains(pattern, case=False, regex=True, na=False)
            matched = country_data[country_data["_code"].eq(code)] if code else country_data.iloc[:0]
            if matched.empty:
                matched = country_data[named]
                if matched.empty:
                    continue
                unknown = sorted(set(matched["_code"]) - {""})
                if unknown:
                    raise FSICSelectionError(f"{country}/{feature}: unrecognized coded series {unknown}; expected {code}")
            elif not (matched["_name"].eq("") | matched["_name"].str.contains(pattern, case=False, regex=True)).all():
                raise FSICSelectionError(f"{country}/{feature}: code/measurement label mismatch for {code}")
            if "unit" in matched:
                units = set(matched["unit"].fillna("").astype(str).str.strip().str.upper())
                if not units <= {"PT", "PERCENT", "%"}:
                    raise FSICSelectionError(f"{country}/{feature}: incompatible or missing units {sorted(units)}")
            if "dataset" in matched:
                datasets = set(matched["dataset"].dropna().astype(str).str.strip().str.upper())
                if not datasets <= {"FSIC"}:
                    raise FSICSelectionError(f"{country}/{feature}: unexpected dataset {sorted(datasets)}")
            period = matched["period"].max()
            latest = matched[matched["period"].eq(period)]
            for dim in sorted(set(latest.columns) - NON_DIMENSION_COLUMNS):
                if latest[dim].astype(str).nunique(dropna=False) > 1:
                    raise FSICSelectionError(f"{country}/{feature}/{period.date()}: ambiguous dimension {dim}")
            values = pd.to_numeric(latest["value"], errors="raise")
            if np.isinf(values.to_numpy(dtype=float)).any():
                raise FSICSelectionError(f"{country}/{feature}: non-finite value")
            unique = sorted(values.dropna().unique())
            if len(unique) > 1:
                raise FSICSelectionError(f"{country}/{feature}/{period.date()}: conflicting latest values {unique}")
            value = float(unique[0]) if unique else np.nan
            result[feature] = value
            result[f"{feature}_year"] = period.year
            if audit is not None:
                audit.append({"country_code": str(country), "feature": feature,
                              "indicator_code": code if latest["_code"].ne("").any() else "legacy_name_only",
                              "period": period.isoformat(), "value": value if np.isfinite(value) else None,
                              "rows_coalesced": len(latest),
                              "frequencies": sorted(latest["frequency"].dropna().astype(str).unique().tolist()) if "frequency" in latest else [],
                              "policy": "identical_values_only_no_frequency_precedence"})
        if len(result) > 1:
            results.append(result)
    return pd.DataFrame(results) if results else pd.DataFrame(columns=["country_code"])
