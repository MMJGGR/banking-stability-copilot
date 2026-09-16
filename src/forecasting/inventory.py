"""Catalogue every available core-source measure without an economic allowlist.

Identities preserve source, measure, frequency, unit and all other available
source dimensions. The normalized caches do not necessarily retain the full
upstream SDMX identity: this module reports ambiguity, never invents metadata.
Inventory is descriptive. It neither trains a model nor chooses predictors.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterator

import numpy as np
import pandas as pd

CORE_SOURCES = ("FSIC", "FSIBSIS", "MFS", "WEO", "WGI")
PERIOD_RE = re.compile(r"^(\d{4})(?:-(Q[1-4]|M(?:0[1-9]|1[0-2])))?$")
NON_DIMENSIONS = {
    "country_code", "country_name", "COUNTRY", "indicator_name", "period",
    "period_str", "value", "latest_actual_year", "observation_status",
    "available_at", "vintage_at", "retrieved_at", "release_date", "source_url",
    "dataset", "identity_kind",
}


class ForecastDataError(ValueError):
    """Source or temporal identity cannot safely be interpreted."""


@dataclass
class Inventory:
    registry: pd.DataFrame
    conflicts: pd.DataFrame
    country_coverage: pd.DataFrame
    year_coverage: pd.DataFrame
    summary: dict


def _require(frame: pd.DataFrame, columns: set[str]) -> None:
    if not frame.columns.is_unique:
        raise ForecastDataError("Duplicate column names")
    missing = columns - set(frame.columns)
    if missing:
        raise ForecastDataError(f"Missing fields: {sorted(missing)}")


def _period(label: str) -> tuple[pd.Timestamp, str]:
    match = PERIOD_RE.fullmatch(str(label))
    if not match:
        raise ForecastDataError(f"Unsupported period: {label}")
    year, part = match.groups()
    if part is None:
        return pd.Timestamp(f"{year}-12-31"), "A"
    if part.startswith("Q"):
        return pd.Period(f"{year}{part}", freq="Q").end_time.normalize(), "Q"
    return pd.Period(f"{year}-{part[1:]}", freq="M").end_time.normalize(), "M"


def source_frames(frame: pd.DataFrame, source: str) -> Iterator[pd.DataFrame]:
    """Adapt all five cache formats, without inferring unpublished unit codes.

    Wide BSIS is processed by frequency to avoid an unnecessary full dense
    expansion. Empty wide cells remain missing, not artificial observations.
    """
    source = source.upper()
    if source not in CORE_SOURCES:
        raise ForecastDataError(f"No registered source adapter: {source}")
    _require(frame, {"country_code"})
    if source in {"FSIC", "MFS", "WEO"}:
        _require(frame, {"indicator_code", "period", "value"})
        data = frame.copy()
        for name in ("unit", "frequency", "indicator_name"):
            if name not in data:
                data[name] = None
        data["identity_kind"] = "source_code"
        yield data
    elif source == "WGI":
        _require(frame, {"year"})
        measures = [c for c in frame if c not in {"country_code", "year"}]
        if not measures or any(not pd.api.types.is_numeric_dtype(frame[c]) for c in measures):
            raise ForecastDataError("Unrecognized WGI measure schema; explicit adapter review needed")
        if frame.year.isna().any() or (frame.year.astype(float) % 1 != 0).any():
            raise ForecastDataError("Invalid WGI year")
        data = frame.melt(id_vars=["country_code", "year"], value_vars=measures,
                          var_name="indicator_code", value_name="value")
        data["period"] = pd.to_datetime(data.pop("year").astype(int).astype(str) + "-12-31")
        data["frequency"], data["unit"] = "A", None
        data["indicator_name"] = data.indicator_code
        data["identity_kind"] = "normalized_measure_name"
        yield data
    else:
        _require(frame, {"INDICATOR", "SECTOR"})
        periods = [c for c in frame if PERIOD_RE.fullmatch(str(c))]
        if not periods:
            raise ForecastDataError("No supported FSIBSIS observation columns")
        dimensions = [c for c in frame if c not in periods]
        for frequency in ("A", "Q", "M"):
            selected = [c for c in periods if _period(str(c))[1] == frequency]
            if not selected:
                continue
            data = frame.melt(id_vars=dimensions, value_vars=selected,
                              var_name="period_str", value_name="value")
            data = data.loc[data.value.notna()].copy()
            if data.empty:
                continue
            data["period"] = data.period_str.map({c: _period(str(c))[0] for c in selected})
            data["indicator_code"] = data.pop("INDICATOR")
            data["indicator_name"] = data.indicator_code
            data["frequency"], data["unit"] = frequency, None
            data["identity_kind"] = "label_only_no_verified_source_code"
            yield data


def _json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    return value.strip() if isinstance(value, str) else value


def catalogue_frame(frame: pd.DataFrame, source: str, cutoff: str) -> Inventory:
    """Summarise all present identities and isolate conflicting period values.

    No cross-frequency substitution, averaging of conflicting measurements,
    target-based selection, interpolation or imputation occurs here.
    """
    _require(frame, {"country_code", "indicator_code", "period", "value"})
    if frame.empty:
        raise ForecastDataError("Empty source partition")
    data = frame.copy()
    for field in ("unit", "frequency", "indicator_name"):
        if field not in data:
            data[field] = None
    if "identity_kind" not in data:
        data["identity_kind"] = "source_code"
    for field in ("country_code", "indicator_code"):
        if data[field].isna().any() or data[field].astype(str).str.strip().eq("").any():
            raise ForecastDataError(f"Missing identity: {field}")
        data[field] = data[field].astype(str).str.strip()
    data["country_code"] = data.country_code.str.upper()
    identity_cols = sorted(set(data.columns) - NON_DIMENSIONS)
    # Unknown extra fields are preserved as dimensions rather than discarded.
    for name in identity_cols:
        data[name] = data[name].map(_json_value).astype(object)
        data[name] = data[name].where(data[name].notna(), None)
    identities = data[identity_cols + ["identity_kind"]].drop_duplicates().copy()
    # A change of identity metadata cannot produce duplicate merge keys.
    if identities.duplicated(identity_cols).any():
        raise ForecastDataError("Inconsistent identity-kind metadata")
    identities["identity_json"] = identities[identity_cols].apply(
        lambda row: json.dumps({"source": source, **row.to_dict()}, sort_keys=True,
                               separators=(",", ":"), allow_nan=False), axis=1)
    identities["feature_id"] = identities.identity_json.map(
        lambda text: source + ":" + hashlib.sha256(text.encode()).hexdigest())
    if identities.feature_id.duplicated().any():
        raise ForecastDataError("Nonunique feature identity")
    data = data.merge(identities[identity_cols + ["feature_id"]], on=identity_cols,
                      how="left", validate="many_to_one")
    data["_period"] = pd.to_datetime(data.period, errors="coerce", utc=True)
    numeric = pd.to_numeric(data.value, errors="coerce")
    data["_value"] = numeric.astype(float)
    data["_valid"] = np.isfinite(data._value)
    if "observation_status" in data:
        data["_status"] = data.observation_status.fillna("unknown").astype(str).str.lower().str.strip()
    else:
        data["_status"] = "unknown"
    cutoff_date = pd.to_datetime(cutoff, utc=True)
    if pd.isna(cutoff_date):
        raise ForecastDataError("Missing cutoff")
    realized_status = data._status.isin(["actual", "estimate", "unknown"])
    eligible = data._valid & data._period.notna() & (data._period <= cutoff_date) & realized_status
    obs = data.loc[eligible, ["country_code", "feature_id", "_period", "_value"]].copy()
    keys = ["country_code", "feature_id", "_period"]
    cells = obs.groupby(keys, observed=True, sort=True)._value.agg(
        rows="size", distinct_values="nunique", minimum="min", maximum="max").reset_index()
    conflicts = cells.loc[cells.distinct_values > 1].copy()
    clean = cells.loc[cells.distinct_values == 1].copy()
    clean["year"] = clean._period.dt.year
    registry = identities.set_index("feature_id")
    all_stats = data.groupby("feature_id", observed=True).agg(
        raw_observations=("value", "size"), countries=("country_code", "nunique"))
    registry = registry.join(all_stats)
    registry["source"] = source
    registry["labels_json"] = data.groupby("feature_id", observed=True).indicator_name.agg(
        lambda s: json.dumps(sorted(set(s.dropna().astype(str)))))
    registry["conflicting_cells"] = conflicts.groupby("feature_id").size().reindex(registry.index, fill_value=0)
    registry["unambiguous_retrospective_cells"] = clean.groupby("feature_id").size().reindex(registry.index, fill_value=0)
    registry["unit_missing"] = registry.unit.isna() | registry.unit.astype(str).str.strip().eq("")
    registry["dimension_status"] = "available_cache_dimensions_only_upstream_review_required"
    registry["model_admission"] = "not_assessed_inventory_only"
    coverage = clean.groupby(["feature_id", "country_code"], observed=True).agg(
        observed_years=("year", "nunique"), observations=("_period", "size"),
        first_period=("_period", "min"), last_period=("_period", "max")).reset_index()
    yearly = clean.groupby(["feature_id", "year"], observed=True).agg(
        countries=("country_code", "nunique"), observations=("_period", "size")).reset_index()
    summary = {
        "source": source, "input_observations": len(data), "features": len(registry),
        "countries": int(data.country_code.nunique()),
        "invalid_numeric_rows": int((data.value.notna() & ~data._valid).sum()),
        "missing_numeric_rows": int(data.value.isna().sum()),
        "invalid_period_rows": int(data._period.isna().sum()),
        "after_cutoff_rows": int((data._period > cutoff_date).sum()),
        "non_realized_status_rows": int((~realized_status).sum()),
        "unknown_status_rows": int(data._status.eq("unknown").sum()),
        "eligible_numeric_rows_before_conflicts": int(eligible.sum()),
        "conflicting_observation_cells": len(conflicts),
        "rows_in_conflicting_cells": int(conflicts.rows.sum()),
        "equal_duplicate_rows_coalesced": int((clean.rows - 1).sum()),
        "unambiguous_retrospective_cells": len(clean),
        "features_with_missing_units": int(registry.unit_missing.sum()),
        "public_availability_rows": int(data.available_at.notna().sum()) if "available_at" in data else 0,
        "vintage_date_rows": int(data.vintage_at.notna().sum()) if "vintage_at" in data else 0,
        "forecast_validation_status": "not_evaluated",
    }
    return Inventory(registry.reset_index().sort_values("feature_id").reset_index(drop=True),
                     conflicts.rename(columns={"_period": "observation_period"}).reset_index(drop=True),
                     coverage, yearly, summary)


def inventory_source(frame: pd.DataFrame, source: str, cutoff: str) -> Inventory:
    source = source.upper()
    parts = [catalogue_frame(part, source, cutoff) for part in source_frames(frame, source)]
    if not parts:
        raise ForecastDataError(f"No source partitions for {source}")
    summary = dict(parts[0].summary)
    for key, value in summary.items():
        if isinstance(value, int) and not isinstance(value, bool):
            summary[key] = sum(part.summary[key] for part in parts)
    registry = pd.concat([p.registry for p in parts], ignore_index=True)
    if registry.feature_id.duplicated().any():
        raise ForecastDataError("Duplicate identities across source partitions")
    summary["countries"] = int(frame.country_code.nunique())
    summary["input_table_rows"] = len(frame)
    if source == "FSIBSIS":
        summary["unobserved_measure_labels"] = sorted(
            set(frame.INDICATOR.dropna().astype(str)) - set(registry.indicator_code))
    return Inventory(registry, pd.concat([p.conflicts for p in parts], ignore_index=True),
                     pd.concat([p.country_coverage for p in parts], ignore_index=True),
                     pd.concat([p.year_coverage for p in parts], ignore_index=True), summary)
