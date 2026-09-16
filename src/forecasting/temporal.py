"""Point-in-time selection and purged forward splits for research.

These utilities do not create historical vintages from a current download.
Unknown availability is a hard error in verified-vintage mode. Retrospective
mode is explicitly named and records the assumptions/revisions it uses.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .inventory import ForecastDataError, _require


@dataclass
class SelectedSnapshot:
    observations: pd.DataFrame
    audit: dict


def _dates(values) -> pd.Series:
    return pd.to_datetime(values, errors="raise", utc=True)


def select_as_of(frame: pd.DataFrame, origin: str, *,
                 mode: str = "verified_vintage",
                 assumed_lag_days: int | None = None) -> SelectedSnapshot:
    """Choose latest eligible observation, then its latest eligible vintage.

    `available_at` is public release time of this recorded value, not retrieval
    time. `vintage_at` identifies its revision. Retrospective mode permits a
    later known revision, but flags it; it never certifies real-time evidence.
    Missing selected values stay missing rather than falling back in time.
    """
    _require(frame, {"country_code", "feature_id", "observation_period", "value", "status"})
    if mode not in {"verified_vintage", "retrospective_latest_vintage"}:
        raise ForecastDataError("Unknown vintage mode")
    if mode == "retrospective_latest_vintage" and (
        not isinstance(assumed_lag_days, int) or isinstance(assumed_lag_days, bool)
        or assumed_lag_days < 0
    ):
        raise ForecastDataError("Retrospective mode requires an explicit nonnegative lag assumption")
    data = frame.copy()
    for col in ("country_code", "feature_id"):
        if data[col].isna().any() or data[col].astype(str).str.strip().eq("").any():
            raise ForecastDataError(f"Missing {col}")
        data[col] = data[col].astype(str).str.strip()
    data["country_code"] = data.country_code.str.upper()
    cutoff = pd.to_datetime(origin, utc=True)
    if pd.isna(cutoff):
        raise ForecastDataError("Missing forecast origin")
    data["observation_period"] = _dates(data.observation_period)
    if data.observation_period.isna().any():
        raise ForecastDataError("Missing observation period")
    data["value"] = pd.to_numeric(data.value, errors="raise").astype(float)
    if np.isinf(data.value).any():
        raise ForecastDataError("Infinite observation")
    for col in ("available_at", "vintage_at"):
        data[col] = _dates(data[col]) if col in data else pd.NaT
        data[col] = pd.to_datetime(data[col], utc=True)
    status = data.status.fillna("unknown").astype(str).str.lower().str.strip()
    allowed = status.isin(["actual", "estimate", "unknown"])
    data["status"] = status
    eligible = allowed & (data.observation_period <= cutoff)
    candidates = data.loc[eligible].copy()
    unknown_dates = candidates.available_at.isna() | candidates.vintage_at.isna()
    if mode == "verified_vintage":
        if unknown_dates.any():
            raise ForecastDataError("Unknown release/vintage dates: real-time selection cannot be verified")
        candidates = candidates.loc[(candidates.available_at <= cutoff) & (candidates.vintage_at <= cutoff)]
    else:
        assumed = candidates.observation_period + pd.to_timedelta(assumed_lag_days, unit="D")
        effective_release = candidates.available_at.fillna(assumed)
        candidates = candidates.loc[effective_release <= cutoff]
    # Unknown revision dates cannot be ordered against known revisions of the
    # same historical cell without an explicit upstream reconciliation.
    cell_keys = ["country_code", "feature_id", "observation_period"]
    if not candidates.empty:
        known = candidates.vintage_at.notna()
        mixed = candidates.assign(_known=known).groupby(cell_keys)._known.nunique()
        if (mixed > 1).any():
            raise ForecastDataError("Mixed known/unknown vintages within an observation")
    keys = ["country_code", "feature_id"]
    latest_period = candidates.groupby(keys).observation_period.transform("max")
    candidates = candidates.loc[candidates.observation_period.eq(latest_period)].copy()
    # Use latest revision, then release date. Unknown retrospective timestamps
    # share the same sentinel, so contradictory unknown revisions fail closed.
    sentinel = pd.Timestamp("1900-01-01", tz="UTC")
    for col in ("vintage_at", "available_at"):
        rank = candidates[col].fillna(sentinel)
        latest = candidates.assign(_rank=rank).groupby(keys)._rank.transform("max")
        candidates = candidates.loc[rank.eq(latest)].copy()
    if not candidates.empty:
        diversity = candidates.groupby(keys).value.nunique(dropna=False)
        if (diversity > 1).any():
            raise ForecastDataError("Conflicting values at identical observation/revision identity")
    # Return only canonical fields: differing irrelevant metadata cannot choose
    # an arbitrary winner. Equal measurements coalesce deterministically.
    columns = keys + ["observation_period", "available_at", "vintage_at", "value"]
    selected = candidates[columns].drop_duplicates().sort_values(keys).reset_index(drop=True)
    selected["forecast_origin"] = cutoff
    audit = {
        "mode": mode, "input_rows": len(frame), "selected_cells": len(selected),
        "excluded_non_realized_or_unknown_status_rows": int((~allowed).sum()),
        "excluded_future_period_rows": int((data.observation_period > cutoff).sum()),
        "assumed_lag_days": assumed_lag_days if mode != "verified_vintage" else None,
        "selected_unknown_release_dates": int(selected.available_at.isna().sum()),
        "selected_unknown_vintage_dates": int(selected.vintage_at.isna().sum()),
        "selected_post_origin_revisions": int((selected.vintage_at > cutoff).sum()),
        "timestamp_rules_passed": mode == "verified_vintage",
        "provenance_caveat": "Timestamp provenance must be independently established; rules do not certify metadata.",
    }
    return SelectedSnapshot(selected, audit)


@dataclass
class ForwardSplit:
    train_positions: np.ndarray
    validation_positions: np.ndarray
    audit: dict


def purged_forward_split(metadata: pd.DataFrame, validation_start: str,
                         validation_end: str, *, embargo_days: int = 0) -> ForwardSplit:
    """Require observed training labels strictly before validation starts.

    Input is one registered target/horizon. `target_available_at` must come
    from that experiment's declared availability policy; this function does
    not certify it or synthesize missing dates. Returned indices are positions.
    """
    _require(metadata, {"country_code", "forecast_origin", "target_end", "target_available_at"})
    if not isinstance(embargo_days, int) or isinstance(embargo_days, bool) or embargo_days < 0:
        raise ForecastDataError("Invalid embargo")
    data = metadata.copy()
    if data.country_code.isna().any() or data.country_code.astype(str).str.strip().eq("").any():
        raise ForecastDataError("Missing country")
    data["country_code"] = data.country_code.astype(str).str.strip().str.upper()
    for col in ("forecast_origin", "target_end", "target_available_at"):
        data[col] = _dates(data[col])
        if data[col].isna().any():
            raise ForecastDataError(f"Missing required date: {col}")
    if data.duplicated(["country_code", "forecast_origin"]).any():
        raise ForecastDataError("Duplicate country-origin: split one target/horizon at a time")
    if (data.target_end <= data.forecast_origin).any():
        raise ForecastDataError("Target must follow forecast origin")
    if (data.target_available_at < data.target_end).any():
        raise ForecastDataError("Realized target cannot be available before its end")
    start, end = pd.to_datetime(validation_start, utc=True), pd.to_datetime(validation_end, utc=True)
    if pd.isna(start) or pd.isna(end) or end < start:
        raise ForecastDataError("Invalid validation window")
    train_before = data.forecast_origin < start
    boundary = start - pd.Timedelta(days=embargo_days)
    train = train_before & (data.target_end < boundary) & (data.target_available_at < boundary)
    validation = data.forecast_origin.between(start, end)
    if not train.any() or not validation.any():
        raise ForecastDataError("Empty training or validation fold")
    return ForwardSplit(np.flatnonzero(train), np.flatnonzero(validation), {
        "train_rows": int(train.sum()), "validation_rows": int(validation.sum()),
        "purged_unresolved_or_embargoed_rows": int((train_before & ~train).sum()),
        "train_label_boundary_exclusive": boundary.isoformat(),
        "embargo_days": embargo_days,
    })
