"""Calculated time-series helpers for the Data Explorer.

These utilities keep Data Explorer calculations explicit and auditable. They
only align observations with the same country, date, and frequency; callers can
decide how to present missing observations instead of silently filling gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalculationSpec:
    """Human-readable description of a calculated series."""

    formula: str
    mode: str
    notes: tuple[str, ...] = ()


ALIGN_KEYS = ["country_code", "date", "frequency"]


def normalize_observation_frame(
    frame: pd.DataFrame,
    indicator_value,
    indicator_col: str,
    label: str,
) -> pd.DataFrame:
    """Return a normalized single-indicator observation frame.

    Required input columns are ``country_code``, ``period`` and ``value``.
    ``frequency`` is preserved when available; otherwise annual frequency is
    assumed to prevent mixed-frequency alignment.
    """
    required = {"country_code", "period", "value", indicator_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Observation frame is missing columns: {sorted(missing)}")

    data = frame.loc[frame[indicator_col] == indicator_value].copy()
    if data.empty:
        return pd.DataFrame(columns=ALIGN_KEYS + ["value", "indicator_label"])

    data["date"] = pd.to_datetime(data["period"].astype(str), errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    if "frequency" not in data.columns:
        data["frequency"] = "A"
    data["frequency"] = data["frequency"].fillna("A").astype(str)
    data = data.dropna(subset=["date", "value"])
    data = data.sort_values("date").drop_duplicates(
        subset=ALIGN_KEYS,
        keep="last",
    )
    data["indicator_label"] = label
    keep = ALIGN_KEYS + ["value", "indicator_label"]
    optional = [
        column
        for column in ("observation_status", "period")
        if column in data.columns
    ]
    return data[keep + optional].reset_index(drop=True)


def restrict_frequency(frame: pd.DataFrame, frequency: str | None) -> pd.DataFrame:
    """Restrict a normalized frame to one frequency when requested."""
    if frequency is None or "frequency" not in frame.columns:
        return frame.copy()
    return frame.loc[frame["frequency"] == frequency].copy()


def filter_time_range(frame: pd.DataFrame, time_range: str) -> pd.DataFrame:
    """Filter a normalized frame to a display range."""
    if frame.empty or time_range == "All Data":
        return frame.copy()
    max_date = frame["date"].max()
    years = {
        "5 Years": 5,
        "10 Years": 10,
        "20 Years": 20,
    }.get(time_range)
    if years is None:
        return frame.copy()
    return frame.loc[frame["date"] >= max_date - pd.DateOffset(years=years)].copy()


def compute_ratio(
    numerator: pd.DataFrame,
    denominator: pd.DataFrame,
    scale: float = 1.0,
) -> pd.DataFrame:
    """Compute ``numerator / denominator * scale`` on exact aligned periods."""
    merged = numerator.merge(
        denominator,
        on=ALIGN_KEYS,
        how="inner",
        suffixes=("_numerator", "_denominator"),
    )
    if merged.empty:
        return pd.DataFrame(
            columns=ALIGN_KEYS
            + [
                "value",
                "numerator_value",
                "denominator_value",
                "calculation_flag",
            ]
        )

    merged["numerator_value"] = pd.to_numeric(
        merged["value_numerator"],
        errors="coerce",
    )
    merged["denominator_value"] = pd.to_numeric(
        merged["value_denominator"],
        errors="coerce",
    )
    valid = (
        merged["numerator_value"].notna()
        & merged["denominator_value"].notna()
        & (merged["denominator_value"] != 0)
    )
    merged["value"] = np.where(
        valid,
        merged["numerator_value"] / merged["denominator_value"] * scale,
        np.nan,
    )
    merged["calculation_flag"] = np.where(valid, "calculated", "invalid_denominator")
    return merged[
        ALIGN_KEYS
        + [
            "value",
            "numerator_value",
            "denominator_value",
            "calculation_flag",
        ]
    ].dropna(subset=["value"]).reset_index(drop=True)


def compute_cross_sectional_share(
    frame: pd.DataFrame,
    group_keys: Iterable[str] = ("date", "frequency"),
    scale: float = 100.0,
) -> pd.DataFrame:
    """Compute country share of the selected-country group total by period."""
    data = frame.copy()
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["value"])
    group_keys = list(group_keys)
    totals = (
        data.groupby(group_keys, dropna=False)["value"]
        .sum()
        .rename("group_total")
        .reset_index()
    )
    data = data.merge(totals, on=group_keys, how="left")
    valid = data["group_total"].notna() & (data["group_total"] != 0)
    data["raw_value"] = data["value"]
    data["value"] = np.where(valid, data["raw_value"] / data["group_total"] * scale, np.nan)
    data["calculation_flag"] = np.where(valid, "calculated", "invalid_group_total")
    return data.dropna(subset=["value"]).reset_index(drop=True)


def compute_temporal_change(
    frame: pd.DataFrame,
    mode: str,
    group_cols: Iterable[str] = ("country_code",),
) -> pd.DataFrame:
    """Compute period change, base-period change, or rebased index.

    ``mode`` values:
    - ``period_pct``: period-over-period percent change
    - ``base_pct``: percent change from first observation in each group
    - ``index_100``: first observation in each group equals 100
    """
    if mode not in {"period_pct", "base_pct", "index_100"}:
        raise ValueError(f"Unsupported temporal mode: {mode}")

    data = frame.copy()
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["date", "value"]).sort_values(list(group_cols) + ["date"])
    group_cols = list(group_cols)

    if mode == "period_pct":
        data["raw_value"] = data["value"]
        data["value"] = data.groupby(group_cols)["raw_value"].pct_change() * 100
        data["calculation_flag"] = "period_pct_change"
    else:
        first = data.groupby(group_cols)["value"].transform("first")
        valid = first.notna() & (first != 0)
        data["raw_value"] = data["value"]
        if mode == "base_pct":
            data["value"] = np.where(valid, (data["raw_value"] / first - 1) * 100, np.nan)
            data["calculation_flag"] = "base_pct_change"
        else:
            data["value"] = np.where(valid, data["raw_value"] / first * 100, np.nan)
            data["calculation_flag"] = "index_100"

    return data.dropna(subset=["value"]).reset_index(drop=True)


def available_frequencies(frame: pd.DataFrame) -> list[str]:
    """Return supported frequencies in display order."""
    if "frequency" not in frame.columns:
        return []
    present = set(frame["frequency"].dropna().astype(str))
    return [frequency for frequency in ("M", "Q", "A") if frequency in present]

