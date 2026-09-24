"""Governed event-ledger and forward-target construction for Phase 6.

Banking and sovereign events use the same contamination/censoring contract but
remain separate labels. A combined event is their explicit union, never an
implicit replacement of either source.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_EPISODE_COLUMNS = {
    "country_code",
    "event_type",
    "start_year",
    "end_year",
    "source_name",
    "source_version",
}


def validate_event_ledger(episodes: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_EPISODE_COLUMNS - set(episodes)
    if missing:
        raise ValueError(f"Event ledger missing columns: {sorted(missing)}")
    frame = episodes.copy()
    frame["country_code"] = frame.country_code.astype(str).str.strip().str.upper()
    frame["event_type"] = frame.event_type.astype(str).str.strip().str.lower()
    frame["start_year"] = pd.to_numeric(frame.start_year, errors="raise").astype(int)
    frame["end_year"] = pd.to_numeric(frame.end_year, errors="raise").astype(int)
    if not frame.country_code.str.fullmatch(r"[A-Z]{3}").all():
        raise ValueError("Event ledger contains invalid country codes")
    if not frame.event_type.isin({"banking", "sovereign"}).all():
        raise ValueError("event_type must be banking or sovereign")
    if (frame.end_year < frame.start_year).any():
        raise ValueError("Event end year cannot precede start year")
    if frame.duplicated(["country_code", "event_type", "start_year", "end_year"]).any():
        raise ValueError("Duplicate event episodes")
    return frame.sort_values(["country_code", "event_type", "start_year"]).reset_index(drop=True)


@dataclass(frozen=True)
class EventTargetPolicy:
    horizon_start_years: int = 1
    horizon_end_years: int = 3
    cooldown_years: int = 3
    label_coverage_end_year: int | None = None

    def __post_init__(self):
        if self.horizon_start_years < 1:
            raise ValueError("horizon_start_years must be at least one")
        if self.horizon_end_years < self.horizon_start_years:
            raise ValueError("horizon_end_years must not precede horizon_start_years")
        if self.cooldown_years < 0:
            raise ValueError("cooldown_years cannot be negative")


def build_forward_event_targets(
    origins: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    policy: EventTargetPolicy = EventTargetPolicy(),
) -> pd.DataFrame:
    """Add banking, sovereign and union targets with explicit exclusions."""
    required = {"entity_code", "forecast_origin_year"}
    if not required <= set(origins):
        raise ValueError(f"Origin table missing columns: {sorted(required-set(origins))}")
    if origins.duplicated(["entity_code", "forecast_origin_year"]).any():
        raise ValueError("Origin rows must be unique")
    events = validate_event_ledger(episodes)
    result = origins.copy()
    result["entity_code"] = result.entity_code.astype(str).str.upper()
    result["forecast_origin_year"] = pd.to_numeric(
        result.forecast_origin_year, errors="raise"
    ).astype(int)

    for event_type in ("banking", "sovereign"):
        subset = events.loc[events.event_type.eq(event_type)]
        by_country = {
            country: group[["start_year", "end_year"]].to_numpy(dtype=int)
            for country, group in subset.groupby("country_code", observed=True)
        }
        targets = np.zeros(len(result), dtype=np.int8)
        active = np.zeros(len(result), dtype=bool)
        cooldown = np.zeros(len(result), dtype=bool)
        start_value = np.full(len(result), np.nan)
        for position, row in enumerate(
            result[["entity_code", "forecast_origin_year"]].itertuples(index=False)
        ):
            periods = by_country.get(row.entity_code)
            if periods is None:
                continue
            year = int(row.forecast_origin_year)
            starts, ends = periods[:, 0], periods[:, 1]
            active[position] = bool(np.any((starts <= year) & (year <= ends)))
            cooldown[position] = bool(
                np.any((ends < year) & (year <= ends + policy.cooldown_years))
            )
            future = starts[
                (starts >= year + policy.horizon_start_years)
                & (starts <= year + policy.horizon_end_years)
            ]
            if len(future):
                targets[position] = 1
                start_value[position] = float(np.min(future))
        result[f"{event_type}_crisis_target_1_3y"] = targets
        result[f"{event_type}_target_event_start_year"] = start_value
        result[f"{event_type}_active_event"] = active
        result[f"{event_type}_post_event_cooldown"] = cooldown

    result["right_censored"] = False
    if policy.label_coverage_end_year is not None:
        result["right_censored"] = (
            result.forecast_origin_year + policy.horizon_end_years
            > policy.label_coverage_end_year
        )
    result["event_contaminated"] = (
        result.banking_active_event
        | result.banking_post_event_cooldown
        | result.sovereign_active_event
        | result.sovereign_post_event_cooldown
    )
    result["event_target_eligible"] = ~(
        result.event_contaminated | result.right_censored
    )
    result["banking_or_sovereign_event_target_1_3y"] = (
        result.banking_crisis_target_1_3y.astype(bool)
        | result.sovereign_crisis_target_1_3y.astype(bool)
    ).astype(np.int8)
    result["event_exclusion_reason"] = np.select(
        [
            result.banking_active_event | result.sovereign_active_event,
            result.banking_post_event_cooldown | result.sovereign_post_event_cooldown,
            result.right_censored,
        ],
        ["active_event", "post_event_cooldown", "right_censored"],
        default="",
    )
    return result


def stock_panel_to_episode_sensitivity(
    stock: pd.DataFrame,
    *,
    country_col: str = "country_code",
    year_col: str = "year",
    amount_col: str = "debt_in_default_usd",
    clean_years: int = 1,
    material_increase_usd: float = 0.0,
    source_name: str = "BoC-BoE Sovereign Default Database",
    source_version: str = "unspecified",
) -> pd.DataFrame:
    """Derive onset episodes from a default-stock panel for sensitivity only.

    The governed sovereign event ledger should be reviewed separately. This
    adapter flags entry from a clean/non-default state into positive default
    stock, or a material renewed increase after the configured clean period.
    """
    if clean_years < 1 or material_increase_usd < 0:
        raise ValueError("Invalid stock-to-episode policy")
    required = {country_col, year_col, amount_col}
    if not required <= set(stock):
        raise ValueError(f"Stock panel missing columns: {sorted(required-set(stock))}")
    frame = stock[[country_col, year_col, amount_col]].copy()
    frame.columns = ["country_code", "year", "amount"]
    frame["country_code"] = frame.country_code.astype(str).str.upper()
    frame["year"] = pd.to_numeric(frame.year, errors="raise").astype(int)
    frame["amount"] = pd.to_numeric(frame.amount, errors="coerce").fillna(0).clip(lower=0)
    if frame.duplicated(["country_code", "year"]).any():
        raise ValueError("Default-stock country-years must be unique")
    records = []
    for country, group in frame.groupby("country_code", observed=True):
        group = group.sort_values("year")
        years = group.year.to_numpy()
        amount = group.amount.to_numpy(float)
        in_episode = False
        start = None
        last_positive_year = None
        for index, (year, value) in enumerate(zip(years, amount)):
            positive = value > 0
            previous_amount = amount[index - 1] if index else 0.0
            previous_clean = True
            if index:
                lower = max(0, index - clean_years)
                previous_clean = bool(np.all(amount[lower:index] <= 0))
            onset = positive and (
                previous_clean
                or (value - previous_amount >= material_increase_usd > 0)
            )
            if onset and not in_episode:
                start = int(year)
                in_episode = True
            if positive:
                last_positive_year = int(year)
            if in_episode and not positive:
                records.append((country, start, int(year) - 1))
                in_episode = False
                start = None
        if in_episode and start is not None and last_positive_year is not None:
            records.append((country, start, last_positive_year))
    return pd.DataFrame(
        records,
        columns=["country_code", "start_year", "end_year"],
    ).assign(
        event_type="sovereign",
        source_name=source_name,
        source_version=source_version,
        event_derivation="default_stock_entry_sensitivity_not_governed_episode_ledger",
    )
