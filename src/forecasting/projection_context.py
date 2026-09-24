"""Separate provider projections from observed/history data.

The current-vintage IMF WEO response contains annual projections beyond the
research cutoff. These rows are useful scenario context but must never be
silently mixed into the historical measurement state, transition targets,
calibration residuals, or realized outcomes.

This module preserves the full raw row and assigns an explicit data role. It
does not infer that pre-cutoff WEO observations are verified actuals; those
remain ``historical_or_estimate_unverified`` unless a separate source-vintage
ledger establishes their publication status.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd


class ProjectionContextError(ValueError):
    """Projection data cannot be safely classified or used."""


FORBIDDEN_PROJECTION_USES = frozenset(
    {
        "measurement_state_fit",
        "transition_target",
        "transition_calibration",
        "realized_outcome",
        "historical_backtest_input",
        "model_selection",
    }
)
ALLOWED_PROJECTION_USES = frozenset(
    {
        "scenario_context",
        "provider_benchmark",
        "analyst_display",
        "conditional_forecast_input",
    }
)


@dataclass(frozen=True)
class ProjectionSplit:
    historical_or_unverified: pd.DataFrame
    provider_projections: pd.DataFrame
    metadata_only: pd.DataFrame
    rejected: pd.DataFrame
    summary: dict


def _period_end(label: object) -> pd.Timestamp:
    text = str(label).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit() or len(text) != 4:
        return pd.NaT
    return pd.Timestamp(f"{text}-12-31")


def assert_projection_use_allowed(usage: str) -> None:
    """Fail closed when provider projections are requested as realized data."""
    if usage in FORBIDDEN_PROJECTION_USES:
        raise ProjectionContextError(
            f"Provider projections cannot be used for {usage}; use observed/history data"
        )
    if usage not in ALLOWED_PROJECTION_USES:
        raise ProjectionContextError(f"Unregistered provider-projection usage: {usage}")


def split_weo_current_vintage(
    frame: pd.DataFrame,
    *,
    cutoff: str | pd.Timestamp,
    retrieved_at: str | pd.Timestamp,
    source_version: str | None = None,
) -> ProjectionSplit:
    """Classify a complete current-vintage WEO response into separate lanes.

    Annual rows whose period end is later than ``cutoff`` are provider
    projections. Rows at or before the cutoff are retained as historical or
    estimate-unverified because the current WEO feed does not provide a
    complete historical actual/estimate boundary for every indicator.

    No values are collapsed, averaged, imputed, or scaled.
    """
    required = {"COUNTRY", "INDICATOR", "TIME_PERIOD", "OBS_VALUE"}
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ProjectionContextError("A DataFrame with unique columns is required")
    if not required <= set(frame.columns):
        raise ProjectionContextError(
            f"Missing WEO projection fields: {sorted(required-set(frame.columns))}"
        )
    cutoff_ts = pd.Timestamp(cutoff)
    retrieved_ts = pd.Timestamp(retrieved_at)
    if pd.isna(cutoff_ts) or pd.isna(retrieved_ts):
        raise ProjectionContextError("Cutoff and retrieval timestamp are required")

    data = frame.copy()
    complete_key = data[["COUNTRY", "INDICATOR", "TIME_PERIOD"]].notna().all(axis=1)
    metadata = data.loc[~complete_key].copy()
    observations = data.loc[complete_key].copy()
    observations["period_end"] = observations.TIME_PERIOD.map(_period_end)
    numeric = pd.to_numeric(observations.OBS_VALUE, errors="coerce")

    invalid_period = observations.period_end.isna()
    invalid_value = observations.OBS_VALUE.notna() & ~np.isfinite(numeric)
    invalid = invalid_period | invalid_value
    rejected = observations.loc[invalid].copy()
    rejected["projection_exclusion_reason"] = np.where(
        invalid_period.loc[invalid], "invalid_period", "invalid_numeric"
    )

    clean = observations.loc[~invalid].copy()
    clean["value"] = numeric.loc[~invalid].astype(float)
    clean["source"] = "WEO"
    clean["source_version"] = source_version or "unrecorded"
    clean["retrieved_at"] = retrieved_ts
    clean["source_cutoff"] = cutoff_ts

    projection_mask = clean.period_end > cutoff_ts
    historical = clean.loc[~projection_mask].copy()
    projection = clean.loc[projection_mask].copy()

    historical["data_role"] = "historical_or_estimate_unverified"
    historical["model_admission"] = "retrospective_history_only_status_unverified"
    projection["data_role"] = "provider_projection"
    projection["model_admission"] = "scenario_only_not_measurement_or_target"
    projection["projection_provider"] = "IMF_WEO"
    projection["projection_year"] = projection.period_end.dt.year.astype(int)

    identity_columns = sorted(
        c
        for c in projection.columns
        if c
        not in {
            "OBS_VALUE",
            "value",
            "retrieved_at",
            "source_cutoff",
            "model_admission",
            "data_role",
        }
    )
    if len(projection):
        projection["projection_row_id"] = projection[identity_columns].apply(
            lambda row: hashlib.sha256(
                json.dumps(
                    {key: None if pd.isna(value) else str(value) for key, value in row.items()},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            axis=1,
        )
        duplicates = int(projection.projection_row_id.duplicated(keep=False).sum())
    else:
        projection["projection_row_id"] = pd.Series(dtype=str)
        duplicates = 0

    summary = {
        "source": "WEO",
        "source_version": source_version or "unrecorded",
        "cutoff": cutoff_ts.isoformat(),
        "retrieved_at": retrieved_ts.isoformat(),
        "input_rows": len(data),
        "metadata_only_rows": len(metadata),
        "historical_or_estimate_unverified_rows": len(historical),
        "provider_projection_rows": len(projection),
        "projection_entities": int(projection.COUNTRY.nunique()) if len(projection) else 0,
        "projection_indicators": int(projection.INDICATOR.nunique()) if len(projection) else 0,
        "projection_year_counts": {
            str(int(year)): int(count)
            for year, count in projection.projection_year.value_counts().sort_index().items()
        },
        "invalid_rows": len(rejected),
        "duplicate_projection_rows": duplicates,
        "historical_actual_status_verified": False,
        "projection_usage": sorted(ALLOWED_PROJECTION_USES),
        "forbidden_projection_usage": sorted(FORBIDDEN_PROJECTION_USES),
    }
    return ProjectionSplit(
        historical.reset_index(drop=True),
        projection.reset_index(drop=True),
        metadata.reset_index(drop=True),
        rejected.reset_index(drop=True),
        summary,
    )
