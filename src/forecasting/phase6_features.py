"""Phase 6 decision-feature construction.

The fixed Phase 2 state remains the broad analytical backbone. This module adds
only the context required by the replacement decision layer:

* exact state velocity and acceleration;
* expanding own-history anomaly measures;
* crisis-specific raw measurements selected by registered semantic families;
* information quality and missingness diagnostics.

Provider projections never enter this feature frame.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .phase1_structure import build_structure_panel
from .phase2_measurement import prepare_observed_cells


SOURCES = ("FSIC", "FSIBSIS", "MFS", "WEO", "WGI")
MAX_RAW_AGE_YEARS = 3

BANKING_PATTERNS = {
    "bank_capital": (
        r"capital adequacy|regulatory capital|tier ?1|common equity|capital to assets|"
        r"risk[- ]weighted assets|leverage ratio"
    ),
    "asset_quality": (
        r"nonperform|non-performing|npl|provision|loan loss|impaired|past due|"
        r"write[- ]off"
    ),
    "bank_liquidity": (
        r"liquid assets|liquidity|customer deposits|deposit liabilities|"
        r"loans to deposits|deposit taker.*deposits|funding"
    ),
    "bank_earnings": (
        r"return on assets|return on equity|net income|profit|interest margin|"
        r"operating expenses|income before tax|income after tax"
    ),
    "credit_cycle": (
        r"credit growth|loans|claims on.*private|claims on.*nonfinancial|"
        r"real estate|household credit|corporate credit"
    ),
    "bank_sovereign_link": (
        r"claims on general government|government securities|sovereign exposure|"
        r"public sector exposure|general government.*assets"
    ),
    "fx_bank_exposure": (
        r"foreign currency.*loan|foreign currency.*liabil|net open position|"
        r"foreign exchange position|nonresident.*liabil"
    ),
}

SOVEREIGN_PATTERNS = {
    "public_debt": (
        r"general government.*debt|government debt|public debt|debt to gdp|"
        r"debt to revenue|gross debt"
    ),
    "fiscal_flow": (
        r"primary balance|fiscal balance|net lending|net borrowing|government revenue|"
        r"government expenditure|interest.*revenue|interest expense"
    ),
    "external_debt_service": (
        r"external debt service|public and publicly guaranteed|ppg|debt service|"
        r"financing need|amortization"
    ),
    "external_liquidity": (
        r"reserve assets|international reserves|reserves to imports|current account|"
        r"external liabilities|net international investment|portfolio liabilities|"
        r"imports of goods|exports of goods|foreign exchange reserves"
    ),
    "macro_pressure": (
        r"inflation|consumer prices|exchange rate|unemployment|real gdp growth|"
        r"output gap|terms of trade|commodity"
    ),
    "institutional_capacity": (
        r"government effectiveness|regulatory quality|rule of law|control of corruption|"
        r"political stability|voice and accountability"
    ),
}


def state_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column for column in frame
        if re.fullmatch(r"state_\d+", str(column))
    ]
    return sorted(columns, key=lambda column: int(column.split("_")[-1]))


def add_state_trajectory_features(
    states: pd.DataFrame,
    information: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Add exact-calendar trajectory and expanding own-history features."""
    columns = state_columns(states)
    required_information = {
        "entity_code", "forecast_origin_year", "state_uncertainty_proxy",
        "observed_share", "effective_information",
    }
    if not required_information <= set(information):
        raise ValueError("Incomplete Phase 2 information table")
    data = states.merge(
        information[list(required_information)],
        on=["entity_code", "forecast_origin_year"],
        how="left",
        validate="one_to_one",
    ).sort_values(["entity_code", "forecast_origin_year"]).reset_index(drop=True)

    velocity_columns = [f"velocity_{column}" for column in columns]
    acceleration_columns = [f"acceleration_{column}" for column in columns]
    for column, velocity_column, acceleration_column in zip(
        columns, velocity_columns, acceleration_columns
    ):
        previous = data.groupby("entity_code", observed=True)[column].shift(1)
        previous_year = data.groupby("entity_code", observed=True).forecast_origin_year.shift(1)
        velocity = data[column] - previous
        velocity = velocity.where(data.forecast_origin_year.eq(previous_year + 1))
        previous_velocity = velocity.groupby(data.entity_code, observed=True).shift(1)
        acceleration = velocity - previous_velocity
        acceleration = acceleration.where(
            velocity.notna() & previous_velocity.notna()
        )
        data[velocity_column] = velocity
        data[acceleration_column] = acceleration

    state_values = data[columns].to_numpy(dtype=float)
    velocity_values = data[velocity_columns].to_numpy(dtype=float)
    acceleration_values = data[acceleration_columns].to_numpy(dtype=float)
    data["state_norm"] = np.sqrt(np.nanmean(state_values ** 2, axis=1))
    data["velocity_norm"] = np.sqrt(np.nanmean(velocity_values ** 2, axis=1))
    data["acceleration_norm"] = np.sqrt(np.nanmean(acceleration_values ** 2, axis=1))
    data["velocity_observed_share"] = np.isfinite(velocity_values).mean(axis=1)
    data["acceleration_observed_share"] = np.isfinite(acceleration_values).mean(axis=1)

    own_anomaly = np.full(len(data), np.nan)
    own_distance = np.full(len(data), np.nan)
    prior_years = np.zeros(len(data), dtype=int)
    for _, index in data.groupby("entity_code", observed=True, sort=False).groups.items():
        positions = list(index)
        values = data.loc[positions, columns].to_numpy(dtype=float)
        for offset in range(len(values)):
            if offset < 3:
                continue
            history = values[:offset]
            current = values[offset]
            mean = np.nanmean(history, axis=0)
            scale = np.nanstd(history, axis=0, ddof=1)
            stable_scale = np.where(scale > 1e-6, scale, 1.0)
            own_anomaly[positions[offset]] = float(
                np.sqrt(np.nanmean(((current - mean) / stable_scale) ** 2))
            )
            own_distance[positions[offset]] = float(
                np.sqrt(np.nanmean((current - mean) ** 2))
            )
            prior_years[positions[offset]] = offset
    data["own_state_anomaly"] = own_anomaly
    data["own_state_distance"] = own_distance
    data["own_history_years"] = prior_years
    data["own_history_supported"] = prior_years >= 3

    # Expanding percentile of the anomaly itself; only earlier anomaly readings
    # from the same country may enter.
    own_percentile = np.full(len(data), np.nan)
    for _, index in data.groupby("entity_code", observed=True, sort=False).groups.items():
        positions = list(index)
        prior = []
        for position in positions:
            value = data.at[position, "own_state_anomaly"]
            if np.isfinite(value) and prior:
                own_percentile[position] = (
                    np.searchsorted(np.sort(prior), value, side="right") / len(prior)
                )
            if np.isfinite(value):
                prior.append(float(value))
    data["own_state_anomaly_percentile"] = own_percentile

    summary = {
        "rows": len(data),
        "countries": int(data.entity_code.nunique()),
        "state_dimensions": len(columns),
        "velocity_dimensions": len(velocity_columns),
        "own_history_supported_rows": int(data.own_history_supported.sum()),
        "provider_projection_rows_read": 0,
    }
    return data, summary


def _semantic_family(metadata: pd.DataFrame) -> pd.DataFrame:
    data = metadata.copy()
    label = (
        data.get("indicator_label", "").fillna("").astype(str)
        + " " + data.get("INDICATOR", "").fillna("").astype(str)
        + " " + data.get("feature_id", "").fillna("").astype(str)
    ).str.lower()
    data["banking_family"] = ""
    data["sovereign_family"] = ""
    for family, pattern in BANKING_PATTERNS.items():
        match = label.str.contains(pattern, regex=True, na=False)
        data.loc[match & data.banking_family.eq(""), "banking_family"] = family
    for family, pattern in SOVEREIGN_PATTERNS.items():
        match = label.str.contains(pattern, regex=True, na=False)
        data.loc[match & data.sovereign_family.eq(""), "sovereign_family"] = family
    data["decision_relevant"] = (
        data.banking_family.ne("") | data.sovereign_family.ne("")
    )
    return data


def load_decision_measurements(
    snapshot: str | Path,
    phase2_directory: str | Path,
    states: pd.DataFrame,
    *,
    maximum_origin_year: int | None = None,
    maximum_age_years: int = MAX_RAW_AGE_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Recover observed crisis-specific level measurements in Phase 2 units."""
    snapshot = Path(snapshot)
    phase2_directory = Path(phase2_directory)
    registry = pd.read_csv(snapshot / "complete-registry.csv", low_memory=False)
    members = set(states.entity_code.astype(str))
    endpoint_frames = []
    for source in SOURCES:
        path = snapshot / source / "annual-endpoints.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        frame["source"] = source
        endpoint_frames.append(frame)
    endpoints = pd.concat(endpoint_frames, ignore_index=True)
    if maximum_origin_year is None:
        maximum_origin_year = int(states.forecast_origin_year.max())
    panel = build_structure_panel(
        endpoints,
        members,
        max_origin_year=maximum_origin_year,
    )
    level = panel.predictors.loc[panel.predictors.representation.eq("level")].copy()
    cells, ledger, preparation = prepare_observed_cells(level, registry)
    scaler = pd.read_csv(phase2_directory / "measurement-scaler.csv.gz")
    cells = cells.merge(
        scaler[["predictor_id", "median", "scale"]],
        on="predictor_id",
        how="inner",
        validate="many_to_one",
    )
    cells["z"] = ((cells.model_value - cells["median"]) / cells["scale"]).clip(-8, 8)
    ledger = _semantic_family(ledger)
    relevant = set(ledger.loc[ledger.decision_relevant, "predictor_id"])
    cells = cells.loc[cells.predictor_id.isin(relevant)].copy()

    # Each measurement observed for origin year t may be carried only a short,
    # explicit number of years. The youngest eligible observation wins.
    expanded = []
    for additional_age in range(maximum_age_years + 1):
        part = cells[[
            "entity_code", "forecast_origin_year", "predictor_id", "feature_id", "z"
        ]].copy()
        part["measurement_origin_year"] = part.forecast_origin_year
        part["forecast_origin_year"] = part.forecast_origin_year + additional_age
        part["measurement_age_years"] = additional_age
        expanded.append(part)
    asof = pd.concat(expanded, ignore_index=True)
    asof = asof.loc[asof.forecast_origin_year.le(maximum_origin_year)].copy()
    asof = asof.sort_values(
        ["entity_code", "forecast_origin_year", "predictor_id", "measurement_age_years"]
    ).drop_duplicates(
        ["entity_code", "forecast_origin_year", "predictor_id"], keep="first"
    )
    asof = asof.merge(
        ledger[[
            "predictor_id", "source", "indicator_label", "INDICATOR", "UNIT",
            "banking_family", "sovereign_family",
        ]].drop_duplicates("predictor_id"),
        on="predictor_id",
        how="left",
        validate="many_to_one",
    )
    summary = {
        "registered_representations": int(len(ledger)),
        "decision_relevant_representations": int(len(relevant)),
        "asof_observed_cells": int(len(asof)),
        "countries": int(asof.entity_code.nunique()),
        "first_origin": int(asof.forecast_origin_year.min()),
        "last_origin": int(asof.forecast_origin_year.max()),
        "maximum_age_years": maximum_age_years,
        "provider_projection_rows_read": 0,
        "measurement_preparation": preparation,
    }
    return asof, ledger, summary


def pivot_decision_measurements(
    asof: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return a sparse wide raw-feature frame and registered feature groups."""
    wide = asof.pivot_table(
        index=["entity_code", "forecast_origin_year"],
        columns="predictor_id",
        values="z",
        aggfunc="last",
    ).reset_index().rename_axis(columns=None)
    groups = {
        "banking_raw": sorted(
            asof.loc[asof.banking_family.ne(""), "predictor_id"].unique()
        ),
        "sovereign_raw": sorted(
            asof.loc[asof.sovereign_family.ne(""), "predictor_id"].unique()
        ),
    }
    groups["all_raw"] = sorted(set(groups["banking_raw"]) | set(groups["sovereign_raw"]))
    return wide, groups


def merge_decision_features(
    state_panel: pd.DataFrame,
    raw_wide: pd.DataFrame,
) -> pd.DataFrame:
    return state_panel.merge(
        raw_wide,
        on=["entity_code", "forecast_origin_year"],
        how="left",
        validate="one_to_one",
    )
