"""Append-only prospective realization scoring for Phase 5 shadow forecasts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def stable_hash(values) -> str:
    return hashlib.sha256(
        json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def score_realizations(
    ledger_path: str | Path,
    quantiles_path: str | Path,
    current_state_path: str | Path,
    realized_states_path: str | Path,
    output_path: str | Path,
    *,
    realization_vintage: str,
) -> pd.DataFrame:
    """Score verified realized states without altering the original forecast."""
    ledger = pd.read_csv(ledger_path)
    quantiles = pd.read_csv(quantiles_path)
    current = pd.read_csv(current_state_path)
    realized = pd.read_csv(realized_states_path)
    state_columns = [f"state_{index}" for index in range(1, 97)]
    required = {"entity_code", "state_year", "data_role", *state_columns}
    if not required <= set(realized):
        raise ValueError(
            f"Realized-state input missing {sorted(required-set(realized))}"
        )
    if realized.data_role.eq("provider_projection").any():
        raise ValueError("Provider projections cannot be scored as realizations")
    if realized.duplicated(["entity_code", "state_year"]).any():
        raise ValueError("Duplicate realized state")

    point = (
        quantiles.pivot_table(
            index=["entity_code", "forecast_year", "horizon"],
            columns="state_coordinate",
            values="point",
            aggfunc="first",
        )
        .reset_index()
    )
    point_columns = [
        column for column in point if str(column).startswith("state_")
    ]
    point_columns = sorted(
        point_columns,
        key=lambda value: int(value.split("_")[-1]),
    )
    if len(point_columns) != 96:
        raise ValueError("Incomplete point forecast coordinates")
    point = point.rename(
        columns={column: f"point_{column}" for column in point_columns}
    )

    current = current[
        ["entity_code", "forecast_origin_year", *state_columns]
    ].rename(
        columns={
            "forecast_origin_year": "state_origin_year",
            **{column: f"current_{column}" for column in state_columns},
        }
    )
    realized = realized.rename(
        columns={
            "state_year": "forecast_year",
            **{column: f"realized_{column}" for column in state_columns},
        }
    )
    scored = (
        ledger.merge(
            point,
            left_on=["entity_code", "forecast_year", "horizon_years"],
            right_on=["entity_code", "forecast_year", "horizon"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            current,
            on=["entity_code", "state_origin_year"],
            how="inner",
            validate="many_to_one",
        )
        .merge(
            realized,
            on=["entity_code", "forecast_year"],
            how="inner",
            validate="one_to_one",
        )
    )
    point_values = scored[
        [f"point_{column}" for column in state_columns]
    ].to_numpy(float)
    current_values = scored[
        [f"current_{column}" for column in state_columns]
    ].to_numpy(float)
    realized_values = scored[
        [f"realized_{column}" for column in state_columns]
    ].to_numpy(float)
    error = realized_values - point_values
    baseline = realized_values - current_values
    scored["state_rmse"] = np.sqrt(np.mean(error**2, axis=1))
    scored["state_mae"] = np.mean(np.abs(error), axis=1)
    scored["no_change_rmse"] = np.sqrt(np.mean(baseline**2, axis=1))
    scored["beats_no_change"] = scored.state_rmse < scored.no_change_rmse
    scored["covered_50"] = scored.state_rmse <= scored.movement_radius_q50
    scored["covered_80"] = scored.state_rmse <= scored.movement_radius_q80
    scored["covered_95"] = scored.state_rmse <= scored.movement_radius_q95
    scored["realization_status"] = "scored"
    scored["realization_vintage"] = realization_vintage
    scored["realized_state_sha256"] = [
        stable_hash([float(value) for value in row])
        for row in realized_values
    ]
    scored["realization_record_id"] = [
        stable_hash(
            {
                "forecast_record_id": record,
                "realization_vintage": realization_vintage,
                "realized_state_sha256": realized_hash,
            }
        )[:24]
        for record, realized_hash in zip(
            scored.forecast_record_id,
            scored.realized_state_sha256,
        )
    ]
    keep = [
        "realization_record_id",
        "forecast_batch_id",
        "forecast_record_id",
        "entity_code",
        "forecast_year",
        "horizon_years",
        "realization_vintage",
        "realized_state_sha256",
        "state_rmse",
        "state_mae",
        "no_change_rmse",
        "beats_no_change",
        "covered_50",
        "covered_80",
        "covered_95",
        "realization_status",
    ]
    result = scored[keep].sort_values(
        ["entity_code", "horizon_years"]
    )
    output = Path(output_path)
    if output.exists():
        existing = pd.read_csv(output)
        overlap = set(existing.realization_record_id) & set(
            result.realization_record_id
        )
        if overlap:
            raise ValueError("Append-only realization record already exists")
        result = pd.concat([existing, result], ignore_index=True)
    result.to_csv(output, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--quantiles", required=True)
    parser.add_argument("--current-state", required=True)
    parser.add_argument("--realized-states", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--realization-vintage", required=True)
    args = parser.parse_args()
    result = score_realizations(
        args.ledger,
        args.quantiles,
        args.current_state,
        args.realized_states,
        args.output,
        realization_vintage=args.realization_vintage,
    )
    print(json.dumps({"rows": len(result), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
