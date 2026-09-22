"""Execute target-independent Phase 2A measurement-state research.

Reads the immutable M2 research artifact, rebuilds the target-independent annual
panel through the source cutoff, selects measurement rank using held-out
observed cells, and fits a fixed-loading missing-aware state model only if the
rank search is not unresolved at its upper boundary.

No supervised targets, crisis labels, production scores or serving models are
read or modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .discovery_execution import verify_snapshot, write_json
from .inventory import ForecastDataError
from .phase1_structure import build_structure_panel
from .phase2_measurement import (
    MaskedLinearStateModel,
    RobustObservedScaler,
    coverage_geometry_diagnostics,
    feature_reliability,
    prepare_observed_cells,
    row_information,
    select_measurement_spec,
)
from src.scripts.audit_forecasting_sources import output_destination, sha256


SOURCES = ("FSIC", "FSIBSIS", "MFS", "WEO", "WGI")


def _parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or any(value < 1 for value in values):
        raise ForecastDataError("Rank grid must contain positive integers")
    if tuple(sorted(set(values))) != values:
        raise ForecastDataError("Rank grid must be sorted and unique")
    return values


def _parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise ForecastDataError("Penalty grid must contain positive values")
    return values


def _load_target_independent_panel(snapshot: Path, cutoff_year: int):
    registry = pd.read_csv(snapshot / "complete-registry.csv", low_memory=False)

    # Membership only: the old M2 predictor file is not used for values or
    # supervised labels. It defines which entities were in the research country
    # store versus context/group entities.
    membership = pd.read_parquet(
        snapshot / "panel" / "predictors.parquet",
        columns=["entity_code"],
    )
    members = set(membership.entity_code.astype(str).unique())

    endpoint_frames = []
    for source in SOURCES:
        path = snapshot / source / "annual-endpoints.parquet"
        if not path.is_file():
            raise ForecastDataError(f"Missing annual endpoints for {source}")
        frame = pd.read_parquet(path)
        frame["source"] = source
        endpoint_frames.append(frame)
    endpoints = pd.concat(endpoint_frames, ignore_index=True)

    panel = build_structure_panel(
        endpoints,
        members,
        max_origin_year=cutoff_year,
    )
    cells, ledger, preparation = prepare_observed_cells(
        panel.predictors,
        registry,
    )
    return panel, registry, cells, ledger, preparation


def _reconstruction_diagnostics(
    model: MaskedLinearStateModel,
    scaled_cells: pd.DataFrame,
    ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction = model.predict(scaled_cells)
    data = scaled_cells.copy()
    data["prediction"] = prediction
    data = data[np.isfinite(data.prediction)].copy()
    data["sq_error"] = (data.z - data.prediction) ** 2

    source = (
        ledger[["predictor_id", "source"]]
        .drop_duplicates("predictor_id")
        .set_index("predictor_id")
        .source
    )
    data["source"] = data.predictor_id.map(source)

    by_year = (
        data.groupby("forecast_origin_year", observed=True)
        .sq_error.agg(cells="size", mse="mean")
        .reset_index()
    )
    by_year["rmse"] = np.sqrt(by_year.mse)
    by_year = by_year.drop(columns="mse")

    by_source = (
        data.groupby("source", observed=True)
        .sq_error.agg(cells="size", mse="mean")
        .reset_index()
    )
    by_source["rmse"] = np.sqrt(by_source.mse)
    by_source = by_source.drop(columns="mse")
    return by_year, by_source


def run(
    snapshot: Path,
    output: Path,
    *,
    ranks=(8, 16, 32, 64),
    l2_grid=(1e-3, 1e-2),
    holdout_fraction=0.05,
    selection_epochs=8,
    final_epochs=12,
    learning_rate=0.03,
    batch_size=65536,
    seed=17,
) -> dict:
    snapshot = Path(snapshot).resolve()
    output = output_destination(snapshot, Path(output))
    _, prior, verified = verify_snapshot(snapshot)
    output.mkdir(parents=True)

    cutoff_year = int(pd.Timestamp(prior["cutoff"]).year)
    panel, registry, cells, ledger, preparation = _load_target_independent_panel(
        snapshot,
        cutoff_year,
    )
    ledger.to_csv(output / "measurement-admission-ledger.csv", index=False)
    write_json(output / "phase2a-input-summary.json", {
        "source_cutoff": prior["cutoff"],
        "M2_files_verified": verified,
        "phase1_panel": panel.summary,
        "measurement_preparation": preparation,
        "targets_read": 0,
        "crisis_labels_read": 0,
        "production_scores_read": 0,
    })

    selection = select_measurement_spec(
        cells,
        ranks=ranks,
        l2_grid=l2_grid,
        holdout_fraction=holdout_fraction,
        epochs=selection_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
    )
    selection["results"].to_csv(output / "rank-selection.csv", index=False)

    rank_record = {
        "chosen_rank_one_se_rule": selection["chosen_rank"],
        "chosen_l2_one_se_rule": selection["chosen_l2"],
        "raw_best_rank": selection["raw_best_rank"],
        "raw_best_l2": selection["raw_best_l2"],
        "rank_search_boundary_reached": selection["rank_search_boundary_reached"],
        "zero_state_baseline": selection["zero_state_baseline"],
        "candidate_ranks": list(ranks),
        "candidate_l2": list(l2_grid),
        "holdout_fraction": holdout_fraction,
    }
    write_json(output / "rank-selection.json", rank_record)

    # A boundary result is informative but not a valid rank choice. Do not
    # pretend the largest tested rank is final merely because the grid ended.
    if selection["rank_search_boundary_reached"]:
        report = {
            "status": "phase2a_rank_search_incomplete_expand_boundary",
            "source_cutoff": prior["cutoff"],
            "inputs_verified": verified,
            "feature_count_cap": None,
            "rank_selection": rank_record,
            "targets_read": 0,
            "crisis_labels_read": 0,
            "production_scores_read": 0,
            "production_modified": False,
            "crisis_classifier_retrained": False,
            "next_action": "expand_rank_grid_before_fitting_final_measurement_state",
        }
        write_json(output / "phase2a-summary.json", report)
        write_json(
            output / "output-checksums.json",
            {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        )
        return report

    scaler = RobustObservedScaler().fit(cells)
    scaled = scaler.transform(cells)
    final_model = MaskedLinearStateModel(
        selection["chosen_rank"],
        l2=selection["chosen_l2"],
        epochs=final_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
    ).fit(scaled)

    states = final_model.state_frame()
    reliability = feature_reliability(final_model, scaled, ledger)
    information = row_information(final_model, scaled, reliability)
    coverage_diagnostics, state_audit = coverage_geometry_diagnostics(
        states,
        information,
    )
    by_year, by_source = _reconstruction_diagnostics(
        final_model,
        scaled,
        ledger,
    )

    states.to_csv(output / "country-year-states.csv.gz", index=False)
    information.to_csv(output / "country-year-information.csv.gz", index=False)
    state_audit.to_csv(output / "state-coverage-audit.csv.gz", index=False)
    reliability.to_csv(output / "feature-reliability.csv.gz", index=False)
    by_year.to_csv(output / "reconstruction-by-year.csv", index=False)
    by_source.to_csv(output / "reconstruction-by-source.csv", index=False)

    loading = pd.DataFrame(
        final_model.loadings_,
        index=pd.Index(final_model.features_, name="predictor_id"),
        columns=[f"state_{i+1}" for i in range(final_model.rank)],
    ).reset_index()
    loading = loading.merge(
        ledger,
        on="predictor_id",
        how="left",
        validate="one_to_one",
    )
    loading.to_csv(output / "measurement-loadings.csv.gz", index=False)

    scaler.stats_.reset_index().to_csv(
        output / "measurement-scaler.csv.gz",
        index=False,
    )

    report = {
        "status": "completed_phase2a_measurement_state_not_transition_validation",
        "source_cutoff": prior["cutoff"],
        "inputs_verified": verified,
        "feature_count_cap": None,
        "state_rank": final_model.rank,
        "state_rows": len(states),
        "model_features": len(final_model.features_),
        "rank_selection": rank_record,
        "coverage_geometry": coverage_diagnostics,
        "final_train_mse": final_model.train_loss_[-1],
        "targets_read": 0,
        "crisis_labels_read": 0,
        "production_scores_read": 0,
        "production_modified": False,
        "crisis_classifier_retrained": False,
        "phase2b_started": False,
        "limitations": [
            "Phase 2A establishes a measurement state, not predictive transition skill.",
            "The first uncertainty measure is a diagonal information approximation.",
            "Historical publication/vintage dates remain unverified.",
            "A fixed-loading model is the initial benchmark; regime-varying loadings remain a challenger if diagnostics require them.",
        ],
    }
    write_json(output / "phase2a-summary.json", report)
    write_json(
        output / "output-checksums.json",
        {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", default="8,16,32,64")
    parser.add_argument("--l2-grid", default="0.001,0.01")
    parser.add_argument("--holdout-fraction", type=float, default=0.05)
    parser.add_argument("--selection-epochs", type=int, default=8)
    parser.add_argument("--final-epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    report = run(
        args.snapshot,
        args.output,
        ranks=_parse_int_list(args.ranks),
        l2_grid=_parse_float_list(args.l2_grid),
        holdout_fraction=args.holdout_fraction,
        selection_epochs=args.selection_epochs,
        final_epochs=args.final_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print("PHASE2A_STATUS", json.dumps(report, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
