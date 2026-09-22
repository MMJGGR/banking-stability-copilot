"""Complete the registered Phase 2 research checkpoint in one execution.

Phase 2A selects and fits a missing-aware measurement state. Phase 2B runs
only if reconstruction, rank and coverage gates pass. No named banking target,
crisis label, production score or serving model is read or changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .discovery_execution import verify_snapshot, write_json
from .inventory import ForecastDataError
from .phase2_measurement import (
    MaskedLinearStateModel,
    RobustObservedScaler,
    coverage_geometry_diagnostics,
    evaluate_reconstruction,
    feature_reliability,
    row_information,
    select_measurement_spec,
)
from .phase2_measurement_execution import (
    _load_target_independent_panel,
    _reconstruction_diagnostics,
)
from .phase2_transition import run_transition_development
from .phase2_weighted_measurement import (
    WeightedMaskedLinearStateModel,
    reliability_weights,
    row_feature_balance,
)
from src.scripts.audit_forecasting_sources import output_destination, sha256


PHASE1_COVERAGE_BASELINE = {
    "distance_vs_coverage_spearman": 0.9483711665297827,
    "movement_vs_abs_coverage_change_spearman": 0.3569698459158592,
}


def _hash_scores(cells: pd.DataFrame) -> np.ndarray:
    scores = np.empty(len(cells), dtype=np.float64)
    for position, row in enumerate(
        cells[["entity_code", "forecast_origin_year", "predictor_id"]].itertuples(index=False)
    ):
        key = f"{row.entity_code}|{int(row.forecast_origin_year)}|{row.predictor_id}".encode()
        scores[position] = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return scores


def deterministic_selection_sample(
    cells: pd.DataFrame,
    max_cells: int,
    *,
    minimum_per_row: int = 8,
    minimum_per_feature: int = 4,
) -> pd.DataFrame:
    """Reduce rank-search cost while retaining broad row/feature support."""
    if len(cells) <= max_cells:
        return cells.copy()
    if max_cells < 1000:
        raise ForecastDataError("Selection sample is too small")
    scores = _hash_scores(cells)
    threshold = max_cells / len(cells)
    selected = scores < threshold
    work = cells[["entity_code", "forecast_origin_year", "predictor_id"]].copy()
    work["_score"] = scores
    work["_selected"] = selected

    row_key = work.entity_code.astype(str) + "|" + work.forecast_origin_year.astype(str)
    for key, minimum in ((row_key, minimum_per_row), (work.predictor_id.astype(str), minimum_per_feature)):
        for _, index in work.groupby(key, sort=False).groups.items():
            index = np.asarray(list(index), dtype=int)
            have = int(work.loc[index, "_selected"].sum())
            need = min(minimum, len(index)) - have
            if need > 0:
                candidates = index[~work.loc[index, "_selected"].to_numpy()]
                order = candidates[np.argsort(work.loc[candidates, "_score"].to_numpy())]
                work.loc[order[:need], "_selected"] = True
    return cells.loc[work._selected.to_numpy()].copy()


def _fit_measurement_mode(
    mode: str,
    train: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    rank: int,
    l2: float,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
):
    if mode == "unweighted":
        return MaskedLinearStateModel(
            rank,
            l2=l2,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed,
        ).fit(train)
    if mode == "balanced":
        return WeightedMaskedLinearStateModel(
            rank,
            l2=l2,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed,
        ).fit(train, row_feature_balance(train))
    if mode == "balanced_reliability":
        preliminary = WeightedMaskedLinearStateModel(
            rank,
            l2=l2,
            epochs=max(3, epochs // 2),
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed,
        ).fit(train, row_feature_balance(train))
        reliability = feature_reliability(preliminary, train, ledger)
        weight = reliability_weights(train, reliability)
        return WeightedMaskedLinearStateModel(
            rank,
            l2=l2,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed,
        ).fit(train, weight)
    raise ForecastDataError(f"Unknown measurement mode {mode}")


def _measurement_mode_selection(
    selection_cells: pd.DataFrame,
    selection: dict,
    ledger: pd.DataFrame,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
):
    holdout = selection["holdout_mask"]
    scaler = selection["scaler"]
    train = scaler.transform(selection_cells.loc[~holdout])
    validation = scaler.transform(selection_cells.loc[holdout])
    records = []
    models = {}
    evaluations = {}
    for mode in ("unweighted", "balanced", "balanced_reliability"):
        model = _fit_measurement_mode(
            mode,
            train,
            ledger,
            rank=selection["chosen_rank"],
            l2=selection["chosen_l2"],
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed,
        )
        evaluation = evaluate_reconstruction(model, validation)
        models[mode] = model
        evaluations[mode] = evaluation
        records.append(
            {
                "mode": mode,
                "validation_cells": evaluation["validation_cells"],
                "row_rmse_mean": evaluation["row_rmse_mean"],
                "row_rmse_se": evaluation["row_rmse_se"],
                "cell_rmse": evaluation["cell_rmse"],
                "final_train_mse": model.train_loss_[-1],
            }
        )
    result = pd.DataFrame(records).sort_values("row_rmse_mean")
    best = result.iloc[0]
    within_one_se = result.loc[
        result.row_rmse_mean <= best.row_rmse_mean + best.row_rmse_se
    ].copy()
    preference = {"balanced_reliability": 0, "balanced": 1, "unweighted": 2}
    within_one_se["_preference"] = within_one_se["mode"].map(preference)
    chosen_mode = within_one_se.sort_values(["_preference", "row_rmse_mean"]).iloc[0]["mode"]

    chosen_model = models[chosen_mode]
    chosen_evaluation = evaluations[chosen_mode]
    reliability = feature_reliability(chosen_model, train, ledger)
    information = row_information(chosen_model, train, reliability)
    heldout_row = (
        chosen_evaluation["cell_results"]
        .groupby(["entity_code", "forecast_origin_year"], observed=True)
        .sq_error.mean()
        .pow(0.5)
        .rename("heldout_row_rmse")
        .reset_index()
    )
    calibration = heldout_row.merge(
        information,
        on=["entity_code", "forecast_origin_year"],
        how="inner",
        validate="one_to_one",
    )
    uncertainty_error_correlation = float(
        calibration.heldout_row_rmse.corr(
            calibration.state_uncertainty_proxy,
            method="spearman",
        )
    )
    return {
        "results": result.drop(columns=[c for c in ["_preference"] if c in result]),
        "chosen_mode": chosen_mode,
        "uncertainty_vs_heldout_error_spearman": uncertainty_error_correlation,
    }


def _fit_final_measurement(
    cells: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    mode: str,
    rank: int,
    l2: float,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
):
    scaler = RobustObservedScaler().fit(cells)
    scaled = scaler.transform(cells)
    model = _fit_measurement_mode(
        mode,
        scaled,
        ledger,
        rank=rank,
        l2=l2,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
    )
    reliability = feature_reliability(model, scaled, ledger)
    information = row_information(model, scaled, reliability)
    states = model.state_frame()
    coverage, state_audit = coverage_geometry_diagnostics(states, information)
    by_year, by_source = _reconstruction_diagnostics(model, scaled, ledger)
    return scaler, scaled, model, reliability, information, states, coverage, state_audit, by_year, by_source


def run(
    snapshot: Path,
    output: Path,
    *,
    ranks=(8, 16, 32, 64, 96),
    l2_grid=(1e-3, 1e-2),
    max_selection_cells=1_200_000,
    holdout_fraction=0.05,
    selection_epochs=5,
    mode_epochs=6,
    final_epochs=8,
    learning_rate=0.03,
    batch_size=131072,
    seed=17,
):
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
    selection_cells = deterministic_selection_sample(cells, max_selection_cells)
    selection = select_measurement_spec(
        selection_cells,
        ranks=ranks,
        l2_grid=l2_grid,
        holdout_fraction=holdout_fraction,
        epochs=selection_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
    )
    selection["results"].to_csv(output / "rank-selection.csv", index=False)

    rank_boundary = selection["chosen_rank"] == max(ranks)
    mode = _measurement_mode_selection(
        selection_cells,
        selection,
        ledger,
        epochs=mode_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
    )
    mode["results"].to_csv(output / "measurement-mode-selection.csv", index=False)

    rank_record = {
        "candidate_ranks": list(ranks),
        "candidate_l2": list(l2_grid),
        "selection_cells": len(selection_cells),
        "total_observed_cells": len(cells),
        "chosen_rank_one_se_rule": selection["chosen_rank"],
        "chosen_l2_one_se_rule": selection["chosen_l2"],
        "raw_best_rank": selection["raw_best_rank"],
        "raw_best_l2": selection["raw_best_l2"],
        "rank_search_boundary_reached": rank_boundary,
        "zero_state_baseline": selection["zero_state_baseline"],
        "chosen_measurement_mode": mode["chosen_mode"],
        "uncertainty_vs_heldout_error_spearman": mode[
            "uncertainty_vs_heldout_error_spearman"
        ],
    }
    write_json(output / "phase2a-selection.json", rank_record)
    if rank_boundary:
        report = {
            "status": "phase2a_incomplete_rank_boundary_expand_search",
            "source_cutoff": prior["cutoff"],
            "inputs_verified": verified,
            "rank_selection": rank_record,
            "phase2b_started": False,
            "production_modified": False,
            "crisis_classifier_retrained": False,
        }
        write_json(output / "phase2-summary.json", report)
        write_json(
            output / "output-checksums.json",
            {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        )
        return report

    (
        scaler,
        scaled,
        model,
        reliability,
        information,
        states,
        coverage,
        state_audit,
        by_year,
        by_source,
    ) = _fit_final_measurement(
        cells,
        ledger,
        mode=mode["chosen_mode"],
        rank=selection["chosen_rank"],
        l2=selection["chosen_l2"],
        epochs=final_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
    )

    states.to_csv(output / "country-year-states.csv.gz", index=False)
    information.to_csv(output / "country-year-information.csv.gz", index=False)
    state_audit.to_csv(output / "state-coverage-audit.csv.gz", index=False)
    reliability.to_csv(output / "feature-reliability.csv.gz", index=False)
    by_year.to_csv(output / "reconstruction-by-year.csv", index=False)
    by_source.to_csv(output / "reconstruction-by-source.csv", index=False)
    scaler.stats_.reset_index().to_csv(output / "measurement-scaler.csv.gz", index=False)
    loading = pd.DataFrame(
        model.loadings_,
        index=pd.Index(model.features_, name="predictor_id"),
        columns=[f"state_{i+1}" for i in range(model.rank)],
    ).reset_index()
    loading.merge(
        ledger,
        on="predictor_id",
        how="left",
        validate="one_to_one",
    ).to_csv(output / "measurement-loadings.csv.gz", index=False)

    chosen_result = mode["results"].set_index("mode").loc[mode["chosen_mode"]]
    baseline_rmse = selection["zero_state_baseline"]["row_rmse_mean"]
    reconstruction_improvement = float(
        (baseline_rmse - chosen_result.row_rmse_mean) / baseline_rmse
    )
    phase2a_gates = {
        "rank_resolved": True,
        "heldout_reconstruction_beats_zero_state": bool(reconstruction_improvement > 0),
        "heldout_reconstruction_relative_improvement": reconstruction_improvement,
        "distance_coverage_materially_better_than_phase1": bool(
            abs(coverage["distance_vs_coverage_spearman"])
            < 0.85 * abs(PHASE1_COVERAGE_BASELINE["distance_vs_coverage_spearman"])
        ),
        "uncertainty_declines_with_coverage": bool(
            coverage["uncertainty_vs_coverage_spearman"] < 0
        ),
        "movement_not_dominated_by_coverage_change": bool(
            abs(coverage["movement_vs_abs_coverage_change_spearman"]) < 0.60
        ),
        "uncertainty_tracks_heldout_error": bool(
            mode["uncertainty_vs_heldout_error_spearman"] > 0
        ),
    }
    phase2a_pass = all(phase2a_gates.values())
    phase2a = {
        "status": (
            "completed_phase2a_measurement_state"
            if phase2a_pass
            else "phase2a_measurement_state_needs_revision"
        ),
        "source_cutoff": prior["cutoff"],
        "inputs_verified": verified,
        "feature_count_cap": None,
        "phase1_panel": panel.summary,
        "preparation": preparation,
        "state_rank": model.rank,
        "state_rows": len(states),
        "model_features": len(model.features_),
        "measurement_mode": mode["chosen_mode"],
        "coverage_geometry": coverage,
        "phase1_coverage_baseline": PHASE1_COVERAGE_BASELINE,
        "gates": phase2a_gates,
        "final_train_mse": model.train_loss_[-1],
        "targets_read": 0,
        "crisis_labels_read": 0,
        "production_scores_read": 0,
        "production_modified": False,
        "crisis_classifier_retrained": False,
    }
    write_json(output / "phase2a-summary.json", phase2a)

    if not phase2a_pass:
        report = {
            "status": "phase2_stopped_after_phase2a_gate_failure",
            "phase2a": phase2a,
            "phase2b_started": False,
            "production_modified": False,
            "crisis_classifier_retrained": False,
        }
        write_json(output / "phase2-summary.json", report)
        write_json(
            output / "output-checksums.json",
            {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        )
        return report

    phase2b = run_transition_development(states, information)
    phase2b["fold_metrics"].to_csv(output / "transition-fold-metrics.csv", index=False)
    phase2b["aggregate_metrics"].to_csv(
        output / "transition-aggregate-metrics.csv", index=False
    )
    phase2b["predictions"].to_csv(
        output / "transition-predictions.csv.gz", index=False
    )
    phase2b["tuning"].to_csv(output / "transition-tuning.csv.gz", index=False)
    phase2b_summary = {
        key: value
        for key, value in phase2b.items()
        if key not in {"fold_metrics", "aggregate_metrics", "predictions", "tuning"}
    }
    phase2b_summary["aggregate_metrics"] = phase2b[
        "aggregate_metrics"
    ].to_dict("records")
    write_json(output / "phase2b-summary.json", phase2b_summary)

    report = {
        "status": "completed_phase2_measurement_and_transition_development",
        "phase2a": phase2a,
        "phase2b": phase2b_summary,
        "production_modified": False,
        "crisis_classifier_retrained": False,
        "final_confirmation_evaluated": False,
        "next_phase": (
            "phase3_probabilistic_future_state_forecasting"
            if any(w["challenger_beats_no_change"] for w in phase2b["winners"])
            else "review_negative_transition_result_before_phase3"
        ),
    }
    write_json(output / "phase2-summary.json", report)
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
    parser.add_argument("--max-selection-cells", type=int, default=1_200_000)
    parser.add_argument("--selection-epochs", type=int, default=5)
    parser.add_argument("--mode-epochs", type=int, default=6)
    parser.add_argument("--final-epochs", type=int, default=8)
    args = parser.parse_args()
    report = run(
        args.snapshot,
        args.output,
        max_selection_cells=args.max_selection_cells,
        selection_epochs=args.selection_epochs,
        mode_epochs=args.mode_epochs,
        final_epochs=args.final_epochs,
    )
    print("PHASE2_STATUS", json.dumps(report, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
