"""Resolve the Phase 2A rank boundary, then complete Phase 2B if gates pass.

This execution reuses the validated lower-rank results from run 35744506248,
fits only the unresolved higher-rank candidates, calibrates state uncertainty
by artificial masking, and preserves the target-independent/production
firewalls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .discovery_execution import verify_snapshot, write_json
from .inventory import ForecastDataError
from .phase2_full_execution import (
    PHASE1_COVERAGE_BASELINE,
    _fit_final_measurement,
    _load_target_independent_panel,
    _measurement_mode_selection,
    deterministic_selection_sample,
)
from .phase2_measurement import select_measurement_spec
from .phase2_transition import run_transition_development
from .phase2_uncertainty import masked_state_calibration
from src.scripts.audit_forecasting_sources import output_destination, sha256


EXTENSION_RANKS = (128, 160, 192, 256, 320)
EXTENSION_L2 = (0.01,)


def _combined_choice(
    prior: pd.DataFrame,
    extension: pd.DataFrame,
) -> dict:
    columns = [
        "rank",
        "l2",
        "validation_cells",
        "row_rmse_mean",
        "row_rmse_se",
        "cell_rmse",
        "final_train_mse",
    ]
    combined = (
        pd.concat([prior[columns], extension[columns]], ignore_index=True)
        .drop_duplicates(["rank", "l2"], keep="last")
        .sort_values(["rank", "l2"])
        .reset_index(drop=True)
    )
    best = combined.loc[combined.row_rmse_mean.idxmin()]
    tolerance = max(float(best.row_rmse_se), 0.005 * float(best.row_rmse_mean))
    eligible = combined.loc[
        combined.row_rmse_mean <= best.row_rmse_mean + tolerance
    ].copy()
    chosen = eligible.sort_values(
        ["rank", "l2"],
        ascending=[True, False],
    ).iloc[0]
    maximum_rank = int(combined["rank"].max())
    return {
        "combined": combined,
        "chosen_rank": int(chosen["rank"]),
        "chosen_l2": float(chosen["l2"]),
        "raw_best_rank": int(best["rank"]),
        "raw_best_l2": float(best["l2"]),
        "selection_tolerance": tolerance,
        "rank_search_boundary_reached": int(chosen["rank"]) == maximum_rank,
        "maximum_rank_tested": maximum_rank,
    }


def run(
    snapshot: Path,
    prior_results: Path,
    output: Path,
    *,
    max_selection_cells=1_200_000,
    extension_epochs=5,
    mode_epochs=4,
    final_epochs=6,
    learning_rate=0.03,
    batch_size=131072,
    seed=17,
):
    snapshot = Path(snapshot).resolve()
    prior_results = Path(prior_results).resolve()
    output = output_destination(snapshot, Path(output))
    _, source_report, verified = verify_snapshot(snapshot)
    output.mkdir(parents=True)

    prior_rank_path = prior_results / "rank-selection.csv"
    prior_selection_path = prior_results / "phase2a-selection.json"
    if not prior_rank_path.is_file() or not prior_selection_path.is_file():
        raise ForecastDataError("Prior rank-boundary evidence is incomplete")
    prior_rank = pd.read_csv(prior_rank_path)
    prior_selection = json.loads(prior_selection_path.read_text())
    if not prior_selection.get("rank_search_boundary_reached"):
        raise ForecastDataError("Prior evidence is not an unresolved boundary")

    cutoff_year = int(pd.Timestamp(source_report["cutoff"]).year)
    panel, registry, cells, ledger, preparation = _load_target_independent_panel(
        snapshot,
        cutoff_year,
    )
    ledger.to_csv(output / "measurement-admission-ledger.csv", index=False)
    selection_cells = deterministic_selection_sample(
        cells,
        max_selection_cells,
    )
    extension = select_measurement_spec(
        selection_cells,
        ranks=EXTENSION_RANKS,
        l2_grid=EXTENSION_L2,
        holdout_fraction=0.05,
        epochs=extension_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
    )
    decision = _combined_choice(prior_rank, extension["results"])
    decision["combined"].to_csv(output / "rank-selection-combined.csv", index=False)
    extension["results"].to_csv(output / "rank-selection-extension.csv", index=False)

    selection = {
        "chosen_rank": decision["chosen_rank"],
        "chosen_l2": decision["chosen_l2"],
        "holdout_mask": extension["holdout_mask"],
        "scaler": extension["scaler"],
    }
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
        "prior_run": 35744506248,
        "prior_artifact": 10701178100,
        "prior_candidate_ranks": prior_selection["candidate_ranks"],
        "extension_ranks": list(EXTENSION_RANKS),
        "extension_l2": list(EXTENSION_L2),
        "selection_cells": len(selection_cells),
        "total_observed_cells": len(cells),
        "chosen_rank": decision["chosen_rank"],
        "chosen_l2": decision["chosen_l2"],
        "raw_best_rank": decision["raw_best_rank"],
        "raw_best_l2": decision["raw_best_l2"],
        "selection_tolerance": decision["selection_tolerance"],
        "maximum_rank_tested": decision["maximum_rank_tested"],
        "rank_search_boundary_reached": decision[
            "rank_search_boundary_reached"
        ],
        "chosen_measurement_mode": mode["chosen_mode"],
    }
    write_json(output / "phase2a-selection.json", rank_record)
    if decision["rank_search_boundary_reached"]:
        report = {
            "status": "phase2a_rank_boundary_remains_unresolved",
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
        rank=decision["chosen_rank"],
        l2=decision["chosen_l2"],
        epochs=final_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
    )
    masking = masked_state_calibration(
        model,
        scaled,
        reliability,
        max_rows=400,
        mask_fraction=0.30,
        epochs=80,
        learning_rate=0.05,
        seed=23,
    )

    states.to_csv(output / "country-year-states.csv.gz", index=False)
    information.to_csv(output / "country-year-information.csv.gz", index=False)
    state_audit.to_csv(output / "state-coverage-audit.csv.gz", index=False)
    reliability.to_csv(output / "feature-reliability.csv.gz", index=False)
    by_year.to_csv(output / "reconstruction-by-year.csv", index=False)
    by_source.to_csv(output / "reconstruction-by-source.csv", index=False)
    masking["row_results"].to_csv(
        output / "uncertainty-masking-rows.csv.gz",
        index=False,
    )
    masking["quartile_calibration"].to_csv(
        output / "uncertainty-quartiles.csv",
        index=False,
    )
    scaler.stats_.reset_index().to_csv(
        output / "measurement-scaler.csv.gz",
        index=False,
    )
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

    chosen_mode_result = (
        mode["results"].set_index("mode").loc[mode["chosen_mode"]]
    )
    baseline_rmse = float(prior_selection["zero_state_baseline"]["row_rmse_mean"])
    reconstruction_improvement = float(
        (baseline_rmse - chosen_mode_result.row_rmse_mean) / baseline_rmse
    )
    gates = {
        "rank_resolved": True,
        "heldout_reconstruction_beats_zero_state": bool(
            reconstruction_improvement > 0
        ),
        "heldout_reconstruction_relative_improvement": reconstruction_improvement,
        "distance_coverage_materially_better_than_phase1": bool(
            abs(coverage["distance_vs_coverage_spearman"])
            < 0.85
            * abs(
                PHASE1_COVERAGE_BASELINE[
                    "distance_vs_coverage_spearman"
                ]
            )
        ),
        "uncertainty_declines_with_coverage": bool(
            coverage["uncertainty_vs_coverage_spearman"] < 0
        ),
        "movement_not_dominated_by_coverage_change": bool(
            abs(
                coverage[
                    "movement_vs_abs_coverage_change_spearman"
                ]
            )
            < 0.60
        ),
        "masked_state_error_tracks_uncertainty": bool(
            masking["state_error_uncertainty_spearman"] > 0.10
        ),
    }
    phase2a_pass = all(gates.values())
    phase2a = {
        "status": (
            "completed_phase2a_measurement_state"
            if phase2a_pass
            else "phase2a_measurement_state_needs_revision"
        ),
        "source_cutoff": source_report["cutoff"],
        "inputs_verified": verified,
        "feature_count_cap": None,
        "phase1_panel": panel.summary,
        "preparation": preparation,
        "state_rank": model.rank,
        "state_rows": len(states),
        "model_features": len(model.features_),
        "measurement_mode": mode["chosen_mode"],
        "rank_selection": rank_record,
        "coverage_geometry": coverage,
        "phase1_coverage_baseline": PHASE1_COVERAGE_BASELINE,
        "uncertainty_masking": {
            key: value
            for key, value in masking.items()
            if key not in {"row_results", "quartile_calibration"}
        },
        "gates": gates,
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
    phase2b["fold_metrics"].to_csv(
        output / "transition-fold-metrics.csv",
        index=False,
    )
    phase2b["aggregate_metrics"].to_csv(
        output / "transition-aggregate-metrics.csv",
        index=False,
    )
    phase2b["predictions"].to_csv(
        output / "transition-predictions.csv.gz",
        index=False,
    )
    phase2b["tuning"].to_csv(
        output / "transition-tuning.csv.gz",
        index=False,
    )
    phase2b_summary = {
        key: value
        for key, value in phase2b.items()
        if key not in {
            "fold_metrics",
            "aggregate_metrics",
            "predictions",
            "tuning",
        }
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
            if any(
                row["challenger_beats_no_change"]
                for row in phase2b["winners"]
            )
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
    parser.add_argument("--prior-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-selection-cells", type=int, default=1_200_000)
    parser.add_argument("--extension-epochs", type=int, default=5)
    parser.add_argument("--mode-epochs", type=int, default=4)
    parser.add_argument("--final-epochs", type=int, default=6)
    args = parser.parse_args()
    report = run(
        args.snapshot,
        args.prior_results,
        args.output,
        max_selection_cells=args.max_selection_cells,
        extension_epochs=args.extension_epochs,
        mode_epochs=args.mode_epochs,
        final_epochs=args.final_epochs,
    )
    print("PHASE2_RESOLVED_STATUS", json.dumps(report, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
