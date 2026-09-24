"""Final Phase 4 advancement decisions from completed development metrics.

The development evaluator may compare and rank challengers, but no latest
crisis probability is admissible unless a challenger improves calibrated
probability quality over the event-rate baseline. This module is deliberately
separate from model fitting so a least-bad challenger is never mistaken for an
approved overlay.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def crisis_overlay_decision(aggregate_metrics: pd.DataFrame) -> dict:
    required = {"model", "brier", "log_loss", "pr_auc", "roc_auc"}
    if not required <= set(aggregate_metrics):
        raise ValueError(f"Missing crisis metrics: {sorted(required-set(aggregate_metrics))}")
    baseline_rows = aggregate_metrics.loc[
        aggregate_metrics.model.eq("event_rate")
    ]
    if len(baseline_rows) != 1:
        raise ValueError("Exactly one event-rate baseline is required")
    baseline = baseline_rows.iloc[0]
    challengers = aggregate_metrics.loc[
        ~aggregate_metrics.model.eq("event_rate")
    ].copy()
    if challengers.empty:
        return {
            "status": "do_not_advance_no_challenger",
            "selected_overlay": None,
            "latest_overlay_rows_admissible": 0,
        }

    challengers["brier_improvement"] = baseline.brier - challengers.brier
    challengers["log_loss_improvement"] = baseline.log_loss - challengers.log_loss
    challengers["pr_auc_improvement"] = challengers.pr_auc - baseline.pr_auc
    challengers["passes_probability_quality_gate"] = (
        challengers.brier_improvement.gt(0)
        & challengers.log_loss_improvement.gt(0)
        & challengers.pr_auc.ge(baseline.pr_auc)
    )
    advancing = challengers.loc[
        challengers.passes_probability_quality_gate
    ].sort_values(["brier", "log_loss", "model"])
    comparison = challengers.astype(object).where(
        pd.notna(challengers), None
    ).to_dict("records")
    if advancing.empty:
        return {
            "status": "do_not_advance_state_crisis_overlay_underperformed_event_rate",
            "selected_overlay": None,
            "latest_overlay_rows_admissible": 0,
            "advancement_rule": (
                "challenger must improve aggregate Brier score and log loss, "
                "without reducing precision-recall AUC versus event-rate baseline"
            ),
            "event_rate_baseline": {
                "brier": float(baseline.brier),
                "log_loss": float(baseline.log_loss),
                "pr_auc": float(baseline.pr_auc),
                "roc_auc": float(baseline.roc_auc),
            },
            "challenger_comparison": comparison,
            "production_classifier_action": "preserve_locked_classifier_unchanged",
        }
    selected = advancing.iloc[0]
    return {
        "status": "advance_research_overlay_only_not_production",
        "selected_overlay": str(selected.model),
        "latest_overlay_rows_admissible": "subject_to_separate_generation",
        "advancement_rule": (
            "challenger improves aggregate Brier score and log loss, without "
            "reducing precision-recall AUC versus event-rate baseline"
        ),
        "event_rate_baseline": {
            "brier": float(baseline.brier),
            "log_loss": float(baseline.log_loss),
            "pr_auc": float(baseline.pr_auc),
            "roc_auc": float(baseline.roc_auc),
        },
        "challenger_comparison": comparison,
        "production_classifier_action": "no_change_without_separate_validation_and_approval",
    }


def write_crisis_overlay_decision(metrics_path: str | Path, output_path: str | Path) -> dict:
    decision = crisis_overlay_decision(pd.read_csv(metrics_path))
    Path(output_path).write_text(
        json.dumps(decision, indent=2, allow_nan=False) + "\n"
    )
    return decision


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(write_crisis_overlay_decision(args.metrics, args.output), indent=2))
