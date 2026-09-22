"""Finalize a completed Phase 2 computation artifact without refitting models.

Strict JSON rejects NaN, which is appropriate. Undefined comparison statistics
(such as movement direction for a no-change forecast) are written as JSON null.
The underlying CSV evidence is preserved byte-for-byte.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


EXPECTED = (
    "phase2a-summary.json",
    "phase2a-selection.json",
    "rank-selection-combined.csv",
    "measurement-mode-selection.csv",
    "transition-fold-metrics.csv",
    "transition-aggregate-metrics.csv",
    "transition-predictions.csv.gz",
    "transition-tuning.csv.gz",
    "country-year-states.csv.gz",
    "country-year-information.csv.gz",
    "feature-reliability.csv.gz",
    "measurement-loadings.csv.gz",
    "uncertainty-masking-rows.csv.gz",
    "uncertainty-quartiles.csv",
)


def _scalar(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "na", "none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return None
    if number.is_integer() and "." not in text.lower() and "e" not in text.lower():
        return int(number)
    return number


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {key: _scalar(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, sort_keys=False) + "\n"
    )


def _gzip_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return max(sum(1 for _ in stream) - 1, 0)


def finalize(
    root: Path,
    *,
    source_run: int,
    source_artifact: int,
    source_zip_sha256: str,
) -> dict:
    root = Path(root).resolve()
    missing = [name for name in EXPECTED if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing completed Phase 2 outputs: {missing}")

    phase2a = json.loads((root / "phase2a-summary.json").read_text())
    if phase2a.get("status") != "completed_phase2a_measurement_state":
        raise RuntimeError(f"Phase 2A not complete: {phase2a.get('status')}")
    if not all(phase2a.get("gates", {}).values()):
        raise RuntimeError("Not all Phase 2A gates passed")

    aggregate = _read_csv(root / "transition-aggregate-metrics.csv")
    folds = _read_csv(root / "transition-fold-metrics.csv")
    if not aggregate or not folds:
        raise RuntimeError("No completed transition evidence")

    winners = []
    for horizon in sorted({int(row["horizon"]) for row in aggregate}):
        rows = [row for row in aggregate if int(row["horizon"]) == horizon]
        baseline = next(row for row in rows if row["model"] == "no_change")
        challengers = [row for row in rows if row["model"] != "no_change"]
        best = min(challengers, key=lambda row: float(row["state_rmse"]))
        baseline_rmse = float(baseline["state_rmse"])
        challenger_rmse = float(best["state_rmse"])
        winners.append(
            {
                "horizon": horizon,
                "no_change_state_rmse": baseline_rmse,
                "best_challenger": best["model"],
                "best_challenger_state_rmse": challenger_rmse,
                "relative_rmse_improvement": (
                    baseline_rmse - challenger_rmse
                )
                / baseline_rmse,
                "challenger_beats_no_change": challenger_rmse < baseline_rmse,
                "fraction_rows_beating_no_change": best.get(
                    "fraction_rows_beating_no_change"
                ),
                "mean_movement_cosine": best.get("mean_movement_cosine"),
            }
        )

    fold_keys = {
        (
            int(row["horizon"]),
            int(row["outer_start"]),
            int(row["outer_end"]),
        )
        for row in folds
    }
    phase2b = {
        "status": "completed_phase2b_retrospective_state_transition_development",
        "development_windows": sorted(
            [
                {"horizon": horizon, "outer_start": start, "outer_end": end}
                for horizon, start, end in fold_keys
            ],
            key=lambda row: (row["horizon"], row["outer_start"]),
        ),
        "model_folds": len(folds),
        "matched_prediction_rows": _gzip_rows(
            root / "transition-predictions.csv.gz"
        ),
        "tuning_rows": _gzip_rows(root / "transition-tuning.csv.gz"),
        "aggregate_metrics": aggregate,
        "winners": winners,
        "skipped_cases": 0,
        "state_rank": phase2a["state_rank"],
        "supervised_banking_targets_read": 0,
        "crisis_labels_read": 0,
        "final_confirmation_evaluated": False,
        "representation_caveat": (
            "The measurement loadings use the retrospective research panel. "
            "These are time-ordered development comparisons, not a real-time "
            "confirmation test."
        ),
    }
    _write_json(root / "phase2b-summary.json", phase2b)

    report = {
        "status": "completed_phase2_measurement_and_transition_development",
        "source_run": source_run,
        "source_artifact": source_artifact,
        "source_zip_sha256": source_zip_sha256,
        "phase2a": phase2a,
        "phase2b": phase2b,
        "production_modified": False,
        "crisis_classifier_retrained": False,
        "final_confirmation_evaluated": False,
        "next_phase": (
            "phase3_probabilistic_future_state_forecasting"
            if any(row["challenger_beats_no_change"] for row in winners)
            else "review_negative_transition_result_before_phase3"
        ),
        "finalization_note": (
            "All model computations completed in the source run. Finalization "
            "converts undefined comparison statistics to JSON null without "
            "refitting any model."
        ),
    }
    _write_json(root / "phase2-summary.json", report)
    checksums = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "output-checksums.json"
    }
    _write_json(root / "output-checksums.json", checksums)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--source-run", type=int, required=True)
    parser.add_argument("--source-artifact", type=int, required=True)
    parser.add_argument("--source-zip-sha256", required=True)
    args = parser.parse_args()
    report = finalize(
        args.results,
        source_run=args.source_run,
        source_artifact=args.source_artifact,
        source_zip_sha256=args.source_zip_sha256,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "state_rank": report["phase2a"]["state_rank"],
                "winners": report["phase2b"]["winners"],
                "next_phase": report["next_phase"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
