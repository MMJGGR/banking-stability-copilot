"""Execute the Phase 6 replacement risk-rating challenge.

The run keeps the Phase 2 measurement state fixed, adds time-safe crisis-
specific raw measurements, evaluates banking/sovereign/combined event hazards,
and builds a 1-10 challenger score from peer position, own history and absolute
three-year event imminence.

Nothing in this module writes production artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .phase6_features import (
    add_state_trajectory_features,
    load_decision_measurements,
    merge_decision_features,
    pivot_decision_measurements,
)
from .phase6_labels import (
    build_horizon_targets,
    load_banking_episodes,
    parse_boc_boe_default_source,
    parse_world_bank_debt_distress,
    status_to_episodes,
)
from .phase6_models import (
    build_oof_replacement_ratings,
    category_diagnostics,
    compare_latest_with_production,
    evaluate_event_family,
    latest_event_probabilities,
    select_development_model,
)


EXPECTED_CLASSIFIER_SHA = "054811a0b12133592bd22de64e2141c6c969d473963170199cff441e8e689aee"
EXPECTED_CLASSIFIER_BYTES = 52580


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_clean(value):
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def write_json(path: str | Path, payload) -> None:
    Path(path).write_text(
        json.dumps(json_clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _load_sovereign_status(
    primary: str | Path | None,
    fallback: str | Path | None,
) -> tuple[pd.DataFrame | None, dict]:
    errors = []
    candidates = []
    if primary:
        candidates.append(("boc_boe_2025_primary", Path(primary), parse_boc_boe_default_source))
    if fallback:
        path = Path(fallback)
        parser = (
            parse_world_bank_debt_distress
            if "non_confidential" in path.name.lower()
            else parse_boc_boe_default_source
        )
        candidates.append(("world_bank_package_fallback", path, parser))
    for role, path, parser in candidates:
        if not path.is_file():
            errors.append({"role": role, "path": str(path), "error": "file_missing"})
            continue
        try:
            frame, report = parser(path)
            if "distress_status" in frame:
                frame = frame.rename(columns={"distress_status": "any_default_status"})
                frame["default_amount_usd"] = np.nan
            required = {"country_code", "year", "any_default_status"}
            if not required <= set(frame):
                raise ValueError(f"missing parsed columns: {sorted(required-set(frame))}")
            frame["country_code"] = frame.country_code.astype(str).str.upper()
            frame["year"] = pd.to_numeric(frame.year, errors="raise").astype(int)
            frame["any_default_status"] = (
                pd.to_numeric(frame.any_default_status, errors="coerce")
                .fillna(0).gt(0).astype("int8")
            )
            frame = frame.loc[frame.country_code.str.fullmatch(r"[A-Z]{3}")].copy()
            frame = frame.groupby(["country_code", "year"], observed=True).agg(
                any_default_status=("any_default_status", "max"),
                default_amount_usd=("default_amount_usd", "sum"),
            ).reset_index()
            years = int(frame.year.max() - frame.year.min() + 1)
            density = len(frame) / max(1, frame.country_code.nunique() * years)
            report = {
                **report,
                "source_role": role,
                "source_path": str(path),
                "source_sha256": sha256(path),
                "panel_density": float(density),
                "parse_errors_before_success": errors,
            }
            if frame.country_code.nunique() < 50 or frame.year.nunique() < 20:
                raise ValueError("parsed sovereign source has insufficient country/year coverage")
            return frame, report
        except Exception as error:
            errors.append({
                "role": role,
                "path": str(path),
                "error": f"{type(error).__name__}: {error}",
            })
    return None, {
        "status": "no_authoritative_sovereign_status_parsed",
        "errors": errors,
    }


def _load_production_reference(risk_model_path: str | Path) -> pd.DataFrame:
    with Path(risk_model_path).open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "country_scores" in payload:
        frame = payload["country_scores"].copy()
    elif isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    else:
        raise ValueError("Unrecognised production risk-model payload")
    rename = {
        "country_code": "entity_code",
        "country_name": "production_country_name",
        "risk_score": "production_risk_score",
        "risk_category": "production_risk_category",
        "crisis_prob": "production_crisis_probability",
    }
    frame = frame.rename(columns=rename)
    required = {"entity_code", "production_risk_score", "production_risk_category"}
    if not required <= set(frame):
        raise ValueError("Production country-score table is incomplete")
    if "production_crisis_probability" not in frame:
        frame["production_crisis_probability"] = np.nan
    if "production_country_name" not in frame:
        frame["production_country_name"] = ""
    return frame[[
        "entity_code", "production_country_name", "production_risk_score",
        "production_risk_category", "production_crisis_probability",
    ]].drop_duplicates("entity_code")


def _save_evaluation(output: Path, prefix: str, result: dict) -> None:
    for key, filename in (
        ("fold_metrics", "fold-metrics.csv"),
        ("aggregate_metrics", "aggregate-metrics.csv"),
        ("test_predictions", "test-predictions.csv.gz"),
        ("reference_predictions", "reference-predictions.csv.gz"),
        ("tuning", "tuning.csv.gz"),
        ("weight_trials", "peer-weight-trials.csv.gz"),
    ):
        frame = result[key]
        frame.to_csv(output / f"{prefix}-{filename}", index=False)
    write_json(output / f"{prefix}-skipped.json", result["skipped"])
    write_json(
        output / f"{prefix}-feature-groups.json",
        {name: {"feature_count": len(values), "features": list(values)}
         for name, values in result["feature_groups"].items()},
    )


def _event_gate(result: dict, horizon: int = 3) -> dict:
    selection = select_development_model(result["aggregate_metrics"], horizon)
    aggregate = result["aggregate_metrics"]
    selected = aggregate.loc[
        aggregate.horizon.eq(horizon) & aggregate.model.eq(selection["model"])
    ]
    raw = aggregate.loc[
        aggregate.horizon.eq(horizon) & aggregate.model.eq("raw_targeted")
    ]
    selected_row = selected.iloc[0] if len(selected) else None
    raw_row = raw.iloc[0] if len(raw) else None
    adds_value_over_raw = None
    if selected_row is not None and raw_row is not None:
        adds_value_over_raw = bool(
            selected_row.brier <= raw_row.brier
            and selected_row.log_loss <= raw_row.log_loss
            and selected_row.pr_auc >= 0.95 * raw_row.pr_auc
            and selection["model"] in {"state_only", "state_trajectory", "hybrid"}
        )
    return {
        "selection": selection,
        "passes_event_rate_gate": selection.get("status") == "passes_event_rate_gate",
        "adds_value_over_raw_benchmark": adds_value_over_raw,
        "raw_benchmark": (
            None if raw_row is None else {
                "brier": float(raw_row.brier),
                "log_loss": float(raw_row.log_loss),
                "pr_auc": float(raw_row.pr_auc),
                "roc_auc": float(raw_row.roc_auc),
                "passes_event_rate_gate": bool(raw_row.passes_event_rate_gate),
            }
        ),
    }


def _latest_probability_wide(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    wide = frame.pivot_table(
        index="country_code", columns="horizon", values="probability", aggfunc="first"
    ).rename(columns={
        1: f"{prefix}_probability_1y",
        2: f"{prefix}_probability_2y",
        3: f"{prefix}_probability_3y",
    }).reset_index()
    return wide


def run(
    *,
    snapshot: str | Path,
    phase2: str | Path,
    banking_episodes_path: str | Path,
    sovereign_primary: str | Path | None,
    sovereign_fallback: str | Path | None,
    production_risk_model: str | Path,
    production_classifier: str | Path,
    output: str | Path,
) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    phase2 = Path(phase2)

    classifier_path = Path(production_classifier)
    classifier_integrity = {
        "bytes": classifier_path.stat().st_size,
        "sha256": sha256(classifier_path),
    }
    if classifier_integrity != {
        "bytes": EXPECTED_CLASSIFIER_BYTES,
        "sha256": EXPECTED_CLASSIFIER_SHA,
    }:
        raise AssertionError(f"Production classifier integrity failure: {classifier_integrity}")

    states = pd.read_csv(phase2 / "country-year-states.csv.gz")
    information = pd.read_csv(phase2 / "country-year-information.csv.gz")
    state_panel, state_summary = add_state_trajectory_features(states, information)
    measurements, measurement_ledger, measurement_summary = load_decision_measurements(
        snapshot, phase2, states,
        maximum_origin_year=int(states.forecast_origin_year.max()),
    )
    raw_wide, raw_groups = pivot_decision_measurements(measurements)
    decision = merge_decision_features(state_panel, raw_wide).rename(
        columns={"entity_code": "country_code"}
    )
    decision["country_code"] = decision.country_code.astype(str).str.upper()

    banking_episodes, banking_report = load_banking_episodes(banking_episodes_path)
    decision, banking_exclusions = build_horizon_targets(
        decision,
        banking_episodes,
        horizons=(1, 2, 3),
        cooldown_years=3,
        coverage_end_year=2025,
        target_prefix="banking_crisis",
    )

    sovereign_status, sovereign_report = _load_sovereign_status(
        sovereign_primary, sovereign_fallback
    )
    sovereign_available = sovereign_status is not None
    if sovereign_available:
        sovereign_episodes = status_to_episodes(
            sovereign_status,
            status_column="any_default_status",
            maximum_internal_gap_years=1,
            event_type="sovereign_distress",
        )
        sovereign_coverage_end = int(sovereign_status.year.max())
        sovereign_countries = set(sovereign_status.country_code.unique())
        decision, sovereign_exclusions = build_horizon_targets(
            decision,
            sovereign_episodes,
            horizons=(1, 2, 3),
            cooldown_years=3,
            coverage_end_year=sovereign_coverage_end,
            target_prefix="sovereign_distress",
        )
        for horizon in (1, 2, 3):
            column = f"sovereign_distress_{horizon}y_eligible"
            decision[column] = (
                decision[column].astype(bool)
                & decision.country_code.isin(sovereign_countries)
            )
        sovereign_label_report = {
            **sovereign_report,
            "episodes": len(sovereign_episodes),
            "episode_countries": int(sovereign_episodes.country_code.nunique()),
            "coverage_end_year": sovereign_coverage_end,
        }
    else:
        sovereign_episodes = pd.DataFrame(columns=[
            "country_code", "start_year", "end_year", "event_type", "event_id"
        ])
        sovereign_exclusions = pd.DataFrame()
        sovereign_label_report = sovereign_report
        for horizon in (1, 2, 3):
            decision[f"sovereign_distress_{horizon}y"] = 0
            decision[f"sovereign_distress_{horizon}y_eligible"] = False
            decision[f"sovereign_distress_{horizon}y_event_start"] = np.nan

    for horizon in (1, 2, 3):
        bank_target = f"banking_crisis_{horizon}y"
        sovereign_target = f"sovereign_distress_{horizon}y"
        bank_eligible = f"banking_crisis_{horizon}y_eligible"
        sovereign_eligible = f"sovereign_distress_{horizon}y_eligible"
        decision[f"any_systemic_event_{horizon}y"] = np.maximum(
            decision[bank_target].astype(int), decision[sovereign_target].astype(int)
        )
        decision[f"any_systemic_event_{horizon}y_eligible"] = (
            decision[bank_eligible].astype(bool)
            & decision[sovereign_eligible].astype(bool)
        )

    measurement_ledger.to_csv(output / "decision-measurement-ledger.csv.gz", index=False)
    measurements.to_csv(output / "decision-measurement-cells.csv.gz", index=False)
    banking_episodes.to_csv(output / "banking-event-ledger.csv", index=False)
    sovereign_episodes.to_csv(output / "sovereign-event-ledger.csv", index=False)
    banking_exclusions.to_csv(output / "banking-event-exclusions.csv.gz", index=False)
    sovereign_exclusions.to_csv(output / "sovereign-event-exclusions.csv.gz", index=False)
    write_json(output / "label-and-feature-audit.json", {
        "state": state_summary,
        "measurements": measurement_summary,
        "banking_labels": banking_report,
        "sovereign_labels": sovereign_label_report,
        "provider_projection_rows_read": 0,
    })

    evaluations = {}
    families = [("banking", "banking_crisis")]
    if sovereign_available:
        families.extend([
            ("sovereign", "sovereign_distress"),
            ("combined", "any_systemic_event"),
        ])
    for family, prefix in families:
        result = evaluate_event_family(
            decision,
            event_family=family,
            target_prefix=prefix,
            raw_groups=raw_groups,
        )
        evaluations[family] = result
        _save_evaluation(output, family, result)

    gates = {
        family: _event_gate(result, 3)
        for family, result in evaluations.items()
    }

    oof_ratings = pd.DataFrame()
    category_table = pd.DataFrame()
    category_summary = {"monotonic_event_rate": False}
    if "combined" in evaluations:
        selected_combined_model = gates["combined"]["selection"]["model"]
        if selected_combined_model != "event_rate":
            oof_ratings = build_oof_replacement_ratings(
                evaluations["combined"],
                selected_model=selected_combined_model,
                horizon=3,
            )
            category_table, category_summary = category_diagnostics(oof_ratings)
    oof_ratings.to_csv(output / "replacement-score-oof.csv.gz", index=False)
    category_table.to_csv(output / "replacement-category-event-rates.csv", index=False)

    latest_year = int(states.forecast_origin_year.max())
    latest_outputs = {}
    model_records = {}
    for family, prefix in families:
        latest, records = latest_event_probabilities(
            decision,
            event_family=family,
            target_prefix=prefix,
            raw_groups=raw_groups,
            evaluation=evaluations[family],
            latest_year=latest_year,
        )
        latest.to_csv(output / f"latest-{family}-probabilities.csv", index=False)
        latest_outputs[family] = latest
        model_records[family] = records

    latest_ratings = pd.DataFrame()
    production_comparison = pd.DataFrame()
    coverage_correlation = np.nan
    if "combined" in latest_outputs:
        combined = latest_outputs["combined"]
        latest_ratings = combined.loc[combined.horizon.eq(3)].copy()
        latest_ratings = latest_ratings.rename(columns={
            "probability": "any_systemic_event_probability_3y"
        })
        latest_ratings = latest_ratings.merge(
            _latest_probability_wide(latest_outputs["banking"], "banking_crisis"),
            on="country_code", how="left", validate="one_to_one",
        ).merge(
            _latest_probability_wide(latest_outputs["sovereign"], "sovereign_distress"),
            on="country_code", how="left", validate="one_to_one",
        ).merge(
            _latest_probability_wide(latest_outputs["combined"], "any_systemic_event"),
            on="country_code", how="left", validate="one_to_one",
            suffixes=("", "_duplicate"),
        )
        latest_ratings = latest_ratings.drop(columns=[
            column for column in latest_ratings if column.endswith("_duplicate")
        ])
        quality = state_panel.loc[
            state_panel.forecast_origin_year.eq(latest_year),
            [
                "entity_code", "observed_share", "state_uncertainty_proxy",
                "effective_information", "own_history_years",
            ],
        ].rename(columns={"entity_code": "country_code"})
        latest_ratings = latest_ratings.merge(
            quality, on="country_code", how="left", validate="one_to_one"
        )
        if latest_ratings.observed_share.notna().sum() > 2:
            coverage_correlation = float(spearmanr(
                latest_ratings.replacement_risk_score,
                latest_ratings.observed_share,
                nan_policy="omit",
            ).statistic)
        production_reference = _load_production_reference(production_risk_model)
        production_comparison = compare_latest_with_production(
            latest_ratings, production_reference
        )
    latest_ratings.to_csv(output / "latest-replacement-ratings.csv", index=False)
    production_comparison.to_csv(
        output / "latest-replacement-vs-production.csv", index=False
    )

    sovereign_label_gate = bool(
        sovereign_available
        and sovereign_label_report.get("episodes", 0) >= 20
        and sovereign_label_report.get("episode_countries", 0) >= 10
    )
    core_model_gates = {
        family: bool(gate["passes_event_rate_gate"])
        for family, gate in gates.items()
    }
    architecture_increment_gate = bool(
        gates.get("combined", {}).get("adds_value_over_raw_benchmark")
    )
    category_gate = bool(
        category_summary.get("monotonic_event_rate")
        and (category_summary.get("high_vs_low_event_rate_ratio") or 0) >= 2.0
    )
    coverage_gate = bool(np.isfinite(coverage_correlation) and abs(coverage_correlation) < 0.4)

    if not sovereign_label_gate:
        final_decision = "blocked_by_missing_authoritative_labels_or_matched_benchmark"
    elif all(core_model_gates.get(name, False) for name in ("banking", "sovereign", "combined")):
        if architecture_increment_gate and category_gate and coverage_gate:
            final_decision = "replacement_challenger_passes_development_gates_shadow_only"
        else:
            final_decision = "partial_components_pass_keep_hybrid"
    else:
        final_decision = "do_not_replace_production"

    summary = {
        "status": "completed_phase6_replacement_challenge",
        "final_decision": final_decision,
        "latest_state_year": latest_year,
        "latest_rating_rows": len(latest_ratings),
        "event_model_gates": gates,
        "latest_model_records": model_records,
        "category_diagnostics": category_summary,
        "coverage_score_spearman": coverage_correlation,
        "coverage_gate": coverage_gate,
        "category_gate": category_gate,
        "architecture_adds_value_over_raw_gate": architecture_increment_gate,
        "sovereign_label_gate": sovereign_label_gate,
        "production_historical_probability_comparison": (
            "not_available_without_retraining_or_historical_locked_predictions"
        ),
        "production_classifier": classifier_integrity,
        "production_classifier_retrained": False,
        "production_modified": False,
        "provider_projection_rows_read": 0,
        "merged": False,
        "deployed": False,
        "limitations": [
            "Latest-vintage retrospective state construction is not a vintage-clean real-time test.",
            "The locked production classifier cannot be reconstructed historically without retraining; no matched historical claim is made.",
            "Phase 6 is development evidence only and cannot authorize production substitution.",
        ],
    }
    write_json(output / "phase6-summary.json", summary)
    write_json(output / "output-checksums.json", {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "output-checksums.json"
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--phase2", required=True)
    parser.add_argument("--banking-episodes", required=True)
    parser.add_argument("--sovereign-primary")
    parser.add_argument("--sovereign-fallback")
    parser.add_argument("--production-risk-model", required=True)
    parser.add_argument("--production-classifier", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(
        snapshot=args.snapshot,
        phase2=args.phase2,
        banking_episodes_path=args.banking_episodes,
        sovereign_primary=args.sovereign_primary,
        sovereign_fallback=args.sovereign_fallback,
        production_risk_model=args.production_risk_model,
        production_classifier=args.production_classifier,
        output=args.output,
    )
    print("PHASE6_COMPLETE", json.dumps(json_clean(report), sort_keys=True))


if __name__ == "__main__":
    main()
