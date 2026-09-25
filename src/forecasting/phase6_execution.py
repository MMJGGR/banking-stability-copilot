"""Execute the Phase 6 replacement-score challenge.

This is a research/development runner. It never writes production artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.crisis_panel import CrisisPanelConfig, IMF_FEATURE_SPECS, build_crisis_panel_result
from src.crisis_validation import classification_metrics
from .phase6_replacement import (
    EpisodeLabels,
    ReplacementPanel,
    _add_state_velocity,
    build_replacement_panel,
    load_imf_observations,
    validate_candidates,
)
from .phase6_score import build_oof_scores, latest_components, score_diagnostics
from .phase6_sovereign import parse_sovereign_default_workbook, sha256

PRODUCTION_CLASSIFIER_SHA256 = (
    "054811a0b12133592bd22de64e2141c6c969d473963170199cff441e8e689aee"
)
PRODUCTION_CLASSIFIER_BYTES = 52580


def write_json(path: str | Path, payload: dict) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, tuple):
            return [clean(item) for item in value]
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    Path(path).write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n")


def _serialize_validation(root: Path, target: str, model: str, result) -> None:
    folder = root / "validation" / target / model
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "summary.json", result.summary)
    result.fold_metrics.to_csv(folder / "fold-metrics.csv", index=False)
    result.ledger.to_csv(folder / "outer-predictions.csv.gz", index=False)
    result.tuning_ledger.to_csv(folder / "inner-predictions.csv.gz", index=False)
    result.bootstrap_cis.to_csv(folder / "bootstrap-cis.csv", index=False)
    write_json(folder / "fold-details.json", {"folds": result.fold_details})


def _event_rate_metrics(panel: pd.DataFrame, target_column: str, ledger: pd.DataFrame) -> dict:
    predictions = []
    for origin in sorted(ledger.origin.astype(int).unique()):
        train = panel.loc[panel.forecast_origin_year < origin]
        probability = float(train[target_column].mean())
        test = ledger.loc[ledger.origin.astype(int).eq(origin), ["y"]].copy()
        test["proba"] = probability
        predictions.append(test)
    frame = pd.concat(predictions, ignore_index=True)
    threshold = float(frame.proba.quantile(0.90))
    return classification_metrics(
        frame.y.astype(int),
        frame.proba.astype(float),
        frame.proba.ge(threshold).astype(int),
    )


def model_summary(results: dict, panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target_columns = {
        "banking": "banking_target",
        "sovereign": "sovereign_target",
        "either": "either_target",
    }
    for target, models in results.items():
        representative = next(iter(models.values())).ledger
        baseline = _event_rate_metrics(panel, target_columns[target], representative)
        rows.append({
            "target": target,
            "model": "event_rate",
            "rows": int(len(representative)),
            "positives": int(representative.y.sum()),
            "brier": float(baseline["brier"]),
            "log_loss": float(baseline["log_loss"]),
            "pr_auc": float(baseline["average_precision"]),
            "roc_auc": float(baseline["roc_auc"]),
            "recall": float(baseline["recall"]),
            "precision": float(baseline["precision"]),
            "alert_burden": float(baseline["alert_burden"]),
        })
        for model, result in models.items():
            summary = result.summary
            rows.append({
                "target": target,
                "model": model,
                "rows": int(len(result.ledger)),
                "positives": int(result.ledger.y.sum()),
                "brier": float(summary["brier"]),
                "log_loss": float(summary["log_loss"]),
                "pr_auc": float(summary["average_precision"]),
                "roc_auc": float(summary["roc_auc"]),
                "recall": float(summary["recall"]),
                "precision": float(summary["precision"]),
                "alert_burden": float(summary["alert_burden"]),
            })
    metrics = pd.DataFrame(rows)
    baseline = metrics.loc[metrics.model.eq("event_rate"), ["target", "brier", "log_loss"]].rename(
        columns={"brier": "event_rate_brier", "log_loss": "event_rate_log_loss"}
    )
    metrics = metrics.merge(baseline, on="target", how="left", validate="many_to_one")
    metrics["brier_skill_vs_event_rate"] = (
        metrics.event_rate_brier - metrics.brier
    ) / metrics.event_rate_brier
    metrics["log_loss_skill_vs_event_rate"] = (
        metrics.event_rate_log_loss - metrics.log_loss
    ) / metrics.event_rate_log_loss
    return metrics.sort_values(["target", "brier", "log_loss"]).reset_index(drop=True)


def _latest_feature_frame(
    snapshot: Path,
    phase2_results: Path,
    replacement: ReplacementPanel,
) -> pd.DataFrame:
    states = pd.read_csv(phase2_results / "country-year-states.csv.gz")
    information = pd.read_csv(phase2_results / "country-year-information.csv.gz")
    latest_year = int(states.forecast_origin_year.max())
    country_universe = sorted(states.entity_code.astype(str).str.upper().unique())
    empty_labels = EpisodeLabels(
        pd.DataFrame(columns=["country_code", "start_year", "end_year"]),
        coverage_end_year=latest_year + 3,
    )
    config = CrisisPanelConfig(
        start_year=latest_year,
        end_year=latest_year,
        horizon_start_years=1,
        horizon_end_years=3,
        feature_lag_years=1,
        exclude_active_crisis=False,
        post_crisis_cooldown_years=0,
        label_coverage_end_year=latest_year + 3,
        drop_right_censored=False,
        family_min_coverage=0.5,
    )
    feature_panel = build_crisis_panel_result(
        weo_df=load_imf_observations(snapshot, "WEO"),
        fsic_df=load_imf_observations(snapshot, "FSIC"),
        labels=empty_labels,
        country_universe=country_universe,
        feature_specs=IMF_FEATURE_SPECS,
        config=config,
    ).panel.rename(columns={"country_code": "entity_code"})
    state_frame, _, _ = _add_state_velocity(states)
    latest = feature_panel.merge(
        state_frame.loc[state_frame.forecast_origin_year.eq(latest_year)],
        on=["entity_code", "forecast_origin_year"],
        how="inner",
        validate="one_to_one",
    ).merge(
        information.loc[information.forecast_origin_year.eq(latest_year)],
        on=["entity_code", "forecast_origin_year"],
        how="left",
        validate="one_to_one",
    )
    for column in replacement.combined_features:
        if column not in latest:
            latest[column] = np.nan
    return latest.sort_values("entity_code").reset_index(drop=True)


def _time_cv_splits(panel: pd.DataFrame, target_column: str, folds: int = 4):
    years = np.array(sorted(panel.forecast_origin_year.unique()), dtype=int)
    first = max(10, len(years) // 2)
    validation_years = np.array_split(years[first:], folds)
    splits = []
    y = panel[target_column].to_numpy(int)
    origin = panel.forecast_origin_year.to_numpy(int)
    for block in validation_years:
        if not len(block):
            continue
        train = np.flatnonzero(origin < int(block.min()))
        valid = np.flatnonzero(np.isin(origin, block))
        if len(train) and len(valid) and np.unique(y[train]).size == 2 and np.unique(y[valid]).size == 2:
            splits.append((train, valid))
    return splits


def fit_latest_probability(
    panel: pd.DataFrame,
    latest: pd.DataFrame,
    features: list[str],
    target_column: str,
) -> np.ndarray:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                C=0.10,
                penalty="l2",
                solver="liblinear",
                max_iter=4000,
                random_state=17,
            ),
        ),
    ])
    splits = _time_cv_splits(panel, target_column)
    if len(splits) >= 2:
        model = CalibratedClassifierCV(
            estimator=numeric,
            method="sigmoid",
            cv=splits,
            ensemble=False,
        )
    else:
        model = numeric
    model.fit(panel[features], panel[target_column].astype(int))
    return model.predict_proba(latest[features])[:, 1]


def _information_status(latest: pd.DataFrame, historical: pd.DataFrame) -> pd.Series:
    lower = float(historical.observed_share.quantile(0.25))
    upper = float(historical.observed_share.quantile(0.75))
    return pd.Series(
        np.select(
            [latest.observed_share < lower, latest.observed_share < upper],
            ["provisional_low_information", "standard_information"],
            default="high_information",
        ),
        index=latest.index,
    )


def production_cross_section(path: str | Path | None) -> pd.DataFrame:
    if path is None or not Path(path).is_file():
        return pd.DataFrame()
    payload = pickle.loads(Path(path).read_bytes())
    scores = payload.get("country_scores") if isinstance(payload, dict) else None
    if not isinstance(scores, pd.DataFrame):
        return pd.DataFrame()
    return scores.copy()


def replacement_gate(metrics: pd.DataFrame, score_report: dict) -> dict:
    indexed = metrics.set_index(["target", "model"])
    checks = {}
    for target in ("banking", "sovereign", "either"):
        challenger = indexed.loc[(target, "combined")]
        baseline = indexed.loc[(target, "event_rate")]
        checks[f"{target}_combined_brier_improves"] = bool(challenger.brier < baseline.brier)
        checks[f"{target}_combined_log_loss_improves"] = bool(
            challenger.log_loss < baseline.log_loss
        )
    checks["score_event_rates_monotonic"] = bool(
        score_report["score_event_rate_monotonic"]
    )
    development_pass = all(checks.values())
    return {
        "development_gate_passed": development_pass,
        "checks": checks,
        "matched_historical_production_comparison": "unavailable_not_fabricated",
        "prospective_confirmation": "pending_frozen_phase5_target_years",
        "replacement_authorized": False,
        "decision": (
            "development_challenger_passed_but_keep_shadow_only"
            if development_pass
            else "do_not_advance_replacement_challenger"
        ),
    }


def run(
    snapshot: str | Path,
    phase2_results: str | Path,
    sovereign_workbook: str | Path,
    output: str | Path,
    *,
    production_risk_model: str | Path | None = None,
) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = Path(snapshot)
    phase2_results = Path(phase2_results)

    sovereign = parse_sovereign_default_workbook(sovereign_workbook)
    sovereign.status.to_csv(output / "sovereign-default-status.csv.gz", index=False)
    sovereign.episodes.to_csv(output / "sovereign-default-episodes.csv", index=False)
    sovereign.domestic_arrears_status.to_csv(
        output / "sovereign-domestic-arrears-episodes.csv", index=False
    )
    write_json(output / "sovereign-source-audit.json", sovereign.audit)

    replacement = build_replacement_panel(snapshot, phase2_results, sovereign)
    write_json(output / "replacement-panel-audit.json", replacement.audit)
    replacement.panel[[
        "entity_code", "forecast_origin_year", "banking_target",
        "sovereign_target", "either_target", "banking_event_id",
        "sovereign_event_id", "either_event_id", "observed_share",
        "state_uncertainty_proxy",
    ]].to_csv(output / "replacement-label-ledger.csv.gz", index=False)
    write_json(output / "feature-sets.json", {
        "state": replacement.state_features,
        "observed": replacement.observed_features,
        "combined": replacement.combined_features,
    })

    results = validate_candidates(replacement)
    for target, models in results.items():
        for model, result in models.items():
            _serialize_validation(output, target, model, result)
    metrics = model_summary(results, replacement.panel)
    metrics.to_csv(output / "candidate-model-summary.csv", index=False)

    score_rows, score_weights = build_oof_scores(results["either"]["combined"])
    score_rows.to_csv(output / "replacement-score-oof.csv.gz", index=False)
    score_weights.to_csv(output / "replacement-score-weights.csv", index=False)
    score_report = score_diagnostics(score_rows)
    write_json(output / "replacement-score-diagnostics.json", score_report)

    latest = _latest_feature_frame(snapshot, phase2_results, replacement)
    model_choice = {}
    latest_probabilities = {}
    for target, target_column in (
        ("banking", "banking_target"),
        ("sovereign", "sovereign_target"),
        ("either", "either_target"),
    ):
        candidates = metrics.loc[
            metrics.target.eq(target) & metrics.model.isin(["state", "observed", "combined"])
        ].sort_values(["brier", "log_loss", "model"])
        selected = str(candidates.iloc[0].model)
        model_choice[target] = selected
        features = {
            "state": replacement.state_features,
            "observed": replacement.observed_features,
            "combined": replacement.combined_features,
        }[selected]
        latest_probabilities[target] = fit_latest_probability(
            replacement.panel, latest, features, target_column
        )

    # The replacement score itself deliberately uses the full combined either-event
    # challenger, even when a simpler benchmark wins; otherwise the replacement
    # claim would evade its own architecture test.
    combined_either_latest = fit_latest_probability(
        replacement.panel,
        latest,
        replacement.combined_features,
        "either_target",
    )
    mean_weights = score_weights[[
        "peer_weight", "history_weight", "absolute_weight"
    ]].mean()
    mean_weights = mean_weights / mean_weights.sum()
    latest_score = latest_components(
        pd.Series(combined_either_latest),
        latest.entity_code,
        int(latest.forecast_origin_year.max()),
        results["either"]["combined"].ledger,
        mean_weights,
    )
    latest_score["banking_crisis_probability_1_to_3y"] = latest_probabilities["banking"]
    latest_score["sovereign_default_probability_1_to_3y"] = latest_probabilities["sovereign"]
    latest_score["either_event_probability_1_to_3y"] = latest_probabilities["either"]
    latest_score["score_model_probability_1_to_3y"] = combined_either_latest
    latest_score["information_status"] = _information_status(latest, replacement.panel)
    latest_score["state_uncertainty_proxy"] = latest.state_uncertainty_proxy.to_numpy()
    latest_score["observed_share"] = latest.observed_share.to_numpy()
    latest_score["model_status"] = "research_replacement_challenger_not_production"
    latest_score["banking_model_family"] = model_choice["banking"]
    latest_score["sovereign_model_family"] = model_choice["sovereign"]
    latest_score["either_model_family"] = model_choice["either"]

    production = production_cross_section(production_risk_model)
    if not production.empty and "country_code" in production:
        keep = [column for column in [
            "country_code", "risk_score", "risk_category", "crisis_prob"
        ] if column in production]
        latest_score = latest_score.merge(
            production[keep].rename(columns={
                "country_code": "country",
                "risk_score": "production_risk_score",
                "risk_category": "production_risk_category",
                "crisis_prob": "production_crisis_probability",
            }),
            on="country",
            how="left",
            validate="one_to_one",
        )
    latest_score.to_csv(output / "latest-replacement-score.csv", index=False)

    gate = replacement_gate(metrics, score_report)
    write_json(output / "replacement-gate.json", gate)
    report = {
        "status": "completed_phase6_replacement_challenge",
        "sovereign_source": sovereign.audit,
        "panel": replacement.audit,
        "candidate_model_rows": len(metrics),
        "selected_latest_model_families": model_choice,
        "score": score_report,
        "mean_score_weights": mean_weights.to_dict(),
        "latest_score_rows": len(latest_score),
        "gate": gate,
        "provider_projection_rows_read": 0,
        "production_classifier_retrained": False,
        "production_modified": False,
        "merged": False,
        "deployed": False,
    }
    write_json(output / "phase6-summary.json", report)
    checksums = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    write_json(output / "output-checksums.json", checksums)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--phase2-results", required=True)
    parser.add_argument("--sovereign-workbook", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--production-risk-model")
    args = parser.parse_args()
    report = run(
        args.snapshot,
        args.phase2_results,
        args.sovereign_workbook,
        args.output,
        production_risk_model=args.production_risk_model,
    )
    print("PHASE6_COMPLETE", json.dumps(report, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
