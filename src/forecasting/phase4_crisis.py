"""Phase 4C: separately governed systemic-crisis overlay on the fixed state.

The 96-dimensional Phase 2 state is never refit or oriented using crisis
labels. This module only asks whether the already-frozen state, its exact
one-year velocity and state-information diagnostics improve one-to-three-year
systemic-crisis onset forecasts on later time windows.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

DEFAULT_WINDOWS = ((1995, 2000), (2001, 2007), (2008, 2014), (2015, 2022))
C_GRID = (0.01, 0.1, 1.0)
HORIZON_START = 1
HORIZON_END = 3
COOLDOWN_YEARS = 3
LABEL_COVERAGE_END_YEAR = 2025


def _state_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column for column in frame
        if column.startswith("state_") and column.split("_")[-1].isdigit()
    ]
    return sorted(columns, key=lambda column: int(column.split("_")[-1]))


def build_crisis_overlay_panel(
    states: pd.DataFrame,
    information: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    label_coverage_end_year: int = LABEL_COVERAGE_END_YEAR,
    horizon_start: int = HORIZON_START,
    horizon_end: int = HORIZON_END,
    cooldown_years: int = COOLDOWN_YEARS,
    include_borderline: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Create exact country-year event targets without altering the state."""
    state_columns = _state_columns(states)
    if not state_columns:
        raise ValueError("No state dimensions")
    info_columns = [
        "entity_code", "forecast_origin_year", "state_uncertainty_proxy",
        "observed_share", "effective_information",
    ]
    if not set(info_columns) <= set(information):
        raise ValueError("Incomplete information table")
    required_episodes = {
        "country_code", "start_year", "label_end_year", "classification",
    }
    if not required_episodes <= set(episodes):
        raise ValueError("Incomplete crisis episode table")
    if horizon_start < 1 or horizon_end < horizon_start or cooldown_years < 0:
        raise ValueError("Invalid horizon/cooldown policy")

    panel = states.merge(
        information[info_columns],
        on=["entity_code", "forecast_origin_year"],
        how="left",
        validate="one_to_one",
    ).copy()
    panel["entity_code"] = panel.entity_code.astype(str).str.upper()
    panel["forecast_origin_year"] = pd.to_numeric(
        panel.forecast_origin_year, errors="raise"
    ).astype(int)

    previous = states[["entity_code", "forecast_origin_year", *state_columns]].copy()
    previous["forecast_origin_year"] += 1
    previous = previous.rename(
        columns={column: f"previous_{column}" for column in state_columns}
    )
    panel = panel.merge(
        previous,
        on=["entity_code", "forecast_origin_year"],
        how="left",
        validate="one_to_one",
    )
    prior = panel[[f"previous_{column}" for column in state_columns]].to_numpy(float)
    current = panel[state_columns].to_numpy(float)
    velocity = current - prior
    observed_velocity = np.isfinite(velocity).all(axis=1)
    velocity[~observed_velocity] = 0.0
    for index, column in enumerate(state_columns):
        panel[f"velocity_{column}"] = velocity[:, index]
    panel["velocity_observed"] = observed_velocity.astype(int)

    events = episodes.copy()
    events["country_code"] = events.country_code.astype(str).str.upper()
    events["start_year"] = pd.to_numeric(events.start_year, errors="raise").astype(int)
    events["label_end_year"] = pd.to_numeric(
        events.label_end_year, errors="raise"
    ).astype(int)
    events["classification"] = events.classification.astype(str).str.lower()
    classifications = ["systemic"] + (["borderline"] if include_borderline else [])
    events = events.loc[events.classification.isin(classifications)].copy()
    by_country = {
        country: group[["start_year", "label_end_year"]].to_numpy(dtype=int)
        for country, group in events.groupby("country_code", observed=True)
    }

    target = np.zeros(len(panel), dtype=np.int8)
    active = np.zeros(len(panel), dtype=bool)
    cooldown = np.zeros(len(panel), dtype=bool)
    event_start = np.full(len(panel), np.nan)
    for row_index, row in enumerate(
        panel[["entity_code", "forecast_origin_year"]].itertuples(index=False)
    ):
        periods = by_country.get(row.entity_code)
        if periods is None:
            continue
        year = int(row.forecast_origin_year)
        starts, ends = periods[:, 0], periods[:, 1]
        active[row_index] = bool(np.any((starts <= year) & (year <= ends)))
        cooldown[row_index] = bool(
            np.any((ends < year) & (year <= ends + cooldown_years))
        )
        future = starts[
            (starts >= year + horizon_start) & (starts <= year + horizon_end)
        ]
        if len(future):
            target[row_index] = 1
            event_start[row_index] = float(np.min(future))

    panel["crisis_target"] = target
    panel["target_event_start_year"] = event_start
    panel["active_crisis"] = active
    panel["post_crisis_cooldown"] = cooldown
    panel["right_censored"] = (
        panel.forecast_origin_year + horizon_end > label_coverage_end_year
    )
    panel["exclusion_reason"] = np.select(
        [panel.active_crisis, panel.post_crisis_cooldown, panel.right_censored],
        ["active_crisis", "post_crisis_cooldown", "right_censored"],
        default="",
    )
    exclusions = panel.loc[panel.exclusion_reason.ne("")].copy()
    eligible = panel.loc[panel.exclusion_reason.eq("")].copy()
    eligible["horizon_start"] = horizon_start
    eligible["horizon_end"] = horizon_end
    summary = {
        "eligible_rows": len(eligible),
        "eligible_countries": int(eligible.entity_code.nunique()),
        "positive_rows": int(eligible.crisis_target.sum()),
        "positive_rate": float(eligible.crisis_target.mean()),
        "systemic_episodes_used": int(events.classification.eq("systemic").sum()),
        "borderline_episodes_used": int(events.classification.eq("borderline").sum()),
        "active_crisis_exclusions": int(exclusions.active_crisis.sum()),
        "cooldown_exclusions": int(exclusions.post_crisis_cooldown.sum()),
        "right_censored_exclusions": int(exclusions.right_censored.sum()),
        "label_coverage_end_year": label_coverage_end_year,
        "provider_projection_rows_read": 0,
    }
    return eligible, exclusions, summary


@dataclass
class PreparedLogit:
    scaler: StandardScaler
    model: LogisticRegression
    feature_columns: list[str]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.feature_columns].to_numpy(dtype=float)
        return self.model.predict_proba(self.scaler.transform(values))[:, 1]


def _fit_logit(train: pd.DataFrame, feature_columns: list[str], c_value: float):
    scaler = StandardScaler().fit(train[feature_columns].to_numpy(dtype=float))
    values = scaler.transform(train[feature_columns].to_numpy(dtype=float))
    model = LogisticRegression(
        C=float(c_value), solver="liblinear", max_iter=4000, random_state=17
    ).fit(values, train.crisis_target.to_numpy(dtype=int))
    return PreparedLogit(scaler, model, feature_columns)


def _safe_metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    y = np.asarray(y, dtype=int)
    result = {
        "rows": len(y),
        "positives": int(y.sum()),
        "event_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "mean_probability": float(probability.mean()),
    }
    if len(np.unique(y)) == 2:
        result["pr_auc"] = float(average_precision_score(y, probability))
        result["roc_auc"] = float(roc_auc_score(y, probability))
    else:
        result["pr_auc"] = np.nan
        result["roc_auc"] = np.nan
    return result


def _review_threshold(y: np.ndarray, probability: np.ndarray, recall_floor=0.60):
    if int(np.sum(y)) == 0:
        return 1.0
    precision, recall, thresholds = precision_recall_curve(y, probability)
    candidates = [
        (float(precision[index]), float(threshold))
        for index, threshold in enumerate(thresholds)
        if recall[index] >= recall_floor
    ]
    if not candidates:
        return float(np.quantile(probability, 0.90))
    return max(candidates, key=lambda item: (item[0], item[1]))[1]


def _alert_metrics(y: np.ndarray, probability: np.ndarray, threshold: float):
    alert = probability >= threshold
    true_positive = int(np.sum(alert & (y == 1)))
    false_positive = int(np.sum(alert & (y == 0)))
    positives = int(np.sum(y == 1))
    alerts = int(np.sum(alert))
    return {
        "threshold": float(threshold),
        "alerts": alerts,
        "alert_rate": float(alerts / len(y)) if len(y) else np.nan,
        "recall": float(true_positive / positives) if positives else np.nan,
        "precision": float(true_positive / alerts) if alerts else np.nan,
        "false_alerts": false_positive,
        "false_alerts_per_true_alert": (
            float(false_positive / true_positive) if true_positive else np.nan
        ),
    }


def _model_specs(state_columns):
    velocity_columns = [f"velocity_{column}" for column in state_columns]
    return {
        "state_only": list(state_columns),
        "state_velocity_uncertainty": [
            *state_columns, *velocity_columns,
            "state_uncertainty_proxy", "observed_share", "velocity_observed",
        ],
    }


def run_crisis_overlay_development(
    panel: pd.DataFrame,
    *,
    windows=DEFAULT_WINDOWS,
    c_grid=C_GRID,
    minimum_training_positives: int = 15,
    minimum_test_positives: int = 2,
) -> dict:
    """Time-ordered development comparison for the separate event overlay."""
    state_columns = _state_columns(panel)
    predictions, fold_metrics, tuning_rows, skipped = [], [], [], []
    for start, end in windows:
        train = panel.loc[panel.forecast_origin_year + HORIZON_END < start].copy()
        test = panel.loc[panel.forecast_origin_year.between(start, end)].copy()
        inner_start = start - 5
        inner_train = train.loc[
            train.forecast_origin_year + HORIZON_END < inner_start
        ].copy()
        inner_test = train.loc[
            train.forecast_origin_year.between(inner_start, start - 1)
        ].copy()
        identity = {
            "outer_start": start, "outer_end": end,
            "train_rows": len(train), "train_positives": int(train.crisis_target.sum()),
            "test_rows": len(test), "test_positives": int(test.crisis_target.sum()),
            "inner_train_rows": len(inner_train),
            "inner_train_positives": int(inner_train.crisis_target.sum()),
            "inner_test_rows": len(inner_test),
            "inner_test_positives": int(inner_test.crisis_target.sum()),
        }
        if (
            identity["train_positives"] < minimum_training_positives
            or identity["test_positives"] < minimum_test_positives
            or identity["inner_train_positives"] < max(8, minimum_training_positives // 2)
            or identity["inner_test_positives"] < 1
        ):
            skipped.append({**identity, "reason": "insufficient_resolved_events"})
            continue

        baseline_probability = float(train.crisis_target.mean())
        baseline = np.full(len(test), baseline_probability)
        fold_metrics.append({
            **identity, "model": "event_rate",
            **_safe_metrics(test.crisis_target.to_numpy(), baseline),
        })
        frame = test[["entity_code", "forecast_origin_year", "crisis_target"]].copy()
        frame["model"], frame["probability"] = "event_rate", baseline
        predictions.append(frame)

        for model_name, feature_columns in _model_specs(state_columns).items():
            candidates = []
            for c_value in c_grid:
                fitted = _fit_logit(inner_train, feature_columns, c_value)
                metric = _safe_metrics(
                    inner_test.crisis_target.to_numpy(),
                    fitted.predict_proba(inner_test),
                )
                candidates.append({"model": model_name, "c_value": c_value, **metric})
            tuning_rows.extend([{**identity, **row} for row in candidates])
            selected = sorted(
                candidates, key=lambda row: (row["brier"], row["log_loss"], row["c_value"])
            )[0]
            fitted = _fit_logit(train, feature_columns, selected["c_value"])
            probability = fitted.predict_proba(test)
            inner_fitted = _fit_logit(
                inner_train, feature_columns, selected["c_value"]
            )
            threshold = _review_threshold(
                inner_test.crisis_target.to_numpy(),
                inner_fitted.predict_proba(inner_test),
            )
            alert = _alert_metrics(
                test.crisis_target.to_numpy(), probability, threshold
            )
            fold_metrics.append({
                **identity, "model": model_name,
                "selected_c": selected["c_value"],
                **_safe_metrics(test.crisis_target.to_numpy(), probability),
                **{f"review_{key}": value for key, value in alert.items()},
            })
            frame = test[["entity_code", "forecast_origin_year", "crisis_target"]].copy()
            frame["model"] = model_name
            frame["probability"] = probability
            frame["selected_c"] = selected["c_value"]
            frame["review_threshold"] = threshold
            predictions.append(frame)

    if not fold_metrics:
        return {
            "status": "crisis_overlay_stopped_insufficient_time_ordered_events",
            "fold_metrics": pd.DataFrame(),
            "aggregate_metrics": pd.DataFrame(),
            "predictions": pd.DataFrame(),
            "tuning": pd.DataFrame(tuning_rows),
            "skipped": skipped,
            "production_classifier_comparison": (
                "not_evaluated_missing_matched_historical_predictions"
            ),
        }

    folds = pd.DataFrame(fold_metrics)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    aggregate = []
    for model, group in prediction_frame.groupby("model", observed=True):
        aggregate.append({
            "model": model,
            **_safe_metrics(
                group.crisis_target.to_numpy(dtype=int),
                group.probability.to_numpy(dtype=float),
            ),
        })
    aggregate_frame = pd.DataFrame(aggregate).sort_values("brier")
    baseline_brier = float(
        aggregate_frame.loc[aggregate_frame.model.eq("event_rate"), "brier"].iloc[0]
    )
    aggregate_frame["brier_skill_vs_event_rate"] = (
        baseline_brier - aggregate_frame.brier
    ) / baseline_brier
    return {
        "status": "completed_phase4c_crisis_overlay_development",
        "fold_metrics": folds,
        "aggregate_metrics": aggregate_frame,
        "predictions": prediction_frame,
        "tuning": pd.DataFrame(tuning_rows),
        "skipped": skipped,
        "production_classifier_comparison": (
            "not_evaluated_missing_matched_historical_predictions"
        ),
        "state_refit_with_crisis_labels": False,
        "provider_projection_rows_read": 0,
    }


def latest_crisis_overlay(
    panel, latest_states, latest_information, selected_model, selected_c
):
    state_columns = _state_columns(latest_states)
    feature_columns = _model_specs(state_columns)[selected_model]
    latest_year = int(latest_states.forecast_origin_year.max())
    latest = latest_states.loc[
        latest_states.forecast_origin_year.eq(latest_year)
    ].copy()
    previous = latest_states.loc[
        latest_states.forecast_origin_year.eq(latest_year - 1),
        ["entity_code", *state_columns],
    ].rename(columns={column: f"previous_{column}" for column in state_columns})
    latest = latest.merge(previous, on="entity_code", how="left", validate="one_to_one")
    prior = latest[[f"previous_{column}" for column in state_columns]].to_numpy(float)
    current = latest[state_columns].to_numpy(float)
    velocity = current - prior
    observed = np.isfinite(velocity).all(axis=1)
    velocity[~observed] = 0.0
    for index, column in enumerate(state_columns):
        latest[f"velocity_{column}"] = velocity[:, index]
    latest["velocity_observed"] = observed.astype(int)
    latest = latest.merge(
        latest_information[[
            "entity_code", "forecast_origin_year",
            "state_uncertainty_proxy", "observed_share",
        ]],
        on=["entity_code", "forecast_origin_year"],
        how="left", validate="one_to_one",
    )
    fitted = _fit_logit(panel, feature_columns, selected_c)
    latest["crisis_probability_1_to_3y"] = fitted.predict_proba(latest)
    latest["overlay_model"] = selected_model
    latest["overlay_c"] = selected_c
    latest["state_year"] = latest_year
    latest["target_window_start_year"] = latest_year + HORIZON_START
    latest["target_window_end_year"] = latest_year + HORIZON_END
    latest["output_status"] = "retrospective_development_overlay_not_production"
    return latest[[
        "entity_code", "state_year", "target_window_start_year",
        "target_window_end_year", "crisis_probability_1_to_3y",
        "state_uncertainty_proxy", "observed_share", "velocity_observed",
        "overlay_model", "overlay_c", "output_status",
    ]].sort_values("crisis_probability_1_to_3y", ascending=False)


def run(states_path, information_path, episodes_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    states = pd.read_csv(states_path)
    information = pd.read_csv(information_path)
    episodes = pd.read_csv(episodes_path)
    panel, exclusions, panel_summary = build_crisis_overlay_panel(
        states, information, episodes
    )
    result = run_crisis_overlay_development(panel)
    panel[[
        "entity_code", "forecast_origin_year", "crisis_target",
        "target_event_start_year", "state_uncertainty_proxy",
        "observed_share", "velocity_observed",
    ]].to_csv(output / "crisis-overlay-panel.csv.gz", index=False)
    exclusions[[
        "entity_code", "forecast_origin_year", "exclusion_reason",
        "active_crisis", "post_crisis_cooldown", "right_censored",
    ]].to_csv(output / "crisis-overlay-exclusions.csv.gz", index=False)
    result["fold_metrics"].to_csv(output / "crisis-overlay-fold-metrics.csv", index=False)
    result["aggregate_metrics"].to_csv(
        output / "crisis-overlay-aggregate-metrics.csv", index=False
    )
    result["predictions"].to_csv(
        output / "crisis-overlay-predictions.csv.gz", index=False
    )
    result["tuning"].to_csv(output / "crisis-overlay-tuning.csv.gz", index=False)

    selected_model = selected_c = None
    latest_rows = 0
    if not result["aggregate_metrics"].empty:
        challengers = result["aggregate_metrics"].loc[
            ~result["aggregate_metrics"].model.eq("event_rate")
        ].sort_values(["brier", "log_loss"])
        if not challengers.empty:
            selected_model = str(challengers.iloc[0].model)
            model_folds = result["fold_metrics"].loc[
                result["fold_metrics"].model.eq(selected_model)
            ]
            if "selected_c" in model_folds and model_folds.selected_c.notna().any():
                selected_c = float(model_folds.selected_c.mode().iloc[0])
                latest = latest_crisis_overlay(
                    panel, states, information, selected_model, selected_c
                )
                latest.to_csv(output / "latest-crisis-overlay.csv", index=False)
                latest_rows = len(latest)

    aggregate_json = result["aggregate_metrics"].astype(object).where(
        pd.notna(result["aggregate_metrics"]), None
    )
    summary = {
        "status": result["status"],
        "panel": panel_summary,
        "development_windows_executed": int(
            result["fold_metrics"].outer_start.nunique()
        ) if not result["fold_metrics"].empty else 0,
        "aggregate_models": aggregate_json.to_dict("records"),
        "selected_latest_overlay": selected_model,
        "selected_latest_c": selected_c,
        "latest_overlay_rows": latest_rows,
        "production_classifier_comparison": result[
            "production_classifier_comparison"
        ],
        "production_classifier_retrained": False,
        "state_refit_with_crisis_labels": False,
        "provider_projection_rows_read": 0,
        "production_modified": False,
        "final_confirmation_evaluated": False,
    }
    (output / "phase4c-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True)
    parser.add_argument("--information", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.states, args.information, args.episodes, args.output), indent=2))
