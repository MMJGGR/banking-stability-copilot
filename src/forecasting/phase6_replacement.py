"""Phase 6 replacement decision layer.

This module joins the fixed broad state to governed banking/sovereign event
labels and time-safe observed early-warning indicators. It evaluates state,
observed and combined challengers, then constructs the owner-required score:
peer position, own-history stress and absolute event imminence.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.crisis_labels import CrisisLabels
from src.crisis_panel import (
    CrisisPanelConfig,
    IMF_FEATURE_SPECS,
    build_crisis_panel_result,
)
from src.crisis_validation import ValidationConfig, evaluate_forward_temporal
from .phase6_sovereign import SovereignDefaultSource

HORIZON_START = 1
HORIZON_END = 3
PANEL_START_YEAR = 1981
PANEL_END_YEAR = 2021
SCORE_WEIGHT_GRID_STEP = 0.10
SCORE_MIN_COMPONENT_WEIGHT = 0.10


class EpisodeLabels:
    """Small adapter exposing externally governed episodes to crisis_panel."""

    def __init__(self, episodes: pd.DataFrame, coverage_end_year: int):
        required = {"country_code", "start_year", "end_year"}
        if not required <= set(episodes):
            raise ValueError(f"Episode table missing {sorted(required-set(episodes))}")
        self.SOURCE_COVERAGE_END_YEAR = int(coverage_end_year)
        self.crises = {
            str(country): [
                (int(row.start_year), int(row.end_year))
                for row in group.itertuples(index=False)
            ]
            for country, group in episodes.groupby("country_code", observed=True)
        }


def _registry_indicator_labels(snapshot: Path, source: str) -> dict[str, str]:
    registry = pd.read_csv(snapshot / source / "registry.csv", low_memory=False)
    if not {"INDICATOR", "indicator_label"} <= set(registry):
        return {}
    subset = registry[["INDICATOR", "indicator_label"]].drop_duplicates("INDICATOR")
    return dict(zip(subset.INDICATOR.astype(str), subset.indicator_label.astype(str)))


def load_imf_observations(snapshot: str | Path, source: str) -> pd.DataFrame:
    """Load annual historical source rows in crisis_panel's normalized schema.

    WEO rows after 2024 are provider projections and are excluded before the
    panel builder sees them. The registered development panel ends in 2021 and
    uses a one-year feature lag, so current-vintage 2024 values cannot enter
    historical validation either.
    """
    snapshot = Path(snapshot)
    source = source.upper()
    if source not in {"WEO", "FSIC"}:
        raise ValueError("Phase 6 observed overlay currently supports WEO and FSIC")
    labels = _registry_indicator_labels(snapshot, source)
    usecols = ["COUNTRY", "INDICATOR", "FREQUENCY", "TIME_PERIOD", "OBS_VALUE"]
    records = []
    for chunk in pd.read_csv(
        snapshot / source / "raw-response.csv.gz",
        usecols=usecols,
        chunksize=250_000,
        low_memory=False,
    ):
        year = pd.to_numeric(chunk.TIME_PERIOD, errors="coerce")
        value = pd.to_numeric(chunk.OBS_VALUE, errors="coerce")
        mask = (
            chunk.FREQUENCY.astype(str).eq("A")
            & year.between(1960, 2024)
            & value.notna()
            & np.isfinite(value)
        )
        if not mask.any():
            continue
        part = pd.DataFrame({
            "country_code": chunk.loc[mask, "COUNTRY"].astype(str).str.upper(),
            "value": value.loc[mask].astype(float),
            "period": pd.to_datetime(year.loc[mask].astype(int).astype(str), format="%Y"),
            "indicator_code": chunk.loc[mask, "INDICATOR"].astype(str),
        })
        part["indicator_name"] = part.indicator_code.map(labels).fillna(part.indicator_code)
        part["observation_status"] = "unknown"
        part["is_direct"] = True
        records.append(part)
    if not records:
        return pd.DataFrame(columns=[
            "country_code", "value", "period", "indicator_code",
            "indicator_name", "observation_status", "is_direct",
        ])
    data = pd.concat(records, ignore_index=True)
    return data.sort_values(["country_code", "indicator_code", "period"]).reset_index(drop=True)


def _state_columns(states: pd.DataFrame) -> list[str]:
    columns = [
        column for column in states
        if column.startswith("state_") and column.split("_")[-1].isdigit()
    ]
    return sorted(columns, key=lambda name: int(name.split("_")[-1]))


def _add_state_velocity(states: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    state_columns = _state_columns(states)
    current = states[["entity_code", "forecast_origin_year", *state_columns]].copy()
    previous = current.copy()
    previous.forecast_origin_year += 1
    previous = previous.rename(columns={column: f"previous_{column}" for column in state_columns})
    current = current.merge(
        previous,
        on=["entity_code", "forecast_origin_year"],
        how="left",
        validate="one_to_one",
    )
    velocity_columns = []
    for column in state_columns:
        velocity = f"velocity_{column}"
        current[velocity] = current[column] - current[f"previous_{column}"]
        velocity_columns.append(velocity)
    current["velocity_observed"] = current[velocity_columns].notna().all(axis=1).astype(int)
    current[velocity_columns] = current[velocity_columns].fillna(0.0)
    return current.drop(columns=[f"previous_{column}" for column in state_columns]), state_columns, velocity_columns


def _combine_event_ids(bank_event: pd.Series, sovereign_event: pd.Series) -> pd.Series:
    bank = bank_event.astype("string")
    sovereign = sovereign_event.astype("string")
    output = pd.Series(pd.NA, index=bank.index, dtype="string")
    both = bank.notna() & sovereign.notna()
    output.loc[both] = "bank::" + bank.loc[both] + "|sovereign::" + sovereign.loc[both]
    output.loc[bank.notna() & ~both] = "bank::" + bank.loc[bank.notna() & ~both]
    output.loc[sovereign.notna() & ~both] = (
        "sovereign::" + sovereign.loc[sovereign.notna() & ~both]
    )
    return output


@dataclass(frozen=True)
class ReplacementPanel:
    panel: pd.DataFrame
    state_features: list[str]
    observed_features: list[str]
    combined_features: list[str]
    audit: dict


def build_replacement_panel(
    snapshot: str | Path,
    phase2_results: str | Path,
    sovereign: SovereignDefaultSource,
) -> ReplacementPanel:
    snapshot = Path(snapshot)
    phase2_results = Path(phase2_results)
    states = pd.read_csv(phase2_results / "country-year-states.csv.gz")
    information = pd.read_csv(phase2_results / "country-year-information.csv.gz")
    states.entity_code = states.entity_code.astype(str).str.upper()
    country_universe = sorted(states.entity_code.unique())
    config = CrisisPanelConfig(
        start_year=PANEL_START_YEAR,
        end_year=PANEL_END_YEAR,
        horizon_start_years=HORIZON_START,
        horizon_end_years=HORIZON_END,
        feature_lag_years=1,
        exclude_active_crisis=True,
        post_crisis_cooldown_years=3,
        label_coverage_end_year=2024,
        drop_right_censored=True,
        family_min_coverage=0.5,
    )
    weo = load_imf_observations(snapshot, "WEO")
    fsic = load_imf_observations(snapshot, "FSIC")
    bank_result = build_crisis_panel_result(
        weo_df=weo,
        fsic_df=fsic,
        labels=CrisisLabels(include_borderline=False),
        country_universe=country_universe,
        feature_specs=IMF_FEATURE_SPECS,
        config=config,
    )
    sovereign_labels = EpisodeLabels(sovereign.episodes, coverage_end_year=2024)
    sovereign_result = build_crisis_panel_result(
        weo_df=None,
        fsic_df=None,
        labels=sovereign_labels,
        country_universe=country_universe,
        feature_specs=(),
        config=config,
    )
    bank = bank_result.panel.rename(columns={
        "country_code": "entity_code",
        "crisis_target": "banking_target",
        "crisis_event_id": "banking_event_id",
        "crisis_start_year": "banking_event_start_year",
    })
    sovereign_panel = sovereign_result.panel.rename(columns={
        "country_code": "entity_code",
        "crisis_target": "sovereign_target",
        "crisis_event_id": "sovereign_event_id",
        "crisis_start_year": "sovereign_event_start_year",
    })
    panel = bank.merge(
        sovereign_panel[[
            "entity_code", "forecast_origin_year", "sovereign_target",
            "sovereign_event_id", "sovereign_event_start_year",
        ]],
        on=["entity_code", "forecast_origin_year"],
        how="inner",
        validate="one_to_one",
    )
    panel["either_target"] = panel[["banking_target", "sovereign_target"]].max(axis=1)
    panel["either_event_id"] = _combine_event_ids(
        panel.banking_event_id, panel.sovereign_event_id
    )

    state_frame, state_columns, velocity_columns = _add_state_velocity(states)
    panel = panel.merge(
        state_frame,
        on=["entity_code", "forecast_origin_year"],
        how="inner",
        validate="one_to_one",
    ).merge(
        information,
        on=["entity_code", "forecast_origin_year"],
        how="left",
        validate="one_to_one",
    )
    state_features = [
        *state_columns,
        *velocity_columns,
        "velocity_observed",
        "state_uncertainty_proxy",
        "observed_share",
        "effective_information",
    ]
    observed_features: list[str] = []
    for spec in IMF_FEATURE_SPECS:
        for column in (
            spec.name,
            f"{spec.name}__age_years",
            f"{spec.name}__available",
            f"{spec.name}__direct",
        ):
            if column in panel:
                observed_features.append(column)
    family_columns = [
        column for column in panel
        if column.startswith("family_") and (
            column.endswith("__coverage_ratio") or column.endswith("__available")
        )
    ]
    observed_features.extend(sorted(family_columns))
    observed_features = list(dict.fromkeys(observed_features))
    combined_features = list(dict.fromkeys([*state_features, *observed_features]))
    panel = panel.sort_values(["forecast_origin_year", "entity_code"]).reset_index(drop=True)
    audit = {
        "rows": len(panel),
        "countries": int(panel.entity_code.nunique()),
        "origin_start": int(panel.forecast_origin_year.min()),
        "origin_end": int(panel.forecast_origin_year.max()),
        "banking_positive_rows": int(panel.banking_target.sum()),
        "sovereign_positive_rows": int(panel.sovereign_target.sum()),
        "either_positive_rows": int(panel.either_target.sum()),
        "state_features": len(state_features),
        "observed_features": len(observed_features),
        "combined_features": len(combined_features),
        "banking_exclusions": int(len(bank_result.exclusions)),
        "sovereign_exclusions": int(len(sovereign_result.exclusions)),
        "sovereign_source": sovereign.audit,
        "provider_projection_rows_read": 0,
    }
    return ReplacementPanel(panel, state_features, observed_features, combined_features, audit)


def estimator_factory(c_value: float = 0.10):
    return lambda: LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="liblinear",
        max_iter=4000,
        random_state=17,
    )


def validation_config() -> ValidationConfig:
    return ValidationConfig(
        outer_splits=5,
        inner_splits=4,
        calibration="sigmoid",
        recall_floor=0.60,
        random_state=17,
        bootstrap_iterations=100,
        temporal_outer_splits=4,
        temporal_inner_splits=3,
        temporal_min_train_periods=12,
        temporal_gap_periods=0,
        purge_overlapping_events=True,
    )


def validate_candidates(replacement: ReplacementPanel) -> dict:
    panel = replacement.panel
    feature_sets = {
        "state": replacement.state_features,
        "observed": replacement.observed_features,
        "combined": replacement.combined_features,
    }
    target_event = {
        "banking": ("banking_target", "banking_event_id"),
        "sovereign": ("sovereign_target", "sovereign_event_id"),
        "either": ("either_target", "either_event_id"),
    }
    results = {}
    for target_name, (target_column, event_column) in target_event.items():
        results[target_name] = {}
        for model_name, columns in feature_sets.items():
            metadata = panel[["entity_code", "forecast_origin_year", event_column]].rename(
                columns={event_column: "event_id"}
            )
            result = evaluate_forward_temporal(
                estimator_factory(),
                panel[columns],
                panel[target_column].astype(int),
                panel.entity_code,
                panel.forecast_origin_year,
                metadata=metadata,
                config=validation_config(),
            )
            results[target_name][model_name] = result
    return results


def _event_rate_ledger(panel: pd.DataFrame, target_column: str, test_times: Iterable[int]):
    records = []
    for year in sorted(set(int(value) for value in test_times)):
        train = panel.loc[panel.forecast_origin_year < year]
        test = panel.loc[panel.forecast_origin_year.eq(year)]
        probability = float(train[target_column].mean()) if len(train) else float(panel[target_column].mean())
        frame = test[["entity_code", "forecast_origin_year", target_column]].copy()
        frame["proba"] = probability
        frame = frame.rename(columns={"entity_code": "country", target_column: "y"})
        records.append(frame)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def candidate_summary(results: dict, panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_name, models in results.items():
        test_times = []
        for model_name, result in models.items():
            summary = result.summary
            test_times.extend(result.ledger.get("time", pd.Series(dtype=float)).dropna().tolist())
            rows.append({
                "target": target_name,
                "model": model_name,
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
        target_column = f"{target_name}_target"
        if target_name == "either":
            target_column = "either_target"
        baseline = _event_rate_ledger(panel, target_column, test_times)
        if not baseline.empty:
            from src.crisis_validation import classification_metrics

            threshold = float(baseline.proba.quantile(0.90))
            metrics = classification_metrics(
                baseline.y, baseline.proba, baseline.proba.ge(threshold).astype(int)
            )
            rows.append({
                "target": target_name,
                "model": "event_rate",
                "rows": len(baseline),
                "positives": int(baseline.y.sum()),
                "brier": float(metrics["brier"]),
                "log_loss": float(metrics["log_loss"]),
                "pr_auc": float(metrics["average_precision"]),
                "roc_auc": float(metrics["roc_auc"]),
                "recall": float(metrics["recall"]),
                "precision": float(metrics["precision"]),
                "alert_burden": float(metrics["alert_burden"]),
            })
    return pd.DataFrame(rows).sort_values(["target", "brier", "log_loss"]).reset_index(drop=True)


def peer_percentile(signal: pd.Series, years: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"signal": signal.to_numpy(), "year": years.to_numpy()})
    return frame.groupby("year", observed=True).signal.rank(pct=True, method="average")


def historical_percentile(
    signal: pd.Series,
    countries: pd.Series,
    years: pd.Series,
    *,
    shrinkage_observations: float = 5.0,
) -> pd.Series:
    frame = pd.DataFrame({
        "signal": signal.to_numpy(dtype=float),
        "country": countries.astype(str).to_numpy(),
        "year": years.to_numpy(dtype=int),
        "position": np.arange(len(signal)),
    }).sort_values(["country", "year", "position"])
    output = np.full(len(frame), 0.5, dtype=float)
    for _, group in frame.groupby("country", observed=True, sort=False):
        history: list[float] = []
        for row in group.itertuples(index=False):
            if history:
                percentile = (
                    np.searchsorted(np.sort(np.asarray(history)), row.signal, side="right")
                    / len(history)
                )
                weight = len(history) / (len(history) + shrinkage_observations)
                value = weight * percentile + (1 - weight) * 0.5
            else:
                value = 0.5
            output[int(row.position)] = value
            history.append(float(row.signal))
    return pd.Series(output, index=signal.index)


def _weight_grid(step: float = SCORE_WEIGHT_GRID_STEP, minimum: float = SCORE_MIN_COMPONENT_WEIGHT):
    units = int(round(1 / step))
    minimum_units = int(round(minimum / step))
    for peer in range(minimum_units, units + 1):
        for history in range(minimum_units, units - peer + 1):
            absolute = units - peer - history
            if absolute < minimum_units:
                continue
            yield (peer / units, history / units, absolute / units)


def select_score_weights(frame: pd.DataFrame) -> dict:
    required = {"peer", "history", "absolute", "y"}
    if not required <= set(frame):
        raise ValueError(f"Score tuning frame missing {sorted(required-set(frame))}")
    best = None
    for weights in _weight_grid():
        index = (
            weights[0] * frame.peer
            + weights[1] * frame.history
            + weights[2] * frame.absolute
        )
        brier = float(np.mean((frame.y.to_numpy(dtype=float) - index.to_numpy()) ** 2))
        loss = float(-np.mean(
            frame.y * np.log(np.clip(index, 1e-8, 1 - 1e-8))
            + (1 - frame.y) * np.log(np.clip(1 - index, 1e-8, 1 - 1e-8))
        ))
        candidate = (brier, loss, -weights[2], weights)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return {
        "peer_weight": best[3][0],
        "history_weight": best[3][1],
        "absolute_weight": best[3][2],
        "tuning_brier": best[0],
        "tuning_log_loss": best[1],
    }


def build_oof_replacement_scores(either_combined_result) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create outer-test scores with weights selected from inner OOF predictions."""
    output, weight_rows = [], []
    ledger = either_combined_result.ledger.copy()
    tuning = either_combined_result.tuning_ledger.copy()
    for fold in sorted(ledger.outer_fold.unique()):
        test = ledger.loc[ledger.outer_fold.eq(fold)].copy()
        tune = tuning.loc[tuning.outer_fold.eq(fold)].copy()
        for frame in (test, tune):
            if "country" not in frame or "time" not in frame:
                raise ValueError("Validation ledgers must include country/time metadata")
            frame["peer"] = peer_percentile(frame.proba, frame.time)
            frame["history"] = historical_percentile(
                frame.proba, frame.country, frame.time
            )
            frame["absolute"] = frame.proba
        selected = select_score_weights(tune)
        test["peer_weight"] = selected["peer_weight"]
        test["history_weight"] = selected["history_weight"]
        test["absolute_weight"] = selected["absolute_weight"]
        test["replacement_index"] = (
            selected["peer_weight"] * test.peer
            + selected["history_weight"] * test.history
            + selected["absolute_weight"] * test.absolute
        )
        test["replacement_score"] = 1 + 9 * test.replacement_index
        test["risk_category"] = pd.cut(
            test.replacement_score,
            bins=[0, 2, 4, 6, 8, 10.000001],
            labels=[
                "1-2: Very Low Risk",
                "3-4: Low Risk",
                "5-6: Moderate Risk",
                "7-8: High Risk",
                "9-10: Very High Risk",
            ],
            include_lowest=True,
        ).astype(str)
        output.append(test)
        weight_rows.append({"outer_fold": int(fold), **selected})
    return pd.concat(output, ignore_index=True), pd.DataFrame(weight_rows)


def score_diagnostics(scores: pd.DataFrame) -> dict:
    scores = scores.copy()
    scores["score_band"] = pd.cut(
        scores.replacement_score,
        bins=[0, 2, 4, 6, 8, 10.000001],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
    )
    category = scores.groupby("score_band", observed=True).agg(
        rows=("y", "size"),
        positives=("y", "sum"),
        event_rate=("y", "mean"),
        mean_score=("replacement_score", "mean"),
    ).reset_index()
    valid = category.loc[category.rows.ge(10)]
    monotonic = bool(valid.event_rate.is_monotonic_increasing) if len(valid) > 1 else False
    brier = float(np.mean((scores.y - scores.replacement_index) ** 2))
    return {
        "rows": len(scores),
        "positives": int(scores.y.sum()),
        "brier": brier,
        "score_event_rate_monotonic": monotonic,
        "category_table": category.to_dict("records"),
        "mean_peer_weight": float(scores.peer_weight.mean()),
        "mean_history_weight": float(scores.history_weight.mean()),
        "mean_absolute_weight": float(scores.absolute_weight.mean()),
    }
