"""Build compact, offline-only hierarchical crisis-risk artifacts.

This command is deliberately a build step, not an application dependency.  It
loads the annual crisis panel and the trusted cross-sectional risk artifact,
fits transparent horizon-specific hazard experts, evaluates them strictly
forward in time, and writes JSON that Streamlit can read without training or
network access.

The output keeps four concepts separate:

* the production operating-environment / structural score;
* one-year and years-two-to-three systemic-crisis hazards;
* mechanism-level evidence; and
* evidence coverage, which is never described as statistical confidence.

Passing validation gates makes an artifact a ``promotion_candidate`` only.  It
does not mutate the production model or promote itself.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.crisis_hazard import (
    HISTORICAL_CORE_FEATURES,
    MODERN_INCREMENTAL_FEATURES,
    ExpertRoutingConfig,
    HazardExpertRouter,
    RegularizedDiscreteTimeHazardModel,
    build_crisis_target_hierarchy,
    cumulative_incidence_from_annual_hazards,
    event_balanced_sample_weights,
    regularized_cloglog_hazard,
)
from src.crisis_labels import CrisisLabels
from src.crisis_model_features import derive_crisis_model_features
from src.mechanism_evidence import (
    AlertPolicyConfig,
    CANONICAL_RISK_DIRECTIONS,
    apply_alert_policy,
    calculate_mechanism_evidence,
)
from src.model_store import load_model_artifact


SCHEMA_VERSION = 1
MODEL_STATUS_RESEARCH = "research_challenger"
MODEL_STATUS_CANDIDATE = "promotion_candidate"

GOVERNMENT_ALIASES: Mapping[str, str] = {
    "govt_gross_debt_gdp": "govt_debt_gdp",
    "govt_fiscal_balance_gdp": "fiscal_balance_gdp",
    "govt_primary_balance_gdp": "primary_balance_gdp",
}

# These aliases preserve the exact transform used in historical training.  A
# field is filled only when its canonical production value is absent.
CURRENT_HAZARD_ALIASES: Mapping[str, tuple[str, float]] = {
    "gdp_growth_3y_avg": ("gdp_growth_3yr_avg", 1.0),
    "inflation_change_3y": ("inflation_acceleration", 1.0),
    "current_account_change_3y": ("ca_deterioration_3yr", 1.0),
    "govt_debt_change_3y": ("debt_buildup_3yr", 1.0),
    "broad_money_to_reserves": ("m2_to_reserves", 1.0),
    # The production field is reserves / annual imports * 100; convert to
    # months of imports as (percent / 100) * 12.
    "reserves_months_imports": (
        "reserves_to_goods_services_imports",
        0.12,
    ),
    "combined_npl_ratio": ("npl_ratio", 1.0),
    "combined_bank_liquidity": ("liquid_assets_total", 1.0),
}


@dataclass(frozen=True)
class BuildConfig:
    """Frozen, deterministic build and forward-evaluation policy."""

    as_of_date: str
    label_coverage_end_year: int = 2025
    # The official systemic series has no non-borderline onset after 2018.
    # Keep the final event-bearing wave (2014-2018) in the untouched test
    # block; 2019+ confirmed negatives remain in that block as a false-alert
    # check rather than consuming the only usable events for threshold tuning.
    validation_start_year: int = 2009
    test_start_year: int = 2014
    modern_start_year: int = 2000
    minimum_training_feature_coverage: float = 0.05
    minimum_training_row_coverage: float = 0.20
    minimum_positive_events: int = 2
    regularization_c: float = 0.25
    amber_recall_floor: float = 0.60
    minimum_amber_validation_precision: float = 0.05
    maximum_amber_validation_burden: float = 0.25
    red_recall_floor: float = 0.30
    minimum_red_validation_precision: float = 0.25
    default_amber_threshold: float = 0.10
    default_red_threshold: float = 0.20

    def __post_init__(self) -> None:
        date.fromisoformat(self.as_of_date)
        if not self.modern_start_year < self.validation_start_year < self.test_start_year:
            raise ValueError(
                "Expected modern_start_year < validation_start_year < test_start_year"
            )
        if self.label_coverage_end_year < self.test_start_year:
            raise ValueError("label coverage must extend into the forward test period")
        for name in (
            "minimum_training_feature_coverage",
            "minimum_training_row_coverage",
            "amber_recall_floor",
            "minimum_amber_validation_precision",
            "maximum_amber_validation_burden",
            "red_recall_floor",
            "minimum_red_validation_precision",
            "default_amber_threshold",
            "default_red_threshold",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.default_red_threshold < self.default_amber_threshold:
            raise ValueError("default Red threshold cannot be below Amber")
        if self.minimum_positive_events < 1:
            raise ValueError("minimum_positive_events must be positive")
        if self.regularization_c <= 0:
            raise ValueError("regularization_c must be positive")


def _json_ready(value: Any) -> Any:
    """Return strict-JSON primitives, replacing non-finite values with null."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        # Stable, compact precision for UI artifacts.
        return round(value, 8)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_country_frame(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if "country_code" not in frame.columns:
        raise ValueError(f"{source} is missing country_code")
    result = frame.copy()
    result["country_code"] = result["country_code"].astype(str).str.strip().str.upper()
    if result["country_code"].eq("").any():
        raise ValueError(f"{source} contains an empty country code")
    duplicates = result.loc[
        result.duplicated("country_code", keep=False), "country_code"
    ].unique()
    if len(duplicates):
        raise ValueError(
            f"{source} has duplicate country rows: {sorted(duplicates)[:10]}"
        )
    return result.sort_values("country_code").reset_index(drop=True)


def merge_candidate_features(
    production: pd.DataFrame,
    candidate: pd.DataFrame | None,
    *,
    source_name: str,
    aliases: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge governed candidate evidence without suffixes or duplicate votes.

    Production values always win.  A staged candidate may fill a missing
    canonical value or add a canonical feature that production does not yet
    contain, but it cannot overwrite an observed production value.  Unknown
    staged columns remain in the source artifact and are reported as ignored.
    """

    base = _normalise_country_frame(production, source="production feature frame")
    if candidate is None:
        return base, {
            "source": source_name,
            "status": "not_supplied",
            "added_features": [],
            "filled_values": {},
            "preserved_production_values": {},
            "ignored_columns": [],
        }
    staged = _normalise_country_frame(candidate, source=source_name)
    aliases = dict(aliases or {})
    allowed = set(CANONICAL_RISK_DIRECTIONS)
    merged = base.set_index("country_code")
    staged = staged.set_index("country_code")
    report: dict[str, Any] = {
        "source": source_name,
        "status": "merged",
        "countries": int(len(staged)),
        "added_features": [],
        "filled_values": {},
        "preserved_production_values": {},
        "ignored_columns": [],
    }
    seen_canonical: set[str] = set()
    for staged_column in staged.columns:
        canonical = aliases.get(staged_column, staged_column)
        if canonical not in allowed:
            report["ignored_columns"].append(staged_column)
            continue
        # When two candidate aliases resolve to the same field, accept only
        # the first deterministic occurrence.  Exact names should be ordered
        # before weaker aliases by the caller/source schema.
        if canonical in seen_canonical:
            report["ignored_columns"].append(staged_column)
            continue
        seen_canonical.add(canonical)
        values = pd.to_numeric(staged[staged_column], errors="coerce").reindex(
            merged.index
        )
        if canonical not in merged.columns:
            merged[canonical] = values
            report["added_features"].append(canonical)
            report["filled_values"][canonical] = int(values.notna().sum())
            report["preserved_production_values"][canonical] = 0
            continue
        current = pd.to_numeric(merged[canonical], errors="coerce")
        fill_mask = current.isna() & values.notna()
        preserve_mask = current.notna() & values.notna()
        merged.loc[fill_mask, canonical] = values.loc[fill_mask]
        report["filled_values"][canonical] = int(fill_mask.sum())
        report["preserved_production_values"][canonical] = int(preserve_mask.sum())

    for key in ("added_features", "ignored_columns"):
        report[key] = sorted(set(report[key]))
    return merged.reset_index(), report


def _event_labels_from_panel(panel: pd.DataFrame) -> SimpleNamespace:
    """Build the minimal ``.crises`` contract used by feature derivation."""

    official = CrisisLabels(include_borderline=False)
    crises = {key: list(periods) for key, periods in official.crises.items()}
    if "crisis_start_year" not in panel.columns:
        return SimpleNamespace(crises=crises)
    columns = ["country_code", "crisis_start_year"]
    if "crisis_end_year" in panel.columns:
        columns.append("crisis_end_year")
    events = panel.loc[panel["crisis_start_year"].notna(), columns].copy()
    if events.empty:
        return SimpleNamespace(crises=crises)
    events["country_code"] = events["country_code"].astype(str).str.upper()
    events["crisis_start_year"] = pd.to_numeric(
        events["crisis_start_year"], errors="raise"
    ).astype(int)
    if "crisis_end_year" not in events:
        events["crisis_end_year"] = events["crisis_start_year"]
    events["crisis_end_year"] = pd.to_numeric(
        events["crisis_end_year"], errors="coerce"
    ).fillna(events["crisis_start_year"]).astype(int)
    for row in events.drop_duplicates().itertuples(index=False):
        period = (int(row.crisis_start_year), int(row.crisis_end_year))
        crises.setdefault(str(row.country_code), [])
        if period not in crises[str(row.country_code)]:
            crises[str(row.country_code)].append(period)
            crises[str(row.country_code)].sort()
    return SimpleNamespace(crises=crises)


def _target_events_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Return exact onset events, falling back to the pinned official labels."""

    if "crisis_start_year" in panel.columns:
        columns = ["country_code", "crisis_start_year"]
        if "crisis_end_year" in panel.columns:
            columns.append("crisis_end_year")
        if "crisis_event_id" in panel.columns:
            columns.append("crisis_event_id")
        events = panel.loc[panel["crisis_start_year"].notna(), columns].copy()
        if not events.empty:
            if "crisis_end_year" not in events:
                events["crisis_end_year"] = events["crisis_start_year"]
            if "crisis_event_id" not in events:
                events["crisis_event_id"] = (
                    events["country_code"].astype(str).str.upper()
                    + "-"
                    + pd.to_numeric(events["crisis_start_year"], errors="raise")
                    .astype(int)
                    .astype(str)
                    + "-"
                    + pd.to_numeric(events["crisis_end_year"], errors="coerce")
                    .fillna(events["crisis_start_year"])
                    .astype(int)
                    .astype(str)
                )
            return events.drop_duplicates().reset_index(drop=True)
    official = CrisisLabels(include_borderline=False).get_episode_table(
        preserve_source_order=True
    )
    official = official.rename(
        columns={
            "start_year": "crisis_start_year",
            "label_end_year": "crisis_end_year",
        }
    )
    official["crisis_event_id"] = (
        official["country_code"].astype(str)
        + "-"
        + official["crisis_start_year"].astype(int).astype(str)
        + "-"
        + official["crisis_end_year"].astype(int).astype(str)
    )
    return official[
        ["country_code", "crisis_start_year", "crisis_end_year", "crisis_event_id"]
    ]


def _load_bis_history(path: Path, *, as_of_date: str) -> tuple[pd.DataFrame, dict]:
    """Reduce optional long-form BIS history to a latest current cross-section."""

    frame = pd.read_parquet(path)
    required = {"country_code", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"BIS history missing required columns: {sorted(missing)}")
    feature_col = "feature" if "feature" in frame else "indicator_code"
    if feature_col not in frame:
        raise ValueError("BIS history needs feature or indicator_code")
    if "year" in frame:
        years = pd.to_numeric(frame["year"], errors="coerce")
    elif "period" in frame:
        years = pd.to_numeric(frame["period"].astype(str).str.slice(0, 4), errors="coerce")
    else:
        raise ValueError("BIS history needs year or period")
    cutoff_year = date.fromisoformat(as_of_date).year
    long = frame.assign(_year=years)
    long = long.loc[long["_year"].notna() & long["_year"].le(cutoff_year)].copy()
    long["country_code"] = long["country_code"].astype(str).str.upper()
    long[feature_col] = long[feature_col].astype(str)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    latest = (
        long.sort_values(["country_code", feature_col, "_year"])
        .drop_duplicates(["country_code", feature_col], keep="last")
    )
    current = latest.pivot(index="country_code", columns=feature_col, values="value")
    current = current.reset_index().rename_axis(columns=None)
    observed_years = latest["_year"]
    return current, {
        "status": "loaded",
        "rows": int(len(frame)),
        "countries": int(frame["country_code"].nunique()),
        "latest_observation_year": (
            int(observed_years.max()) if not observed_years.empty else None
        ),
        "feature_coverage": {
            str(key): int(value)
            for key, value in latest.groupby(feature_col)["country_code"].nunique().items()
        },
    }


def _merge_bis_panel_history(panel: pd.DataFrame, path: Path) -> pd.DataFrame:
    """As-of merge optional BIS observations into the annual training panel."""

    long = pd.read_parquet(path)
    if not {"country_code", "value"}.issubset(long.columns):
        raise ValueError("BIS history needs country_code and value")
    feature_col = "feature" if "feature" in long else "indicator_code"
    if feature_col not in long:
        raise ValueError("BIS history needs feature or indicator_code")
    years = (
        pd.to_numeric(long["year"], errors="coerce")
        if "year" in long
        else pd.to_numeric(long["period"].astype(str).str.slice(0, 4), errors="coerce")
    )
    observations = long.assign(_year=years)
    observations["country_code"] = observations["country_code"].astype(str).str.upper()
    observations[feature_col] = observations[feature_col].astype(str)
    observations["value"] = pd.to_numeric(observations["value"], errors="coerce")
    observations = observations.dropna(subset=["_year", "value"])
    result = panel.copy()
    cutoff_col = (
        "feature_cutoff_year" if "feature_cutoff_year" in result else "forecast_origin_year"
    )
    for feature in sorted(observations[feature_col].unique()):
        source = observations.loc[
            observations[feature_col].eq(feature),
            ["country_code", "_year", "value"],
        ].sort_values(["_year", "country_code"])
        left = result[["country_code", cutoff_col]].copy()
        left["_row"] = np.arange(len(left))
        left = left.sort_values([cutoff_col, "country_code"])
        matched = pd.merge_asof(
            left,
            source,
            left_on=cutoff_col,
            right_on="_year",
            by="country_code",
            direction="backward",
            allow_exact_matches=True,
        ).sort_values("_row")
        age = pd.to_numeric(result[cutoff_col], errors="coerce").to_numpy() - matched[
            "_year"
        ].to_numpy()
        values = matched["value"].to_numpy(dtype=float)
        available = np.isfinite(values) & np.isfinite(age) & (age >= 0) & (age <= 2)
        # A pre-existing panel field remains authoritative, including its
        # original availability flag.  The optional BIS source may only fill
        # absent values; it cannot mark an existing direct observation stale.
        prior_available = (
            result[f"{feature}__available"].fillna(False).astype(bool)
            if f"{feature}__available" in result
            else pd.to_numeric(result.get(feature), errors="coerce").notna()
            if feature in result
            else pd.Series(False, index=result.index)
        )
        if feature in result:
            existing = pd.to_numeric(result[feature], errors="coerce")
            fill = existing.isna().to_numpy() & available
            result.loc[fill, feature] = values[fill]
        else:
            result[feature] = np.where(available, values, np.nan)
        result[f"{feature}__available"] = pd.to_numeric(
            result[feature], errors="coerce"
        ).notna() & (prior_available.to_numpy() | available)
    return result


def _prepare_training_features(
    panel: pd.DataFrame,
    config: BuildConfig,
    *,
    bis_history_path: Path | None = None,
) -> pd.DataFrame:
    required = {"country_code", "forecast_origin_year"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"annual crisis panel missing columns: {sorted(missing)}")
    work = panel.copy()
    work["country_code"] = work["country_code"].astype(str).str.upper()
    work["forecast_origin_year"] = pd.to_numeric(
        work["forecast_origin_year"], errors="raise"
    ).astype(int)
    if "crisis_target" not in work:
        work["crisis_target"] = 0
    if bis_history_path is not None:
        work = _merge_bis_panel_history(work, bis_history_path)
    labels = _event_labels_from_panel(work)
    existing_target_inputs = {"years_to_crisis", "crisis_event_id"}.issubset(
        work.columns
    )
    targets = build_crisis_target_hierarchy(
        work,
        events=None if existing_target_inputs else _target_events_from_panel(work),
        label_coverage_end_year=config.label_coverage_end_year,
    )
    derived = derive_crisis_model_features(targets, labels)
    # Preserve the exact leakage/contamination flags from the annual panel.
    keep = pd.Series(True, index=derived.index)
    for column in ("active_crisis", "post_crisis_cooldown"):
        if column in derived:
            keep &= ~derived[column].fillna(False).astype(bool)
    return derived.loc[keep].sort_values(
        ["country_code", "forecast_origin_year"]
    ).reset_index(drop=True)


def _prepare_current_features(
    artifact: Mapping[str, Any],
    *,
    external_candidates: pd.DataFrame | None,
    government_candidates: pd.DataFrame | None,
    bis_current: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[dict]]:
    scores = artifact.get("country_scores")
    values = artifact.get("feature_values")
    if not isinstance(scores, pd.DataFrame) or not isinstance(values, pd.DataFrame):
        raise TypeError("risk model needs DataFrame country_scores and feature_values")
    scores = _normalise_country_frame(scores, source="risk_model.country_scores")
    values = _normalise_country_frame(values, source="risk_model.feature_values")
    current = scores.merge(values, on="country_code", how="left", suffixes=("", "_feature"))

    reports: list[dict] = []
    current, report = merge_candidate_features(
        current,
        government_candidates,
        source_name="government_liquidity_candidates",
        aliases=GOVERNMENT_ALIASES,
    )
    reports.append(report)
    current, report = merge_candidate_features(
        current,
        external_candidates,
        source_name="external_liquidity_candidates",
    )
    reports.append(report)
    current, report = merge_candidate_features(
        current,
        bis_current,
        source_name="bis_financial_history",
    )
    reports.append(report)

    for canonical, (source, multiplier) in CURRENT_HAZARD_ALIASES.items():
        if source not in current:
            continue
        source_values = pd.to_numeric(current[source], errors="coerce") * multiplier
        if canonical in current:
            canonical_values = pd.to_numeric(current[canonical], errors="coerce")
            current[canonical] = canonical_values.combine_first(source_values)
        else:
            current[canonical] = source_values
        current[f"{canonical}__available"] = pd.to_numeric(
            current[canonical], errors="coerce"
        ).notna()

    if "crisis_recency_10y" not in current and "years_since_banking_crisis" in current:
        years = pd.to_numeric(current["years_since_banking_crisis"], errors="coerce")
        current["crisis_recency_10y"] = (10.0 - years).clip(lower=0)
        current["crisis_recency_10y__available"] = years.notna()
    return current.sort_values("country_code").reset_index(drop=True), reports


def _select_training_features(
    frame: pd.DataFrame,
    features: Sequence[str],
    minimum_coverage: float,
) -> list[str]:
    selected = []
    for feature in features:
        if feature not in frame:
            continue
        coverage = pd.to_numeric(frame[feature], errors="coerce").notna().mean()
        if coverage >= minimum_coverage:
            selected.append(feature)
    return selected


def _fit_one_expert(
    frame: pd.DataFrame,
    *,
    expert: str,
    features: Sequence[str],
    config: BuildConfig,
    label_available_before: int | None = None,
) -> tuple[dict[int, RegularizedDiscreteTimeHazardModel], dict[str, Any]]:
    models: dict[int, RegularizedDiscreteTimeHazardModel] = {}
    report: dict[str, Any] = {"expert": expert, "horizons": {}}
    for horizon in (1, 2, 3):
        target_col = f"crisis_hazard_year_{horizon}"
        observed = pd.to_numeric(frame[target_col], errors="coerce").notna()
        fit_frame = frame.loc[observed].copy()
        if label_available_before is not None:
            # A horizon-h label resolves in ``origin + h``.  Requiring that
            # year to precede the next evaluation origin creates a proper
            # horizon embargo and prevents one crisis episode from leaking
            # across train/validation or validation/test waves.
            label_year = pd.to_numeric(
                fit_frame["forecast_origin_year"], errors="coerce"
            ).add(horizon)
            fit_frame = fit_frame.loc[label_year.lt(label_available_before)].copy()
        selected = _select_training_features(
            fit_frame, features, config.minimum_training_feature_coverage
        )
        if selected:
            row_coverage = fit_frame[selected].apply(
                pd.to_numeric, errors="coerce"
            ).notna().mean(axis=1)
            fit_frame = fit_frame.loc[
                row_coverage.ge(config.minimum_training_row_coverage)
            ].copy()
        y = pd.to_numeric(fit_frame.get(target_col), errors="coerce")
        positive_events = int(
            fit_frame.loc[y.eq(1), "hazard_event_id"].astype("string").nunique()
        ) if "hazard_event_id" in fit_frame else 0
        horizon_report = {
            "target": target_col,
            "horizon_years": horizon,
            "label_available_before": label_available_before,
            "training_rows": int(len(fit_frame)),
            "training_countries": int(fit_frame["country_code"].nunique()),
            "positive_rows": int(y.eq(1).sum()),
            "positive_events": positive_events,
            "selected_features": selected,
            "status": "unavailable",
        }
        if not fit_frame.empty:
            horizon_report["training_year_start"] = int(
                fit_frame["forecast_origin_year"].min()
            )
            horizon_report["training_year_end"] = int(
                fit_frame["forecast_origin_year"].max()
            )
        if not selected:
            horizon_report["reason"] = "no feature met the training coverage contract"
        elif y.nunique(dropna=True) < 2:
            horizon_report["reason"] = "training target contains fewer than two classes"
        elif positive_events < config.minimum_positive_events:
            horizon_report["reason"] = (
                "too few distinct crisis episodes for governed fitting"
            )
        else:
            weights = event_balanced_sample_weights(
                fit_frame,
                target_col=target_col,
                event_id_col="hazard_event_id",
            )
            model = regularized_cloglog_hazard(
                C=config.regularization_c,
                class_weight=None,
                metadata={
                    "expert": expert,
                    "target": target_col,
                    "as_of_date": config.as_of_date,
                    "training_year_start": int(fit_frame["forecast_origin_year"].min()),
                    "training_year_end": int(fit_frame["forecast_origin_year"].max()),
                    "label_available_before": label_available_before,
                    "event_balanced": True,
                },
            )
            model.fit(
                fit_frame[selected],
                y.astype(int).to_numpy(),
                sample_weight=weights.to_numpy(),
            )
            models[horizon] = model
            horizon_report["status"] = "fitted"
            horizon_report["optimization_success"] = bool(
                model.optimization_success_
            )
        report["horizons"][str(horizon)] = horizon_report
    report["complete"] = set(models) == {1, 2, 3}
    return models, report


def _fit_experts(
    features: pd.DataFrame,
    *,
    config: BuildConfig,
    router_config: ExpertRoutingConfig,
    label_available_before: int | None,
) -> tuple[dict[str, dict[int, RegularizedDiscreteTimeHazardModel]], dict]:
    subset = features
    expert_models: dict[str, dict[int, RegularizedDiscreteTimeHazardModel]] = {}
    reports: dict[str, Any] = {}
    specifications = {
        "historical_core": router_config.historical_features,
        "modern_full": router_config.modern_full_features,
    }
    for expert, feature_names in specifications.items():
        expert_frame = subset
        if expert == "modern_full":
            expert_frame = expert_frame.loc[
                expert_frame["forecast_origin_year"].ge(config.modern_start_year)
            ]
        models, report = _fit_one_expert(
            expert_frame,
            expert=expert,
            features=feature_names,
            config=config,
            label_available_before=label_available_before,
        )
        reports[expert] = report
        if set(models) == {1, 2, 3}:
            expert_models[expert] = models
    return expert_models, reports


def _predict_routed(
    frame: pd.DataFrame,
    models: Mapping[str, Mapping[int, RegularizedDiscreteTimeHazardModel]],
    router_config: ExpertRoutingConfig,
) -> pd.DataFrame:
    router = HazardExpertRouter(router_config)
    routing = router.route(frame)
    historical_available = "historical_core" in models
    modern_available = "modern_full" in models
    selected = routing["selected_expert"].copy()
    if not modern_available:
        fallback = selected.eq("modern_full") & historical_available
        selected.loc[fallback] = "historical_core"
        routing.loc[fallback, "routing_reason"] = (
            "modern expert unavailable; fell back to eligible historical core"
        )
    if not historical_available:
        selected.loc[selected.eq("historical_core")] = "insufficient_evidence"
    if not modern_available:
        selected.loc[selected.eq("modern_full")] = "insufficient_evidence"
    routing["selected_expert"] = selected

    hazards = np.full((len(frame), 3), np.nan, dtype=float)
    for expert, feature_names in (
        ("historical_core", router_config.historical_features),
        ("modern_full", router_config.modern_full_features),
    ):
        if expert not in models:
            continue
        mask = selected.eq(expert)
        if not mask.any():
            continue
        positions = np.flatnonzero(mask.to_numpy())
        inputs = frame.reindex(columns=list(feature_names)).loc[mask]
        for horizon in (1, 2, 3):
            hazards[positions, horizon - 1] = models[expert][horizon].predict_hazard(
                inputs
            )
    probability_columns = [
        "annual_hazard_1y",
        "annual_hazard_year_2",
        "annual_hazard_year_3",
        "probability_1y",
        "probability_within_2y",
        "probability_years_2_3",
        "probability_within_3y",
    ]
    probabilities = pd.DataFrame(
        np.nan, index=frame.index, columns=probability_columns
    )
    complete = np.isfinite(hazards).all(axis=1)
    if complete.any():
        positions = np.flatnonzero(complete)
        derived = cumulative_incidence_from_annual_hazards(
            hazards[complete], index=frame.index[positions]
        )
        probabilities.loc[derived.index, derived.columns] = derived
    return routing.join(probabilities)


def _confusion(y: np.ndarray, prediction: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def _operating_point(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    predicted = (probability >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "flagged": int(predicted.sum()),
        "confusion_matrix": _confusion(y, predicted),
    }


def _select_threshold(
    y: pd.Series,
    probability: pd.Series,
    *,
    recall_floor: float,
    fallback: float,
) -> dict[str, Any]:
    observed = pd.to_numeric(y, errors="coerce").notna() & pd.to_numeric(
        probability, errors="coerce"
    ).notna()
    y_array = pd.to_numeric(y.loc[observed], errors="raise").astype(int).to_numpy()
    p_array = pd.to_numeric(probability.loc[observed], errors="raise").to_numpy()
    if len(y_array) == 0 or len(np.unique(y_array)) < 2 or y_array.sum() < 2:
        return {
            "threshold": float(fallback),
            "source": "provisional_default_insufficient_validation_events",
            "recall_floor": recall_floor,
        }
    candidates: list[tuple[float, float, float, float]] = []
    for threshold in np.unique(p_array):
        point = _operating_point(y_array, p_array, float(threshold))
        if point["recall"] + 1e-12 >= recall_floor:
            candidates.append(
                (point["precision"], point["f1"], float(threshold), point["recall"])
            )
    if not candidates:
        return {
            "threshold": float(fallback),
            "source": "provisional_default_unmet_recall_floor",
            "recall_floor": recall_floor,
        }
    precision, f1, threshold, recall = max(candidates)
    return {
        "threshold": float(np.clip(threshold, 1e-6, 1 - 1e-6)),
        "source": "forward_validation",
        "recall_floor": recall_floor,
        "validation_precision": precision,
        "validation_recall": recall,
        "validation_f1": f1,
        "validation_flagged": int((p_array >= threshold).sum()),
        "validation_rows": int(len(p_array)),
        "validation_alert_burden": float((p_array >= threshold).mean()),
    }


def _metric_block(
    y: pd.Series,
    probability: pd.Series,
    *,
    operating_points: Mapping[str, float],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    observed = pd.to_numeric(y, errors="coerce").notna() & pd.to_numeric(
        probability, errors="coerce"
    ).notna()
    y_array = pd.to_numeric(y.loc[observed], errors="raise").astype(int).to_numpy()
    p_array = pd.to_numeric(probability.loc[observed], errors="raise").to_numpy()
    positions = np.flatnonzero(observed.to_numpy())
    countries = frame.iloc[positions]["country_code"].nunique() if len(positions) else 0
    event_ids = (
        frame.iloc[positions].loc[
            pd.Series(y_array, index=frame.index[positions]).eq(1).to_numpy(),
            "hazard_event_id",
        ].astype("string").nunique()
        if len(positions) and "hazard_event_id" in frame
        else 0
    )
    block: dict[str, Any] = {
        "rows": int(len(y_array)),
        "countries": int(countries),
        "positive_rows": int(y_array.sum()) if len(y_array) else 0,
        "positive_events": int(event_ids),
        "prevalence": float(y_array.mean()) if len(y_array) else np.nan,
        "roc_auc": (
            float(roc_auc_score(y_array, p_array))
            if len(np.unique(y_array)) == 2
            else np.nan
        ),
        "average_precision": (
            float(average_precision_score(y_array, p_array))
            if len(y_array) and y_array.sum()
            else np.nan
        ),
        "brier": (
            float(brier_score_loss(y_array, p_array)) if len(y_array) else np.nan
        ),
    }
    for label, threshold in operating_points.items():
        point = (
            _operating_point(y_array, p_array, threshold)
            if len(y_array)
            else {
                "threshold": threshold,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "flagged": 0,
                "confusion_matrix": {
                    "true_negative": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "true_positive": 0,
                },
            }
        )
        if len(y_array):
            observed_frame = frame.loc[observed].reset_index(drop=True)
            predicted = p_array >= threshold
            event_ids = observed_frame.get(
                "hazard_event_id", pd.Series(pd.NA, index=observed_frame.index)
            ).astype("string")
            positive_events = set(event_ids[(y_array == 1) & event_ids.notna()].tolist())
            captured_events = set(
                event_ids[(y_array == 1) & predicted & event_ids.notna()].tolist()
            )
            point["unique_events"] = len(positive_events)
            point["captured_events"] = len(captured_events)
            point["unique_event_recall"] = (
                len(captured_events) / len(positive_events)
                if positive_events else np.nan
            )
            point["false_alerts_per_100_country_years"] = (
                100.0 * point["confusion_matrix"]["false_positive"] / len(y_array)
            )
        block[label] = point
    return block


def _forward_validation(
    features: pd.DataFrame,
    *,
    config: BuildConfig,
    router_config: ExpertRoutingConfig,
) -> tuple[dict, dict[str, Any], dict[str, Any]]:
    train_models, train_reports = _fit_experts(
        features,
        config=config,
        router_config=router_config,
        label_available_before=config.validation_start_year,
    )
    validation_frame = features.loc[
        features["forecast_origin_year"].between(
            config.validation_start_year, config.test_start_year - 1
        )
    ].reset_index(drop=True)
    validation_predictions = _predict_routed(
        validation_frame, train_models, router_config
    )
    validation_origin = pd.to_numeric(
        validation_frame["forecast_origin_year"], errors="coerce"
    )
    safe_one_year = validation_origin.add(1).lt(config.test_start_year)
    safe_three_year = validation_origin.add(3).lt(config.test_start_year)

    amber = _select_threshold(
        validation_frame.loc[safe_one_year, "crisis_hazard_year_1"],
        validation_predictions.loc[safe_one_year, "probability_1y"],
        recall_floor=config.amber_recall_floor,
        fallback=config.default_amber_threshold,
    )
    amber_precision = amber.get("validation_precision")
    amber_burden = amber.get("validation_alert_burden")
    if (
        amber.get("source") != "forward_validation"
        or amber_precision is None
        or not np.isfinite(amber_precision)
        or amber_precision < config.minimum_amber_validation_precision
        or amber_burden is None
        or not np.isfinite(amber_burden)
        or amber_burden > config.maximum_amber_validation_burden
    ):
        amber = {
            **amber,
            "threshold": 1.0,
            "source": "disabled_no_usable_validation_operating_point",
            "minimum_validation_precision": config.minimum_amber_validation_precision,
            "maximum_validation_alert_burden": config.maximum_amber_validation_burden,
            "reason": (
                "Amber reviews remain disabled until temporal validation meets "
                "both the precision and analyst-capacity burden contracts"
            ),
        }
    red = _select_threshold(
        validation_frame.loc[safe_one_year, "crisis_hazard_year_1"],
        validation_predictions.loc[safe_one_year, "probability_1y"],
        recall_floor=config.red_recall_floor,
        fallback=config.default_red_threshold,
    )
    red["threshold"] = max(float(red["threshold"]), float(amber["threshold"]))
    # A Red tier is meaningful only if the forward validation block actually
    # demonstrates precision.  With a weak ranker, relaxing the recall floor
    # can return the same low threshold as Amber.  Disable Red rather than
    # dressing that result up as a precision-oriented alert.
    red_precision = red.get("validation_precision")
    if (
        red["source"] != "forward_validation"
        or red_precision is None
        or not np.isfinite(red_precision)
        or red_precision < config.minimum_red_validation_precision
        or float(red["threshold"]) <= float(amber["threshold"]) + 1e-12
    ):
        red = {
            "threshold": 1.0,
            "source": "disabled_no_validated_precision_separation",
            "recall_floor": config.red_recall_floor,
            "minimum_validation_precision": config.minimum_red_validation_precision,
            "validation_precision": red_precision,
            "reason": (
                "Red alerts remain disabled until forward validation demonstrates "
                "a distinct precision-oriented operating point"
            ),
        }
    medium = _select_threshold(
        validation_frame.loc[safe_three_year, "crisis_onset_in_years_2_3"],
        validation_predictions.loc[safe_three_year, "probability_years_2_3"],
        recall_floor=config.amber_recall_floor,
        fallback=config.default_amber_threshold,
    )
    cumulative = _select_threshold(
        validation_frame.loc[safe_three_year, "crisis_onset_within_3y"],
        validation_predictions.loc[safe_three_year, "probability_within_3y"],
        recall_floor=config.amber_recall_floor,
        fallback=config.default_amber_threshold,
    )
    thresholds = {
        "policy_version": "forward-validation-v1",
        "status": (
            "validation_selected"
            if amber["source"] == red["source"] == "forward_validation"
            else "provisional"
        ),
        "one_year": {"amber": amber, "red": red},
        "years_2_3_evaluation": medium,
        "within_3y_evaluation": cumulative,
        "confidence_semantics": (
            "evidence_confidence is observed feature coverage, not statistical confidence"
        ),
    }

    evaluation_models, evaluation_reports = _fit_experts(
        features,
        config=config,
        router_config=router_config,
        label_available_before=config.test_start_year,
    )
    test_frame = features.loc[
        features["forecast_origin_year"].ge(config.test_start_year)
    ].reset_index(drop=True)
    test_predictions = _predict_routed(test_frame, evaluation_models, router_config)
    metrics = {
        "1y": _metric_block(
            test_frame["crisis_hazard_year_1"],
            test_predictions["probability_1y"],
            operating_points={
                "amber": float(amber["threshold"]),
                "red": float(red["threshold"]),
            },
            frame=test_frame,
        ),
        "2_3y": _metric_block(
            test_frame["crisis_onset_in_years_2_3"],
            test_predictions["probability_years_2_3"],
            operating_points={"review": float(medium["threshold"])},
            frame=test_frame,
        ),
        "3y": _metric_block(
            test_frame["crisis_onset_within_3y"],
            test_predictions["probability_within_3y"],
            operating_points={"review": float(cumulative["threshold"])},
            frame=test_frame,
        ),
    }
    reports = {
        "threshold_training_experts": train_reports,
        "forward_test_experts": evaluation_reports,
    }
    return metrics, thresholds, reports


def _promotion_gates(metrics: Mapping[str, Any]) -> dict[str, Any]:
    one = metrics["1y"]
    three = metrics["3y"]
    gate_specs = (
        ("1y_roc_auc", one["roc_auc"], 0.70, ">="),
        ("1y_average_precision", one["average_precision"], max(0.15, 2 * (one["prevalence"] or 0)), ">="),
        ("1y_amber_recall", one["amber"]["recall"], 0.60, ">="),
        ("1y_red_precision", one["red"]["precision"], 0.50, ">="),
        ("3y_roc_auc", three["roc_auc"], 0.70, ">="),
        ("3y_average_precision", three["average_precision"], max(0.20, 2 * (three["prevalence"] or 0)), ">="),
        ("forward_test_countries", one["countries"], 50, ">="),
        ("forward_test_crisis_events", one["positive_events"], 5, ">="),
    )
    checks = []
    for name, actual, required, comparison in gate_specs:
        passed = actual is not None and np.isfinite(actual) and actual >= required
        checks.append(
            {
                "name": name,
                "actual": actual,
                "required": required,
                "comparison": comparison,
                "passed": bool(passed),
            }
        )
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "effect": (
            "eligible for owner review; this builder does not promote"
            if all(check["passed"] for check in checks)
            else "remain research/challenger"
        ),
    }


def _direction(signal_strength: float | None) -> str:
    if signal_strength is None or not np.isfinite(signal_strength):
        return "unavailable"
    if signal_strength >= 0.67:
        return "adverse"
    if signal_strength <= 0.33:
        return "supportive"
    return "balanced"


def _alert_reason(row: pd.Series, thresholds: Mapping[str, Any]) -> str:
    status = str(row["alert_level"])
    if status == "red":
        return "Red threshold met with corroborating mechanism evidence."
    if status == "amber":
        blockers = tuple(row.get("red_blockers", ()))
        if blockers:
            return "Amber review; Red conditions not met: " + ", ".join(blockers) + "."
        return "One-year hazard meets the Amber review threshold."
    if status == "insufficient_evidence":
        return "Feature evidence is insufficient for a governed alert."
    if thresholds["one_year"]["amber"].get("source", "").startswith("disabled"):
        return "No review tier is issued because validation found no usable operating point."
    amber = thresholds["one_year"]["amber"]["threshold"]
    return f"One-year hazard is below the Amber threshold ({amber:.1%})."


def _country_records(
    current: pd.DataFrame,
    predictions: pd.DataFrame,
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evidence = calculate_mechanism_evidence(
        current,
        reference=current,
        identifier_columns=("country_code",),
    )
    policy = AlertPolicyConfig(
        amber_hazard_threshold=float(thresholds["one_year"]["amber"]["threshold"]),
        red_hazard_threshold=float(thresholds["one_year"]["red"]["threshold"]),
        require_persistence_for_red=False,
        policy_version=str(thresholds["policy_version"]),
    )
    alerts = apply_alert_policy(
        predictions["probability_1y"], evidence, config=policy
    )
    records: list[dict[str, Any]] = []
    for position, row in current.reset_index(drop=True).iterrows():
        prediction = predictions.iloc[position]
        alert = alerts.iloc[position]
        mechanism_rows = evidence.mechanism_evidence.loc[
            evidence.mechanism_evidence["row_position"].eq(position)
        ]
        mechanisms = []
        for mechanism in mechanism_rows.itertuples(index=False):
            strength = (
                float(mechanism.risk_evidence) / 100.0
                if pd.notna(mechanism.risk_evidence)
                else None
            )
            mechanisms.append(
                {
                    "key": str(mechanism.mechanism),
                    "name": str(mechanism.mechanism_label),
                    "signal_strength": strength,
                    "direction": _direction(strength),
                    "evidence_confidence": (
                        float(mechanism.evidence_confidence)
                        if pd.notna(mechanism.evidence_confidence)
                        else None
                    ),
                    "dominant_signal_label": (
                        str(mechanism.dominant_signal_label)
                        if pd.notna(mechanism.dominant_signal_label)
                        else None
                    ),
                }
            )
        hazard_coverage = prediction.get("evidence_coverage_score")
        mechanism_coverage = alert.get("overall_evidence_confidence")
        coverages = [
            float(value)
            for value in (hazard_coverage, mechanism_coverage)
            if pd.notna(value) and np.isfinite(value)
        ]
        evidence_confidence = min(coverages) if coverages else None
        record = {
            "country_code": str(row["country_code"]),
            "country_name": (
                str(row["country_name"]) if pd.notna(row.get("country_name")) else None
            ),
            "operating_environment_score": (
                float(row["economic_pillar"])
                if pd.notna(row.get("economic_pillar"))
                else None
            ),
            "structural_vulnerability_score": (
                float(row["risk_score"]) if pd.notna(row.get("risk_score")) else None
            ),
            "systemic_hazard_1y": prediction.get("probability_1y"),
            "systemic_hazard_2_3y": prediction.get("probability_years_2_3"),
            "systemic_hazard_3y": prediction.get("probability_within_3y"),
            "selected_expert": str(prediction["selected_expert"]),
            "hazard_expert": str(prediction["selected_expert"]),
            "evidence_confidence": evidence_confidence,
            "evidence_confidence_label": str(
                prediction.get("evidence_confidence", "insufficient")
            ),
            "hazard_evidence_coverage": hazard_coverage,
            "mechanism_evidence_coverage": mechanism_coverage,
            "alert_status": str(alert["alert_level"]),
            "alert_reason": _alert_reason(alert, thresholds),
            "alert_blockers": list(alert.get("red_blockers", ())),
            "dominant_mechanism": (
                str(alert["dominant_mechanism"])
                if pd.notna(alert.get("dominant_mechanism"))
                else None
            ),
            "dominant_mechanism_label": (
                str(alert["dominant_mechanism_label"])
                if pd.notna(alert.get("dominant_mechanism_label"))
                else None
            ),
            "mechanisms": mechanisms,
        }
        records.append(_json_ready(record))
    return records


def _source_vintage(
    path: Path,
    *,
    rows: int | None = None,
    countries: int | None = None,
    period: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "artifact": path.name,
        "sha256": _sha256(path),
        "rows": rows,
        "countries": countries,
        "period": dict(period or {}),
    }


def build_hierarchical_risk_snapshot(
    *,
    annual_panel_path: Path | str,
    risk_model_path: Path | str,
    config: BuildConfig,
    external_candidates_path: Path | str | None = None,
    government_candidates_path: Path | str | None = None,
    bis_history_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build snapshot and validation payloads without writing or promotion."""

    annual_panel_path = Path(annual_panel_path)
    risk_model_path = Path(risk_model_path)
    optional_paths = {
        "external": Path(external_candidates_path) if external_candidates_path else None,
        "government": Path(government_candidates_path) if government_candidates_path else None,
        "bis": Path(bis_history_path) if bis_history_path else None,
    }
    for path in (annual_panel_path, risk_model_path, *optional_paths.values()):
        if path is not None and not path.exists():
            raise FileNotFoundError(path)

    panel = pd.read_parquet(annual_panel_path)
    artifact = load_model_artifact(risk_model_path, verify_checksum=False)
    external = (
        pd.read_parquet(optional_paths["external"])
        if optional_paths["external"] is not None
        else None
    )
    government = (
        pd.read_parquet(optional_paths["government"])
        if optional_paths["government"] is not None
        else None
    )
    bis_current = None
    bis_report = {"status": "not_supplied"}
    if optional_paths["bis"] is not None:
        bis_current, bis_report = _load_bis_history(
            optional_paths["bis"], as_of_date=config.as_of_date
        )

    features = _prepare_training_features(
        panel,
        config,
        bis_history_path=optional_paths["bis"],
    )
    try:
        from src.crisis_model_features import BIS_CHALLENGER_FEATURES
    except ImportError:
        BIS_CHALLENGER_FEATURES = ()
    bis_features = tuple(BIS_CHALLENGER_FEATURES) if optional_paths["bis"] else ()
    router_config = ExpertRoutingConfig(
        historical_features=tuple(HISTORICAL_CORE_FEATURES),
        modern_incremental_features=tuple(
            dict.fromkeys((*MODERN_INCREMENTAL_FEATURES, *bis_features))
        ),
    )

    metrics, thresholds, validation_fit_reports = _forward_validation(
        features, config=config, router_config=router_config
    )
    gates = _promotion_gates(metrics)
    model_status = MODEL_STATUS_CANDIDATE if gates["passed"] else MODEL_STATUS_RESEARCH

    final_models, final_fit_reports = _fit_experts(
        features,
        config=config,
        router_config=router_config,
        label_available_before=None,
    )
    current, merge_reports = _prepare_current_features(
        artifact,
        external_candidates=external,
        government_candidates=government,
        bis_current=bis_current,
    )
    current_predictions = _predict_routed(current, final_models, router_config)
    countries = _country_records(current, current_predictions, thresholds)

    source_vintages: dict[str, Any] = {
        "annual_crisis_panel": _source_vintage(
            annual_panel_path,
            rows=len(panel),
            countries=int(panel["country_code"].nunique()),
            period={
                "forecast_origin_start": int(panel["forecast_origin_year"].min()),
                "forecast_origin_end": int(panel["forecast_origin_year"].max()),
                "label_coverage_end_year": config.label_coverage_end_year,
            },
        ),
        "risk_model": _source_vintage(
            risk_model_path,
            rows=int(len(artifact["country_scores"])),
            countries=int(artifact["country_scores"]["country_code"].nunique()),
            period={"training_date": artifact.get("training_date")},
        ),
    }
    for key, frame in (("external", external), ("government", government)):
        path = optional_paths[key]
        if path is not None and frame is not None:
            source_vintages[f"{key}_candidates"] = _source_vintage(
                path,
                rows=len(frame),
                countries=int(frame["country_code"].nunique()),
                period={
                    "observation_vintage": "not_provided_by_staged_artifact",
                    "snapshot_as_of_date": config.as_of_date,
                },
            )
    if optional_paths["bis"] is not None:
        source_vintages["bis_history"] = {
            **_source_vintage(optional_paths["bis"]),
            **bis_report,
        }

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "as_of_date": config.as_of_date,
        "model_status": model_status,
        "source_vintages": source_vintages,
        "threshold_policy": thresholds,
        "confidence_semantics": (
            "evidence_confidence measures observed feature coverage; it is not "
            "statistical confidence, a confidence interval, or probability certainty"
        ),
        "countries": countries,
    }
    test_end = min(
        int(features["forecast_origin_year"].max()),
        config.label_coverage_end_year - 1,
    )
    validation = {
        "schema_version": SCHEMA_VERSION,
        "as_of_date": config.as_of_date,
        "status": model_status,
        "model_status": model_status,
        "design": "horizon-embargoed forward-time train/validation/test holdout",
        "evaluation_period": {
            "training": {
                "origin_start": int(features["forecast_origin_year"].min()),
                "label_cutoff_exclusive": config.validation_start_year,
                "origin_end_by_horizon": {
                    "1y": config.validation_start_year - 2,
                    "2y": config.validation_start_year - 3,
                    "3y": config.validation_start_year - 4,
                },
            },
            "threshold_validation": {
                "origin_start": config.validation_start_year,
                "label_cutoff_exclusive": config.test_start_year,
                "origin_end_by_horizon": {
                    "1y": config.test_start_year - 2,
                    "2_3y": config.test_start_year - 4,
                    "3y": config.test_start_year - 4,
                },
            },
            "forward_test": {
                "origin_start": config.test_start_year,
                "origin_end": test_end,
            },
        },
        "sample_description": {
            "annual_rows": int(len(features)),
            "countries": int(features["country_code"].nunique()),
            "target_definition": (
                "systemic crisis onset; conditional annual hazards with separately "
                "reported years-two-to-three and cumulative three-year incidence"
            ),
            "expert_routing": (
                "modern-full when banking/BIS coverage meets contract, otherwise "
                "historical-core; insufficient evidence is not imputed to low risk"
            ),
        },
        "metrics": metrics,
        "promotion_gates": gates,
        "threshold_policy": thresholds,
        "fit_diagnostics": {
            **validation_fit_reports,
            "final_experts": final_fit_reports,
        },
        "candidate_merge_diagnostics": merge_reports,
        "source_vintages": source_vintages,
        "confidence_semantics": snapshot["confidence_semantics"],
        "model_artifacts": {
            expert: {
                str(horizon): model.to_dict()
                for horizon, model in sorted(models.items())
            }
            for expert, models in sorted(final_models.items())
        },
    }
    return _json_ready(snapshot), _json_ready(validation)


def write_json_artifacts(
    snapshot: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    snapshot_output: Path | str,
    validation_output: Path | str,
) -> None:
    """Write strict, deterministic JSON with stable key and country ordering."""

    outputs = (
        (Path(snapshot_output), snapshot),
        (Path(validation_output), validation),
    )
    for path, payload in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(
            _json_ready(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        path.write_text(text, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-panel", type=Path, required=True)
    parser.add_argument("--risk-model", type=Path, required=True)
    parser.add_argument("--external-candidates", type=Path)
    parser.add_argument("--government-candidates", type=Path)
    parser.add_argument("--bis-history", type=Path)
    parser.add_argument("--snapshot-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--label-coverage-end-year", type=int, default=2025)
    parser.add_argument("--validation-start-year", type=int, default=2009)
    parser.add_argument("--test-start-year", type=int, default=2014)
    parser.add_argument("--modern-start-year", type=int, default=2000)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    config = BuildConfig(
        as_of_date=args.as_of_date,
        label_coverage_end_year=args.label_coverage_end_year,
        validation_start_year=args.validation_start_year,
        test_start_year=args.test_start_year,
        modern_start_year=args.modern_start_year,
    )
    snapshot, validation = build_hierarchical_risk_snapshot(
        annual_panel_path=args.annual_panel,
        risk_model_path=args.risk_model,
        config=config,
        external_candidates_path=args.external_candidates,
        government_candidates_path=args.government_candidates,
        bis_history_path=args.bis_history,
    )
    write_json_artifacts(
        snapshot,
        validation,
        snapshot_output=args.snapshot_output,
        validation_output=args.validation_output,
    )
    print(
        json.dumps(
            {
                "model_status": snapshot["model_status"],
                "countries": len(snapshot["countries"]),
                "snapshot_output": str(args.snapshot_output),
                "validation_output": str(args.validation_output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
