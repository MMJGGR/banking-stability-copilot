"""Phase 4B broad observable-outcome validation.

The fixed Phase 2 state-transition architecture is tested against every
eligible annual level identity. There is no economic allowlist or top-k
screen. WEO provider projections are excluded from outcomes and calibration.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

CURRENCY_UNITS = {"XDC", "USD", "EUR", "XDR"}
WINDOWS = ((2016, 2018), (2019, 2021))
MIN_WINDOW_ROWS = 30
MIN_WINDOW_COUNTRIES = 10
MIN_WINDOW_ORIGINS = 2
FDR_ALPHA = 0.10


def _normalize_key(series: pd.Series) -> pd.Series:
    def clean(value):
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if text.endswith(".0"):
            try:
                number = float(text)
                if number.is_integer():
                    return str(int(number))
            except ValueError:
                pass
        return text

    return series.map(clean)


def _registry_maps(registry: pd.DataFrame, eligible_features: set[str]):
    registry = registry.loc[registry.feature_id.isin(eligible_features)].copy()
    specifications = {
        "FSIC": ["SECTOR", "INDICATOR", "FREQUENCY"],
        "FSIBSIS": ["SECTOR", "INDICATOR", "FREQUENCY"],
        "MFS": ["INDICATOR", "TYPE_OF_TRANSFORMATION", "FREQUENCY", "SCALE"],
        "WEO": ["INDICATOR", "FREQUENCY", "SCALE", "UNIT"],
    }
    result = {}
    for source, keys in specifications.items():
        columns = list(dict.fromkeys([*keys, "feature_id", "UNIT", "indicator_label"]))
        mapping = registry.loc[registry.source.eq(source), columns].copy()
        for key in keys:
            mapping[key] = _normalize_key(mapping[key])
        if mapping.duplicated(keys).any():
            raise ValueError(f"nonunique {source} registry join")
        result[source] = keys, mapping
    return result


def load_observed_levels(
    snapshot: str | Path,
    loadings: pd.DataFrame,
    scaler: pd.DataFrame,
    states: pd.DataFrame,
    maximum_year: int = 2022,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Recover exact annual historical level observations in Phase 2 units."""
    snapshot = Path(snapshot)
    level = loadings.loc[loadings.predictor_id.str.endswith("::level")].copy()
    mappings = _registry_maps(
        pd.read_csv(snapshot / "complete-registry.csv", dtype=str, keep_default_na=False),
        set(level.feature_id),
    )
    member_entities = set(states.entity_code.astype(str))
    frames, audit = [], []
    common = ["COUNTRY", "FREQUENCY", "TIME_PERIOD", "OBS_VALUE"]

    for source in ("FSIC", "FSIBSIS", "MFS", "WEO"):
        keys, mapping = mappings[source]
        columns = list(dict.fromkeys([*common, *keys]))
        input_rows = annual_rows = mapped_rows = 0
        parts = []
        for chunk in pd.read_csv(
            snapshot / source / "raw-response.csv.gz",
            usecols=columns,
            chunksize=250_000,
            low_memory=False,
        ):
            input_rows += len(chunk)
            year = pd.to_numeric(chunk.TIME_PERIOD, errors="coerce")
            value = pd.to_numeric(chunk.OBS_VALUE, errors="coerce")
            mask = (
                chunk.FREQUENCY.astype(str).eq("A")
                & year.notna()
                & value.notna()
                & np.isfinite(value)
                & year.le(maximum_year)
                & chunk.COUNTRY.astype(str).isin(member_entities)
            )
            part = chunk.loc[mask].copy()
            if part.empty:
                continue
            part["observation_year"] = year.loc[mask].astype(int)
            part["value"] = value.loc[mask].astype(float)
            annual_rows += len(part)
            for key in keys:
                part[key] = _normalize_key(part[key])
            part = part.merge(mapping, on=keys, how="inner", validate="many_to_one")
            mapped_rows += len(part)
            parts.append(part[["COUNTRY", "observation_year", "feature_id", "UNIT", "value"]])
        if parts:
            frames.append(pd.concat(parts, ignore_index=True))
        audit.append({
            "source": source,
            "raw_rows": input_rows,
            "eligible_annual_rows": annual_rows,
            "mapped_rows": mapped_rows,
            "unmapped_rows": annual_rows - mapped_rows,
        })

    wgi_path = snapshot / "WGI" / "worldbank_WGI_2026-09-16T180808Z.csv"
    wgi = pd.read_csv(wgi_path)
    wgi_registry = pd.read_csv(
        snapshot / "WGI" / "registry.csv", dtype=str, keep_default_na=False
    )
    wgi = wgi.loc[
        wgi.country_code.astype(str).isin(member_entities)
        & pd.to_numeric(wgi.year, errors="coerce").le(maximum_year)
        & pd.to_numeric(wgi.value, errors="coerce").notna()
    ].copy()
    wgi = wgi.merge(
        wgi_registry[["INDICATOR", "feature_id", "UNIT"]],
        left_on="indicator_code",
        right_on="INDICATOR",
        how="inner",
        validate="many_to_one",
    )
    if len(wgi):
        frames.append(
            wgi.rename(
                columns={"country_code": "COUNTRY", "year": "observation_year"}
            )[["COUNTRY", "observation_year", "feature_id", "UNIT", "value"]]
        )
    audit.append({
        "source": "WGI",
        "raw_rows": len(pd.read_csv(wgi_path, usecols=["year"])),
        "eligible_annual_rows": len(wgi),
        "mapped_rows": len(wgi),
        "unmapped_rows": 0,
    })

    raw = pd.concat(frames, ignore_index=True)
    raw["observation_year"] = pd.to_numeric(raw.observation_year).astype(int)
    raw["value"] = pd.to_numeric(raw.value).astype(float)
    keys = ["COUNTRY", "feature_id", "observation_year"]
    cells = raw.groupby(keys, observed=True).value.agg(
        rows="size", distinct_values="nunique", minimum="min", maximum="max"
    ).reset_index()
    conflicts = cells.loc[cells.distinct_values.gt(1)].copy()
    clean = cells.loc[
        cells.distinct_values.eq(1), keys + ["minimum"]
    ].rename(columns={"minimum": "value"})
    metadata = level[[
        "feature_id", "predictor_id", "source", "INDICATOR",
        "indicator_label", "UNIT", "unit_policy",
    ]].drop_duplicates("feature_id")
    clean = clean.merge(metadata, on="feature_id", how="inner", validate="many_to_one")
    clean = clean.sort_values(["COUNTRY", "predictor_id", "observation_year"])

    # Reproduce the Phase 2 causal own-history normalization for amounts.
    clean["model_value"] = clean.value
    monetary = clean.loc[clean.UNIT.isin(CURRENCY_UNITS)].copy()
    if len(monetary):
        group = [monetary.COUNTRY, monetary.predictor_id]
        count = monetary.groupby(["COUNTRY", "predictor_id"], observed=True).cumcount()
        absolute = monetary.value.abs()
        previous_sum = absolute.groupby(group, observed=True).cumsum() - absolute
        denominator = previous_sum / count.replace(0, np.nan)
        supported = count.ge(2) & denominator.gt(0)
        transformed = pd.Series(np.nan, index=monetary.index, dtype=float)
        transformed.loc[supported] = np.arcsinh(
            monetary.loc[supported, "value"] / denominator.loc[supported]
        )
        clean.loc[monetary.index, "model_value"] = transformed
    clean = clean.loc[np.isfinite(clean.model_value)].copy()
    clean = clean.merge(
        scaler[["predictor_id", "median", "scale"]],
        on="predictor_id",
        how="inner",
        validate="many_to_one",
    )
    clean["z"] = ((clean.model_value - clean["median"]) / clean["scale"]).clip(-8, 8)
    clean = clean.rename(columns={"COUNTRY": "entity_code"})
    summary = {
        "observed_level_cells": len(clean),
        "level_features": int(clean.predictor_id.nunique()),
        "entities": int(clean.entity_code.nunique()),
        "conflicting_cells": len(conflicts),
        "provider_projection_rows_read": 0,
        "maximum_observation_year": maximum_year,
    }
    return clean, conflicts, pd.DataFrame(audit), summary


class StateScale:
    def __init__(self, mean, scale):
        self.mean, self.scale = mean, scale

    @classmethod
    def fit(cls, values):
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        return cls(mean, np.where(scale > 1e-8, scale, 1.0))

    def transform(self, values):
        return (values - self.mean) / self.scale

    def inverse(self, values):
        return values * self.scale + self.mean


def _transition_pairs(states, information, horizon):
    state_columns = [f"state_{index}" for index in range(1, 97)]
    current = states[["entity_code", "forecast_origin_year", *state_columns]].copy()
    future = current.copy()
    future.forecast_origin_year -= horizon
    future = future.rename(columns={column: f"future_{column}" for column in state_columns})
    pairs = current.merge(
        future, on=["entity_code", "forecast_origin_year"], validate="one_to_one"
    )
    pairs["target_year"] = pairs.forecast_origin_year + horizon
    current_info = information[[
        "entity_code", "forecast_origin_year", "state_uncertainty_proxy"
    ]]
    future_info = current_info.copy()
    future_info.forecast_origin_year -= horizon
    pairs = pairs.merge(
        current_info.rename(columns={"state_uncertainty_proxy": "current_uncertainty"}),
        on=["entity_code", "forecast_origin_year"], how="left",
    ).merge(
        future_info.rename(columns={"state_uncertainty_proxy": "future_uncertainty"}),
        on=["entity_code", "forecast_origin_year"], how="left",
    )
    return pairs, state_columns


def state_predictions(states, information):
    records = []
    for horizon in (1, 2):
        pairs, state_columns = _transition_pairs(states, information, horizon)
        for start, end in WINDOWS:
            train = pairs.loc[
                (pairs.forecast_origin_year < start) & (pairs.target_year < start)
            ].copy()
            test = pairs.loc[pairs.forecast_origin_year.between(start, end)].copy()
            scale = StateScale.fit(train[state_columns].to_numpy(float))
            x = scale.transform(train[state_columns].to_numpy(float))
            y = scale.transform(
                train[[f"future_{column}" for column in state_columns]].to_numpy(float)
            )
            test_x = scale.transform(test[state_columns].to_numpy(float))
            uncertainty = (
                train.current_uncertainty.to_numpy(float) ** 2
                + train.future_uncertainty.to_numpy(float) ** 2
            )
            weight = 1 / np.maximum(uncertainty, 1e-4)
            lower, upper = np.quantile(weight, [0.02, 0.98])
            weight = np.clip(weight, lower, upper)
            weight /= weight.mean()
            if horizon == 1:
                model = Ridge(alpha=100.0).fit(x, y - x, sample_weight=weight)
                prediction = test_x + model.predict(test_x)
                model_name = "ridge_delta"
            else:
                prediction = np.empty_like(test_x)
                for index in range(test_x.shape[1]):
                    model = Ridge(alpha=0.1).fit(
                        x[:, [index]], y[:, index], sample_weight=weight
                    )
                    prediction[:, index] = model.predict(test_x[:, [index]])
                model_name = "diagonal_ar"
            predicted = scale.inverse(prediction)
            current = test[state_columns].to_numpy(float)
            future = test[
                [f"future_{column}" for column in state_columns]
            ].to_numpy(float)
            for index, row in enumerate(
                test[["entity_code", "forecast_origin_year", "target_year"]]
                .itertuples(index=False)
            ):
                records.append({
                    "entity_code": row.entity_code,
                    "forecast_origin_year": int(row.forecast_origin_year),
                    "target_year": int(row.target_year),
                    "horizon": horizon,
                    "outer_start": start,
                    "outer_end": end,
                    "model": model_name,
                    "current_state": current[index],
                    "future_state": future[index],
                    "predicted_state": predicted[index],
                })
    return records


def _bh_adjust(pvalues: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.dropna().sort_values()
    if not len(valid):
        return result
    adjusted = np.minimum.accumulate(
        (valid.to_numpy() * len(valid) / np.arange(1, len(valid) + 1))[::-1]
    )[::-1]
    result.loc[valid.index] = np.clip(adjusted, 0, 1)
    return result


def evaluate_outcomes(observations, loadings, prediction_records):
    level = loadings.loc[loadings.predictor_id.str.endswith("::level")].copy()
    state_columns = [f"state_{index}" for index in range(1, 97)]
    loading_matrix = level[state_columns].to_numpy(float)
    feature_index = {value: index for index, value in enumerate(level.predictor_id)}
    observations = observations[[
        "entity_code", "observation_year", "predictor_id", "z"
    ]].copy()
    fold_rows, loss_rows = [], []

    for horizon in (1, 2):
        for start, end in WINDOWS:
            records = [
                row for row in prediction_records
                if row["horizon"] == horizon and row["outer_start"] == start
            ]
            if not records:
                continue
            row_map = {
                (row["entity_code"], row["forecast_origin_year"]): index
                for index, row in enumerate(records)
            }
            delta = np.vstack([
                row["predicted_state"] - row["current_state"] for row in records
            ])
            predicted_all = delta @ loading_matrix.T
            current = observations.copy()
            current["forecast_origin_year"] = current.observation_year + 1
            current = current.rename(columns={"z": "current_z"})
            future = observations.copy()
            future["forecast_origin_year"] = future.observation_year - horizon + 1
            future = future.rename(columns={"z": "future_z"})
            pairs = current[[
                "entity_code", "forecast_origin_year", "predictor_id", "current_z"
            ]].merge(
                future[[
                    "entity_code", "forecast_origin_year", "predictor_id", "future_z"
                ]],
                on=["entity_code", "forecast_origin_year", "predictor_id"],
                how="inner", validate="one_to_one",
            )
            pairs = pairs.loc[pairs.forecast_origin_year.between(start, end)].copy()
            pairs["row_index"] = [
                row_map.get((entity, int(year)), -1)
                for entity, year in zip(pairs.entity_code, pairs.forecast_origin_year)
            ]
            pairs["feature_index"] = pairs.predictor_id.map(feature_index).fillna(-1).astype(int)
            pairs = pairs.loc[(pairs.row_index >= 0) & (pairs.feature_index >= 0)].copy()
            pairs["actual_change"] = pairs.future_z - pairs.current_z
            pairs["prediction"] = predicted_all[
                pairs.row_index.to_numpy(), pairs.feature_index.to_numpy()
            ]
            pairs["baseline_sq_error"] = pairs.actual_change ** 2
            pairs["model_sq_error"] = (pairs.actual_change - pairs.prediction) ** 2
            pairs["loss_improvement"] = pairs.baseline_sq_error - pairs.model_sq_error
            pairs["horizon"] = horizon
            pairs["outer_start"] = start
            pairs["outer_end"] = end
            loss_rows.append(pairs)

            grouped = pairs.groupby("predictor_id", observed=True)
            fold = grouped.agg(
                rows=("actual_change", "size"),
                countries=("entity_code", "nunique"),
                origins=("forecast_origin_year", "nunique"),
                distinct_changes=("actual_change", "nunique"),
                baseline_mse=("baseline_sq_error", "mean"),
                model_mse=("model_sq_error", "mean"),
                mean_loss_improvement=("loss_improvement", "mean"),
            ).reset_index()
            fold["horizon"] = horizon
            fold["outer_start"] = start
            fold["outer_end"] = end
            fold["baseline_rmse"] = np.sqrt(fold.pop("baseline_mse"))
            fold["state_model_rmse"] = np.sqrt(fold.pop("model_mse"))
            fold["relative_rmse_improvement"] = (
                fold.baseline_rmse - fold.state_model_rmse
            ) / fold.baseline_rmse.replace(0, np.nan)
            fold["admitted"] = (
                fold.rows.ge(MIN_WINDOW_ROWS)
                & fold.countries.ge(MIN_WINDOW_COUNTRIES)
                & fold.origins.ge(MIN_WINDOW_ORIGINS)
                & fold.distinct_changes.gt(1)
            )
            fold_rows.append(fold)

    losses = pd.concat(loss_rows, ignore_index=True)
    folds = pd.concat(fold_rows, ignore_index=True)
    metadata = level[[
        "predictor_id", "feature_id", "source", "INDICATOR",
        "indicator_label", "UNIT", "unit_policy",
    ]].drop_duplicates()
    grid = pd.MultiIndex.from_product(
        [level.predictor_id.unique(), (1, 2), (2016, 2019)],
        names=["predictor_id", "horizon", "outer_start"],
    ).to_frame(index=False)
    folds = grid.merge(
        folds, on=["predictor_id", "horizon", "outer_start"], how="left"
    )
    folds["admitted"] = folds.admitted.astype("boolean").fillna(False).astype(bool)
    folds = folds.merge(metadata, on="predictor_id", how="left", validate="many_to_one")

    entity_loss = losses.groupby(
        ["predictor_id", "horizon", "entity_code"], observed=True
    ).loss_improvement.mean().reset_index()
    entity_stats = entity_loss.groupby(
        ["predictor_id", "horizon"], observed=True
    ).loss_improvement.agg(["size", "mean", "std"]).reset_index()
    entity_stats["entity_cluster_ttest_pvalue"] = stats.t.sf(
        entity_stats["mean"] / (
            entity_stats["std"] / np.sqrt(entity_stats["size"])
        ),
        df=entity_stats["size"] - 1,
    )
    entity_stats.loc[
        entity_stats["size"].lt(10) | entity_stats["std"].le(0),
        "entity_cluster_ttest_pvalue",
    ] = np.nan

    pooled = losses.groupby(
        ["predictor_id", "horizon"], observed=True
    ).agg(
        rows=("actual_change", "size"),
        countries=("entity_code", "nunique"),
        baseline_mse=("baseline_sq_error", "mean"),
        model_mse=("model_sq_error", "mean"),
    ).reset_index()
    pooled["baseline_rmse"] = np.sqrt(pooled.pop("baseline_mse"))
    pooled["state_model_rmse"] = np.sqrt(pooled.pop("model_mse"))
    pooled["pooled_relative_rmse_improvement"] = (
        pooled.baseline_rmse - pooled.state_model_rmse
    ) / pooled.baseline_rmse.replace(0, np.nan)
    admitted = folds.loc[folds.admitted].pivot_table(
        index=["predictor_id", "horizon"],
        columns="outer_start",
        values="relative_rmse_improvement",
        aggfunc="first",
    ).rename(columns={2016: "window_2016_improvement", 2019: "window_2019_improvement"})
    pooled = pooled.merge(admitted.reset_index(), on=["predictor_id", "horizon"], how="left")
    pooled["both_windows_admitted"] = (
        pooled.window_2016_improvement.notna() & pooled.window_2019_improvement.notna()
    )
    pooled["positive_both_windows"] = (
        pooled.both_windows_admitted
        & pooled.window_2016_improvement.gt(0)
        & pooled.window_2019_improvement.gt(0)
    )
    pooled = pooled.merge(
        entity_stats[["predictor_id", "horizon", "entity_cluster_ttest_pvalue"]],
        on=["predictor_id", "horizon"], how="left",
    ).merge(metadata, on="predictor_id", how="left", validate="many_to_one")
    pooled["fdr_qvalue"] = np.nan
    for horizon, index in pooled.groupby("horizon").groups.items():
        pooled.loc[index, "fdr_qvalue"] = _bh_adjust(
            pooled.loc[index, "entity_cluster_ttest_pvalue"]
        ).to_numpy()
    pooled["stable_state_predictable"] = (
        pooled.both_windows_admitted
        & pooled.positive_both_windows
        & pooled.pooled_relative_rmse_improvement.gt(0)
        & pooled.fdr_qvalue.le(FDR_ALPHA)
    )
    return folds, pooled


def outcome_admission(level, observations, pooled):
    metadata = level[[
        "predictor_id", "feature_id", "source", "INDICATOR",
        "indicator_label", "UNIT", "unit_policy", "measurement_state",
    ]].drop_duplicates()
    counts = observations.groupby("predictor_id", observed=True).agg(
        observed_cells=("z", "size"), countries=("entity_code", "nunique"),
        first_year=("observation_year", "min"), last_year=("observation_year", "max"),
    ).reset_index()
    ledger = metadata.merge(counts, on="predictor_id", how="left")
    eligible = pooled.groupby("predictor_id").both_windows_admitted.any()
    stable = pooled.groupby("predictor_id").stable_state_predictable.any()
    ledger["outcome_admission_state"] = "insufficient_exact_future_pairs"
    ledger.loc[
        ledger.unit_policy.eq("quarantined_unresolved_unit"),
        "outcome_admission_state",
    ] = "unresolved_unit_quarantined"
    eligible_mask = ledger.predictor_id.map(eligible).astype("boolean").fillna(False).astype(bool)
    stable_mask = ledger.predictor_id.map(stable).astype("boolean").fillna(False).astype(bool)
    ledger.loc[eligible_mask, "outcome_admission_state"] = "eligible_evaluated"
    ledger.loc[stable_mask, "outcome_admission_state"] = "stable_state_predictable"
    return ledger


def run(snapshot, phase2, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    phase2 = Path(phase2)
    loadings = pd.read_csv(phase2 / "measurement-loadings.csv.gz")
    scaler = pd.read_csv(phase2 / "measurement-scaler.csv.gz")
    states = pd.read_csv(phase2 / "country-year-states.csv.gz")
    information = pd.read_csv(phase2 / "country-year-information.csv.gz")
    observations, conflicts, audit, observation_summary = load_observed_levels(
        snapshot, loadings, scaler, states
    )
    folds, pooled = evaluate_outcomes(
        observations, loadings, state_predictions(states, information)
    )
    level = loadings.loc[loadings.predictor_id.str.endswith("::level")].copy()
    ledger = outcome_admission(level, observations, pooled)
    source_summary = pooled.groupby(["source", "horizon"], observed=True).agg(
        identities=("predictor_id", "size"),
        admitted_both_windows=("both_windows_admitted", "sum"),
        stable_state_predictable=("stable_state_predictable", "sum"),
        median_improvement=("pooled_relative_rmse_improvement", "median"),
        share_positive=("pooled_relative_rmse_improvement", lambda values: float((values > 0).mean())),
    ).reset_index()
    stable = pooled.loc[pooled.stable_state_predictable].sort_values(
        ["horizon", "pooled_relative_rmse_improvement"], ascending=[True, False]
    )

    conflicts.to_csv(output / "observable-conflicts.csv.gz", index=False)
    audit.to_csv(output / "source-mapping-audit.csv", index=False)
    folds.to_csv(output / "observable-fold-results.csv.gz", index=False)
    pooled.to_csv(output / "observable-pooled-results.csv.gz", index=False)
    ledger.to_csv(output / "observable-outcome-admission-ledger.csv", index=False)
    source_summary.to_csv(output / "observable-source-summary.csv", index=False)
    stable.to_csv(output / "stable-state-predictable-outcomes.csv", index=False)

    report = {
        "status": "completed_phase4b_broad_observable_validation",
        "feature_count_cap": None,
        "registered_level_identities": int(len(level)),
        "observed_level_identities": int(observations.predictor_id.nunique()),
        "observed_level_cells": int(len(observations)),
        "both_window_evaluations": int(pooled.both_windows_admitted.sum()),
        "stable_state_predictable_evaluations": int(pooled.stable_state_predictable.sum()),
        "stable_unique_identities": int(stable.predictor_id.nunique()),
        "stable_one_year_evaluations": int(stable.horizon.eq(1).sum()),
        "stable_two_year_evaluations": int(stable.horizon.eq(2).sum()),
        "provider_projection_rows_read": 0,
        "minimum_window_rows": MIN_WINDOW_ROWS,
        "minimum_window_countries": MIN_WINDOW_COUNTRIES,
        "minimum_window_origins": MIN_WINDOW_ORIGINS,
        "fdr_alpha": FDR_ALPHA,
        "observation_summary": observation_summary,
        "production_modified": False,
    }
    (output / "phase4b-summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--phase2", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.snapshot, args.phase2, args.output), indent=2))
