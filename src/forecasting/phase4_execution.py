from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, silhouette_score

SOURCES = ("FSIC", "FSIBSIS", "MFS", "WEO")
CURRENCY_UNITS = {"XDC", "USD", "EUR", "XDR"}
TRANSFORM_UNIT = {
    "XDC":"XDC","USD":"USD","EUR":"EUR","XDR":"XDR",
    "SA_XDC":"XDC","SA_USD":"USD","SA_EUR":"EUR",
    "PT":"PT","EOP_PT":"PT","PCH_CP_A_PT":"PT",
    "RMBBMPT_A_PT":"PT","PT_A_PT":"PT","PT_A":"PT",
    "IX":"IX","SAF_IX":"IX","EOP_IX":"IX","PA_IX":"IX",
}
STATE_WINDOWS = ((2016, 2018), (2019, 2021))
OWN_ALPHAS = (0.1, 1.0, 10.0, 100.0)
STATE_ALPHAS = (10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0)
STOPWORDS = {
    "the","and","of","to","in","for","on","as","by","at","from","with",
    "total","other","all","ratio","percent","percentage","domestic","currency",
    "annual","end","period","assets","liabilities","financial","general",
    "government","gross","net","value","index","prices","price","rate",
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_scalar(value) -> str:
    if pd.isna(value):
        return "__NULL__"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def load_source_cells(root: Path, source: str, chunksize: int = 250_000) -> pd.DataFrame:
    registry = pd.read_csv(root / source / "registry.csv", low_memory=False)
    identity_cols = list(registry.columns[: registry.columns.get_loc("identity_json")])
    header = pd.read_csv(root / source / "raw-response.csv.gz", nrows=0).columns.tolist()
    usecols = ["COUNTRY", "TIME_PERIOD", "OBS_VALUE", "STATUS"] + [
        c for c in identity_cols if c in header
    ]
    usecols = [c for c in dict.fromkeys(usecols) if c in header]

    lookup = registry[identity_cols + ["feature_id", "indicator_label", "source"]].copy()
    for column in identity_cols:
        lookup["_k_" + column] = lookup[column].map(clean_scalar)
    keys = ["_k_" + column for column in identity_cols]
    lookup = lookup[keys + ["feature_id", "indicator_label", "source"]]

    parts = []
    raw_total = matched = 0
    for raw in pd.read_csv(
        root / source / "raw-response.csv.gz",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        for column in identity_cols:
            if column not in raw:
                raw[column] = np.nan
        if source == "MFS":
            inferred = raw["TYPE_OF_TRANSFORMATION"].map(TRANSFORM_UNIT)
            raw["UNIT"] = raw["UNIT"].where(raw["UNIT"].notna(), inferred)

        mask = raw.COUNTRY.notna() & raw.TIME_PERIOD.notna() & raw.OBS_VALUE.notna()
        if "FREQUENCY" in raw:
            mask &= raw.FREQUENCY.astype(str).eq("A")
        data = raw.loc[mask].copy()
        if data.empty:
            continue
        data["year"] = pd.to_numeric(data.TIME_PERIOD, errors="coerce")
        data["value"] = pd.to_numeric(data.OBS_VALUE, errors="coerce")
        data = data.loc[data.year.notna() & np.isfinite(data.value)].copy()
        if data.empty:
            continue
        data["year"] = data.year.astype(int)
        for column in identity_cols:
            data["_k_" + column] = data[column].map(clean_scalar)
        data = data.merge(lookup, on=keys, how="left", validate="many_to_one")
        raw_total += len(data)
        matched += int(data.feature_id.notna().sum())
        status = (
            data.STATUS.astype("string").fillna("UNKNOWN")
            if "STATUS" in data
            else pd.Series("UNKNOWN", index=data.index, dtype="string")
        )
        parts.append(
            pd.DataFrame(
                {
                    "entity_code": data.COUNTRY.astype(str).str.strip().str.upper(),
                    "feature_id": data.feature_id,
                    "year": data.year.astype(int),
                    "value": data.value.astype(float),
                    "status": status.astype(str),
                    "source": source,
                    "indicator_label": data.indicator_label,
                }
            )
        )
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    result = result.loc[result.feature_id.notna()].copy()
    if raw_total != matched:
        raise ValueError(f"{source}: {raw_total-matched} annual observations did not map to registry")
    return result


def load_historical_cells(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frames = [load_source_cells(root, source) for source in SOURCES]

    wgi = pd.read_csv(root / "WGI" / "worldbank_WGI_2026-09-16T180808Z.csv", low_memory=False)
    wgi_registry = pd.read_csv(root / "WGI" / "registry.csv")
    mapping = wgi_registry.set_index("INDICATOR").feature_id
    wgi_frame = pd.DataFrame(
        {
            "entity_code": wgi.country_code.astype("string").str.strip().str.upper(),
            "feature_id": wgi.indicator_code.map(mapping),
            "year": pd.to_numeric(wgi.year, errors="coerce"),
            "value": pd.to_numeric(wgi.value, errors="coerce"),
            "status": "UNKNOWN",
            "source": "WGI",
            "indicator_label": wgi.feature_name,
        }
    )
    wgi_frame = wgi_frame.loc[
        wgi_frame.entity_code.notna()
        & wgi_frame.feature_id.notna()
        & wgi_frame.year.notna()
        & np.isfinite(wgi_frame.value)
    ].copy()
    wgi_frame["year"] = wgi_frame.year.astype(int)
    frames.append(wgi_frame)

    raw = pd.concat(frames, ignore_index=True)
    raw["data_role"] = np.where(
        raw.source.eq("WEO") & raw.year.ge(2026),
        "provider_projection",
        np.where(
            raw.source.eq("WEO") & raw.year.eq(2025),
            "historical_or_estimate_unverified",
            "historical_observed_or_unverified",
        ),
    )

    grouped = raw.groupby(["entity_code", "feature_id", "year"], observed=True).value.agg(
        rows="size", distinct_values="nunique", minimum="min", maximum="max"
    ).reset_index()
    conflicts = grouped.loc[grouped.distinct_values.gt(1)].copy()
    clean = grouped.loc[grouped.distinct_values.eq(1), ["entity_code", "feature_id", "year", "minimum"]]
    clean = clean.rename(columns={"minimum": "value"})
    role = raw[
        ["entity_code", "feature_id", "year", "source", "data_role", "status"]
    ].drop_duplicates(["entity_code", "feature_id", "year"])
    clean = clean.merge(role, on=["entity_code", "feature_id", "year"], how="left", validate="one_to_one")

    registry = pd.read_csv(root / "complete-registry.csv", low_memory=False)
    metadata_cols = [
        "feature_id", "source", "INDICATOR", "indicator_label", "UNIT", "SCALE",
        "FREQUENCY", "SECTOR", "TYPE_OF_TRANSFORMATION", "unit_resolution",
    ]
    metadata_cols = [c for c in metadata_cols if c in registry]
    metadata = registry[metadata_cols].drop_duplicates(["feature_id", "source"])
    clean = clean.merge(metadata, on=["feature_id", "source"], how="left", validate="many_to_one")

    summary = {
        "raw_annual_rows": len(raw),
        "canonical_cells": len(clean),
        "conflicting_cells": len(conflicts),
        "provider_projection_cells": int(clean.data_role.eq("provider_projection").sum()),
        "historical_cells": int((~clean.data_role.eq("provider_projection")).sum()),
        "features": int(clean.feature_id.nunique()),
        "entities": int(clean.entity_code.nunique()),
    }
    return clean, conflicts, summary


def transform_historical_values(cells: pd.DataFrame, state_entities: set[str]) -> tuple[pd.DataFrame, dict]:
    data = cells.loc[
        ~cells.data_role.eq("provider_projection") & cells.entity_code.isin(state_entities)
    ].copy()
    data = data.sort_values(["entity_code", "feature_id", "year"]).reset_index(drop=True)
    data["model_value"] = data.value.astype(float)

    amount_mask = data.UNIT.isin(CURRENCY_UNITS)
    amount = data.loc[amount_mask, ["entity_code", "feature_id", "year", "value"]].copy()
    keys = ["entity_code", "feature_id"]
    amount["prior_count"] = amount.groupby(keys, observed=True).cumcount()
    absolute = amount.value.abs()
    amount["prior_sum"] = absolute.groupby(
        [amount.entity_code, amount.feature_id], observed=True
    ).cumsum() - absolute
    amount["denominator"] = amount.prior_sum / amount.prior_count.replace(0, np.nan)
    supported = amount.prior_count.ge(2) & amount.denominator.gt(0)
    amount["transformed"] = np.nan
    amount.loc[supported, "transformed"] = np.arcsinh(
        amount.loc[supported, "value"] / amount.loc[supported, "denominator"]
    )
    data.loc[amount.index, "model_value"] = amount.transformed
    data = data.loc[np.isfinite(data.model_value)].copy()

    return data, {
        "historical_model_cells": len(data),
        "features": int(data.feature_id.nunique()),
        "entities": int(data.entity_code.nunique()),
        "currency_cells_without_supported_prior_scale": int((~supported).sum()),
        "provider_projection_rows_read": 0,
    }


def varimax(phi: np.ndarray, gamma: float = 1.0, max_iter: int = 100, tol: float = 1e-6):
    p, k = phi.shape
    rotation = np.eye(k)
    objective = 0.0
    for iteration in range(max_iter):
        old = objective
        rotated = phi @ rotation
        u, singular, vh = np.linalg.svd(
            phi.T @ (
                rotated**3
                - (gamma / p) * rotated @ np.diag(np.diag(rotated.T @ rotated))
            ),
            full_matrices=False,
        )
        rotation = u @ vh
        objective = float(singular.sum())
        if old and objective / old < 1 + tol:
            break
    return rotation, iteration + 1


def top_terms(labels: pd.Series, n: int = 6) -> str:
    counts: dict[str, float] = {}
    for label in labels.dropna().astype(str):
        for token in re.findall(r"[a-z][a-z0-9]+", label.lower()):
            if len(token) < 3 or token in STOPWORDS:
                continue
            counts[token] = counts.get(token, 0.0) + 1.0
    return " | ".join([key for key, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:n]])


def build_interpretation(
    phase2: Path,
    phase3: Path,
    output: Path,
    *,
    top_n: int = 12,
    bootstrap_repeats: int = 8,
) -> dict:
    loadings = pd.read_csv(phase2 / "measurement-loadings.csv.gz", low_memory=False)
    reliability = pd.read_csv(phase2 / "feature-reliability.csv.gz", low_memory=False)
    states = pd.read_csv(phase2 / "country-year-states.csv.gz")
    quantiles = pd.read_csv(phase3 / "latest-state-coordinate-quantiles.csv.gz")
    forecast_summary = pd.read_csv(phase3 / "latest-forecast-summary.csv")

    state_cols = sorted(
        [c for c in loadings if c.startswith("state_")],
        key=lambda value: int(value.split("_")[-1]),
    )
    variance = reliability.set_index("predictor_id").shrunk_residual_variance
    mapped_variance = loadings.predictor_id.map(variance).fillna(float(variance.median())).to_numpy()
    original = loadings[state_cols].to_numpy(dtype=float)

    weighted = original / np.sqrt(np.maximum(mapped_variance, 1e-6))[:, None]
    row_norm = np.linalg.norm(weighted, axis=1, keepdims=True)
    normalized = weighted / np.maximum(row_norm, 1e-12)
    rotation, iterations = varimax(normalized)
    rotated = original @ rotation

    rotated_frame = pd.DataFrame(
        rotated,
        columns=[f"display_state_{i+1}" for i in range(rotated.shape[1])],
    )
    enriched = pd.concat([loadings.reset_index(drop=True), rotated_frame], axis=1)

    top_rows = []
    cards = []
    display_cols = rotated_frame.columns.tolist()
    for index, column in enumerate(display_cols):
        values = enriched[column].to_numpy(dtype=float)
        order = np.argsort(np.abs(values))[::-1]
        chosen = order[: top_n * 2]
        subset = enriched.iloc[chosen].copy()
        subset["loading"] = values[chosen]
        subset["abs_loading"] = np.abs(values[chosen])
        subset["rank"] = np.arange(1, len(subset) + 1)
        subset["display_dimension"] = column
        top_rows.append(
            subset[
                [
                    "display_dimension", "rank", "predictor_id", "feature_id", "source",
                    "INDICATOR", "indicator_label", "UNIT", "unit_policy", "loading", "abs_loading",
                ]
            ]
        )
        energy = values**2
        total = max(float(energy.sum()), 1e-12)
        source_energy = (
            pd.DataFrame({"source": enriched.source, "energy": energy})
            .groupby("source", observed=True).energy.sum().sort_values(ascending=False)
        )
        positive = subset.loc[subset.loading.gt(0)].nlargest(top_n, "abs_loading")
        negative = subset.loc[subset.loading.lt(0)].nlargest(top_n, "abs_loading")
        cards.append(
            {
                "display_dimension": column,
                "original_state_mix": json.dumps(
                    {
                        state_cols[j]: float(rotation[j, index])
                        for j in np.argsort(np.abs(rotation[:, index]))[::-1][:8]
                    },
                    sort_keys=True,
                ),
                "top20_loading_energy_share": float(np.sort(energy)[::-1][:20].sum() / total),
                "top100_loading_energy_share": float(np.sort(energy)[::-1][:100].sum() / total),
                "dominant_source": source_energy.index[0] if len(source_energy) else "",
                "dominant_source_energy_share": float(source_energy.iloc[0] / total) if len(source_energy) else np.nan,
                "suggested_terms": top_terms(subset.indicator_label),
                "top_positive_labels": " || ".join(positive.indicator_label.dropna().astype(str).head(5)),
                "top_negative_labels": " || ".join(negative.indicator_label.dropna().astype(str).head(5)),
                "interpretation_status": (
                    "relatively_concentrated"
                    if np.sort(energy)[::-1][:100].sum() / total >= 0.30
                    else "mixed_high_dimensional"
                ),
            }
        )
    top_features = pd.concat(top_rows, ignore_index=True)
    cards_frame = pd.DataFrame(cards)
    top_features.to_csv(output / "dimension-top-features.csv.gz", index=False)
    cards_frame.to_csv(output / "dimension-cards.csv", index=False)

    profiles = np.abs(rotated.T) / np.sqrt(np.maximum(mapped_variance, 1e-6))[None, :]
    profiles /= np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-12)
    similarity = np.clip(profiles @ profiles.T, -1, 1)
    distance = np.clip(1 - similarity, 0, 2)
    np.fill_diagonal(distance, 0)
    rng = np.random.default_rng(20260924)
    diagnostics = []
    labels_by_k = {}
    for clusters in range(6, 21):
        labels = AgglomerativeClustering(
            n_clusters=clusters, metric="precomputed", linkage="average"
        ).fit_predict(distance)
        labels_by_k[clusters] = labels
        silhouette = float(silhouette_score(distance, labels, metric="precomputed"))
        stability_values = []
        for _ in range(bootstrap_repeats):
            mask = rng.random(profiles.shape[1]) < 0.70
            boot = profiles[:, mask]
            boot /= np.maximum(np.linalg.norm(boot, axis=1, keepdims=True), 1e-12)
            boot_distance = np.clip(1 - boot @ boot.T, 0, 2)
            np.fill_diagonal(boot_distance, 0)
            boot_labels = AgglomerativeClustering(
                n_clusters=clusters, metric="precomputed", linkage="average"
            ).fit_predict(boot_distance)
            stability_values.append(adjusted_rand_score(labels, boot_labels))
        stability = float(np.mean(stability_values))
        diagnostics.append(
            {
                "clusters": clusters,
                "silhouette": silhouette,
                "bootstrap_stability_ari": stability,
                "selection_score": silhouette * max(stability, 0.0) - 0.001 * clusters,
            }
        )
    diagnostic_frame = pd.DataFrame(diagnostics)
    diagnostic_frame.to_csv(output / "theme-cluster-diagnostics.csv", index=False)
    best = diagnostic_frame.sort_values(
        ["selection_score", "clusters"], ascending=[False, True]
    ).iloc[0]
    best_k = int(best.clusters)
    candidate_labels = labels_by_k[best_k] + 1
    cluster_sizes = pd.Series(candidate_labels).value_counts()
    compact_supported = (
        float(best.silhouette) >= 0.15
        and float(best.bootstrap_stability_ari) >= 0.70
        and float(cluster_sizes.max() / cluster_sizes.sum()) <= 0.65
        and int((cluster_sizes == 1).sum()) == 0
    )
    compact_status = (
        "compact_themes_supported"
        if compact_supported
        else "compact_themes_not_reliably_separated"
    )
    membership = pd.DataFrame(
        {
            "display_dimension": display_cols,
            "theme_cluster": candidate_labels,
            "compact_theme_status": compact_status,
        }
    )
    membership.to_csv(output / "state-theme-membership.csv", index=False)

    latest_year = int(states.forecast_origin_year.max())
    latest = states.loc[states.forecast_origin_year.eq(latest_year)].copy()
    original_latest = latest[state_cols].to_numpy(dtype=float)
    display_latest = original_latest @ rotation
    display_current = pd.DataFrame(
        display_latest, index=latest.entity_code, columns=display_cols
    )

    point = quantiles.pivot_table(
        index=["entity_code", "horizon"], columns="state_coordinate", values="point"
    )
    point = point.reindex(columns=state_cols)
    country_records = []
    for (entity, horizon), row in point.iterrows():
        if entity not in display_current.index or row.isna().any():
            continue
        future = row.to_numpy(dtype=float) @ rotation
        current = display_current.loc[entity].to_numpy(dtype=float)
        delta = future - current
        current_order = np.argsort(np.abs(current))[::-1][:8]
        movement_order = np.argsort(np.abs(delta))[::-1][:8]
        forecast_row = forecast_summary.loc[
            forecast_summary.entity_code.eq(entity) & forecast_summary.horizon.eq(horizon)
        ]
        context = forecast_row.iloc[0].to_dict() if len(forecast_row) else {}
        country_records.append(
            {
                "entity_code": entity,
                "current_state_year": latest_year,
                "horizon": int(horizon),
                "forecast_year": int(context.get("forecast_year", latest_year + int(horizon))),
                "top_current_display_dimensions": json.dumps(
                    {display_cols[i]: float(current[i]) for i in current_order}, sort_keys=True
                ),
                "top_forecast_movement_dimensions": json.dumps(
                    {display_cols[i]: float(delta[i]) for i in movement_order}, sort_keys=True
                ),
                "display_movement_rms": float(np.sqrt(np.mean(delta**2))),
                "current_display_distance_rms": float(np.sqrt(np.mean(current**2))),
                "future_display_distance_rms": float(np.sqrt(np.mean(future**2))),
                "movement_radius_q80": context.get("movement_radius_q80"),
                "movement_radius_q95": context.get("movement_radius_q95"),
                "state_uncertainty_proxy": context.get("state_uncertainty_proxy"),
                "forecast_quality": context.get("forecast_quality"),
                "compact_theme_status": compact_status,
            }
        )
    pd.DataFrame(country_records).to_csv(
        output / "latest-country-state-interpretation.csv.gz", index=False
    )

    return {
        "state_dimensions": len(state_cols),
        "varimax_iterations": iterations,
        "best_compact_cluster_count": best_k,
        "best_compact_silhouette": float(best.silhouette),
        "best_compact_stability_ari": float(best.bootstrap_stability_ari),
        "largest_candidate_cluster_dimensions": int(cluster_sizes.max()),
        "singleton_candidate_clusters": int((cluster_sizes == 1).sum()),
        "compact_theme_status": compact_status,
        "dimension_cards": len(cards_frame),
        "country_interpretation_rows": len(country_records),
        "geometry_preserved_by_orthogonal_rotation": True,
    }


def robust_stats(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return median, scale


def robust_scale_state(train: pd.DataFrame, test: pd.DataFrame, state_cols: list[str]):
    train_values = train[state_cols].to_numpy(dtype=float)
    test_values = test[state_cols].to_numpy(dtype=float)
    median = np.median(train_values, axis=0)
    q25 = np.quantile(train_values, 0.25, axis=0)
    q75 = np.quantile(train_values, 0.75, axis=0)
    scale = (q75 - q25) / 1.349
    standard = np.std(train_values, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(standard > 1e-8, standard, 1.0))
    return (
        np.clip((train_values - median) / scale, -5, 5),
        np.clip((test_values - median) / scale, -5, 5),
    )


def build_all_feature_pairs(
    historical: pd.DataFrame,
    state_keys: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    current = historical[["entity_code", "feature_id", "year", "model_value"]].rename(
        columns={"year": "current_year", "model_value": "current_value"}
    )
    target = historical[["entity_code", "feature_id", "year", "model_value"]].copy()
    target["current_year"] = target.year - horizon - 1
    target = target.rename(
        columns={"year": "target_year", "model_value": "target_value"}
    )[["entity_code", "feature_id", "current_year", "target_year", "target_value"]]
    pairs = current.merge(
        target,
        on=["entity_code", "feature_id", "current_year"],
        how="inner",
        validate="one_to_one",
    )
    pairs["forecast_origin_year"] = pairs.current_year + 1
    previous = historical[["entity_code", "feature_id", "year", "model_value"]].copy()
    previous["current_year"] = previous.year + 1
    previous = previous.rename(columns={"model_value": "previous_value"})[
        ["entity_code", "feature_id", "current_year", "previous_value"]
    ]
    pairs = pairs.merge(
        previous,
        on=["entity_code", "feature_id", "current_year"],
        how="left",
        validate="one_to_one",
    )
    pairs["prior_delta"] = (pairs.current_value - pairs.previous_value).fillna(0.0)
    pairs["prior_available"] = pairs.previous_value.notna().astype(float)
    pairs["target_available_year"] = pairs.target_year + 1
    pairs = pairs.merge(
        state_keys,
        on=["entity_code", "forecast_origin_year"],
        how="inner",
        validate="many_to_one",
    )
    pairs["horizon"] = horizon
    return pairs


def tune_ridge(x_train, y_train, x_validation, y_validation, alphas):
    trials = []
    for alpha in alphas:
        model = Ridge(alpha=alpha, solver="cholesky").fit(x_train, y_train)
        prediction = model.predict(x_validation)
        trials.append((float(np.sqrt(np.mean((prediction - y_validation) ** 2))), alpha))
    return min(trials, key=lambda item: (item[0], -item[1]))[1]


def evaluate_overlay_case(
    feature_id: str,
    horizon: int,
    pairs: pd.DataFrame,
    states_info: pd.DataFrame,
    state_cols: list[str],
) -> tuple[list[dict], list[pd.DataFrame]]:
    pairs = pairs.merge(
        states_info,
        on=["entity_code", "forecast_origin_year"],
        how="inner",
        validate="many_to_one",
    )
    fold_records = []
    prediction_frames = []
    own_columns = ["current_value", "prior_delta", "prior_available"]
    for start, end in STATE_WINDOWS:
        train = pairs.loc[
            pairs.forecast_origin_year.lt(start) & pairs.target_available_year.lt(start)
        ].copy()
        test = pairs.loc[pairs.forecast_origin_year.between(start, end)].copy()
        inner_end = start - horizon - 2
        inner_start = inner_end - 2
        inner_train = train.loc[
            train.forecast_origin_year.lt(inner_start)
            & train.target_available_year.lt(inner_start)
        ].copy()
        inner_test = train.loc[
            train.forecast_origin_year.between(inner_start, inner_end)
            & train.target_available_year.lt(start)
        ].copy()
        if min(len(train), len(inner_train)) < 40 or min(len(test), len(inner_test)) < 10:
            continue

        inner_median, inner_scale = robust_stats(inner_train.target_value.to_numpy())
        inner_y = (inner_train.target_value.to_numpy() - inner_median) / inner_scale
        validation_y = (inner_test.target_value.to_numpy() - inner_median) / inner_scale

        inner_own = inner_train[own_columns].to_numpy(dtype=float)
        validation_own = inner_test[own_columns].to_numpy(dtype=float)
        own_mean = inner_own[:, :2].mean(axis=0)
        own_std = inner_own[:, :2].std(axis=0)
        own_std = np.where(own_std > 1e-8, own_std, 1.0)
        inner_own[:, :2] = (inner_own[:, :2] - own_mean) / own_std
        validation_own[:, :2] = (validation_own[:, :2] - own_mean) / own_std
        inner_state, validation_state = robust_scale_state(inner_train, inner_test, state_cols)

        own_alpha = tune_ridge(
            inner_own, inner_y, validation_own, validation_y, OWN_ALPHAS
        )
        own_model = Ridge(alpha=own_alpha, solver="cholesky").fit(inner_own, inner_y)
        inner_residual = inner_y - own_model.predict(inner_own)
        validation_base = own_model.predict(validation_own)

        state_alpha = None
        best = None
        for alpha in STATE_ALPHAS:
            state_model = Ridge(alpha=alpha, solver="cholesky").fit(inner_state, inner_residual)
            prediction = validation_base + state_model.predict(validation_state)
            rmse = float(np.sqrt(np.mean((prediction - validation_y) ** 2)))
            candidate = (rmse, alpha)
            if best is None or candidate < best:
                best = candidate
                state_alpha = alpha

        direct_state_alpha = tune_ridge(
            inner_state, inner_y, validation_state, validation_y, STATE_ALPHAS
        )

        target_median, target_scale = robust_stats(train.target_value.to_numpy())
        train_y = (train.target_value.to_numpy() - target_median) / target_scale
        test_y = (test.target_value.to_numpy() - target_median) / target_scale
        no_change = (test.current_value.to_numpy() - target_median) / target_scale

        train_own = train[own_columns].to_numpy(dtype=float)
        test_own = test[own_columns].to_numpy(dtype=float)
        own_mean = train_own[:, :2].mean(axis=0)
        own_std = train_own[:, :2].std(axis=0)
        own_std = np.where(own_std > 1e-8, own_std, 1.0)
        train_own[:, :2] = (train_own[:, :2] - own_mean) / own_std
        test_own[:, :2] = (test_own[:, :2] - own_mean) / own_std
        train_state, test_state = robust_scale_state(train, test, state_cols)

        own_model = Ridge(alpha=own_alpha, solver="cholesky").fit(train_own, train_y)
        own_prediction = own_model.predict(test_own)
        residual = train_y - own_model.predict(train_own)
        overlay_model = Ridge(alpha=state_alpha, solver="cholesky").fit(train_state, residual)
        overlay_prediction = own_prediction + overlay_model.predict(test_state)
        state_model = Ridge(alpha=direct_state_alpha, solver="cholesky").fit(train_state, train_y)
        state_prediction = state_model.predict(test_state)

        predictions = {
            "no_change": no_change,
            "own_history": own_prediction,
            "state_only": state_prediction,
            "state_overlay": overlay_prediction,
        }
        baseline_error = np.abs(no_change - test_y)
        own_error = np.abs(own_prediction - test_y)
        for model_name, prediction in predictions.items():
            error = prediction - test_y
            absolute = np.abs(error)
            fold_records.append(
                {
                    "feature_id": feature_id,
                    "horizon": horizon,
                    "outer_start": start,
                    "outer_end": end,
                    "model": model_name,
                    "rows": len(test),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(absolute)),
                    "fraction_beating_no_change": (
                        float(np.mean(absolute < baseline_error))
                        if model_name != "no_change"
                        else 0.0
                    ),
                    "fraction_beating_own_history": (
                        float(np.mean(absolute < own_error))
                        if model_name == "state_overlay"
                        else np.nan
                    ),
                    "alpha": (
                        own_alpha
                        if model_name == "own_history"
                        else state_alpha
                        if model_name == "state_overlay"
                        else direct_state_alpha
                        if model_name == "state_only"
                        else np.nan
                    ),
                }
            )
            frame = test[["entity_code", "forecast_origin_year"]].copy()
            frame["feature_id"] = feature_id
            frame["horizon"] = horizon
            frame["outer_start"] = start
            frame["model"] = model_name
            frame["absolute_error"] = absolute
            frame["no_change_absolute_error"] = baseline_error
            frame["own_history_absolute_error"] = own_error
            prediction_frames.append(frame)
    return fold_records, prediction_frames


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.fillna(1.0).to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    n = len(values)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for position in range(n - 1, -1, -1):
        rank = position + 1
        running = min(running, ranked[position] * n / rank)
        adjusted[position] = min(running, 1.0)
    result = np.empty(n, dtype=float)
    result[order] = adjusted
    return pd.Series(result, index=p_values.index)


def _support_registry_for_horizon(
    historical: pd.DataFrame,
    pairs: pd.DataFrame,
    horizon: int,
    *,
    min_pairs: int = 200,
    min_countries: int = 20,
    min_origin_years: int = 8,
) -> pd.DataFrame:
    metadata = historical.sort_values("year").groupby("feature_id", observed=True).tail(1)
    metadata = metadata.set_index("feature_id")
    support = pairs.groupby("feature_id", observed=True).agg(
        pairs=("entity_code", "size"),
        countries=("entity_code", "nunique"),
        origin_years=("forecast_origin_year", "nunique"),
        first_origin=("forecast_origin_year", "min"),
        last_origin=("forecast_origin_year", "max"),
    ).reset_index()
    all_features = pd.DataFrame({"feature_id": historical.feature_id.unique()})
    support = all_features.merge(support, on="feature_id", how="left")
    for column in ["pairs", "countries", "origin_years"]:
        support[column] = support[column].fillna(0).astype(int)
    support["horizon"] = horizon
    for column in ["source", "INDICATOR", "indicator_label", "UNIT", "unit_resolution"]:
        support[column] = support.feature_id.map(metadata[column])
    reasons = []
    states = []
    for row in support.itertuples(index=False):
        reason = []
        if pd.isna(row.UNIT) or row.unit_resolution == "unresolved":
            reason.append("unresolved_unit")
        if row.pairs < min_pairs:
            reason.append("insufficient_pairs")
        if row.countries < min_countries:
            reason.append("insufficient_countries")
        if row.origin_years < min_origin_years:
            reason.append("insufficient_origin_years")
        reasons.append("|".join(reason))
        states.append("eligible" if not reason else "not_admitted")
    support["provider_projection_rows"] = 0
    support["admission_state"] = states
    support["exclusion_reasons"] = reasons
    return support


def _summarize_overlay_case(
    feature_id: str,
    horizon: int,
    records: list[dict],
    predictions: list[pd.DataFrame],
) -> tuple[list[dict], dict | None, pd.DataFrame]:
    if not records or not predictions:
        return records, None, pd.DataFrame()
    fold_frame = pd.DataFrame(records)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    aggregate_records = []
    for model, data in fold_frame.groupby("model", observed=True):
        weight = data.rows.to_numpy(dtype=float)
        aggregate_records.append(
            {
                "feature_id": feature_id,
                "horizon": int(horizon),
                "model": model,
                "folds": len(data),
                "rows": int(data.rows.sum()),
                "rmse": float(np.sqrt(np.average(data.rmse**2, weights=weight))),
                "mae": float(np.average(data.mae, weights=weight)),
                "fraction_beating_no_change": float(
                    np.average(data.fraction_beating_no_change, weights=weight)
                ),
                "fraction_beating_own_history": (
                    float(np.average(data.fraction_beating_own_history, weights=weight))
                    if data.fraction_beating_own_history.notna().any()
                    else np.nan
                ),
            }
        )
    aggregate = pd.DataFrame(aggregate_records).set_index("model")
    required = {"no_change", "own_history", "state_only", "state_overlay"}
    if not required <= set(aggregate.index):
        return records, None, prediction_frame
    own = prediction_frame.loc[prediction_frame.model.eq("own_history")][
        ["entity_code", "forecast_origin_year", "outer_start", "absolute_error"]
    ].rename(columns={"absolute_error": "own_error"})
    overlay = prediction_frame.loc[prediction_frame.model.eq("state_overlay")][
        ["entity_code", "forecast_origin_year", "outer_start", "absolute_error"]
    ].rename(columns={"absolute_error": "overlay_error"})
    paired = own.merge(
        overlay,
        on=["entity_code", "forecast_origin_year", "outer_start"],
        validate="one_to_one",
    )
    difference = paired.own_error - paired.overlay_error
    if len(difference) >= 10 and not np.allclose(difference, 0):
        try:
            p_value = float(wilcoxon(difference, alternative="greater").pvalue)
        except ValueError:
            p_value = 1.0
    else:
        p_value = 1.0
    no_rmse = float(aggregate.loc["no_change", "rmse"])
    own_rmse = float(aggregate.loc["own_history", "rmse"])
    overlay_rmse = float(aggregate.loc["state_overlay", "rmse"])
    state_rmse = float(aggregate.loc["state_only", "rmse"])
    summary = {
        "feature_id": feature_id,
        "horizon": int(horizon),
        "rows": int(aggregate.loc["state_overlay", "rows"]),
        "no_change_rmse": no_rmse,
        "own_history_rmse": own_rmse,
        "state_only_rmse": state_rmse,
        "state_overlay_rmse": overlay_rmse,
        "own_history_improvement_vs_no_change": (no_rmse - own_rmse) / no_rmse,
        "state_overlay_improvement_vs_no_change": (no_rmse - overlay_rmse) / no_rmse,
        "state_incremental_improvement_vs_own": (own_rmse - overlay_rmse) / own_rmse,
        "fraction_rows_overlay_beats_own": float(np.mean(difference > 0)),
        "paired_wilcoxon_p": p_value,
    }
    return aggregate_records, summary, prediction_frame


def run_overlay_screen(
    historical: pd.DataFrame,
    states: pd.DataFrame,
    information: pd.DataFrame,
    output: Path,
) -> dict:
    state_cols = sorted(
        [c for c in states if c.startswith("state_")],
        key=lambda value: int(value.split("_")[-1]),
    )
    states_info = states.merge(
        information[
            ["entity_code", "forecast_origin_year", "state_uncertainty_proxy", "observed_share"]
        ],
        on=["entity_code", "forecast_origin_year"],
        how="left",
        validate="one_to_one",
    )
    state_keys = states_info[["entity_code", "forecast_origin_year"]].drop_duplicates()
    admission_frames = []
    fold_records: list[dict] = []
    aggregate_records: list[dict] = []
    outcome_records: list[dict] = []
    executed = 0
    started = time.time()

    for horizon in (1, 2):
        pairs = build_all_feature_pairs(historical, state_keys, horizon)
        registry = _support_registry_for_horizon(historical, pairs, horizon)
        admission_frames.append(registry)
        eligible_ids = registry.loc[registry.admission_state.eq("eligible"), "feature_id"].tolist()
        grouped = pairs.groupby("feature_id", observed=True, sort=False)
        for feature_id in eligible_ids:
            if feature_id not in grouped.groups:
                continue
            records, predictions = evaluate_overlay_case(
                feature_id,
                horizon,
                grouped.get_group(feature_id),
                states_info,
                state_cols,
            )
            aggregates, summary, _ = _summarize_overlay_case(
                feature_id, horizon, records, predictions
            )
            fold_records.extend(records)
            aggregate_records.extend(aggregates)
            if summary is not None:
                outcome_records.append(summary)
                executed += 1
            if executed and executed % 50 == 0:
                print(
                    f"overlay_progress {executed} elapsed={time.time()-started:.1f}s",
                    flush=True,
                )
        del pairs, grouped

    admission = pd.concat(admission_frames, ignore_index=True)
    folds = pd.DataFrame(fold_records)
    aggregate = pd.DataFrame(aggregate_records)
    outcomes = pd.DataFrame(outcome_records)
    folds.to_csv(output / "observable-overlay-fold-metrics.csv.gz", index=False)
    aggregate.to_csv(output / "observable-overlay-model-metrics.csv.gz", index=False)

    if len(outcomes):
        outcomes["paired_bh_q"] = outcomes.groupby(
            "horizon", group_keys=False
        ).paired_wilcoxon_p.apply(benjamini_hochberg)
        fold_compare = folds.pivot_table(
            index=["feature_id", "horizon", "outer_start", "outer_end"],
            columns="model",
            values="rmse",
            aggfunc="first",
        ).reset_index()
        fold_compare["fold_state_incremental_improvement"] = (
            fold_compare["own_history"] - fold_compare["state_overlay"]
        ) / fold_compare["own_history"]
        fold_stability = (
            fold_compare.groupby(["feature_id", "horizon"], observed=True)
            .fold_state_incremental_improvement
            .agg(
                executed_windows="size",
                positive_windows=lambda values: int((values > 0).sum()),
                min_fold_incremental="min",
                median_fold_incremental="median",
                max_fold_incremental="max",
            )
            .reset_index()
        )
        outcomes = outcomes.merge(
            fold_stability,
            on=["feature_id", "horizon"],
            how="left",
            validate="one_to_one",
        )

        def evidence_tier(row):
            improvement = float(row.state_incremental_improvement_vs_own)
            q_value = float(row.paired_bh_q) if pd.notna(row.paired_bh_q) else 1.0
            fraction = float(row.fraction_rows_overlay_beats_own)
            windows = int(row.executed_windows)
            positive = int(row.positive_windows)
            own_improvement = float(row.own_history_improvement_vs_no_change)
            if improvement <= -0.05:
                return "state_overlay_harmful"
            if (
                windows >= 2
                and positive == windows
                and improvement >= 0.01
                and fraction > 0.50
                and q_value <= 0.10
            ):
                return "state_incremental_material_stable"
            if improvement > 0 and q_value <= 0.10 and fraction > 0.50:
                return "state_incremental_small_or_partial"
            if improvement > 0:
                return "state_incremental_exploratory"
            if own_improvement > 0:
                return "own_history_only"
            return "no_confirmed_gain"

        outcomes["overlay_evidence_tier"] = outcomes.apply(evidence_tier, axis=1)
        outcomes = outcomes.merge(
            admission[
                [
                    "feature_id", "horizon", "source", "INDICATOR", "indicator_label",
                    "UNIT", "unit_resolution",
                ]
            ],
            on=["feature_id", "horizon"],
            how="left",
            validate="one_to_one",
        )
    outcomes.to_csv(output / "observable-overlay-summary.csv.gz", index=False)

    if len(outcomes):
        admission = admission.merge(
            outcomes[["feature_id", "horizon", "executed_windows", "overlay_evidence_tier"]],
            on=["feature_id", "horizon"],
            how="left",
            validate="one_to_one",
        )
        admission["executed_windows"] = admission.executed_windows.fillna(0).astype(int)
        admission["execution_state"] = np.where(
            admission.admission_state.ne("eligible"),
            "not_admitted",
            np.where(
                admission.executed_windows.eq(0),
                "eligible_but_no_registered_window",
                np.where(
                    admission.executed_windows.eq(1),
                    "executed_one_window",
                    "executed_two_windows",
                ),
            ),
        )
    admission.to_csv(output / "observable-outcome-admission-ledger.csv.gz", index=False)
    tier_counts = outcomes.overlay_evidence_tier.value_counts().to_dict() if len(outcomes) else {}
    return {
        "registered_outcome_horizons": len(admission),
        "eligible_outcome_horizons": int(admission.admission_state.eq("eligible").sum()),
        "executed_outcome_horizons": int(outcomes.shape[0]),
        "outcome_features_executed": int(outcomes.feature_id.nunique()) if len(outcomes) else 0,
        "tier_counts": {str(key): int(value) for key, value in tier_counts.items()},
        "provider_projection_rows_read": 0,
        "selected_banking_targets_predeclared": 0,
        "final_confirmation_evaluated": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-root", type=Path, required=True)
    parser.add_argument("--phase2-results", type=Path, required=True)
    parser.add_argument("--phase3-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("Output directory must be new")
    args.output.mkdir(parents=True)

    started = time.time()
    states = pd.read_csv(args.phase2_results / "country-year-states.csv.gz")
    information = pd.read_csv(args.phase2_results / "country-year-information.csv.gz")

    interpretation = build_interpretation(
        args.phase2_results, args.phase3_results, args.output
    )
    cells, conflicts, raw_summary = load_historical_cells(args.m2_root)
    conflicts.to_csv(args.output / "outcome-source-conflicts.csv.gz", index=False)
    transformed, transform_summary = transform_historical_values(
        cells, set(states.entity_code)
    )
    overlay = run_overlay_screen(transformed, states, information, args.output)

    report = {
        "status": "completed_phase4a_interpretation_and_phase4b_observable_overlay_development",
        "elapsed_seconds": time.time() - started,
        "interpretation": interpretation,
        "source_observation_summary": raw_summary,
        "outcome_transformation_summary": transform_summary,
        "observable_overlays": overlay,
        "phase4c_crisis_overlay_status": "deferred_pending_event_label_alignment_and_comparable_production_benchmark",
        "provider_projection_rows_read": 0,
        "measurement_state_refit": False,
        "production_modified": False,
        "production_classifier_retrained": False,
        "final_confirmation_evaluated": False,
        "limitations": [
            "The outcome screen is retrospective latest-vintage development evidence.",
            "Compact theme clustering may legitimately fail when the state remains highly mixed.",
            "Currency outcomes use causal own-history scaling.",
            "Wilcoxon/BH evidence is exploratory and does not replace vintage-clean final confirmation.",
            "The crisis overlay remains a separate Phase 4 slice because comparable event-label and production-benchmark alignment must be verified rather than inferred.",
        ],
    }
    write_json(args.output / "phase4-summary.json", report)
    checksums = {
        str(path.relative_to(args.output)): file_sha256(path)
        for path in sorted(args.output.rglob("*"))
        if path.is_file() and path.name != "output-checksums.json"
    }
    write_json(args.output / "output-checksums.json", checksums)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
