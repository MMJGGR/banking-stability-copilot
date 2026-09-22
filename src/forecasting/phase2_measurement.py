"""Phase 2A: target-independent, missing-aware measurement state.

The model is linear low-rank matrix factorization fitted only to genuinely
observed cells. Missing cells never enter the objective as median/zero
pseudo-observations. PyTorch is used only as an efficient optimizer for this
linear model; this is not a nonlinear neural-network architecture.

See docs/prd/broad-feature-phase2-v0.6.md.
"""
from __future__ import annotations

import hashlib
import random
import numpy as np
import pandas as pd

from .inventory import ForecastDataError

try:
    import torch
except ImportError:  # pragma: no cover - explicit research dependency
    torch = None


CURRENCY_UNITS = frozenset({"XDC", "USD", "EUR", "XDR"})


def _require_torch():
    if torch is None:
        raise ForecastDataError(
            "Phase 2A research requires PyTorch for masked linear-factor optimization"
        )


def deterministic_holdout(cells: pd.DataFrame, fraction: float = 0.05) -> pd.Series:
    """Stable target-independent cell holdout; retain training support per row/feature."""
    if not 0 < fraction < 0.5:
        raise ForecastDataError("Holdout fraction must lie in (0, 0.5)")
    required = {"entity_code", "forecast_origin_year", "predictor_id"}
    if not required <= set(cells):
        raise ForecastDataError("Incomplete observed-cell identity")

    scores = []
    for row in cells[
        ["entity_code", "forecast_origin_year", "predictor_id"]
    ].itertuples(index=False):
        key = (
            f"{row.entity_code}|{int(row.forecast_origin_year)}|{row.predictor_id}"
        ).encode()
        scores.append(
            int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        )
    holdout = pd.Series(np.asarray(scores) < fraction, index=cells.index, dtype=bool)

    row_key = (
        cells.entity_code.astype(str)
        + "|"
        + cells.forecast_origin_year.astype(str)
    )
    for key in (row_key, cells.predictor_id.astype(str)):
        total = key.groupby(key, sort=False).transform("size")
        held = holdout.groupby(key, sort=False).transform("sum")
        unsupported = holdout & held.ge(total)
        if unsupported.any():
            first = (
                cells.loc[unsupported]
                .groupby(key.loc[unsupported], sort=False)
                .head(1)
                .index
            )
            holdout.loc[first] = False
    return holdout


def prepare_observed_cells(
    predictors: pd.DataFrame,
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply unit policy while keeping the matrix long and genuinely sparse."""
    required = {
        "entity_code",
        "predictor_id",
        "feature_id",
        "forecast_origin_year",
        "observation_year",
        "value",
        "break_in_year",
    }
    if not required <= set(predictors):
        raise ForecastDataError(
            f"Incomplete Phase 2 predictors: {sorted(required-set(predictors))}"
        )
    if not registry.feature_id.is_unique:
        raise ForecastDataError("Feature registry identities must be unique")

    data = predictors[list(required)].copy()
    data["value"] = pd.to_numeric(data.value, errors="raise").astype(float)
    if not np.isfinite(data.value).all():
        raise ForecastDataError("Nonfinite staged predictor value")

    r = registry.set_index("feature_id")
    if not set(data.feature_id) <= set(r.index):
        raise ForecastDataError("Unknown feature identity")

    meta = (
        data[["predictor_id", "feature_id"]]
        .drop_duplicates()
        .set_index("predictor_id")
    )
    metadata = [
        c for c in ["source", "INDICATOR", "indicator_label", "UNIT", "SCALE"]
        if c in r
    ]
    meta = meta.join(r[metadata], on="feature_id")
    meta["unit_policy"] = np.where(
        meta.UNIT.isna(),
        "quarantined_unresolved_unit",
        np.where(
            meta.UNIT.isin(CURRENCY_UNITS),
            "causal_own_history_amount",
            "observed_native_unit",
        ),
    )

    data["model_value"] = data.value
    data.loc[data.feature_id.map(r.UNIT).isna(), "model_value"] = np.nan

    amount = data.feature_id.map(r.UNIT).isin(CURRENCY_UNITS)
    monetary = data.loc[
        amount,
        [
            "entity_code",
            "predictor_id",
            "forecast_origin_year",
            "value",
            "break_in_year",
        ],
    ].copy()
    monetary = monetary.sort_values(
        ["entity_code", "predictor_id", "forecast_origin_year"]
    )
    unsupported_amount = 0
    if len(monetary):
        keys = ["entity_code", "predictor_id"]
        monetary["_segment"] = monetary.groupby(
            keys, observed=True, sort=False
        ).break_in_year.cumsum()
        group_keys = [monetary[k] for k in keys + ["_segment"]]
        previous_count = monetary.groupby(
            keys + ["_segment"], observed=True, sort=False
        ).cumcount()
        abs_value = monetary.value.abs()
        previous_sum = (
            abs_value.groupby(group_keys, observed=True, sort=False).cumsum()
            - abs_value
        )
        denominator = previous_sum / previous_count.replace(0, np.nan)
        supported = previous_count.ge(2) & denominator.gt(0)
        normalized = pd.Series(np.nan, index=monetary.index, dtype=float)
        normalized.loc[supported] = np.arcsinh(
            monetary.loc[supported, "value"] / denominator.loc[supported]
        )
        data.loc[monetary.index, "model_value"] = normalized
        unsupported_amount = int((~supported).sum())

    cells = data.loc[
        np.isfinite(data.model_value),
        [
            "entity_code",
            "forecast_origin_year",
            "predictor_id",
            "feature_id",
            "model_value",
        ],
    ].copy()

    stats = cells.groupby("predictor_id", observed=True).model_value.agg(
        observed_cells="size",
        distinct_values="nunique",
    )
    ledger = meta.join(stats).reset_index()
    ledger["measurement_state"] = np.where(
        ledger.unit_policy.eq("quarantined_unresolved_unit"),
        "quarantined_unit",
        np.where(
            ledger.observed_cells.fillna(0).lt(2),
            "insufficient_observed_cells",
            np.where(
                ledger.distinct_values.fillna(0).lt(2),
                "constant_observed",
                "eligible",
            ),
        ),
    )
    summary = {
        "representations": len(ledger),
        "eligible_representations": int(
            ledger.measurement_state.eq("eligible").sum()
        ),
        "observed_cells": len(cells),
        "missing_values_imputed": 0,
        "feature_count_cap": None,
        "amount_cells_without_supported_prior_scale": unsupported_amount,
    }
    return cells, ledger, summary


class RobustObservedScaler:
    """Feature-wise robust scaling fitted only on observed training cells."""

    def fit(self, cells: pd.DataFrame):
        if cells.empty:
            raise ForecastDataError("Empty scaler input")
        grouped = cells.groupby("predictor_id", observed=True).model_value
        quantiles = grouped.quantile([0.25, 0.50, 0.75]).unstack()
        quantiles.columns = ["q25", "median", "q75"]
        stats = (
            quantiles.join(grouped.std().rename("std"))
            .join(grouped.size().rename("n"))
        )
        robust_scale = (stats.q75 - stats.q25) / 1.349
        stats["scale"] = robust_scale.where(robust_scale > 1e-12, stats["std"])
        self.stats_ = stats
        self.active_ = stats.index[
            stats.scale.notna() & stats.scale.gt(1e-12) & stats.n.ge(2)
        ].tolist()
        return self

    def transform(self, cells: pd.DataFrame) -> pd.DataFrame:
        data = cells.loc[cells.predictor_id.isin(self.active_)].copy()
        data = data.join(
            self.stats_[["median", "scale"]],
            on="predictor_id",
        )
        data["z"] = (
            (data.model_value - data["median"]) / data["scale"]
        ).clip(-8, 8)
        return data.drop(columns=["median", "scale"])


class MaskedLinearStateModel:
    """Linear fixed-loading state model optimized only on observed cells."""

    def __init__(
        self,
        rank: int,
        *,
        l2: float = 1e-3,
        learning_rate: float = 0.03,
        epochs: int = 10,
        batch_size: int = 65536,
        seed: int = 17,
    ):
        if rank < 1 or l2 <= 0 or learning_rate <= 0:
            raise ForecastDataError("Invalid Phase 2A model configuration")
        if epochs < 1 or batch_size < 1:
            raise ForecastDataError("Invalid fit iteration configuration")
        self.rank = int(rank)
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    def fit(self, cells: pd.DataFrame):
        _require_torch()
        required = {
            "z",
            "entity_code",
            "forecast_origin_year",
            "predictor_id",
        }
        if not required <= set(cells):
            raise ForecastDataError("Incomplete scaled-cell schema")

        rows = (
            cells[["entity_code", "forecast_origin_year"]]
            .drop_duplicates()
            .sort_values(["entity_code", "forecast_origin_year"])
            .reset_index(drop=True)
        )
        rows["row_idx"] = np.arange(len(rows))
        features = sorted(cells.predictor_id.unique())
        feature_map = {name: idx for idx, name in enumerate(features)}
        if self.rank >= min(len(rows), len(features)):
            raise ForecastDataError("State rank must be below row and feature counts")

        data = cells.merge(
            rows,
            on=["entity_code", "forecast_origin_year"],
            validate="many_to_one",
        )
        data["col_idx"] = data.predictor_id.map(feature_map)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        row_factor = torch.nn.Embedding(len(rows), self.rank)
        loading = torch.nn.Embedding(len(features), self.rank)
        feature_bias = torch.nn.Embedding(len(features), 1)
        torch.nn.init.normal_(row_factor.weight, std=0.03)
        torch.nn.init.normal_(loading.weight, std=0.03)
        torch.nn.init.zeros_(feature_bias.weight)

        optimizer = torch.optim.Adam(
            [row_factor.weight, loading.weight, feature_bias.weight],
            lr=self.learning_rate,
        )
        row_idx = torch.as_tensor(data.row_idx.to_numpy(), dtype=torch.long)
        col_idx = torch.as_tensor(data.col_idx.to_numpy(), dtype=torch.long)
        observed = torch.as_tensor(data.z.to_numpy(), dtype=torch.float32)
        generator = torch.Generator().manual_seed(self.seed)

        losses = []
        for _ in range(self.epochs):
            permutation = torch.randperm(len(data), generator=generator)
            total = 0.0
            count = 0
            for start in range(0, len(data), self.batch_size):
                index = permutation[start : start + self.batch_size]
                u = row_factor(row_idx[index])
                v = loading(col_idx[index])
                prediction = (
                    (u * v).sum(1)
                    + feature_bias(col_idx[index]).squeeze(1)
                )
                mse = torch.mean((prediction - observed[index]) ** 2)
                regularization = self.l2 * (
                    u.pow(2).mean() + v.pow(2).mean()
                )
                loss = mse + regularization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(mse.detach()) * len(index)
                count += len(index)
            losses.append(total / count)

        u = row_factor.weight.detach().cpu().numpy()
        v = loading.weight.detach().cpu().numpy()
        bias = feature_bias.weight.detach().cpu().numpy().ravel()

        # Remove the arbitrary state mean while preserving fitted values.
        state_mean = u.mean(axis=0)
        u = u - state_mean
        bias = bias + v @ state_mean

        # Fix the otherwise arbitrary factor rotation and sign so stored
        # coordinates are reproducible and ordered.
        _, _, qt = np.linalg.svd(u, full_matrices=False)
        rotation = qt.T
        u = u @ rotation
        v = v @ rotation
        for component in range(self.rank):
            anchor = int(np.argmax(np.abs(v[:, component])))
            sign = 1.0 if v[anchor, component] >= 0 else -1.0
            u[:, component] *= sign
            v[:, component] *= sign

        self.rows_ = rows
        self.features_ = features
        self.feature_map_ = feature_map
        self.row_map_ = {
            (row.entity_code, int(row.forecast_origin_year)): int(row.row_idx)
            for row in rows.itertuples()
        }
        self.states_ = u
        self.loadings_ = v
        self.feature_bias_ = bias
        self.train_loss_ = losses
        return self

    def predict(self, cells: pd.DataFrame) -> np.ndarray:
        values = np.full(len(cells), np.nan, dtype=float)
        for position, row in enumerate(cells.itertuples(index=False)):
            i = self.row_map_.get(
                (row.entity_code, int(row.forecast_origin_year))
            )
            j = self.feature_map_.get(row.predictor_id)
            if i is not None and j is not None:
                values[position] = (
                    self.feature_bias_[j]
                    + self.states_[i] @ self.loadings_[j]
                )
        return values

    def state_frame(self) -> pd.DataFrame:
        result = self.rows_.drop(columns="row_idx").copy()
        for component in range(self.rank):
            result[f"state_{component+1}"] = self.states_[:, component]
        return result


def evaluate_reconstruction(
    model: MaskedLinearStateModel,
    cells: pd.DataFrame,
) -> dict:
    prediction = model.predict(cells)
    valid = np.isfinite(prediction)
    data = cells.loc[
        valid,
        ["entity_code", "forecast_origin_year", "predictor_id", "z"],
    ].copy()
    data["prediction"] = prediction[valid]
    data["sq_error"] = (data.prediction - data.z) ** 2
    if data.empty:
        raise ForecastDataError("No evaluable held-out cells")

    row_rmse = np.sqrt(
        data.groupby(
            ["entity_code", "forecast_origin_year"],
            observed=True,
        ).sq_error.mean()
    )
    return {
        "validation_cells": len(data),
        "row_rmse_mean": float(row_rmse.mean()),
        "row_rmse_se": (
            float(row_rmse.std(ddof=1) / np.sqrt(len(row_rmse)))
            if len(row_rmse) > 1
            else 0.0
        ),
        "cell_rmse": float(np.sqrt(data.sq_error.mean())),
        "cell_results": data,
    }


def select_measurement_spec(
    cells: pd.DataFrame,
    *,
    ranks=(8, 16, 32, 64),
    l2_grid=(1e-3, 1e-2),
    holdout_fraction=0.05,
    epochs=8,
    learning_rate=0.03,
    batch_size=65536,
    seed=17,
) -> dict:
    """Select rank from held-out observed cells; flag an unresolved upper boundary."""
    holdout = deterministic_holdout(cells, holdout_fraction)
    raw_train = cells.loc[~holdout].copy()
    raw_validation = cells.loc[holdout].copy()

    scaler = RobustObservedScaler().fit(raw_train)
    train = scaler.transform(raw_train)
    validation = scaler.transform(raw_validation)

    n_rows = len(
        train[["entity_code", "forecast_origin_year"]].drop_duplicates()
    )
    n_features = int(train.predictor_id.nunique())
    records = []

    for rank in ranks:
        for l2 in l2_grid:
            if rank >= min(n_rows, n_features):
                continue
            model = MaskedLinearStateModel(
                rank,
                l2=l2,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                seed=seed,
            ).fit(train)
            evaluation = evaluate_reconstruction(model, validation)
            records.append(
                {
                    "rank": rank,
                    "l2": l2,
                    "validation_cells": evaluation["validation_cells"],
                    "row_rmse_mean": evaluation["row_rmse_mean"],
                    "row_rmse_se": evaluation["row_rmse_se"],
                    "cell_rmse": evaluation["cell_rmse"],
                    "final_train_mse": model.train_loss_[-1],
                }
            )

    results = pd.DataFrame(records)
    if results.empty:
        raise ForecastDataError("No candidate measurement specification fitted")

    best = results.loc[results.row_rmse_mean.idxmin()]
    eligible = results.loc[
        results.row_rmse_mean <= best.row_rmse_mean + best.row_rmse_se
    ].sort_values(["rank", "l2"], ascending=[True, False])
    chosen = eligible.iloc[0]

    return {
        "chosen_rank": int(chosen["rank"]),
        "chosen_l2": float(chosen["l2"]),
        "raw_best_rank": int(best["rank"]),
        "raw_best_l2": float(best["l2"]),
        "rank_search_boundary_reached": bool(
            int(best["rank"]) == int(results["rank"].max())
        ),
        "results": results,
        "holdout_mask": holdout,
        "scaler": scaler,
    }


def feature_reliability(
    model: MaskedLinearStateModel,
    cells: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    prior_strength: float = 25.0,
) -> pd.DataFrame:
    prediction = model.predict(cells)
    data = cells.copy()
    data["prediction"] = prediction
    data = data[np.isfinite(data.prediction)].copy()
    data["sq_error"] = (data.z - data.prediction) ** 2

    residual = (
        data.groupby("predictor_id", observed=True)
        .sq_error.agg(
            residual_variance="mean",
            residual_cells="size",
        )
        .reset_index()
    )
    global_variance = float(data.sq_error.mean())
    residual["shrunk_residual_variance"] = (
        residual.residual_cells * residual.residual_variance
        + prior_strength * global_variance
    ) / (residual.residual_cells + prior_strength)

    metadata = [
        c
        for c in [
            "predictor_id",
            "feature_id",
            "source",
            "INDICATOR",
            "indicator_label",
            "UNIT",
            "unit_policy",
            "measurement_state",
        ]
        if c in ledger
    ]
    return residual.merge(
        ledger[metadata].drop_duplicates("predictor_id"),
        on="predictor_id",
        how="left",
        validate="one_to_one",
    )


def row_information(
    model: MaskedLinearStateModel,
    cells: pd.DataFrame,
    reliability: pd.DataFrame,
) -> pd.DataFrame:
    """Diagonal information approximation; uncertainty rises with sparse/noisy data."""
    variance = (
        reliability.set_index("predictor_id")
        .shrunk_residual_variance.to_dict()
    )
    observed = cells.groupby(
        ["entity_code", "forecast_origin_year"],
        sort=False,
    ).predictor_id.agg(list)

    records = []
    total_features = len(model.features_)
    for row in model.rows_.itertuples():
        features = observed.get(
            (row.entity_code, row.forecast_origin_year),
            [],
        )
        information = np.full(model.rank, model.l2, dtype=float)
        used = 0
        for feature in features:
            column = model.feature_map_.get(feature)
            if column is None:
                continue
            feature_variance = max(
                float(variance.get(feature, 1.0)),
                1e-4,
            )
            information += (
                model.loadings_[column] ** 2 / feature_variance
            )
            used += 1
        records.append(
            {
                "entity_code": row.entity_code,
                "forecast_origin_year": int(row.forecast_origin_year),
                "observed_model_features": used,
                "observed_share": used / max(total_features, 1),
                "state_uncertainty_proxy": float(
                    np.sqrt(
                        np.mean(1 / np.maximum(information, 1e-9))
                    )
                ),
                "effective_information": float(np.mean(information)),
            }
        )
    return pd.DataFrame(records)


def coverage_geometry_diagnostics(
    states: pd.DataFrame,
    information: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    data = states.merge(
        information,
        on=["entity_code", "forecast_origin_year"],
        validate="one_to_one",
    )
    state_columns = [c for c in states if c.startswith("state_")]
    data["state_distance"] = np.sqrt(
        np.mean(data[state_columns].to_numpy() ** 2, axis=1)
    )
    data = data.sort_values(
        ["entity_code", "forecast_origin_year"]
    ).reset_index(drop=True)
    data["movement"] = np.nan
    data["coverage_change"] = np.nan

    for _, index in data.groupby("entity_code").groups.items():
        index = list(index)
        subset = data.loc[index]
        years = subset.forecast_origin_year.to_numpy()
        state = subset[state_columns].to_numpy()
        coverage = subset.observed_share.to_numpy()
        movement = np.full(len(subset), np.nan)
        coverage_change = np.full(len(subset), np.nan)
        for position in range(1, len(subset)):
            if years[position] == years[position - 1] + 1:
                movement[position] = np.sqrt(
                    np.mean(
                        (state[position] - state[position - 1]) ** 2
                    )
                )
                coverage_change[position] = (
                    coverage[position] - coverage[position - 1]
                )
        data.loc[index, "movement"] = movement
        data.loc[index, "coverage_change"] = coverage_change

    diagnostics = {
        "distance_vs_coverage_spearman": float(
            data.state_distance.corr(
                data.observed_share,
                method="spearman",
            )
        ),
        "uncertainty_vs_coverage_spearman": float(
            data.state_uncertainty_proxy.corr(
                data.observed_share,
                method="spearman",
            )
        ),
        "movement_vs_abs_coverage_change_spearman": float(
            data.movement.corr(
                data.coverage_change.abs(),
                method="spearman",
            )
        ),
    }
    return diagnostics, data
