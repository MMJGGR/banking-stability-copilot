"""Weighted sensitivities for the Phase 2A linear measurement state.

Weights change how repeated observations influence the fit; they do not delete
feature identities. The same target-independent observed-cell objective and
fixed-loading state definition are retained.
"""
from __future__ import annotations

import random
import numpy as np
import pandas as pd

from .inventory import ForecastDataError
from .phase2_measurement import MaskedLinearStateModel, _require_torch

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def row_feature_balance(cells: pd.DataFrame) -> pd.Series:
    """Equalize row/feature energy without imposing source or economic weights."""
    required = {"entity_code", "forecast_origin_year", "predictor_id"}
    if not required <= set(cells):
        raise ForecastDataError("Incomplete weighted-cell identity")
    row_key = cells.entity_code.astype(str) + "|" + cells.forecast_origin_year.astype(str)
    row_count = row_key.groupby(row_key, sort=False).transform("size").to_numpy(dtype=float)
    feature_count = cells.groupby("predictor_id", observed=True).predictor_id.transform("size").to_numpy(dtype=float)
    weight = 1.0 / np.sqrt(np.maximum(row_count * feature_count, 1.0))
    weight /= np.mean(weight)
    return pd.Series(weight, index=cells.index, dtype=float)


def reliability_weights(
    cells: pd.DataFrame,
    feature_reliability: pd.DataFrame,
    *,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
) -> pd.Series:
    """Combine row/feature balance with residual measurement reliability."""
    if "shrunk_residual_variance" not in feature_reliability:
        raise ForecastDataError("Missing feature reliability variance")
    variance = feature_reliability.set_index("predictor_id").shrunk_residual_variance
    mapped = cells.predictor_id.map(variance)
    if mapped.isna().any():
        raise ForecastDataError("Missing reliability for fitted feature")
    weight = row_feature_balance(cells).to_numpy() / np.maximum(mapped.to_numpy(dtype=float), 1e-4)
    lo, hi = np.quantile(weight, [lower_quantile, upper_quantile])
    weight = np.clip(weight, lo, hi)
    weight /= np.mean(weight)
    return pd.Series(weight, index=cells.index, dtype=float)


class WeightedMaskedLinearStateModel(MaskedLinearStateModel):
    """Masked linear factor model with positive observation weights."""

    def fit(self, cells: pd.DataFrame, observation_weight=None):
        _require_torch()
        required = {"z", "entity_code", "forecast_origin_year", "predictor_id"}
        if not required <= set(cells):
            raise ForecastDataError("Incomplete scaled-cell schema")
        if observation_weight is None:
            weight = np.ones(len(cells), dtype=float)
        else:
            series = pd.Series(observation_weight, index=cells.index, dtype=float)
            if not np.isfinite(series).all() or not series.gt(0).all():
                raise ForecastDataError("Observation weights must be positive and finite")
            weight = series.to_numpy(dtype=float)
            weight /= np.mean(weight)

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

        data = cells.merge(rows, on=["entity_code", "forecast_origin_year"], validate="many_to_one")
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
        weights = torch.as_tensor(weight, dtype=torch.float32)
        generator = torch.Generator().manual_seed(self.seed)

        losses = []
        for _ in range(self.epochs):
            permutation = torch.randperm(len(data), generator=generator)
            total = 0.0
            total_weight = 0.0
            for start in range(0, len(data), self.batch_size):
                index = permutation[start : start + self.batch_size]
                u = row_factor(row_idx[index])
                v = loading(col_idx[index])
                prediction = (u * v).sum(1) + feature_bias(col_idx[index]).squeeze(1)
                batch_weight = weights[index]
                squared = (prediction - observed[index]) ** 2
                mse = torch.sum(batch_weight * squared) / torch.sum(batch_weight)
                regularization = self.l2 * (u.pow(2).mean() + v.pow(2).mean())
                loss = mse + regularization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(torch.sum(batch_weight * squared).detach())
                total_weight += float(torch.sum(batch_weight).detach())
            losses.append(total / total_weight)

        u = row_factor.weight.detach().cpu().numpy()
        v = loading.weight.detach().cpu().numpy()
        bias = feature_bias.weight.detach().cpu().numpy().ravel()
        state_mean = u.mean(axis=0)
        u = u - state_mean
        bias = bias + v @ state_mean
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
        self.weight_summary_ = {
            "minimum": float(np.min(weight)),
            "median": float(np.median(weight)),
            "maximum": float(np.max(weight)),
            "mean": float(np.mean(weight)),
        }
        return self
