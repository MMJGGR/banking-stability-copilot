"""Direct calibration of Phase 2A state uncertainty by observation masking.

Feature-reconstruction error and state-estimation error are different objects.
This module hides observed cells, re-estimates country-year states with frozen
loadings, and checks whether the information-based uncertainty proxy tracks the
resulting state error.
"""
from __future__ import annotations

import hashlib
import random
import numpy as np
import pandas as pd

from .inventory import ForecastDataError
from .phase2_measurement import _require_torch

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _row_hash(entity: str, year: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{entity}|{int(year)}".encode()).digest()[:8],
        "big",
    )


def _cell_hash(entity: str, year: int, predictor: str) -> float:
    value = int.from_bytes(
        hashlib.sha256(
            f"{entity}|{int(year)}|{predictor}|uncertainty-mask".encode()
        ).digest()[:8],
        "big",
    )
    return value / 2**64


def masked_state_calibration(
    model,
    scaled_cells: pd.DataFrame,
    reliability: pd.DataFrame,
    *,
    max_rows: int = 400,
    mask_fraction: float = 0.30,
    epochs: int = 80,
    learning_rate: float = 0.05,
    seed: int = 23,
) -> dict:
    """Re-estimate sampled row states after hiding observed measurements."""
    _require_torch()
    if not 0 < mask_fraction < 0.8:
        raise ForecastDataError("Invalid uncertainty mask fraction")
    required = {"entity_code", "forecast_origin_year", "predictor_id", "z"}
    if not required <= set(scaled_cells):
        raise ForecastDataError("Incomplete scaled-cell schema")

    rows = model.rows_.drop(columns="row_idx").copy()
    rows["_hash"] = [
        _row_hash(row.entity_code, row.forecast_origin_year)
        for row in rows.itertuples()
    ]
    rows = rows.sort_values("_hash").head(max_rows).reset_index(drop=True)
    row_map = {
        (row.entity_code, int(row.forecast_origin_year)): i
        for i, row in enumerate(rows.itertuples())
    }
    data = scaled_cells.loc[
        [
            (entity, int(year)) in row_map
            for entity, year in zip(
                scaled_cells.entity_code,
                scaled_cells.forecast_origin_year,
            )
        ]
    ].copy()
    data["_row"] = [
        row_map[(entity, int(year))]
        for entity, year in zip(data.entity_code, data.forecast_origin_year)
    ]
    data["_col"] = data.predictor_id.map(model.feature_map_)
    data = data.loc[data._col.notna()].copy()
    data["_col"] = data._col.astype(int)
    data["_mask_score"] = [
        _cell_hash(entity, year, predictor)
        for entity, year, predictor in zip(
            data.entity_code,
            data.forecast_origin_year,
            data.predictor_id,
        )
    ]
    data["_retained"] = data._mask_score >= mask_fraction
    for _, index in data.groupby("_row", sort=False).groups.items():
        index = list(index)
        minimum = min(12, len(index))
        have = int(data.loc[index, "_retained"].sum())
        if have < minimum:
            order = data.loc[index].sort_values("_mask_score", ascending=False).index
            data.loc[order[:minimum], "_retained"] = True
    retained = data.loc[data._retained].copy()
    if retained.empty:
        raise ForecastDataError("No retained uncertainty-calibration cells")

    variance = (
        reliability.set_index("predictor_id")
        .shrunk_residual_variance.to_dict()
    )
    observation_weight = 1.0 / np.maximum(
        retained.predictor_id.map(variance).to_numpy(dtype=float),
        1e-4,
    )
    lo, hi = np.quantile(observation_weight, [0.02, 0.98])
    observation_weight = np.clip(observation_weight, lo, hi)
    observation_weight /= np.mean(observation_weight)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    state = torch.nn.Embedding(len(rows), model.rank)
    torch.nn.init.zeros_(state.weight)
    loadings = torch.as_tensor(model.loadings_, dtype=torch.float32)
    bias = torch.as_tensor(model.feature_bias_, dtype=torch.float32)
    row_index = torch.as_tensor(retained._row.to_numpy(), dtype=torch.long)
    col_index = torch.as_tensor(retained._col.to_numpy(), dtype=torch.long)
    observed = torch.as_tensor(retained.z.to_numpy(), dtype=torch.float32)
    weights = torch.as_tensor(observation_weight, dtype=torch.float32)
    optimizer = torch.optim.Adam([state.weight], lr=learning_rate)

    for _ in range(epochs):
        prediction = (
            (state(row_index) * loadings[col_index]).sum(1)
            + bias[col_index]
        )
        squared = (prediction - observed) ** 2
        loss = torch.sum(weights * squared) / torch.sum(weights)
        loss = loss + model.l2 * state.weight.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    inferred = state.weight.detach().cpu().numpy()
    truth = np.vstack(
        [
            model.states_[
                model.row_map_[(row.entity_code, int(row.forecast_origin_year))]
            ]
            for row in rows.itertuples()
        ]
    )
    state_error = np.sqrt(np.mean((inferred - truth) ** 2, axis=1))

    information = np.full((len(rows), model.rank), model.l2, dtype=float)
    retained_count = np.zeros(len(rows), dtype=int)
    for row in retained.itertuples():
        feature_variance = max(float(variance[row.predictor_id]), 1e-4)
        information[row._row] += (
            model.loadings_[row._col] ** 2 / feature_variance
        )
        retained_count[row._row] += 1
    uncertainty = np.sqrt(
        np.mean(1 / np.maximum(information, 1e-9), axis=1)
    )
    result = rows[["entity_code", "forecast_origin_year"]].copy()
    result["retained_cells"] = retained_count
    result["state_error"] = state_error
    result["uncertainty_proxy"] = uncertainty
    correlation = float(
        result.state_error.corr(
            result.uncertainty_proxy,
            method="spearman",
        )
    )
    result["uncertainty_quartile"] = pd.qcut(
        result.uncertainty_proxy,
        4,
        duplicates="drop",
    ).astype(str)
    calibration = (
        result.groupby("uncertainty_quartile", observed=True)
        .agg(
            rows=("state_error", "size"),
            mean_state_error=("state_error", "mean"),
            mean_uncertainty=("uncertainty_proxy", "mean"),
            mean_retained_cells=("retained_cells", "mean"),
        )
        .reset_index()
    )
    return {
        "rows": len(result),
        "mask_fraction": mask_fraction,
        "state_error_uncertainty_spearman": correlation,
        "mean_state_error": float(result.state_error.mean()),
        "mean_uncertainty": float(result.uncertainty_proxy.mean()),
        "row_results": result,
        "quartile_calibration": calibration,
    }
