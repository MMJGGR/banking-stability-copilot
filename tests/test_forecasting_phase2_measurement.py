import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.forecasting.phase2_measurement import (
    MaskedLinearStateModel,
    RobustObservedScaler,
    coverage_geometry_diagnostics,
    deterministic_holdout,
    evaluate_reconstruction,
    feature_reliability,
    row_information,
    select_measurement_spec,
)


def synthetic_cells(seed=7, true_rank=3, missing_rate=0.35):
    rng = np.random.default_rng(seed)
    entities = [f"E{i:02d}" for i in range(16)]
    years = list(range(2012, 2020))
    features = [f"f{j:02d}" for j in range(28)]
    latent = rng.normal(size=(len(entities) * len(years), true_rank))
    loadings = rng.normal(size=(len(features), true_rank))
    rows = []
    position = 0
    for entity in entities:
        for year in years:
            for column, feature in enumerate(features):
                if rng.random() < missing_rate:
                    continue
                value = float(
                    latent[position] @ loadings[column]
                    + rng.normal(scale=0.20)
                )
                rows.append(
                    (entity, year, feature, "raw_" + feature, value)
                )
            position += 1
    return pd.DataFrame(
        rows,
        columns=[
            "entity_code",
            "forecast_origin_year",
            "predictor_id",
            "feature_id",
            "model_value",
        ],
    )


def test_holdout_is_deterministic_and_keeps_training_support():
    cells = synthetic_cells()
    first = deterministic_holdout(cells, 0.10)
    second = deterministic_holdout(cells.sample(frac=1, random_state=9), 0.10)
    lookup = dict(zip(
        cells.entity_code.astype(str)
        + "|"
        + cells.forecast_origin_year.astype(str)
        + "|"
        + cells.predictor_id,
        first,
    ))
    shuffled_key = (
        second.index.to_series().map(
            lambda i: (
                cells.loc[i, "entity_code"]
                + "|"
                + str(cells.loc[i, "forecast_origin_year"])
                + "|"
                + cells.loc[i, "predictor_id"]
            )
        )
    )
    assert all(bool(second.loc[i]) == bool(lookup[k]) for i, k in shuffled_key.items())

    training = cells.loc[~first]
    assert (
        training.groupby(["entity_code", "forecast_origin_year"]).size() > 0
    ).all()
    assert (training.groupby("predictor_id").size() > 0).all()


def test_missing_cells_never_enter_fit_matrix():
    cells = synthetic_cells()
    scaler = RobustObservedScaler().fit(cells)
    scaled = scaler.transform(cells)
    assert len(scaled) <= len(cells)
    assert scaled.z.notna().all()
    assert not hasattr(scaler, "imputed_")


def test_masked_linear_model_reconstructs_held_observations():
    cells = synthetic_cells()
    holdout = deterministic_holdout(cells, 0.10)
    scaler = RobustObservedScaler().fit(cells.loc[~holdout])
    train = scaler.transform(cells.loc[~holdout])
    validation = scaler.transform(cells.loc[holdout])

    rank1 = MaskedLinearStateModel(
        1,
        epochs=15,
        learning_rate=0.04,
        batch_size=4096,
        seed=4,
    ).fit(train)
    rank4 = MaskedLinearStateModel(
        4,
        epochs=15,
        learning_rate=0.04,
        batch_size=4096,
        seed=4,
    ).fit(train)

    error1 = evaluate_reconstruction(rank1, validation)["cell_rmse"]
    error4 = evaluate_reconstruction(rank4, validation)["cell_rmse"]
    assert error4 < error1


def test_rank_search_flags_unresolved_upper_boundary():
    cells = synthetic_cells()
    result = select_measurement_spec(
        cells,
        ranks=(1, 2, 4),
        l2_grid=(0.001, 0.01),
        holdout_fraction=0.10,
        epochs=12,
        learning_rate=0.04,
        batch_size=4096,
        seed=2,
    )
    assert len(result["results"]) == 6
    if result["raw_best_rank"] == 4:
        assert result["rank_search_boundary_reached"]


def test_uncertainty_reflects_observed_information():
    cells = synthetic_cells(missing_rate=0.45)
    scaler = RobustObservedScaler().fit(cells)
    scaled = scaler.transform(cells)
    model = MaskedLinearStateModel(
        4,
        epochs=12,
        learning_rate=0.04,
        batch_size=4096,
        seed=3,
    ).fit(scaled)

    feature_names = sorted(scaled.predictor_id.unique())
    ledger = pd.DataFrame(
        {
            "predictor_id": feature_names,
            "feature_id": ["raw_" + f for f in feature_names],
            "source": "SYNTHETIC",
            "measurement_state": "eligible",
        }
    )
    reliability = feature_reliability(model, scaled, ledger)
    information = row_information(model, scaled, reliability)
    states = model.state_frame()
    diagnostics, _ = coverage_geometry_diagnostics(states, information)

    assert diagnostics["uncertainty_vs_coverage_spearman"] < 0


def test_phase2a_source_has_supervised_target_firewall():
    source = Path("src/forecasting/phase2_measurement.py").read_text()
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(
                (node.module or "")
                + ":"
                + ",".join(alias.name for alias in node.names)
            )

    assert "target-pairs.csv" not in source
    assert all("TARGETS" not in name for name in imported)
    assert all("crisis_classifier" not in name for name in imported)
    assert all("risk_model" not in name for name in imported)
