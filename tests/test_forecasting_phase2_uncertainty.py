import numpy as np
import pandas as pd

from src.forecasting.phase2_measurement import (
    MaskedLinearStateModel,
    RobustObservedScaler,
    feature_reliability,
)
from src.forecasting.phase2_uncertainty import masked_state_calibration


def synthetic_cells(seed=41):
    rng = np.random.default_rng(seed)
    rows = []
    entities = [f"E{i:02d}" for i in range(18)]
    features = [f"f{j:02d}" for j in range(24)]
    loadings = rng.normal(size=(len(features), 3))
    for entity in entities:
        for year in range(2010, 2018):
            state = rng.normal(size=3)
            keep_probability = 0.25 + 0.65 * rng.random()
            for j, feature in enumerate(features):
                if rng.random() > keep_probability:
                    continue
                rows.append(
                    {
                        "entity_code": entity,
                        "forecast_origin_year": year,
                        "predictor_id": feature,
                        "feature_id": feature,
                        "model_value": float(
                            state @ loadings[j]
                            + rng.normal(scale=0.12)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_masked_state_calibration_returns_direct_state_error_diagnostic():
    cells = synthetic_cells()
    scaler = RobustObservedScaler().fit(cells)
    scaled = scaler.transform(cells)
    model = MaskedLinearStateModel(
        4,
        epochs=12,
        learning_rate=0.04,
        batch_size=4096,
        seed=9,
    ).fit(scaled)
    ledger = pd.DataFrame(
        {
            "predictor_id": sorted(scaled.predictor_id.unique()),
            "feature_id": sorted(scaled.predictor_id.unique()),
            "source": "SYNTHETIC",
            "measurement_state": "eligible",
        }
    )
    reliability = feature_reliability(model, scaled, ledger)
    result = masked_state_calibration(
        model,
        scaled,
        reliability,
        max_rows=80,
        mask_fraction=0.35,
        epochs=40,
        learning_rate=0.05,
        seed=5,
    )
    assert result["rows"] == 80
    assert np.isfinite(result["state_error_uncertainty_spearman"])
    assert result["row_results"].state_error.gt(0).all()
    assert result["row_results"].uncertainty_proxy.gt(0).all()
    assert result["quartile_calibration"].rows.sum() == 80
