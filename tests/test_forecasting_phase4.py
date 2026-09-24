from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.phase4_execution import (
    benjamini_hochberg,
    transform_historical_values,
    varimax,
)


def test_varimax_is_an_orthogonal_display_rotation():
    rng = np.random.default_rng(17)
    loadings = rng.normal(size=(120, 8))
    rotation, _ = varimax(loadings)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(8), atol=1e-10)
    before = loadings @ loadings.T
    after = (loadings @ rotation) @ (loadings @ rotation).T
    np.testing.assert_allclose(before, after, atol=1e-9)


def test_benjamini_hochberg_preserves_index_and_bounds():
    p = pd.Series([0.001, 0.20, 0.03, np.nan], index=list("abcd"))
    q = benjamini_hochberg(p)
    assert q.index.equals(p.index)
    assert q.between(0, 1).all()
    assert q.loc["a"] <= q.loc["c"] <= q.loc["b"] <= q.loc["d"]


def test_provider_projections_never_enter_outcome_transform():
    cells = pd.DataFrame(
        {
            "entity_code": ["AAA", "AAA", "AAA", "AAA", "AAA"],
            "feature_id": ["f"] * 5,
            "year": [2022, 2023, 2024, 2026, 2027],
            "value": [100.0, 110.0, 120.0, 130.0, 140.0],
            "source": ["WEO"] * 5,
            "data_role": [
                "historical_observed_or_unverified",
                "historical_observed_or_unverified",
                "historical_observed_or_unverified",
                "provider_projection",
                "provider_projection",
            ],
            "UNIT": ["XDC"] * 5,
        }
    )
    result, summary = transform_historical_values(cells, {"AAA"})
    assert result.year.max() == 2024
    assert summary["provider_projection_rows_read"] == 0
    assert result.year.tolist() == [2024]


def test_no_artificial_feature_cap_in_phase4_contract():
    source = (
        Path(__file__).parents[1] / "src/forecasting/phase4_execution.py"
    ).read_text(encoding="utf-8")
    assert "top-k" not in source.lower()
    assert '"provider_projection_rows_read": 0' in source
    assert '"selected_banking_targets_predeclared": 0' in source
