import json
from pathlib import Path

import pandas as pd
import pytest

from src.forecasting.phase5_prospective import score_realizations
from src.forecasting.phase5_shadow import (
    _country_name,
    sha256,
    stable_hash,
    write_csv,
)
from src.forecasting.phase5_shadow_view import load_shadow_bundle


def test_deterministic_helpers(tmp_path):
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
    frame = pd.DataFrame({"b": [2, 3], "a": [1.5, 2.5]})
    first = tmp_path / "a" / "file.csv.gz"
    second = tmp_path / "b" / "file.csv.gz"
    first.parent.mkdir()
    second.parent.mkdir()
    write_csv(frame, first)
    write_csv(frame, second)
    assert sha256(first) == sha256(second)
    assert _country_name("XKX", {}) == (
        "Kosovo",
        "static_iso_or_research_mapping",
    )


def _minimal_realization_inputs(tmp_path, *, provider=False):
    states = [f"state_{index}" for index in range(1, 97)]
    ledger = pd.DataFrame(
        [
            {
                "forecast_batch_id": "batch",
                "forecast_record_id": "record",
                "entity_code": "AAA",
                "state_origin_year": 2026,
                "forecast_year": 2027,
                "horizon_years": 1,
                "movement_radius_q50": 0.1,
                "movement_radius_q80": 0.2,
                "movement_radius_q95": 0.3,
            }
        ]
    )
    quantiles = pd.DataFrame(
        [
            {
                "entity_code": "AAA",
                "forecast_year": 2027,
                "horizon": 1,
                "state_coordinate": state,
                "point": float(index),
            }
            for index, state in enumerate(states)
        ]
    )
    current = pd.DataFrame(
        [
            {
                "entity_code": "AAA",
                "forecast_origin_year": 2026,
                **{state: 0.0 for state in states},
            }
        ]
    )
    realized = pd.DataFrame(
        [
            {
                "entity_code": "AAA",
                "state_year": 2027,
                "data_role": (
                    "provider_projection" if provider else "observed"
                ),
                **{
                    state: float(index)
                    for index, state in enumerate(states)
                },
            }
        ]
    )
    paths = []
    for name, frame in [
        ("ledger.csv", ledger),
        ("quantiles.csv", quantiles),
        ("current.csv", current),
        ("realized.csv", realized),
    ]:
        path = tmp_path / name
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def test_prospective_scoring_is_append_only(tmp_path):
    ledger, quantiles, current, realized = _minimal_realization_inputs(tmp_path)
    output = tmp_path / "scored.csv"
    scored = score_realizations(
        ledger,
        quantiles,
        current,
        realized,
        output,
        realization_vintage="v1",
    )
    assert len(scored) == 1
    assert scored.state_rmse.iloc[0] == 0
    assert bool(scored.covered_50.iloc[0])
    with pytest.raises(ValueError, match="already exists"):
        score_realizations(
            ledger,
            quantiles,
            current,
            realized,
            output,
            realization_vintage="v1",
        )


def test_prospective_scoring_rejects_provider_projection(tmp_path):
    ledger, quantiles, current, realized = _minimal_realization_inputs(
        tmp_path,
        provider=True,
    )
    with pytest.raises(ValueError, match="Provider projections"):
        score_realizations(
            ledger,
            quantiles,
            current,
            realized,
            tmp_path / "scored.csv",
            realization_vintage="provider",
        )


def test_shadow_loader_rejects_research_crisis_probability(tmp_path):
    manifest = {
        "status": "phase5_shadow_bundle_frozen_prospective_confirmation_open",
        "research_crisis_probabilities_served": 0,
        "provider_projection_rows_read_by_baseline": 0,
        "files": {},
    }
    (tmp_path / "shadow-manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame(
        [
            {
                "entity_code": "AAA",
                "horizon": 1,
                "research_crisis_probability": 0.1,
                "research_crisis_overlay_status": "not_approved",
            }
        ]
    ).to_csv(tmp_path / "shadow-country-horizon-summary.csv", index=False)
    with pytest.raises(ValueError, match="must be null"):
        load_shadow_bundle(tmp_path, verify_hashes=False)


def test_viewer_is_separate_and_read_only_contract():
    source = Path("research_shadow_app.py").read_text()
    assert "app.py" not in source
    assert "research_probability" not in source
    assert "not served" in source
    assert "to_csv(" not in source
    assert "requests." not in source
