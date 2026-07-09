import pickle

import pandas as pd
import pytest

import src.model_store as model_store
from src.model_store import load_data_manifest, load_model_artifact


def test_load_model_artifact_without_training_stack(tmp_path):
    artifact_path = tmp_path / "risk_model.pkl"
    artifact = {
        "country_scores": pd.DataFrame(
            [{"country_code": "TST", "risk_score": 5.0}]
        ),
        "trained": True,
        "training_date": "2026-01-01T00:00:00",
        "countries_trained": 1,
        "feature_values": None,
        "pca_info": {},
    }
    with artifact_path.open("wb") as model_file:
        pickle.dump(artifact, model_file)

    loaded = load_model_artifact(artifact_path)

    assert loaded["countries_trained"] == 1
    assert loaded["country_scores"].iloc[0]["country_code"] == "TST"


def test_load_model_artifact_rejects_incomplete_dictionary(tmp_path):
    artifact_path = tmp_path / "risk_model.pkl"
    with artifact_path.open("wb") as model_file:
        pickle.dump({"trained": True}, model_file)

    with pytest.raises(ValueError, match="missing required keys"):
        load_model_artifact(artifact_path)


def test_load_data_manifest_is_optional(tmp_path):
    missing = tmp_path / "missing.json"
    assert load_data_manifest(missing) == {}

    present = tmp_path / "manifest.json"
    present.write_text('{"snapshot_id": "2025-12-31"}', encoding="utf-8")
    assert load_data_manifest(present)["snapshot_id"] == "2025-12-31"


def test_default_model_load_rejects_checksum_mismatch(monkeypatch, tmp_path):
    artifact_path = tmp_path / "risk_model.pkl"
    manifest_path = tmp_path / "manifest.json"
    artifact = {
        "country_scores": pd.DataFrame(),
        "trained": True,
        "training_date": "2026-01-01T00:00:00",
        "countries_trained": 0,
    }
    with artifact_path.open("wb") as model_file:
        pickle.dump(artifact, model_file)
    manifest_path.write_text(
        '{"artifacts":{"cache/risk_model.pkl":{"sha256":"wrong"}}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(model_store, "MODEL_PATH", artifact_path)
    monkeypatch.setattr(model_store, "MANIFEST_PATH", manifest_path)

    with pytest.raises(ValueError, match="checksum"):
        load_model_artifact()
