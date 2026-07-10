import json
import pickle

import pandas as pd
import pytest

import src.model_store as model_store


def _artifact_dict():
    return {
        "country_scores": pd.DataFrame(
            {"country_code": ["USA"], "risk_score": [3.0]}
        ),
        "trained": True,
        "training_date": "2026-07-01T00:00:00",
        "countries_trained": 1,
    }


def _write_bundle(directory, with_manifest=True, corrupt=False):
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "risk_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(_artifact_dict(), handle)
    if corrupt:
        model_path.write_bytes(model_path.read_bytes()[:-4] + b"XXXX")
    if with_manifest:
        checksum = model_store._sha256_file(model_path)
        (directory / "snapshot_manifest.json").write_text(
            json.dumps(
                {
                    "snapshot_id": directory.name,
                    "snapshot_status": "verified",
                    "artifacts": {"cache/risk_model.pkl": {"sha256": checksum}},
                }
            ),
            encoding="utf-8",
        )


def test_fallback_serves_newest_valid_archived_bundle(tmp_path, monkeypatch):
    archive = tmp_path / "snapshots"
    _write_bundle(archive / "2025-12-31")
    _write_bundle(archive / "2026-06-30")
    monkeypatch.setattr(model_store, "MODEL_PATH", tmp_path / "missing.pkl")
    monkeypatch.setattr(model_store, "SNAPSHOT_ARCHIVE", archive)

    artifact, manifest, status = model_store.load_model_artifact_with_fallback()
    assert artifact["trained"]
    assert status["mode"] == "fallback"
    assert status["fallback_snapshot"] == "2026-06-30"
    assert "missing" in status["active_error"] or "FileNotFoundError" in status["active_error"]
    assert manifest["snapshot_id"] == "2026-06-30"


def test_fallback_skips_checksum_mismatched_bundle(tmp_path, monkeypatch):
    archive = tmp_path / "snapshots"
    _write_bundle(archive / "2025-12-31")
    newest = archive / "2026-06-30"
    _write_bundle(newest)
    # Tamper with the newest bundle after its manifest was written.
    with (newest / "risk_model.pkl").open("wb") as handle:
        pickle.dump(_artifact_dict() | {"training_date": "tampered"}, handle)
    monkeypatch.setattr(model_store, "MODEL_PATH", tmp_path / "missing.pkl")
    monkeypatch.setattr(model_store, "SNAPSHOT_ARCHIVE", archive)

    artifact, manifest, status = model_store.load_model_artifact_with_fallback()
    assert status["fallback_snapshot"] == "2025-12-31"
    assert artifact["training_date"] != "tampered"


def test_all_paths_failing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "MODEL_PATH", tmp_path / "missing.pkl")
    monkeypatch.setattr(model_store, "SNAPSHOT_ARCHIVE", tmp_path / "none")
    with pytest.raises(RuntimeError, match="No serveable model artifact"):
        model_store.load_model_artifact_with_fallback()
