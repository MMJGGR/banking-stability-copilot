import hashlib
import json
from types import SimpleNamespace

import pytest

from src.classifier_integrity import load_validated_classifier, verify_classifier_file
from src.snapshot_reuse import restore_source_snapshot, SOURCES


def fixture_files(tmp_path, payload=b"trusted-test-model"):
    artifact = tmp_path / "classifier.pkl"
    artifact.write_bytes(payload)
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}))
    return artifact, lock


def test_verified_bytes(tmp_path):
    artifact, lock = fixture_files(tmp_path)
    assert verify_classifier_file(artifact, lock)["bytes"] == artifact.stat().st_size


@pytest.mark.parametrize("failure", ["missing", "pointer", "checksum", "size", "invalid_lock"])
def test_fail_closed_before_deserialization(tmp_path, failure):
    artifact, lock = fixture_files(tmp_path)
    if failure == "missing":
        artifact.unlink()
    elif failure == "pointer":
        artifact.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    elif failure == "checksum":
        artifact.write_bytes(b"untrustd-test-model")
    elif failure == "size":
        artifact.write_bytes(b"x")
    else:
        lock.write_text("{}")
    calls = []
    with pytest.raises(RuntimeError):
        load_validated_classifier(artifact, lock, lambda path: calls.append(path))
    assert calls == []


def test_valid_fitted_classifier(tmp_path):
    artifact, lock = fixture_files(tmp_path)
    classifier = SimpleNamespace(fitted_=True, feature_names_=["capital_adequacy"])
    assert load_validated_classifier(artifact, lock, lambda path: classifier) is classifier


def test_unloadable_classifier_never_retrains(tmp_path):
    artifact, lock = fixture_files(tmp_path)
    def broken(path):
        raise ValueError("invalid pickle")
    with pytest.raises(RuntimeError, match="not fallback retraining"):
        load_validated_classifier(artifact, lock, broken)


def test_unfitted_classifier_blocked(tmp_path):
    artifact, lock = fixture_files(tmp_path)
    with pytest.raises(RuntimeError, match="not fitted"):
        load_validated_classifier(artifact, lock, lambda path: SimpleNamespace(fitted_=False))


def test_mutation_during_load_blocked(tmp_path):
    artifact, lock = fixture_files(tmp_path)
    def mutating(path):
        artifact.write_bytes(b"changed")
        return SimpleNamespace(fitted_=True, feature_names_=["x"])
    with pytest.raises(RuntimeError):
        load_validated_classifier(artifact, lock, mutating)


def source_fixture(tmp_path):
    root = tmp_path / "source"
    (root / "cache").mkdir(parents=True)
    (root / "artifacts").mkdir()
    manifest = {"snapshot_status": "verified", "as_of_date": "2026-09-12", "artifacts": {}, "retrieval": {}}
    for name in SOURCES:
        relative = f"cache/{name}_cache.parquet"
        payload = name.encode()
        (root / relative).write_bytes(payload)
        manifest["artifacts"][relative] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        manifest["retrieval"][name] = {"retrieved_at": "2026-09-12"}
    (root / "artifacts/data_manifest.json").write_text(json.dumps(manifest))
    (root / "cache/crisis_classifier.pkl").write_bytes(b"must not copy")
    return root


def test_source_reuse_copies_only_verified_sources(tmp_path):
    root = source_fixture(tmp_path)
    target = tmp_path / "output"
    restore_source_snapshot(root, "2026-09-12", target)
    assert len(list(target.iterdir())) == 5
    assert not (target / "crisis_classifier.pkl").exists()


def test_source_reuse_validates_all_before_copying(tmp_path):
    root = source_fixture(tmp_path)
    (root / "cache/WGI_cache.parquet").write_bytes(b"corrupt")
    target = tmp_path / "output"
    with pytest.raises(RuntimeError, match="checksum"):
        restore_source_snapshot(root, "2026-09-12", target)
    assert not target.exists()


def test_source_reuse_cannot_advance_vintage(tmp_path):
    root = source_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="cutoff"):
        restore_source_snapshot(root, "2026-09-15", tmp_path / "output")
