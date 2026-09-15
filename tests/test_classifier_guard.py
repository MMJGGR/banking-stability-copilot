import hashlib
import json

import pytest

from src.scripts.classifier_guard import LFS_HEADER, verify_classifier


def fixture_artifact(tmp_path, content=b"approved classifier fixture"):
    path = tmp_path / "cache" / "crisis_classifier.pkl"
    path.parent.mkdir()
    path.write_bytes(content)
    manifest = tmp_path / "baseline.json"
    manifest.write_text(json.dumps({"artifacts": {"cache/crisis_classifier.pkl": {
        "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}}}))
    return path, manifest


def test_selected_bytes_pass(tmp_path):
    _, manifest = fixture_artifact(tmp_path)
    assert verify_classifier(tmp_path, manifest)["preserved"] is True


def test_missing_classifier_fails(tmp_path):
    path, manifest = fixture_artifact(tmp_path)
    path.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        verify_classifier(tmp_path, manifest)


def test_lfs_pointer_fails_even_with_matching_pin(tmp_path):
    _, manifest = fixture_artifact(tmp_path, LFS_HEADER + b"\noid sha256:abc\n")
    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        verify_classifier(tmp_path, manifest)


def test_changed_classifier_fails(tmp_path):
    path, manifest = fixture_artifact(tmp_path)
    path.write_bytes(b"unapproved classifier bytes")
    with pytest.raises(RuntimeError, match="differs"):
        verify_classifier(tmp_path, manifest)


def test_same_size_wrong_hash_fails(tmp_path):
    path, manifest = fixture_artifact(tmp_path)
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(RuntimeError, match="differs"):
        verify_classifier(tmp_path, manifest)


def test_unpinned_manifest_fails(tmp_path):
    _, manifest = fixture_artifact(tmp_path)
    manifest.write_text('{"artifacts": {}}')
    with pytest.raises(RuntimeError, match="SHA-256 pin"):
        verify_classifier(tmp_path, manifest)


def test_false_size_pin_fails(tmp_path):
    _, manifest = fixture_artifact(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["artifacts"]["cache/crisis_classifier.pkl"]["bytes"] = True
    manifest.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="byte-size pin"):
        verify_classifier(tmp_path, manifest)
