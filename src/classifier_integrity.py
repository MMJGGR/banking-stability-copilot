"""Fail-closed integrity gate for classifier-preserving data refreshes."""

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_classifier_file(path=None, lock_path=None):
    """Verify trusted release metadata before any pickle deserialization."""
    path = Path(path) if path is not None else ROOT / "cache/crisis_classifier.pkl"
    lock_path = Path(lock_path) if lock_path is not None else ROOT / "artifacts/validated_classifier.json"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Classifier release lock is missing or invalid; refresh blocked") from exc
    expected = lock.get("sha256", "")
    size = lock.get("bytes")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError("Classifier release lock has an invalid SHA-256")
    if type(size) is not int or size <= 0:
        raise RuntimeError("Classifier release lock has an invalid byte count")
    if not path.is_file():
        raise RuntimeError("Validated classifier is missing; fetch its Git LFS object before refresh")
    with path.open("rb") as stream:
        header = stream.read(128)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError("Classifier is a Git LFS pointer, not model bytes; refresh blocked")
    actual_size = path.stat().st_size
    actual_sha = sha256_file(path)
    if actual_size != size or actual_sha != expected:
        raise RuntimeError("Classifier checksum/size differs from the approved baseline; refresh blocked")
    return {"sha256": actual_sha, "bytes": actual_size, "baseline_commit": lock.get("baseline_commit")}


def load_validated_classifier(path=None, lock_path=None, loader=None):
    """Load only verified, fitted classifier bytes; never train as a fallback."""
    path = Path(path) if path is not None else ROOT / "cache/crisis_classifier.pkl"
    before = verify_classifier_file(path, lock_path)
    if loader is None:
        from src.crisis_classifier import CrisisClassifier
        loader = CrisisClassifier.load
    try:
        classifier = loader(str(path))
    except Exception as exc:
        raise RuntimeError("Validated classifier cannot be loaded; explicit model-release review is required, not fallback retraining") from exc
    if not getattr(classifier, "fitted_", False) or not getattr(classifier, "feature_names_", None):
        raise RuntimeError("Validated classifier is not fitted or has no feature schema")
    after = verify_classifier_file(path, lock_path)
    if before != after:
        raise RuntimeError("Classifier changed during loading; refresh blocked")
    return classifier
