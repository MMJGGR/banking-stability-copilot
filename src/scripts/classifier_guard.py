"""Fail closed when a data-only refresh cannot preserve its selected classifier."""

import argparse
import hashlib
import json
import re
from pathlib import Path


CLASSIFIER_PATH = "cache/crisis_classifier.pkl"
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_classifier(root, manifest_path, load=False):
    """Verify against a manifest captured BEFORE refresh; never learn a new pin."""
    root = Path(root)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    record = manifest.get("artifacts", {}).get(CLASSIFIER_PATH, {})
    expected_hash = record.get("sha256", "")
    expected_size = record.get("bytes")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise RuntimeError("Selected baseline has no valid classifier SHA-256 pin")
    if type(expected_size) is not int or expected_size <= 0:
        raise RuntimeError("Selected baseline has no valid classifier byte-size pin")
    path = root / CLASSIFIER_PATH
    if not path.is_file():
        raise RuntimeError("Selected classifier is missing; refresh must not retrain it")
    with path.open("rb") as stream:
        if stream.read(len(LFS_HEADER)).startswith(LFS_HEADER):
            raise RuntimeError("Classifier is a Git LFS pointer; download the pinned object first")
    actual_hash = sha256_file(path)
    if path.stat().st_size != expected_size or actual_hash != expected_hash:
        raise RuntimeError("Classifier differs from the selected baseline; data-only refresh blocked")
    if load:
        # Deserialize only AFTER authentication against the user-selected manifest.
        from src.crisis_classifier import CrisisClassifier
        classifier = CrisisClassifier.load(str(path))
        names = getattr(classifier, "feature_names_", None)
        if names is None or len(names) == 0:
            raise RuntimeError("Pinned classifier has no trained input-feature schema")
    return {"path": CLASSIFIER_PATH, "sha256": actual_hash,
            "bytes": expected_size, "preserved": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_classifier(args.root, args.manifest, args.load), indent=2))


if __name__ == "__main__":
    main()
