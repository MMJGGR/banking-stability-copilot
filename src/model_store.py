"""Lightweight access to the trusted serving artifact.

The Streamlit application only needs the serialized data dictionary. Importing
the full training module also imports XGBoost, SHAP, scikit-learn, matplotlib,
and the feature-engineering stack, which materially increases cold-start time.
"""

from pathlib import Path
import json
import pickle

from src.config import CACHE_DIR
from src.lfs_resolver import ensure_lfs_file


MODEL_PATH = Path(CACHE_DIR) / "risk_model.pkl"
MANIFEST_PATH = Path(CACHE_DIR).parent / "artifacts" / "data_manifest.json"
REQUIRED_KEYS = {
    "country_scores",
    "trained",
    "training_date",
    "countries_trained",
}


def load_model_artifact(path=None, verify_checksum: bool = True) -> dict:
    """Load and validate the trusted, dictionary-based serving artifact."""
    artifact_path = Path(path) if path is not None else MODEL_PATH
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model not found: {artifact_path}")
    ensure_lfs_file(artifact_path)

    if path is None and verify_checksum:
        manifest = load_data_manifest()
        expected = (
            manifest.get("artifacts", {})
            .get("cache/risk_model.pkl", {})
            .get("sha256")
        )
        if expected:
            actual = _sha256_file(artifact_path)
            if actual != expected:
                raise ValueError(
                    "Risk model checksum does not match the serving manifest"
                )

    with artifact_path.open("rb") as model_file:
        artifact = pickle.load(model_file)

    if not isinstance(artifact, dict):
        raise TypeError("Risk model artifact must contain a dictionary")

    missing = REQUIRED_KEYS.difference(artifact)
    if missing:
        raise ValueError(
            f"Risk model artifact is missing required keys: {sorted(missing)}"
        )
    if not artifact["trained"]:
        raise ValueError("Risk model artifact is not marked as trained")

    return artifact


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_data_manifest(path=None) -> dict:
    """Load optional serving-manifest metadata."""
    manifest_path = Path(path) if path is not None else MANIFEST_PATH
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, dict):
        raise TypeError("Data manifest must contain a JSON object")
    return manifest
