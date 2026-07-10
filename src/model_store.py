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
SNAPSHOT_ARCHIVE = Path(CACHE_DIR).parent / "artifacts" / "snapshots"
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


def load_model_artifact_with_fallback() -> tuple:
    """Load the active artifact, falling back to the last archived snapshot.

    Returns ``(artifact, manifest, serving_status)``. ``serving_status`` is a
    dictionary with:

    - ``mode``: ``"active"`` when the checksum-verified active artifact
      loaded, ``"fallback"`` when an archived last-known-good bundle is being
      served instead.
    - ``fallback_snapshot``: name of the archived bundle in use (fallback
      mode only).
    - ``active_error``: why the active artifact was rejected (fallback mode
      only).

    Fallback candidates are the bundles under ``artifacts/snapshots/``, newest
    first. Each candidate's ``risk_model.pkl`` is checksum-verified against
    that bundle's own manifest before it is served.
    """
    try:
        artifact = load_model_artifact()
        return artifact, load_data_manifest(), {"mode": "active"}
    except Exception as active_error:  # noqa: BLE001 - degrade, don't die
        failure = f"{type(active_error).__name__}: {active_error}"

    errors = [f"active: {failure}"]
    if SNAPSHOT_ARCHIVE.exists():
        candidates = sorted(
            (
                bundle for bundle in SNAPSHOT_ARCHIVE.iterdir()
                if bundle.is_dir()
                and (bundle / "risk_model.pkl").exists()
                # Challenger bundles are archived for review, not approved
                # for serving; they must never be picked up as a fallback.
                and "challenger" not in bundle.name
            ),
            key=lambda bundle: bundle.name,
            reverse=True,
        )
    else:
        candidates = []

    for bundle in candidates:
        try:
            artifact, manifest = _load_bundle(bundle)
            return artifact, manifest, {
                "mode": "fallback",
                "fallback_snapshot": bundle.name,
                "active_error": failure,
            }
        except Exception as bundle_error:  # noqa: BLE001
            errors.append(
                f"{bundle.name}: {type(bundle_error).__name__}: {bundle_error}"
            )

    raise RuntimeError(
        "No serveable model artifact: " + "; ".join(errors)
    )


def _load_bundle(bundle: Path) -> tuple:
    """Load one archived snapshot bundle, checksum-verified when possible."""
    model_path = bundle / "risk_model.pkl"
    ensure_lfs_file(model_path)
    manifest = {}
    for manifest_name in ("snapshot_manifest.json", "data_manifest.json"):
        if (bundle / manifest_name).exists():
            manifest = load_data_manifest(bundle / manifest_name)
            break
    expected = (
        manifest.get("artifacts", {})
        .get("cache/risk_model.pkl", {})
        .get("sha256")
    )
    if expected and _sha256_file(model_path) != expected:
        raise ValueError(
            "archived risk model checksum does not match the bundle manifest"
        )
    return load_model_artifact(model_path), manifest


def list_archived_snapshots() -> list:
    """Names of archived snapshot bundles, newest first."""
    if not SNAPSHOT_ARCHIVE.exists():
        return []
    return sorted(
        (
            bundle.name for bundle in SNAPSHOT_ARCHIVE.iterdir()
            if bundle.is_dir() and (bundle / "risk_model.pkl").exists()
        ),
        reverse=True,
    )


def load_archived_snapshot(name: str) -> tuple:
    """Load a named archived bundle for read-only inspection.

    Unlike the fallback chain this may load challenger bundles: the user is
    explicitly choosing to inspect them, and the caller is responsible for
    labelling unapproved snapshots.
    """
    bundle = SNAPSHOT_ARCHIVE / name
    if not bundle.is_dir():
        raise FileNotFoundError(f"No archived snapshot named {name!r}")
    return _load_bundle(bundle)


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
