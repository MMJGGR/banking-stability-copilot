"""Read-only loader for the Phase 5 research shadow bundle."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_shadow_bundle(
    root: str | Path,
    *,
    verify_hashes: bool = True,
) -> dict:
    root = Path(root)
    manifest_path = root / "shadow-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing shadow manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != (
        "phase5_shadow_bundle_frozen_prospective_confirmation_open"
    ):
        raise ValueError("Shadow bundle is not in the accepted frozen state")
    if manifest.get("research_crisis_probabilities_served") != 0:
        raise ValueError("Rejected research crisis probabilities are present")
    if manifest.get("provider_projection_rows_read_by_baseline") != 0:
        raise ValueError("Provider projections entered the model baseline")
    if verify_hashes:
        for name, expected in manifest.get("files", {}).items():
            path = root / name
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Shadow bundle hash mismatch: {name}")
    summary = pd.read_csv(root / "shadow-country-horizon-summary.csv")
    if summary.research_crisis_probability.notna().any():
        raise ValueError("Research crisis probability must be null")
    if not summary.research_crisis_overlay_status.eq("not_approved").all():
        raise ValueError("Unexpected research crisis-overlay status")
    return {"root": root, "manifest": manifest, "summary": summary}


def load_country_record(
    bundle: dict,
    entity_code: str,
    horizon: int,
) -> dict:
    entity_code = str(entity_code).strip().upper()
    horizon = int(horizon)
    if horizon not in (1, 2):
        raise ValueError("Shadow horizon must be one or two years")
    path = bundle["root"] / "shadow-country-records.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            summary = record["summary"]
            if (
                summary["entity_code"] == entity_code
                and int(summary["horizon"]) == horizon
            ):
                if record["crisis_overlay"]["research_probability"] is not None:
                    raise ValueError("Rejected crisis probability in country record")
                return record
    raise KeyError(f"No shadow record for {entity_code}, horizon {horizon}")


def country_options(bundle: dict) -> pd.DataFrame:
    frame = (
        bundle["summary"][["entity_code", "entity_name"]]
        .drop_duplicates()
        .sort_values(["entity_name", "entity_code"])
        .reset_index(drop=True)
    )
    frame["display"] = frame.entity_name + " (" + frame.entity_code + ")"
    return frame
