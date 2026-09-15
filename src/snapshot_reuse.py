"""Restore only the five core source caches from a verified candidate.

The original retrieval dates and provenance are retained. No classifier or
fitted pipeline is copied from the candidate being repaired.
"""

import json
from pathlib import Path
import shutil

from src.classifier_integrity import sha256_file

SOURCES = ("WEO", "FSIC", "MFS", "FSIBSIS", "WGI")


def restore_source_snapshot(snapshot_dir, as_of_date, cache_dir):
    root = Path(snapshot_dir).resolve()
    manifest = json.loads((root / "artifacts/data_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("snapshot_status") != "verified":
        raise RuntimeError("Source candidate manifest is not verified")
    if manifest.get("as_of_date") != as_of_date:
        raise RuntimeError("Rebuild cutoff must equal the source candidate cutoff")
    verified = []
    for name in SOURCES:
        relative = f"cache/{name}_cache.parquet"
        source = root / relative
        expected = manifest.get("artifacts", {}).get(relative, {})
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"Missing or unsafe source cache: {relative}")
        if source.stat().st_size != expected.get("bytes") or sha256_file(source) != expected.get("sha256"):
            raise RuntimeError(f"Source cache checksum mismatch: {relative}")
        if name not in manifest.get("retrieval", {}):
            raise RuntimeError(f"Missing original source retrieval provenance: {name}")
        verified.append(source)
    # Validate every source before changing any active cache.
    target = Path(cache_dir)
    target.mkdir(parents=True, exist_ok=True)
    for source in verified:
        destination = target / source.name
        if destination.resolve() == source.resolve():
            raise RuntimeError("Source evidence and rebuild output must be separate")
    for source in verified:
        shutil.copy2(source, target / source.name)
    return manifest
