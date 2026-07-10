"""Generate an auditable data and model snapshot manifest.

Contract (remediation plan backlog item 35): this lightweight builder derives
counts, checksums, and freshness from the current cache state. It cannot
reconstruct retrieval provenance (official download URLs, dataflow versions,
retrieval timestamps) or validation results — those are produced by
``refresh_data.py`` / ``build_local_snapshot.py`` at build time. To keep an
official-refresh manifest's provenance intact, this script PRESERVES the
``retrieval``, ``source_mode``, and ``validation`` blocks from an existing
manifest at the output path by default. Pass ``--fresh`` to drop them
deliberately (e.g. when the caches were rebuilt outside a recorded refresh
and the old provenance no longer applies).
"""

import argparse
import json
from pathlib import Path

from src.config import BASE_DIR
from src.snapshot_manifest import (
    build_snapshot_manifest,
    write_snapshot_manifest,
)


PRESERVED_KEYS = ("retrieval", "source_mode", "validation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True, help="Snapshot cutoff YYYY-MM-DD")
    parser.add_argument(
        "--output",
        default=str(Path(BASE_DIR) / "artifacts" / "data_manifest.json"),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Do not carry over retrieval/source_mode/validation metadata "
            "from an existing manifest at the output path."
        ),
    )
    args = parser.parse_args()

    manifest = build_snapshot_manifest(args.as_of)

    output_path = Path(args.output)
    if not args.fresh and output_path.exists():
        try:
            previous = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        preserved = [
            key for key in PRESERVED_KEYS
            if key in previous and key not in manifest
        ]
        for key in preserved:
            manifest[key] = previous[key]
        if preserved:
            print(f"Preserved metadata from previous manifest: {preserved}")

    output = write_snapshot_manifest(manifest, args.output)
    print(f"Wrote snapshot manifest: {output}")
    print(f"Snapshot status: {manifest['snapshot_status']}")


if __name__ == "__main__":
    main()
