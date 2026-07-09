"""Generate an auditable data and model snapshot manifest."""

import argparse
from pathlib import Path

from src.config import BASE_DIR
from src.snapshot_manifest import (
    build_snapshot_manifest,
    write_snapshot_manifest,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True, help="Snapshot cutoff YYYY-MM-DD")
    parser.add_argument(
        "--output",
        default=str(Path(BASE_DIR) / "artifacts" / "data_manifest.json"),
    )
    args = parser.parse_args()

    manifest = build_snapshot_manifest(args.as_of)
    output = write_snapshot_manifest(manifest, args.output)
    print(f"Wrote snapshot manifest: {output}")
    print(f"Snapshot status: {manifest['snapshot_status']}")


if __name__ == "__main__":
    main()
