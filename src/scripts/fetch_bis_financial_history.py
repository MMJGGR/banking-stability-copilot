"""Build a local official-BIS financial-history artifact for crisis modelling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.bis_financial import (
    BIS_BULK_SPECS,
    DEFAULT_CACHE_PATH,
    DEFAULT_MANIFEST_PATH,
    bis_coverage_report,
    build_bis_financial_history,
    download_bis_bulk_dataset,
    local_bis_download_record,
    write_bis_financial_history,
)


def _parse_local_inputs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --input {value!r}; expected DATASET=PATH"
            )
        dataset, path = value.split("=", 1)
        dataset = dataset.strip()
        if dataset not in BIS_BULK_SPECS:
            raise ValueError(
                f"Unknown BIS dataset {dataset!r}; choose from "
                f"{sorted(BIS_BULK_SPECS)}"
            )
        result[dataset] = Path(path).expanduser()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download official BIS bulk files and build a local, auditable "
            "crisis-model input artifact. No retrieval occurs in the app."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(BIS_BULK_SPECS),
        help="Dataset to build; repeat as needed (default: all).",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Use a pre-downloaded official BIS ZIP/CSV instead of downloading.",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/bis"))
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()

    datasets = list(dict.fromkeys(args.dataset or BIS_BULK_SPECS.keys()))
    local_inputs = _parse_local_inputs(args.input)
    unused = sorted(set(local_inputs).difference(datasets))
    if unused:
        parser.error(f"--input supplied for unselected datasets: {unused}")

    downloads = []
    inputs = {}
    for dataset in datasets:
        if dataset in local_inputs:
            record = local_bis_download_record(dataset, local_inputs[dataset])
        else:
            record = download_bis_bulk_dataset(
                dataset,
                args.raw_dir,
                timeout=args.timeout,
                attempts=args.attempts,
            )
        downloads.append(record)
        inputs[dataset] = Path(record.path)

    metadata = {download.dataset: download for download in downloads}
    observations = build_bis_financial_history(
        inputs,
        metadata=metadata,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    output_path, manifest_path = write_bis_financial_history(
        observations,
        output_path=args.output,
        manifest_path=args.manifest,
        downloads=downloads,
    )
    print(
        json.dumps(
            {
                "output_path": str(output_path.resolve()),
                "manifest_path": str(manifest_path.resolve()),
                "coverage": bis_coverage_report(observations),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
