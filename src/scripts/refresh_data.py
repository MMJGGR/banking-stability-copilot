"""Fetch, validate, normalize, train, and package a dated snapshot."""

import argparse
from pathlib import Path

from src.config import BASE_DIR
from src.data_loader import FSIBSISLoader, IMFDataLoader, WGILoader
from src.snapshot_manifest import build_snapshot_manifest, write_snapshot_manifest
from src.sources import build_source_adapters
from train_model import BankingRiskModel, validate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True, help="Snapshot cutoff YYYY-MM-DD")
    parser.add_argument(
        "--download-dir",
        default=str(Path(BASE_DIR) / "data" / "raw"),
    )
    parser.add_argument(
        "--manifest",
        default=str(Path(BASE_DIR) / "artifacts" / "data_manifest.json"),
    )
    args = parser.parse_args()

    download_dir = Path(args.download_dir)
    adapters = build_source_adapters()
    fetched = {
        name: adapter.fetch(download_dir, BASE_DIR)
        for name, adapter in adapters.items()
    }

    loader = IMFDataLoader()
    fsic_df = loader.load_fsic(fetched["FSIC"].path)
    weo_df = loader.load_weo(fetched["WEO"].path)
    mfs_df = loader.load_mfs(fetched["MFS"].path)
    loader.save_cache()

    FSIBSISLoader(fetched["FSIBSIS"].path).load()
    WGILoader(fetched["WGI"].path).load(force_refresh=True)

    model = BankingRiskModel()
    results = model.train(
        fsic_df,
        weo_df,
        mfs_df,
        as_of_date=args.as_of,
    )
    validate_model(results)
    model.save()

    manifest = build_snapshot_manifest(args.as_of)
    manifest["retrieval"] = {
        name: result.to_dict() for name, result in fetched.items()
    }
    output = write_snapshot_manifest(manifest, args.manifest)
    print(f"Published candidate snapshot manifest: {output}")
    print(f"Snapshot status: {manifest['snapshot_status']}")


if __name__ == "__main__":
    main()
