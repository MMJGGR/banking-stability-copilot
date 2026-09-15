"""Apply narrow, context-checked source edits on the isolated review branch.

The review workflow commits these two source-file changes after tests pass.
This script is idempotent and never stages cache or serving artifact changes.
"""
from pathlib import Path


def replace_once(text, old, new):
    if new in text:
        return text
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one unmodified patch context: {old[:90]!r}")
    return text.replace(old, new, 1)


path = Path("train_model.py")
text = path.read_text()
old = '''            try:
                classifier = CrisisClassifier.load()
                metrics = {"cached_classifier": True}
                print("  Loaded cached crisis classifier for snapshot scoring.")
            except Exception as e:
                print(
                    "  Cached crisis classifier unavailable; retraining "
                    f"for this snapshot: {e}"
                )
                classifier, metrics = train_crisis_model(
                    weo_df=weo_df,
                    fsic_df=fsic_df,
                    as_of_date=cutoff,
                )'''
new = '''            from src.classifier_integrity import load_validated_classifier
            classifier = load_validated_classifier()
            metrics = {"cached_classifier": True}
            print("  Loaded checksum-verified baseline classifier; retraining disabled.")'''
text = replace_once(text, old, new)
path.write_text(text)

path = Path("src/scripts/refresh_data.py")
text = path.read_text()
text = replace_once(text, "from src.config import BASE_DIR\n", "from src.config import BASE_DIR, CACHE_DIR\nfrom src.classifier_integrity import load_validated_classifier, verify_classifier_file\nfrom src.snapshot_reuse import restore_source_snapshot\n")
text = replace_once(text, "    args = parser.parse_args()\n", '''    parser.add_argument(
        "--source-snapshot",
        help="Rebuild the identical cutoff from checksum-verified candidate source caches; retain original retrieval dates.",
    )
    args = parser.parse_args()
''')
text = replace_once(text, '''    download_dir = Path(args.download_dir)
    if args.retrieval_mode == "official":''', '''    # Fail before any network retrieval or active-cache mutation.
    classifier_before = None
    if not args.retrain_classifier:
        load_validated_classifier()
        classifier_before = verify_classifier_file()
    source_snapshot = None
    download_dir = Path(args.download_dir)
    if args.source_snapshot:
        if args.retrieval_mode != "official" or args.reuse_downloads:
            parser.error("--source-snapshot cannot be combined with legacy mode or --reuse-downloads")
        source_snapshot = restore_source_snapshot(args.source_snapshot, args.as_of, CACHE_DIR)
        loader = IMFDataLoader()
        if not loader.load_from_cache():
            raise RuntimeError("Verified source caches could not be loaded")
        FSIBSISLoader().load()
        WGILoader().load(force_refresh=False)
        fsic_df = loader._data_cache.get("FSIC")
        weo_df = loader._data_cache.get("WEO")
        mfs_df = loader._data_cache.get("MFS")
        fetched = {}
    elif args.retrieval_mode == "official":''')
text = replace_once(text, "    model.save()\n", '''    if classifier_before is not None and verify_classifier_file() != classifier_before:
        raise RuntimeError("Classifier changed during refresh; candidate publication blocked")
    model.save()
''')
text = replace_once(text, "    output = write_snapshot_manifest(manifest, args.manifest)\n", '''    manifest["classifier_preservation"] = {
        "mode": "explicit_retraining" if args.retrain_classifier else "preserved",
        "verified_baseline": classifier_before,
        "human_promotion_approved": False,
        "pillar_policy": "Pillar pipeline is refitted by the existing build; review the frozen-pipeline comparison separately.",
    }
    if source_snapshot is not None:
        manifest["retrieval"] = source_snapshot["retrieval"]
        manifest["source_mode"] = "verified_candidate_cache_rebuild"
        manifest["source_reuse"] = {
            "original_as_of_date": source_snapshot["as_of_date"],
            "original_generated_at": source_snapshot.get("generated_at"),
            "note": "Source retrieval dates are original, not the rebuild date. Only the five source caches were reused.",
        }
    output = write_snapshot_manifest(manifest, args.manifest)
''')
path.write_text(text)
print("Classifier-preserving source edits applied (or already present).")
