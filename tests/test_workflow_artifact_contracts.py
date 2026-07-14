import hashlib
from pathlib import Path

from src.snapshot_manifest import (
    PROMOTED_SOURCE_CACHE_PATHS,
    build_artifact_inventory,
    snapshot_artifact_paths,
)


ROOT = Path(__file__).resolve().parents[1]


GOVERNMENT_REFERENCE_PATHS = (
    "artifacts/government_liquidity_features_report.json",
    "data/reference/government_liquidity_features.parquet",
    "data/reference/government_liquidity_observations.parquet",
    "data/reference/government_liquidity_features_report.json",
)


def test_candidate_and_promotion_workflows_share_government_reference_contract():
    refresh = (ROOT / ".github" / "workflows" / "refresh-data.yml").read_text(
        encoding="utf-8"
    )
    promotion = (ROOT / ".github" / "workflows" / "promote-snapshot.yml").read_text(
        encoding="utf-8"
    )

    for path in GOVERNMENT_REFERENCE_PATHS:
        assert path in refresh
        assert path in promotion

    # Broad cache globs can accidentally package an unhydrated LFS pointer or
    # an unrelated local file.  Both sides enumerate the intended contract.
    assert "cache/*_cache.parquet" not in refresh
    assert "candidate/cache/*.parquet" not in promotion
    assert "candidate/cache/*.pkl" not in promotion


def test_every_promoted_source_cache_is_covered_by_the_snapshot_manifest(tmp_path):
    promotion = (ROOT / ".github" / "workflows" / "promote-snapshot.yml").read_text(
        encoding="utf-8"
    )
    refresh = (ROOT / ".github" / "workflows" / "refresh-data.yml").read_text(
        encoding="utf-8"
    )
    root = tmp_path / "repository"
    cache = root / "cache"
    governed = {
        path.relative_to(root).as_posix()
        for path in snapshot_artifact_paths(root, cache)
    }

    for relative_path in PROMOTED_SOURCE_CACHE_PATHS:
        artifact = root / relative_path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(relative_path.encode("utf-8"))
        assert relative_path in governed
        assert relative_path in refresh
        assert relative_path in promotion
        assert Path(relative_path).name in promotion
    assert '"candidate/cache/$filename"' in promotion

    inventory = build_artifact_inventory(snapshot_artifact_paths(root, cache), root)
    for relative_path in PROMOTED_SOURCE_CACHE_PATHS:
        assert inventory[relative_path] == {
            "bytes": len(relative_path.encode("utf-8")),
            "sha256": hashlib.sha256(relative_path.encode("utf-8")).hexdigest(),
        }


def test_checksummed_government_json_is_lf_pinned_and_written_portably():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    expected = {
        "artifacts/government_liquidity_features_report.json text eol=lf",
        "data/reference/government_liquidity_features_report.json text eol=lf",
    }
    assert expected.issubset(set(attributes))

    source = (ROOT / "src" / "government_liquidity.py").read_text(encoding="utf-8")
    assert 'report_path.open("w", encoding="utf-8", newline="\\n")' in source


def test_government_reference_reconciliation_occurs_after_candidate_training():
    source = (ROOT / "src" / "scripts" / "refresh_data.py").read_text(
        encoding="utf-8"
    )
    training = source.index("results = model.train(")
    reconciliation = source.index("fiscal_observations=government_observations")
    manifest = source.index("manifest = build_snapshot_manifest")
    assert training < reconciliation < manifest


def test_live_check_runs_after_master_on_schedule_and_manually():
    workflow = (ROOT / ".github" / "workflows" / "live-app-check.yml").read_text(
        encoding="utf-8"
    )
    assert "push:" in workflow
    assert "- master" in workflow
    assert "schedule:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "src.scripts.check_live_streamlit" in workflow
    assert "vars.STREAMLIT_APP_URL" in workflow
