from pathlib import Path


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
