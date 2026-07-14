import hashlib
import json
from pathlib import Path

from src.model_evidence import (
    active_model_feature_codes,
    crisis_validation_display_state,
    liquidity_feature_role,
)
from src.model_store import load_model_artifact


def test_active_feature_roles_come_from_served_loading_maps():
    pca_info = {
        "economic_loadings": {
            "govt_interest_to_revenue": 0.2,
            "govt_revenue_gdp": -0.1,
        },
        "industry_loadings": {"npl_ratio": 0.4},
    }
    active = active_model_feature_codes(pca_info)
    stale_report = {
        "active_features": ["govt_interest_to_revenue"],
        "candidate_features": ["govt_revenue_gdp", "portfolio_liabilities_gdp"],
    }

    assert liquidity_feature_role("govt_revenue_gdp", stale_report, active) == (
        "Active score input"
    )
    assert liquidity_feature_role(
        "portfolio_liabilities_gdp", stale_report, active
    ) == "Candidate / monitoring"


def test_current_government_loading_roles_override_stale_candidate_report():
    artifact = load_model_artifact()
    active = active_model_feature_codes(artifact["pca_info"])
    report = json.loads(
        Path("artifacts/liquidity_candidate_score_movement.json").read_text(
            encoding="utf-8"
        )
    )
    now_active = {
        "govt_revenue_gdp",
        "govt_interest_to_revenue_change_3y",
        "govt_debt_to_revenue_change_3y",
        "govt_revenue_gdp_change_3y",
    }

    assert now_active <= set(report["candidate_features"])
    assert now_active <= active
    assert all(
        liquidity_feature_role(feature, report, active) == "Active score input"
        for feature in now_active
    )


def test_validation_display_gate_fails_closed_for_legacy_reports():
    legacy = {
        "validation_status": "invalid_superseded",
        "clean_validation": False,
        "display_metrics": False,
        "roc_auc": 0.99,
    }
    assert crisis_validation_display_state(legacy)["display_metrics"] is False
    assert crisis_validation_display_state({"roc_auc": 0.99})["display_metrics"] is False


def test_validation_display_gate_requires_all_clean_flags():
    clean = {
        "validation_status": "validated_clean",
        "clean_validation": True,
        "display_metrics": True,
    }
    assert crisis_validation_display_state(clean)["display_metrics"] is True

    for key in ("clean_validation", "display_metrics"):
        incomplete = dict(clean)
        incomplete[key] = False
        assert crisis_validation_display_state(incomplete)["display_metrics"] is False


def test_packaged_legacy_summary_is_explicitly_withheld():
    summary = json.loads(
        Path("artifacts/crisis_validation_summary.json").read_text(encoding="utf-8")
    )
    state = crisis_validation_display_state(summary)

    assert summary["validation_status"] == "invalid_superseded"
    assert state["display_metrics"] is False


def test_policy_audit_uses_exact_wp_26_94_episode_counts():
    audit_path = Path("artifacts/model_policy_audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert audit["crisis_labels"] == {
        "borderline_episodes_excluded": 3,
        "source_coverage_end_year": 2025,
        "source_version": "IMF Working Paper WP/26/94 (May 2026)",
        "systemic_countries": 120,
        "systemic_episodes": 161,
    }

    manifest = json.loads(
        Path("artifacts/data_manifest.json").read_text(encoding="utf-8")
    )
    recorded = manifest["artifacts"]["artifacts/model_policy_audit.json"]
    assert recorded["bytes"] == audit_path.stat().st_size
    assert recorded["sha256"] == hashlib.sha256(audit_path.read_bytes()).hexdigest()


def test_mobile_layout_clears_fixed_streamlit_toolbar():
    styles = Path("src/dashboard/styles.py").read_text(encoding="utf-8")

    assert "@media (max-width: 640px)" in styles
    assert "padding-top: max(" in styles
    assert "4.5rem" in styles
    assert "env(safe-area-inset-top)" in styles


def test_methodology_confusion_matrix_is_guarded_by_clean_display_state():
    app_source = Path("app.py").read_text(encoding="utf-8")

    assert (
        'if validation_state["display_metrics"] and '
        "CRISIS_CONFUSION_MATRIX_PATH.exists():"
    ) in app_source
