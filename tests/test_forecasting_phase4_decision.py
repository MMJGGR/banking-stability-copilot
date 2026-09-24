import pandas as pd

from src.forecasting.phase4_decision import crisis_overlay_decision


def test_crisis_overlay_does_not_advance_when_event_rate_is_better():
    metrics = pd.DataFrame([
        {
            "model": "event_rate",
            "brier": 0.0342,
            "log_loss": 0.1652,
            "pr_auc": 0.0531,
            "roc_auc": 0.6897,
        },
        {
            "model": "state_only",
            "brier": 0.0446,
            "log_loss": 0.2068,
            "pr_auc": 0.0342,
            "roc_auc": 0.4619,
        },
        {
            "model": "state_velocity_uncertainty",
            "brier": 0.0512,
            "log_loss": 0.2414,
            "pr_auc": 0.0644,
            "roc_auc": 0.6366,
        },
    ])
    result = crisis_overlay_decision(metrics)
    assert result["status"] == (
        "do_not_advance_state_crisis_overlay_underperformed_event_rate"
    )
    assert result["selected_overlay"] is None
    assert result["latest_overlay_rows_admissible"] == 0
    assert result["production_classifier_action"] == (
        "preserve_locked_classifier_unchanged"
    )


def test_crisis_overlay_advances_only_when_probability_quality_improves():
    metrics = pd.DataFrame([
        {
            "model": "event_rate",
            "brier": 0.040,
            "log_loss": 0.180,
            "pr_auc": 0.060,
            "roc_auc": 0.60,
        },
        {
            "model": "state_overlay",
            "brier": 0.035,
            "log_loss": 0.170,
            "pr_auc": 0.065,
            "roc_auc": 0.63,
        },
    ])
    result = crisis_overlay_decision(metrics)
    assert result["status"] == "advance_research_overlay_only_not_production"
    assert result["selected_overlay"] == "state_overlay"
