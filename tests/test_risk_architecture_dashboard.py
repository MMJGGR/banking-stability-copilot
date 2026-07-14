import pytest

from src.dashboard.risk_architecture import (
    AlertStatus,
    CountryRiskView,
    MechanismSignal,
    ModelStatus,
    ValidationMetric,
    ValidationSummary,
    build_country_risk_html,
    build_validation_html,
    mechanism_signals_from_records,
)


def _view(**overrides):
    values = {
        "country_name": "Kenya",
        "operating_environment_score": 6.4,
        "systemic_hazard_1y": 0.082,
        "systemic_hazard_2_3y": 0.174,
        "alert_status": AlertStatus.AMBER,
        "evidence_confidence": 0.78,
        "dominant_mechanism": "Sovereign liquidity",
        "mechanisms": (
            MechanismSignal("Bank solvency", 0.34, "Stable", 0.81),
            MechanismSignal("Sovereign liquidity", 0.72, "Deteriorating", 0.76),
        ),
        "model_status": ModelStatus.RESEARCH,
        "as_of_date": "2026-06-30",
        "evidence_basis": "Modern full expert; 78% observed coverage",
        "overall_risk_score": 6.8,
        "banking_system_score": 7.1,
        "risk_percentile": 0.74,
        "data_coverage": 0.82,
        "risk_tier": "High",
        "hazard_input_coverage": 0.64,
        "mechanism_evidence_coverage": 0.71,
    }
    values.update(overrides)
    return CountryRiskView(**values)


def test_country_view_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="systemic_hazard_1y"):
        _view(systemic_hazard_1y=1.01)

    with pytest.raises(ValueError, match="operating_environment_score"):
        _view(operating_environment_score=-0.1)

    with pytest.raises(ValueError, match="data_coverage"):
        _view(data_coverage=1.01)


def test_research_view_is_explicitly_not_production_scoring():
    markup = build_country_risk_html(_view(model_status="challenger"))

    assert '<span class="be-model-status research">Research</span>' in markup
    assert markup.count("Research estimate; not used in production scoring") == 2
    assert "Hazard input coverage" in markup
    assert "Mechanism evidence coverage" in markup
    assert "Dominant mechanism" in markup
    assert "Active assessment" in markup
    assert "Overall risk score" in markup
    assert "Banking system" in markup
    assert "74th" in markup

    research_markup = build_country_risk_html(
        _view(model_status="research_challenger")
    )
    assert '<span class="be-model-status research">Research</span>' in research_markup


def test_production_view_uses_production_status_and_hazard_language():
    markup = build_country_risk_html(_view(model_status="live"))

    assert '<span class="be-model-status production">Production</span>' in markup
    assert markup.count("Production onset probability") == 2
    assert "not used in production scoring" not in markup


def test_layers_can_be_presented_separately_for_failed_gate_research():
    structural = build_country_risk_html(_view(), show_early_warning=False)
    warning = build_country_risk_html(_view(), show_structural=False)

    assert "Active assessment" in structural
    assert "Systemic early warning" not in structural
    assert "Systemic early warning" in warning
    assert "Overall risk score" not in warning

    with pytest.raises(ValueError, match="at least one"):
        build_country_risk_html(
            _view(), show_structural=False, show_early_warning=False
        )


def test_failed_gate_can_render_not_issued_without_country_probabilities():
    markup = build_country_risk_html(
        _view(
            systemic_hazard_1y=None,
            systemic_hazard_2_3y=None,
            alert_status="not issued",
            alert_reason="Forward validation failed.",
        ),
        show_structural=False,
    )

    assert "Not issued" in markup
    assert markup.count("Not reportable") == 2
    assert "Forward validation failed." in markup
    assert "1-year crisis-onset probability" in markup
    assert "Crisis-onset probability in years 2–3" in markup


def test_mechanism_table_marks_only_explicit_dominant_signal():
    markup = build_country_risk_html(_view())

    assert markup.count('class="be-mechanism-row dominant"') == 1
    assert markup.count("Dominant</span>") == 1
    assert "72%" in markup
    assert "76%" in markup


def test_all_user_supplied_text_is_escaped():
    markup = build_country_risk_html(
        _view(
            country_name='<script>alert("country")</script>',
            dominant_mechanism="Funding <stress>",
            mechanisms=(
                MechanismSignal(
                    '<img src=x onerror="alert(1)">',
                    0.4,
                    '<b>Worse</b>',
                    0.5,
                    'quote" onmouseover="bad',
                ),
            ),
        ),
        title="Risk <view>",
    )

    assert "<script>" not in markup
    assert "<img src=x" not in markup
    assert "<b>Worse</b>" not in markup
    assert "Risk &lt;view&gt;" in markup
    assert "Funding &lt;stress&gt;" in markup
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in markup


def test_missing_values_are_rendered_without_false_precision():
    markup = build_country_risk_html(
        _view(
            operating_environment_score=None,
            systemic_hazard_1y=None,
            systemic_hazard_2_3y=None,
            evidence_confidence=None,
            dominant_mechanism=None,
            mechanisms=(),
            alert_status="insufficient evidence",
        )
    )

    assert markup.count("Not available") >= 4
    assert "Insufficient evidence" in markup
    assert "No mechanism outputs available" in markup


def test_validation_summary_surfaces_design_metrics_and_limits():
    summary = ValidationSummary(
        model_name="Discrete-time hazard challenger",
        validation_design="Rolling forward, leave-wave-out",
        evaluation_period="2008–2018",
        sample_description="929 country-years; 18 positives",
        operating_threshold="Frozen before test window",
        status="Research gate",
        metrics=(
            ValidationMetric("ROC AUC", "0.72", "95% CI 0.64–0.79"),
            ValidationMetric("Precision", "0.31"),
            ValidationMetric("Recall", "0.61"),
            ValidationMetric("False alerts", "2.8 / 100 country-years"),
        ),
        limitations=("Rare events; confidence intervals remain wide.",),
    )

    markup = build_validation_html(summary)

    assert "Rolling forward, leave-wave-out" in markup
    assert "Frozen before test window" in markup
    assert "False alerts" in markup
    assert "How the model family is organized" in markup
    assert "Mechanism models" in markup
    assert "Rare events; confidence intervals remain wide." in markup


def test_record_adapter_preserves_typed_contract():
    signals = mechanism_signals_from_records(
        [
            {
                "name": "External and FX",
                "signal_strength": 0.64,
                "direction": "Deteriorating",
                "evidence_confidence": 0.88,
            }
        ]
    )

    assert signals == (
        MechanismSignal("External and FX", 0.64, "Deteriorating", 0.88),
    )
