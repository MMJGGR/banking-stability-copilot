"""Presentation components for BankEnv's hierarchical risk architecture.

This module deliberately contains no scoring or alert-policy logic.  The model
layer owns hazards, mechanism signals, confidence and the governed alert tier;
the dashboard only validates and presents those outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import html
import math
from typing import Iterable, Mapping, Sequence

import streamlit as st


class AlertStatus(str, Enum):
    """Governed analyst-review status supplied by the model policy."""

    RED = "red"
    AMBER = "amber"
    CLEAR = "clear"
    INSUFFICIENT = "insufficient"
    NOT_ISSUED = "not_issued"


class ModelStatus(str, Enum):
    """Whether the displayed hierarchy is deployed or an analytical challenger."""

    PRODUCTION = "production"
    RESEARCH = "research"


_ALERT_PRESENTATION = {
    AlertStatus.RED: ("Red alert", "red"),
    AlertStatus.AMBER: ("Amber watch", "amber"),
    AlertStatus.CLEAR: ("No active alert", "clear"),
    AlertStatus.INSUFFICIENT: ("Insufficient evidence", "insufficient"),
    AlertStatus.NOT_ISSUED: ("Not issued", "insufficient"),
}


@dataclass(frozen=True)
class MechanismSignal:
    """One independently modelled pathway into systemic stress.

    ``signal_strength`` is a normalized risk signal from 0 to 1. It is not
    described as a probability because mechanism models may use calibrated
    probabilities, percentiles or governed composite signals.
    """

    name: str
    signal_strength: float | None
    direction: str | None = None
    evidence_confidence: float | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("mechanism name must not be blank")
        _validate_unit_interval(self.signal_strength, "signal_strength")
        _validate_unit_interval(self.evidence_confidence, "evidence_confidence")


@dataclass(frozen=True)
class CountryRiskView:
    """Country-level outputs required by the hierarchical risk surface."""

    country_name: str
    operating_environment_score: float | None
    systemic_hazard_1y: float | None
    systemic_hazard_2_3y: float | None
    alert_status: AlertStatus | str
    evidence_confidence: float | None
    dominant_mechanism: str | None
    mechanisms: Sequence[MechanismSignal] = field(default_factory=tuple)
    model_status: ModelStatus | str = ModelStatus.RESEARCH
    as_of_date: str | None = None
    operating_environment_label: str | None = None
    alert_reason: str | None = None
    evidence_basis: str | None = None
    overall_risk_score: float | None = None
    banking_system_score: float | None = None
    risk_percentile: float | None = None
    data_coverage: float | None = None
    risk_tier: str | None = None
    hazard_input_coverage: float | None = None
    mechanism_evidence_coverage: float | None = None

    def __post_init__(self) -> None:
        if not str(self.country_name).strip():
            raise ValueError("country_name must not be blank")
        if self.operating_environment_score is not None:
            score = _as_finite_float(
                self.operating_environment_score, "operating_environment_score"
            )
            if not 0 <= score <= 10:
                raise ValueError("operating_environment_score must be between 0 and 10")
        for field_name in ("overall_risk_score", "banking_system_score"):
            value = getattr(self, field_name)
            if value is None:
                continue
            score = _as_finite_float(value, field_name)
            if not 0 <= score <= 10:
                raise ValueError(f"{field_name} must be between 0 and 10")
        _validate_unit_interval(self.systemic_hazard_1y, "systemic_hazard_1y")
        _validate_unit_interval(self.systemic_hazard_2_3y, "systemic_hazard_2_3y")
        _validate_unit_interval(self.evidence_confidence, "evidence_confidence")
        _validate_unit_interval(self.risk_percentile, "risk_percentile")
        _validate_unit_interval(self.data_coverage, "data_coverage")
        _validate_unit_interval(self.hazard_input_coverage, "hazard_input_coverage")
        _validate_unit_interval(
            self.mechanism_evidence_coverage, "mechanism_evidence_coverage"
        )
        object.__setattr__(self, "alert_status", _coerce_alert_status(self.alert_status))
        object.__setattr__(self, "model_status", _coerce_model_status(self.model_status))
        object.__setattr__(self, "mechanisms", tuple(self.mechanisms))
        if not all(isinstance(item, MechanismSignal) for item in self.mechanisms):
            raise TypeError("mechanisms must contain MechanismSignal instances")


@dataclass(frozen=True)
class ValidationMetric:
    """One compact metric shown in the methodology validation block."""

    label: str
    value: str
    note: str | None = None

    def __post_init__(self) -> None:
        if not str(self.label).strip() or not str(self.value).strip():
            raise ValueError("validation metric label and value must not be blank")


@dataclass(frozen=True)
class ValidationSummary:
    """Governed validation evidence for one deployed model/version."""

    model_name: str
    validation_design: str
    evaluation_period: str
    metrics: Sequence[ValidationMetric]
    sample_description: str | None = None
    operating_threshold: str | None = None
    status: str | None = None
    limitations: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for field_name in ("model_name", "validation_design", "evaluation_period"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be blank")
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        if not self.metrics:
            raise ValueError("validation summary must include at least one metric")
        if not all(isinstance(item, ValidationMetric) for item in self.metrics):
            raise TypeError("metrics must contain ValidationMetric instances")


@dataclass(frozen=True)
class ArchitectureLayer:
    """Concise methodology description for a model layer."""

    name: str
    purpose: str
    output: str


DEFAULT_ARCHITECTURE_LAYERS = (
    ArchitectureLayer(
        "Operating environment",
        "Describes structural banking-system vulnerability and resilience.",
        "Score, not a crisis probability",
    ),
    ArchitectureLayer(
        "Mechanism models",
        "Organize distinct banking, funding, credit, sovereign, external and shock evidence.",
        "Comparable evidence signals, not probabilities",
    ),
    ArchitectureLayer(
        "Systemic hazard",
        "Combines pre-event evidence into separate near- and medium-term onset horizons.",
        "1-year and 2–3-year probabilities",
    ),
    ArchitectureLayer(
        "Review policy",
        "Maps frozen model outputs to recall-oriented Amber and precision-oriented Red tiers.",
        "Analyst review status",
    ),
    ArchitectureLayer(
        "Conditional severity",
        "Would estimate loss or system damage after onset from a separately observed target.",
        "Not active until a governed severity label exists",
    ),
    ArchitectureLayer(
        "Evidence coverage",
        "Reports coverage, vintage and staleness independently of economic risk.",
        "Coverage, never hidden imputation",
    ),
)


RISK_SURFACE_STYLES = """
<style>
  .be-risk-surface {
    --be-border: rgba(128, 128, 128, 0.28);
    --be-muted: color-mix(in srgb, var(--text-color) 66%, transparent);
    --be-subtle: color-mix(in srgb, var(--text-color) 8%, transparent);
    color: var(--text-color);
    width: 100%;
    container-type: inline-size;
  }
  .be-risk-surface * { box-sizing: border-box; }
  .be-risk-heading {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: .75rem;
    margin: 0 0 .65rem;
  }
  .be-risk-title { font-size: 1.05rem; font-weight: 650; letter-spacing: -.01em; }
  .be-risk-date { color: var(--be-muted); font-size: .76rem; white-space: nowrap; }
  .be-risk-meta { display: flex; align-items: center; justify-content: flex-end; gap: .45rem; }
  .be-model-status {
    --be-status-color: var(--be-border);
    display: inline-flex;
    align-items: center;
    border: 1px solid var(--be-status-color);
    border-radius: 999px;
    padding: .19rem .43rem;
    font-size: .65rem;
    font-weight: 650;
    letter-spacing: .035em;
    line-height: 1;
    text-transform: uppercase;
    color: var(--text-color);
  }
  .be-model-status.production { --be-status-color: #238636; background: rgba(35, 134, 54, .10); }
  .be-model-status.research { --be-status-color: #d29922; background: rgba(210, 153, 34, .12); }
  .be-risk-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: .55rem;
  }
  .be-layer-heading {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: .75rem;
    margin: .78rem 0 .38rem;
  }
  .be-layer-heading:first-of-type { margin-top: 0; }
  .be-layer-title { font-size: .88rem; font-weight: 650; }
  .be-layer-note { color: var(--be-muted); font-size: .7rem; }
  .be-structural-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: .55rem;
  }
  .be-risk-card {
    min-width: 0;
    min-height: 92px;
    padding: .78rem .82rem;
    border: 1px solid var(--be-border);
    border-radius: 7px;
    background: var(--secondary-background-color);
  }
  .be-risk-label {
    color: var(--be-muted);
    font-size: .69rem;
    font-weight: 600;
    letter-spacing: .045em;
    line-height: 1.25;
    text-transform: uppercase;
  }
  .be-risk-value {
    margin-top: .26rem;
    font-size: 1.32rem;
    font-variant-numeric: tabular-nums;
    font-weight: 680;
    line-height: 1.1;
  }
  .be-risk-detail {
    color: var(--be-muted);
    font-size: .72rem;
    line-height: 1.3;
    margin-top: .28rem;
    overflow-wrap: anywhere;
  }
  .be-alert-pill {
    --be-status-color: var(--be-border);
    display: inline-flex;
    align-items: center;
    margin-top: .37rem;
    padding: .26rem .48rem;
    border: 1px solid var(--be-status-color);
    border-radius: 999px;
    font-size: .77rem;
    font-weight: 650;
    line-height: 1;
    white-space: normal;
    color: var(--text-color);
  }
  .be-alert-pill.red { --be-status-color: #cf3f4f; background: rgba(207, 63, 79, .10); }
  .be-alert-pill.amber { --be-status-color: #d29922; background: rgba(210, 153, 34, .12); }
  .be-alert-pill.clear { --be-status-color: #238636; background: rgba(35, 134, 54, .10); }
  .be-alert-pill.insufficient { color: var(--be-muted); background: var(--be-subtle); }
  .be-risk-context {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 2fr);
    gap: .55rem;
    margin-top: .55rem;
  }
  .be-context-block {
    min-width: 0;
    padding: .65rem .82rem;
    border-left: 3px solid var(--primary-color);
    background: var(--be-subtle);
    border-radius: 0 6px 6px 0;
  }
  .be-context-value { font-size: .88rem; font-weight: 620; margin-top: .16rem; }
  .be-mechanisms { margin-top: .8rem; }
  .be-section-heading {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: .75rem;
    margin-bottom: .35rem;
  }
  .be-section-title { font-size: .9rem; font-weight: 650; }
  .be-section-note { color: var(--be-muted); font-size: .72rem; }
  .be-mechanism-head, .be-mechanism-row {
    display: grid;
    grid-template-columns: minmax(150px, 1.45fr) minmax(160px, 1.2fr) minmax(90px, .75fr) minmax(82px, .65fr);
    gap: .65rem;
    align-items: center;
  }
  .be-mechanism-head {
    color: var(--be-muted);
    padding: .36rem .55rem;
    border-bottom: 1px solid var(--be-border);
    font-size: .67rem;
    font-weight: 600;
    letter-spacing: .035em;
    text-transform: uppercase;
  }
  .be-mechanism-row {
    padding: .46rem .55rem;
    border-bottom: 1px solid var(--be-border);
    font-size: .77rem;
  }
  .be-mechanism-row:last-child { border-bottom: 0; }
  .be-mechanism-row.dominant { background: var(--be-subtle); }
  .be-mechanism-name { min-width: 0; font-weight: 560; }
  .be-mechanism-tag {
    color: var(--primary-color);
    display: block;
    font-size: .64rem;
    font-weight: 600;
    margin-top: .08rem;
    text-transform: uppercase;
  }
  .be-signal-wrap { display: grid; grid-template-columns: minmax(60px, 1fr) 38px; gap: .45rem; align-items: center; }
  .be-signal-track { height: 7px; overflow: hidden; border-radius: 999px; background: rgba(128, 128, 128, .22); }
  .be-signal-fill { height: 100%; border-radius: inherit; background: var(--primary-color); }
  .be-signal-number { font-variant-numeric: tabular-nums; text-align: right; }
  .be-mechanism-muted { color: var(--be-muted); }
  .be-validation-meta {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: .55rem;
    margin-bottom: .65rem;
  }
  .be-validation-metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: .5rem;
    margin-bottom: .7rem;
  }
  .be-validation-metric {
    min-width: 0;
    padding: .58rem .62rem;
    border: 1px solid var(--be-border);
    border-radius: 6px;
    background: var(--secondary-background-color);
  }
  .be-validation-value { font-size: 1.08rem; font-weight: 650; margin-top: .15rem; }
  .be-architecture-table { border-top: 1px solid var(--be-border); margin-top: .35rem; }
  .be-architecture-row {
    display: grid;
    grid-template-columns: minmax(125px, .7fr) minmax(230px, 1.7fr) minmax(150px, 1fr);
    gap: .65rem;
    padding: .46rem 0;
    border-bottom: 1px solid var(--be-border);
    font-size: .76rem;
    line-height: 1.35;
  }
  .be-architecture-name { font-weight: 620; }
  .be-limitations { color: var(--be-muted); font-size: .75rem; line-height: 1.42; margin-top: .55rem; }
  .be-limitations ul { margin: .25rem 0 0 1.05rem; padding: 0; }
  @container (max-width: 850px) {
    .be-risk-grid, .be-structural-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .be-risk-card:last-child { grid-column: span 2; }
    .be-validation-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }
  @container (max-width: 590px) {
    .be-risk-context, .be-validation-meta { grid-template-columns: 1fr; }
    .be-mechanism-head { display: none; }
    .be-mechanism-row { grid-template-columns: minmax(0, 1fr) minmax(115px, .85fr); gap: .32rem .65rem; }
    .be-mechanism-row > :nth-child(3), .be-mechanism-row > :nth-child(4) { color: var(--be-muted); font-size: .7rem; }
    .be-architecture-row { grid-template-columns: 1fr; gap: .18rem; }
  }
  @media (max-width: 640px) {
    .be-risk-heading { align-items: flex-start; flex-direction: column; gap: .15rem; }
    .be-risk-grid, .be-structural-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .be-risk-card { min-height: 84px; padding: .68rem; }
    .be-risk-card:last-child { grid-column: span 2; }
    .be-risk-value { font-size: 1.17rem; }
  }
  @media (max-width: 420px) {
    .be-risk-grid, .be-structural-grid { grid-template-columns: 1fr; }
    .be-risk-card:last-child { grid-column: auto; }
    .be-validation-metrics { grid-template-columns: 1fr; }
  }
</style>
"""


def _as_finite_float(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric or None") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_unit_interval(value: float | None, name: str) -> None:
    if value is None:
        return
    number = _as_finite_float(value, name)
    if not 0 <= number <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _coerce_alert_status(value: AlertStatus | str) -> AlertStatus:
    if isinstance(value, AlertStatus):
        return value
    normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "red": AlertStatus.RED,
        "red alert": AlertStatus.RED,
        "amber": AlertStatus.AMBER,
        "amber watch": AlertStatus.AMBER,
        "clear": AlertStatus.CLEAR,
        "none": AlertStatus.CLEAR,
        "no alert": AlertStatus.CLEAR,
        "no active alert": AlertStatus.CLEAR,
        "insufficient": AlertStatus.INSUFFICIENT,
        "insufficient evidence": AlertStatus.INSUFFICIENT,
        "unavailable": AlertStatus.INSUFFICIENT,
        "not issued": AlertStatus.NOT_ISSUED,
        "suppressed": AlertStatus.NOT_ISSUED,
    }
    if normalized not in aliases:
        supported = ", ".join(item.value for item in AlertStatus)
        raise ValueError(f"unknown alert status {value!r}; expected one of {supported}")
    return aliases[normalized]


def _coerce_model_status(value: ModelStatus | str) -> ModelStatus:
    if isinstance(value, ModelStatus):
        return value
    normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "production": ModelStatus.PRODUCTION,
        "live": ModelStatus.PRODUCTION,
        "deployed": ModelStatus.PRODUCTION,
        "research": ModelStatus.RESEARCH,
        "challenger": ModelStatus.RESEARCH,
        "research challenger": ModelStatus.RESEARCH,
        "research only": ModelStatus.RESEARCH,
        "research gate failed": ModelStatus.RESEARCH,
        "staged": ModelStatus.RESEARCH,
    }
    if normalized not in aliases:
        raise ValueError("model_status must be production/live or research/challenger")
    return aliases[normalized]


def _safe(value: object | None, fallback: str = "Not available") -> str:
    text = fallback if value is None or not str(value).strip() else str(value).strip()
    return html.escape(text, quote=True)


def _format_score(value: float | None) -> str:
    return "Not available" if value is None else f"{float(value):.1f}/10"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "Not available"
    percentage = float(value) * 100
    decimals = 1 if 0 < percentage < 10 else 0
    return f"{percentage:.{decimals}f}%"


def _format_percentile(value: float | None) -> str:
    if value is None:
        return "Not available"
    percentile = int(round(float(value) * 100))
    if 10 <= percentile % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(percentile % 10, "th")
    return f"{percentile}{suffix}"


def _confidence_label(value: float | None) -> str:
    if value is None:
        return "Coverage not assessed"
    if value >= 0.8:
        return "High evidence coverage"
    if value >= 0.6:
        return "Moderate evidence coverage"
    return "Limited evidence coverage"


def _render_metric_card(label: str, value: str, detail: str | None = None) -> str:
    detail_html = f'<div class="be-risk-detail">{_safe(detail)}</div>' if detail else ""
    return (
        '<div class="be-risk-card">'
        f'<div class="be-risk-label">{_safe(label)}</div>'
        f'<div class="be-risk-value">{_safe(value)}</div>'
        f"{detail_html}</div>"
    )


def _render_alert_card(view: CountryRiskView) -> str:
    label, css_class = _ALERT_PRESENTATION[view.alert_status]
    detail_html = (
        f'<div class="be-risk-detail">{_safe(view.alert_reason)}</div>'
        if view.alert_reason
        else ""
    )
    return (
        '<div class="be-risk-card">'
        '<div class="be-risk-label">Review status</div>'
        f'<div class="be-alert-pill {css_class}">{_safe(label)}</div>'
        f"{detail_html}</div>"
    )


def _render_mechanism_rows(view: CountryRiskView) -> str:
    if not view.mechanisms:
        return (
            '<div class="be-mechanism-row">'
            '<div class="be-mechanism-muted">No mechanism outputs available</div>'
            '<div class="be-mechanism-muted">Not available</div>'
            '<div></div><div></div></div>'
        )

    dominant = (view.dominant_mechanism or "").strip().casefold()
    rows: list[str] = []
    for mechanism in view.mechanisms:
        is_dominant = mechanism.name.strip().casefold() == dominant
        tag = '<span class="be-mechanism-tag">Dominant</span>' if is_dominant else ""
        if mechanism.signal_strength is None:
            signal = '<span class="be-mechanism-muted">Not available</span>'
        else:
            width = max(0.0, min(float(mechanism.signal_strength) * 100, 100.0))
            signal = (
                '<div class="be-signal-wrap">'
                '<div class="be-signal-track" aria-hidden="true">'
                f'<div class="be-signal-fill" style="width:{width:.1f}%"></div></div>'
                f'<div class="be-signal-number">{_format_percent(mechanism.signal_strength)}</div>'
                "</div>"
            )
        direction = _safe(mechanism.direction, "Not assessed")
        confidence = _format_percent(mechanism.evidence_confidence)
        note = f' title="{_safe(mechanism.note)}"' if mechanism.note else ""
        rows.append(
            f'<div class="be-mechanism-row{" dominant" if is_dominant else ""}"{note}>'
            f'<div class="be-mechanism-name">{_safe(mechanism.name)}{tag}</div>'
            f"<div>{signal}</div><div>{direction}</div>"
            f'<div class="be-mechanism-muted">{_safe(confidence)}</div></div>'
        )
    return "".join(rows)


def build_country_risk_html(
    view: CountryRiskView,
    *,
    title: str | None = "Country risk assessment",
    show_structural: bool = True,
    show_early_warning: bool = True,
) -> str:
    """Build one surface while keeping structural and warning layers distinct."""

    if not show_structural and not show_early_warning:
        raise ValueError("at least one country risk layer must be shown")

    title_html = ""
    status_text = "Production" if view.model_status is ModelStatus.PRODUCTION else "Research"
    status_class = view.model_status.value
    date = f"As of {_safe(view.as_of_date)}" if view.as_of_date else ""
    if title:
        title_html = (
            '<div class="be-risk-heading">'
            f'<div class="be-risk-title">{_safe(title)}</div>'
            f'<div class="be-risk-date">{date}</div></div>'
        )
    else:
        title_html = f'<div class="be-risk-heading"><div></div><div class="be-risk-date">{date}</div></div>'

    if view.alert_status is AlertStatus.NOT_ISSUED:
        hazard_detail = "Suppressed because the forward validation gate did not pass"
        one_year_value = "Not reportable"
        medium_value = "Not reportable"
    else:
        hazard_detail = (
            "Production onset probability"
            if view.model_status is ModelStatus.PRODUCTION
            else "Research estimate; not used in production scoring"
        )
        one_year_value = _format_percent(view.systemic_hazard_1y)
        medium_value = _format_percent(view.systemic_hazard_2_3y)

    structural_cards: list[str] = []
    if view.overall_risk_score is not None:
        structural_cards.append(
            _render_metric_card(
                "Overall risk score",
                _format_score(view.overall_risk_score),
                view.risk_tier or "Active composite risk",
            )
        )
    structural_cards.append(
        _render_metric_card(
            "Operating environment",
            _format_score(view.operating_environment_score),
            view.operating_environment_label or "Structural conditions",
        )
    )
    if view.banking_system_score is not None:
        structural_cards.append(
            _render_metric_card(
                "Banking system",
                _format_score(view.banking_system_score),
                "System-level resilience and vulnerability",
            )
        )
    if view.risk_percentile is not None:
        structural_cards.append(
            _render_metric_card(
                "Risk percentile",
                _format_percentile(view.risk_percentile),
                "Share of countries with a lower active score",
            )
        )
    if view.data_coverage is not None:
        structural_cards.append(
            _render_metric_card(
                "Data coverage",
                _format_percent(view.data_coverage),
                "Active scoring inputs observed",
            )
        )

    hazard_cards = [
        _render_metric_card(
            "1-year crisis-onset probability",
            one_year_value,
            hazard_detail,
        ),
        _render_metric_card(
            "Crisis-onset probability in years 2–3",
            medium_value,
            hazard_detail,
        ),
        _render_alert_card(view),
        _render_metric_card(
            "Hazard input coverage",
            _format_percent(view.hazard_input_coverage),
            "Observed inputs for the selected hazard expert",
        ),
        _render_metric_card(
            "Mechanism evidence coverage",
            _format_percent(view.mechanism_evidence_coverage),
            "Observed signal families in the full mechanism taxonomy",
        ),
        _render_metric_card(
            "Selected hazard expert",
            view.evidence_basis or "Not available",
            "Routing is based on observed input coverage",
        ),
    ]

    structural_html = (
        '<div class="be-layer-heading"><div class="be-layer-title">Active assessment</div>'
        '<div class="be-layer-note">Relative score; not a default probability</div></div>'
        f'<div class="be-structural-grid">{"".join(structural_cards)}</div>'
    )
    warning_html = (
        '<div class="be-layer-heading"><div class="be-layer-title">Systemic early warning</div>'
        '<div class="be-risk-meta">'
        f'<span class="be-model-status {status_class}">{status_text}</span>'
        '<span class="be-layer-note">Onset risk by horizon</span></div></div>'
        f'<div class="be-risk-grid">{"".join(hazard_cards)}</div>'
        '<div class="be-risk-context">'
        '<div class="be-context-block"><div class="be-risk-label">Dominant mechanism</div>'
        f'<div class="be-context-value">{_safe(view.dominant_mechanism)}</div></div>'
        '<div class="be-context-block"><div class="be-risk-label">Interpretation</div>'
        '<div class="be-context-value">Mechanism strength and evidence coverage are reported separately; missing data do not imply low risk.</div></div></div>'
        '<div class="be-mechanisms">'
        '<div class="be-section-heading"><div class="be-section-title">Mechanism signals</div>'
        '<div class="be-section-note">Normalized evidence, not separate default probabilities</div></div>'
        '<div class="be-mechanism-head"><div>Pathway</div><div>Signal strength</div>'
        '<div>Direction</div><div>Evidence</div></div>'
        f"{_render_mechanism_rows(view)}</div>"
    )
    return (
        '<section class="be-risk-surface" aria-label="BankEnv hierarchical risk view">'
        f"{title_html}"
        f"{structural_html if show_structural else ''}"
        f"{warning_html if show_early_warning else ''}"
        "</section>"
    )


def _render_architecture_layers(layers: Sequence[ArchitectureLayer]) -> str:
    rows = []
    for layer in layers:
        rows.append(
            '<div class="be-architecture-row">'
            f'<div class="be-architecture-name">{_safe(layer.name)}</div>'
            f"<div>{_safe(layer.purpose)}</div>"
            f'<div class="be-mechanism-muted">{_safe(layer.output)}</div></div>'
        )
    return '<div class="be-architecture-table">' + "".join(rows) + "</div>"


def build_validation_html(
    summary: ValidationSummary,
    *,
    architecture_layers: Sequence[ArchitectureLayer] = DEFAULT_ARCHITECTURE_LAYERS,
) -> str:
    """Build the methodology and validation summary markup."""

    status = summary.status or "Status not supplied"
    meta = (
        '<div class="be-validation-meta">'
        + _render_metric_card("Model", summary.model_name, summary.sample_description)
        + _render_metric_card("Validation design", summary.validation_design, summary.evaluation_period)
        + _render_metric_card("Governance", status, summary.operating_threshold)
        + "</div>"
    )
    metrics = "".join(
        '<div class="be-validation-metric">'
        f'<div class="be-risk-label">{_safe(metric.label)}</div>'
        f'<div class="be-validation-value">{_safe(metric.value)}</div>'
        + (f'<div class="be-risk-detail">{_safe(metric.note)}</div>' if metric.note else "")
        + "</div>"
        for metric in summary.metrics
    )
    limitations = ""
    if summary.limitations:
        items = "".join(f"<li>{_safe(item)}</li>" for item in summary.limitations)
        limitations = (
            '<div class="be-limitations"><strong>Interpretation limits</strong>'
            f"<ul>{items}</ul></div>"
        )
    architecture = _render_architecture_layers(tuple(architecture_layers))
    return (
        '<section class="be-risk-surface" aria-label="Model methodology and validation">'
        f"{meta}<div class=\"be-validation-metrics\">{metrics}</div>"
        '<div class="be-section-title">How the model family is organized</div>'
        f"{architecture}{limitations}</section>"
    )


def render_country_risk_view(
    view: CountryRiskView,
    *,
    title: str | None = "Country risk assessment",
    show_structural: bool = True,
    show_early_warning: bool = True,
) -> None:
    """Render the country risk hierarchy without deriving model decisions."""

    st.markdown(RISK_SURFACE_STYLES, unsafe_allow_html=True)
    st.markdown(
        build_country_risk_html(
            view,
            title=title,
            show_structural=show_structural,
            show_early_warning=show_early_warning,
        ),
        unsafe_allow_html=True,
    )


def render_methodology_validation(
    summary: ValidationSummary,
    *,
    architecture_layers: Sequence[ArchitectureLayer] = DEFAULT_ARCHITECTURE_LAYERS,
    expanded: bool = False,
    label: str = "Model architecture and validation",
) -> None:
    """Render concise methodology and validation evidence in an expander."""

    st.markdown(RISK_SURFACE_STYLES, unsafe_allow_html=True)
    with st.expander(label, expanded=expanded):
        st.markdown(
            build_validation_html(summary, architecture_layers=architecture_layers),
            unsafe_allow_html=True,
        )


def render_risk_architecture(
    view: CountryRiskView,
    validation: ValidationSummary | None = None,
    *,
    title: str | None = "Country risk assessment",
    validation_expanded: bool = False,
    show_structural: bool = True,
    show_early_warning: bool = True,
) -> None:
    """Render the country surface and, when supplied, its validation context."""

    render_country_risk_view(
        view,
        title=title,
        show_structural=show_structural,
        show_early_warning=show_early_warning,
    )
    if validation is not None:
        render_methodology_validation(validation, expanded=validation_expanded)


def mechanism_signals_from_records(
    records: Iterable[Mapping[str, object]],
) -> tuple[MechanismSignal, ...]:
    """Adapt model-output records to the typed presentation contract."""

    return tuple(
        MechanismSignal(
            name=str(record["name"]),
            signal_strength=record.get("signal_strength"),
            direction=record.get("direction"),
            evidence_confidence=record.get("evidence_confidence"),
            note=record.get("note"),
        )
        for record in records
    )


__all__ = [
    "AlertStatus",
    "ArchitectureLayer",
    "CountryRiskView",
    "DEFAULT_ARCHITECTURE_LAYERS",
    "MechanismSignal",
    "ModelStatus",
    "ValidationMetric",
    "ValidationSummary",
    "build_country_risk_html",
    "build_validation_html",
    "mechanism_signals_from_records",
    "render_country_risk_view",
    "render_methodology_validation",
    "render_risk_architecture",
]
