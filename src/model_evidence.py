"""Presentation policy for model evidence and active feature roles.

This module is deliberately independent of Streamlit so the trust rules used
by the Methodology tab can be tested without importing the application.
"""

from __future__ import annotations

from collections.abc import Mapping


CLEAN_CRISIS_VALIDATION_STATUS = "validated_clean"


def active_model_feature_codes(pca_info: Mapping | None) -> set[str]:
    """Return the feature codes that have loadings in the served pillar model.

    The feature-value sidecar contains both scored and contextual columns, so
    column presence alone is not evidence that a field affects the score.  The
    persisted economic/industry loading maps are the serving artifact's source
    of truth for pillar membership.
    """

    if not isinstance(pca_info, Mapping):
        return set()

    active: set[str] = set()
    for key in ("economic_loadings", "industry_loadings"):
        loadings = pca_info.get(key)
        if isinstance(loadings, Mapping):
            active.update(str(feature) for feature in loadings)
    return active


def liquidity_feature_role(
    feature: str,
    report: Mapping | None,
    active_features: set[str] | None = None,
) -> str:
    """Resolve a liquidity field's current role without trusting a stale report.

    When the active artifact's loading set is available it takes precedence
    over the historical candidate report.  The report is only a fallback for
    older artifacts that do not persist loading maps.
    """

    if active_features is not None:
        return (
            "Active score input"
            if feature in active_features
            else "Candidate / monitoring"
        )

    reported_active = set((report or {}).get("active_features") or [])
    return "Active score input" if feature in reported_active else "Candidate / monitoring"


def crisis_validation_display_state(summary: Mapping | None) -> dict[str, object]:
    """Return whether packaged classifier metrics are safe to show as valid.

    Display is fail-closed.  A future report must explicitly identify itself as
    clean, opt in to display, and carry the exact governed status token.  Old
    reports without these fields therefore remain audit artifacts rather than
    silently appearing as current validation evidence.
    """

    if not isinstance(summary, Mapping) or not summary:
        return {
            "display_metrics": False,
            "status": "unavailable",
            "reason": "No structured crisis-validation evidence is packaged.",
        }

    status = str(summary.get("validation_status") or "unlabelled").strip().lower()
    clean = summary.get("clean_validation") is True
    opted_in = summary.get("display_metrics") is True
    display = status == CLEAN_CRISIS_VALIDATION_STATUS and clean and opted_in

    if display:
        reason = "Validation evidence passed the governed clean-display gate."
    else:
        reason = str(summary.get("status_reason") or "").strip() or (
            "Metrics are withheld because this report has not passed the governed "
            "clean-validation display gate."
        )
    return {
        "display_metrics": display,
        "status": status,
        "reason": reason,
    }
