"""Safe loading and governance checks for optional early-warning artifacts.

The active structural score must remain usable when a research JSON artifact is
missing, malformed, stale, or internally inconsistent.  This module therefore
normalises the compact snapshot before Streamlit sees it and exposes one shared
gate-state definition for the country, global, explorer, and methodology views.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROBABILITY_COLUMNS = (
    "systemic_hazard_1y",
    "systemic_hazard_2_3y",
    "systemic_hazard_3y",
)
COVERAGE_COLUMNS = (
    "evidence_confidence",
    "hazard_evidence_coverage",
    "mechanism_evidence_coverage",
)
VALID_ALERT_STATUSES = {"red", "amber", "none", "clear", "insufficient_evidence"}
VALID_MODEL_STATUSES = {
    "production",
    "research",
    "research_challenger",
    "promotion_candidate",
}


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _country_records(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    records = snapshot.get("countries", [])
    if isinstance(records, dict):
        records = [
            {"country_code": code, **(record if isinstance(record, dict) else {})}
            for code, record in records.items()
        ]
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def _normalise_frame(snapshot: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(_country_records(snapshot))
    if frame.empty or "country_code" not in frame:
        return pd.DataFrame()

    aliases = {
        "probability_1y": "systemic_hazard_1y",
        "probability_years_2_3": "systemic_hazard_2_3y",
        "probability_within_3y": "systemic_hazard_3y",
        "selected_expert": "hazard_expert",
        "alert_level": "alert_status",
        "overall_evidence_confidence": "evidence_confidence",
    }
    for source, target in aliases.items():
        if target not in frame and source in frame:
            frame[target] = frame[source]

    if "dominant_mechanism_label" in frame:
        if "dominant_mechanism" in frame:
            frame["dominant_mechanism_key"] = frame["dominant_mechanism"]
        frame["dominant_mechanism"] = frame["dominant_mechanism_label"].combine_first(
            frame.get("dominant_mechanism", pd.Series(index=frame.index, dtype=object))
        )

    country_codes = frame["country_code"].astype("string").str.strip().str.upper()
    frame["country_code"] = country_codes
    frame = frame[country_codes.str.fullmatch(r"[A-Z]{3}", na=False)].copy()
    if frame.empty:
        return pd.DataFrame()
    if frame["country_code"].duplicated(keep=False).any():
        return pd.DataFrame()

    for column in (*PROBABILITY_COLUMNS, *COVERAGE_COLUMNS):
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = numeric.where(numeric.between(0.0, 1.0))

    if "alert_status" in frame:
        statuses = frame["alert_status"].astype(str).str.strip().str.lower()
        frame["alert_status"] = statuses.where(
            statuses.isin(VALID_ALERT_STATUSES), "insufficient_evidence"
        )
    else:
        frame["alert_status"] = "insufficient_evidence"

    snapshot_status = str(snapshot.get("model_status") or "research").strip().lower()
    if snapshot_status not in VALID_MODEL_STATUSES:
        snapshot_status = "research"
    if "model_status" in frame:
        row_statuses = frame["model_status"].astype(str).str.strip().str.lower()
        frame["model_status"] = row_statuses.where(
            row_statuses.isin(VALID_MODEL_STATUSES), "research"
        )
    else:
        frame["model_status"] = snapshot_status
    frame["as_of_date"] = snapshot.get("as_of_date")
    return frame.reset_index(drop=True)


def load_hierarchical_artifacts(
    snapshot_path: str | Path,
    validation_path: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Load optional research artifacts without allowing them to crash serving."""

    snapshot = _read_mapping(Path(snapshot_path))
    validation = _read_mapping(Path(validation_path))
    return snapshot, _normalise_frame(snapshot), validation


def validation_gate_state(validation: dict[str, Any] | None) -> str:
    """Return ``passed``, ``failed``, or ``unavailable`` without guessing."""

    if not isinstance(validation, dict):
        return "unavailable"
    gates = validation.get("promotion_gates")
    if not isinstance(gates, dict):
        return "unavailable"
    passed = gates.get("passed")
    if passed is True:
        return "passed"
    if passed is False:
        return "failed"
    return "unavailable"


def artifacts_match_snapshot(
    snapshot: dict[str, Any] | None,
    expected_as_of: str | None,
) -> bool:
    """Require a dated research artifact to match the active structural cutoff."""

    if not isinstance(snapshot, dict) or not snapshot:
        return False
    actual = str(snapshot.get("as_of_date") or "").strip()[:10]
    expected = str(expected_as_of or "").strip()[:10]
    return bool(actual and expected and actual == expected)


def outputs_reportable(
    snapshot: dict[str, Any] | None,
    validation: dict[str, Any] | None,
    *,
    expected_as_of: str | None = None,
) -> bool:
    """Country probabilities and review tiers require a passed, matched gate."""

    if validation_gate_state(validation) != "passed":
        return False
    if expected_as_of is not None and not artifacts_match_snapshot(snapshot, expected_as_of):
        return False
    return True


def outputs_are_production(
    snapshot: dict[str, Any] | None,
    frame: pd.DataFrame,
    validation: dict[str, Any] | None,
    *,
    expected_as_of: str | None = None,
) -> bool:
    """Operational alerts require approval in addition to statistical gates."""

    if not outputs_reportable(
        snapshot, validation, expected_as_of=expected_as_of
    ) or frame.empty:
        return False
    snapshot_status = str((snapshot or {}).get("model_status") or "").lower()
    row_statuses = set(frame.get("model_status", pd.Series(dtype=object)).dropna().astype(str).str.lower())
    return snapshot_status == "production" and row_statuses == {"production"}


__all__ = [
    "artifacts_match_snapshot",
    "load_hierarchical_artifacts",
    "outputs_are_production",
    "outputs_reportable",
    "validation_gate_state",
]
