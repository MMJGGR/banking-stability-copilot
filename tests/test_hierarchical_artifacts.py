import json

import pandas as pd

from src.hierarchical_artifacts import (
    load_hierarchical_artifacts,
    outputs_are_production,
    outputs_reportable,
    validation_gate_state,
)


def test_malformed_optional_artifacts_disable_research_without_crashing(tmp_path):
    snapshot = tmp_path / "snapshot.json"
    validation = tmp_path / "validation.json"
    snapshot.write_text("{broken", encoding="utf-8")
    validation.write_text("[]", encoding="utf-8")

    payload, frame, report = load_hierarchical_artifacts(snapshot, validation)

    assert payload == {}
    assert frame.empty
    assert report == {}
    assert validation_gate_state(report) == "unavailable"


def test_loader_sanitizes_invalid_optional_rows(tmp_path):
    snapshot = tmp_path / "snapshot.json"
    validation = tmp_path / "validation.json"
    snapshot.write_text(
        json.dumps(
            {
                "as_of_date": "2026-06-30",
                "model_status": "research_challenger",
                "countries": [
                    {
                        "country_code": "usa",
                        "probability_1y": 1.2,
                        "probability_years_2_3": 0.04,
                        "alert_level": "invented",
                        "evidence_confidence": -0.1,
                    },
                    {"country_code": "bad-code"},
                ],
            }
        ),
        encoding="utf-8",
    )
    validation.write_text(
        json.dumps({"promotion_gates": {"passed": False}}), encoding="utf-8"
    )

    payload, frame, report = load_hierarchical_artifacts(snapshot, validation)

    assert list(frame["country_code"]) == ["USA"]
    assert pd.isna(frame.loc[0, "systemic_hazard_1y"])
    assert frame.loc[0, "systemic_hazard_2_3y"] == 0.04
    assert pd.isna(frame.loc[0, "evidence_confidence"])
    assert frame.loc[0, "alert_status"] == "insufficient_evidence"
    assert validation_gate_state(report) == "failed"
    assert not outputs_reportable(payload, report, expected_as_of="2026-06-30")


def test_operational_status_requires_gate_cutoff_and_unanimous_approval():
    snapshot = {"as_of_date": "2026-06-30", "model_status": "production"}
    validation = {"promotion_gates": {"passed": True}}
    frame = pd.DataFrame({"model_status": ["production", "production"]})

    assert outputs_reportable(snapshot, validation, expected_as_of="2026-06-30")
    assert outputs_are_production(
        snapshot, frame, validation, expected_as_of="2026-06-30"
    )
    assert not outputs_are_production(
        snapshot, frame, validation, expected_as_of="2025-12-31"
    )
    frame.loc[1, "model_status"] = "research"
    assert not outputs_are_production(
        snapshot, frame, validation, expected_as_of="2026-06-30"
    )
