import json

import pandas as pd
import pytest

from src.scripts.discover_external_sources import (
    CANDIDATES,
    parse_dataflow_structure,
)
from src.scripts.fetch_external_sources import _coverage, normalize_long_csv


def test_candidate_registry_covers_the_backlog_families():
    assert set(CANDIDATES) == {
        "BOP", "IIP", "IRFCL", "CPIS", "CDIS", "FM", "GFS", "QEDS",
    }
    for family, candidates in CANDIDATES.items():
        assert candidates, family
        for agency, dataflow_id in candidates:
            assert agency.startswith("IMF."), (family, agency)
            assert dataflow_id


def test_parse_dataflow_structure_orders_dimensions_and_reads_version():
    payload = {
        "data": {
            "dataflows": [{"version": "4.0.1", "name": "Balance of Payments"}],
            "dataStructures": [
                {
                    "version": "4.0.0",
                    "dataStructureComponents": {
                        "dimensionList": {
                            "dimensions": [
                                {"id": "INDICATOR", "position": 2},
                                {"id": "COUNTRY", "position": 1},
                                {"id": "FREQUENCY", "position": 3},
                                {"id": "TIME_PERIOD", "position": 4},
                            ]
                        }
                    },
                }
            ],
        }
    }
    parsed = parse_dataflow_structure(payload)
    assert parsed["dataflow_version"] == "4.0.1"
    assert parsed["structure_version"] == "4.0.0"
    assert parsed["dimensions"] == ["COUNTRY", "INDICATOR", "FREQUENCY"]


def test_parse_dataflow_structure_handles_empty_payload():
    parsed = parse_dataflow_structure({"data": {}})
    assert parsed["dimensions"] == []
    assert parsed["dataflow_version"] is None


def _long_csv(tmp_path, country_column="COUNTRY"):
    frame = pd.DataFrame(
        {
            "STRUCTURE_ID": ["IMF.STA:BOP(4.0.1)"] * 4,
            country_column: ["KEN", "KEN", "MOZ", "MOZ"],
            "INDICATOR": ["CAB", "CAB", "CAB", "RES"],
            "FREQUENCY": ["A", "Q", "A", "A"],
            "TIME_PERIOD": ["2024", "2025-Q4", "2024", "not-a-period"],
            "OBS_VALUE": ["-1200.5", "-310.2", "", "88.0"],
            "OBS_STATUS": ["A", "E", "A", "A"],
        }
    )
    path = tmp_path / "bop.csv"
    frame.to_csv(path, index=False)
    return path


def test_normalize_long_csv_parses_periods_and_drops_bad_rows(tmp_path):
    normalized = normalize_long_csv(_long_csv(tmp_path), "BOP")
    # Blank value and unparsable period rows are dropped.
    assert len(normalized) == 2
    assert set(normalized["country_code"]) == {"KEN"}
    assert (normalized["source"] == "BOP").all()
    assert normalized["dataset_version"].iloc[0] == "IMF.STA:BOP(4.0.1)"
    assert "dim_INDICATOR" in normalized.columns
    assert "dim_FREQUENCY" in normalized.columns
    quarterly = normalized[normalized["period_label"] == "2025-Q4"]
    assert quarterly["period"].iloc[0].month == 12
    assert quarterly["observation_status"].iloc[0] == "E"

    coverage = _coverage(normalized)
    assert coverage["rows"] == 2
    assert coverage["countries"] == 1
    assert coverage["indicators"] == 1
    assert coverage["latest_period"] == "2025-12-31"


def test_normalize_long_csv_accepts_ref_area_country_dimension(tmp_path):
    normalized = normalize_long_csv(
        _long_csv(tmp_path, country_column="REF_AREA"), "IIP"
    )
    assert set(normalized["country_code"]) == {"KEN"}


def test_normalize_long_csv_rejects_valueless_files(tmp_path):
    frame = pd.DataFrame({"TIME_PERIOD": ["2024"], "OBS_VALUE": [""]})
    path = tmp_path / "empty.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="no country dimension"):
        normalize_long_csv(path, "BOP")


def test_discovery_output_is_json_serializable():
    # The report structure must round-trip so the workflow artifact is valid.
    from src.scripts.discover_external_sources import DISCOVERY_PATH

    sample = {
        "discovered_at": "2026-07-10T00:00:00+00:00",
        "families": {
            "BOP": {"status": "resolved", "agency": "IMF.STA",
                    "dataflow_id": "BOP", "dimensions": ["COUNTRY"],
                    "dataflow_version": "1", "attempts": []},
        },
    }
    assert json.loads(json.dumps(sample)) == sample
    assert DISCOVERY_PATH.name == "external_sources_discovery.json"
