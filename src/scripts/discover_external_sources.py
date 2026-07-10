"""Discover the IMF SDMX dataflows for the external-liquidity data block.

Backlog ranks 5-14 need BOP, IIP, IRFCL, CPIS, CDIS, Fiscal Monitor, GFS, and
QEDS sources. Their exact dataflow IDs and key structures on the official
SDMX 3.0 API are not documented anywhere machine-readable, so this script
probes a candidate list against the structure endpoint and records what
resolves: agency, dataflow ID, version, and dimension order (needed to build
wildcard data keys).

Run where api.imf.org is reachable (GitHub Actions or the owner's machine;
the Claude sandbox egress proxy blocks it):

    python -m src.scripts.discover_external_sources
    python -m src.scripts.discover_external_sources --families BOP IIP

Output: config/external_sources_discovery.json, consumed by
``fetch_external_sources.py`` in the same workflow run.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import BASE_DIR
from src.sources.base import SourceUnavailableError
from src.sources.sdmx import IMF_SDMX_BASE, _retrying_get

DISCOVERY_PATH = Path(BASE_DIR) / "config" / "external_sources_discovery.json"

# Candidate (agency, dataflow_id) pairs per source family. Discovery keeps
# the first candidate that resolves. IDs follow the naming pattern of the
# verified flows (WEO, FSIC, MFS_DC, FSIBSIS) plus the IMF's public dataset
# names; wrong guesses cost one cheap structure request each.
CANDIDATES = {
    "BOP": [("IMF.STA", "BOP"), ("IMF.STA", "BOP_DSD"), ("IMF.STA", "BOPSY")],
    "IIP": [("IMF.STA", "IIP"), ("IMF.STA", "BOP_IIP")],
    "IRFCL": [("IMF.STA", "IRFCL"), ("IMF.STA", "IRFCL_DSD")],
    "CPIS": [("IMF.STA", "CPIS")],
    "CDIS": [("IMF.STA", "CDIS")],
    "FM": [("IMF.FAD", "FM"), ("IMF.RES", "FM"), ("IMF.FAD", "FISCALMONITOR")],
    "GFS": [
        ("IMF.STA", "GFS_MAB"),
        ("IMF.STA", "QGFS"),
        ("IMF.STA", "GFS"),
        ("IMF.STA", "GFSR"),
    ],
    "QEDS": [("IMF.STA", "QEDS"), ("IMF.STA", "QEDS_GDDS")],
}


def parse_dataflow_structure(payload: dict) -> dict:
    """Extract version and ordered non-time dimensions from SDMX-JSON."""
    data = payload.get("data", {})
    flows = data.get("dataflows", [])
    structures = data.get("dataStructures", [])
    dimensions = []
    if structures:
        dimension_list = (
            structures[0]
            .get("dataStructureComponents", {})
            .get("dimensionList", {})
        )
        entries = dimension_list.get("dimensions", [])
        entries = sorted(entries, key=lambda item: item.get("position", 0))
        dimensions = [
            entry["id"] for entry in entries
            if entry.get("id") and entry["id"] != "TIME_PERIOD"
        ]
    return {
        "dataflow_version": flows[0].get("version") if flows else None,
        "dataflow_name": (
            (flows[0].get("name") or flows[0].get("names", {}).get("en"))
            if flows else None
        ),
        "structure_version": structures[0].get("version") if structures else None,
        "dimensions": dimensions,
    }


def probe_dataflow(agency: str, dataflow_id: str, timeout=60) -> dict:
    url = f"{IMF_SDMX_BASE}/structure/dataflow/{agency}/{dataflow_id}/+"
    response = _retrying_get(
        url,
        headers={"Accept": "application/vnd.sdmx.structure+json"},
        timeout=timeout,
        retries=2,
    )
    parsed = parse_dataflow_structure(response.json())
    if not parsed["dimensions"]:
        raise SourceUnavailableError(
            f"{agency}/{dataflow_id} resolved but exposed no dimensions"
        )
    return parsed


def discover(families=None) -> dict:
    results = {}
    for family, candidates in CANDIDATES.items():
        if families and family not in families:
            continue
        family_result = {"status": "unresolved", "attempts": []}
        for agency, dataflow_id in candidates:
            try:
                parsed = probe_dataflow(agency, dataflow_id)
                family_result = {
                    "status": "resolved",
                    "agency": agency,
                    "dataflow_id": dataflow_id,
                    **parsed,
                    "attempts": family_result["attempts"],
                }
                break
            except Exception as error:  # noqa: BLE001 - record and continue
                family_result["attempts"].append(
                    f"{agency}/{dataflow_id}: {type(error).__name__}: {error}"
                )
        results[family] = family_result
    return {
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": IMF_SDMX_BASE,
        "families": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--output", default=str(DISCOVERY_PATH))
    args = parser.parse_args()

    report = discover(args.families)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    resolved = [
        name for name, entry in report["families"].items()
        if entry["status"] == "resolved"
    ]
    unresolved = [
        name for name, entry in report["families"].items()
        if entry["status"] != "resolved"
    ]
    print(f"Discovery written to: {output}")
    print(f"Resolved: {resolved or 'none'}")
    print(f"Unresolved: {unresolved or 'none'}")
    for name in resolved:
        entry = report["families"][name]
        print(
            f"  {name}: {entry['agency']}:{entry['dataflow_id']}"
            f"({entry['dataflow_version']}) key={'.'.join(entry['dimensions'])}"
        )


if __name__ == "__main__":
    main()
