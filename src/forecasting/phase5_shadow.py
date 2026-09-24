"""Phase 5 shadow-serving bundle and prospective ledger.

This module packages accepted Phase 2-4 research evidence into a read-only,
versioned shadow contract. Production values remain a separate benchmark lane.
WEO provider projections are represented only as scenario availability flags.
The rejected Phase 4 crisis overlay is never served.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "banking-copilot-shadow-v1.0"
VINTAGE_MODE = "retrospective_latest_vintage_research"
WEO_STATUS_CAVEAT = (
    "WEO values dated 2025 may include estimates; actual/estimate status is not "
    "fully verified for every indicator."
)
CRISIS_STATUS = "not_approved"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _clean_records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict("records")


def _country_name(code: str, production_names: dict[str, str]) -> tuple[str, str]:
    production_name = production_names.get(code)
    if isinstance(production_name, str) and production_name.strip():
        return production_name.strip(), "production_reference"
    special = {
        "AIA": "Anguilla",
        "ASM": "American Samoa",
        "BMU": "Bermuda",
        "CUB": "Cuba",
        "CUW": "Curaçao",
        "CYM": "Cayman Islands",
        "GRL": "Greenland",
        "GUM": "Guam",
        "MCO": "Monaco",
        "MSR": "Montserrat",
        "NCL": "New Caledonia",
        "PRK": "Korea, Democratic People's Republic of",
        "PSE": "Palestine",
        "PYF": "French Polynesia",
        "SXM": "Sint Maarten (Dutch part)",
        "VIR": "Virgin Islands, U.S.",
        "XKX": "Kosovo",
    }
    if code in special:
        return special[code], "static_iso_or_research_mapping"
    return code, "unresolved_code_only"


def write_csv(frame: pd.DataFrame, path: str | Path, *, index: bool = False) -> None:
    path = Path(path)
    if path.suffix == ".gz":
        frame.to_csv(
            path,
            index=index,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
    else:
        frame.to_csv(path, index=index)


def write_jsonl_gzip(records, path: str | Path) -> None:
    path = Path(path)
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename=path.name, mode="wb", fileobj=raw, mtime=0
        ) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as handle:
                for record in records:
                    handle.write(json.dumps(record, allow_nan=False, default=str) + "\n")


def validate_inputs(phase2: Path, phase3: Path, phase4: Path) -> dict:
    p2 = _read_json(phase2 / "phase2-summary.json")
    p3 = _read_json(phase3 / "phase3-summary.json")
    p4a = _read_json(phase4 / "phase4a_interpretation" / "phase4a-summary.json")
    p4b = _read_json(
        phase4 / "phase4b_observable_validation" / "phase4b-summary.json"
    )
    p4c = _read_json(
        phase4 / "phase4c_crisis_overlay" / "phase4c-final-decision.json"
    )
    if p2.get("status") != "completed_phase2_measurement_and_transition_development":
        raise ValueError("Phase 2 evidence is not accepted")
    if not p3.get("transition_distribution_acceptance_passed"):
        raise ValueError("Phase 3 calibration did not pass")
    if p3.get("provider_projection_rows_read") != 0:
        raise ValueError("Provider projections entered the Phase 3 baseline")
    if p4a.get("labels_are_model_constraints") is not False:
        raise ValueError("Phase 4 interpretation altered the model")
    if p4b.get("provider_projection_rows_read") != 0:
        raise ValueError("Provider projections entered observable validation")
    if p4c.get("status") != (
        "do_not_advance_state_crisis_overlay_underperformed_event_rate"
    ):
        raise ValueError("Unexpected Phase 4 crisis-overlay decision")
    if p4c.get("latest_overlay_rows_admissible") != 0:
        raise ValueError("Rejected crisis overlay has admissible latest rows")
    return {
        "phase2": p2,
        "phase3": p3,
        "phase4a": p4a,
        "phase4b": p4b,
        "phase4c": p4c,
    }


def provider_availability(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(
            columns=["entity_code", "forecast_year", "provider_projection_count"]
        )
    frame = pd.read_csv(
        path,
        usecols=["COUNTRY", "projection_year", "data_role", "model_admission"],
    )
    if not frame.data_role.eq("provider_projection").all():
        raise ValueError("Unexpected non-projection row in provider ledger")
    if not frame.model_admission.eq(
        "scenario_only_not_measurement_or_target"
    ).all():
        raise ValueError("Provider projection admission policy mismatch")
    return (
        frame.groupby(["COUNTRY", "projection_year"], observed=True)
        .size()
        .rename("provider_projection_count")
        .reset_index()
        .rename(
            columns={"COUNTRY": "entity_code", "projection_year": "forecast_year"}
        )
    )


def build_bundle(
    phase2: str | Path,
    phase3: str | Path,
    phase4: str | Path,
    output: str | Path,
    *,
    production_reference: str | Path | None = None,
    provider_projections: str | Path | None = None,
    issued_at: str = "2026-09-24",
    research_commit: str = "31968c9596d928df9a12e29b5ea2ea442853213d",
    production_commit: str = "5957ca779dafa21f2e098c819bfb060f43243206",
) -> dict:
    phase2, phase3, phase4, output = map(Path, (phase2, phase3, phase4, output))
    output.mkdir(parents=True, exist_ok=False)
    evidence = validate_inputs(phase2, phase3, phase4)

    forecasts = pd.read_csv(phase3 / "latest-forecast-summary.csv")
    quantiles = pd.read_csv(phase3 / "latest-state-coordinate-quantiles.csv.gz")
    peers = pd.read_csv(phase3 / "nearest-peer-frequencies.csv.gz")
    analogues = pd.read_csv(phase3 / "historical-analogues.csv.gz")
    observable_implications = pd.read_csv(
        phase3 / "observable-implications-top.csv.gz"
    )
    states = pd.read_csv(phase2 / "country-year-states.csv.gz")
    information = pd.read_csv(phase2 / "country-year-information.csv.gz")
    state_explanations = pd.read_csv(
        phase4
        / "phase4a_interpretation"
        / "latest-country-state-explanations.csv.gz"
    )
    movement_explanations = pd.read_csv(
        phase4
        / "phase4a_interpretation"
        / "latest-country-movement-explanations.csv.gz"
    )
    stable_outcomes = pd.read_csv(
        phase4
        / "phase4b_observable_validation"
        / "stable-state-predictable-outcomes.csv"
    )
    crisis_decision = evidence["phase4c"]

    if forecasts.entity_code.nunique() != 213 or len(forecasts) != 426:
        raise ValueError("Unexpected Phase 3 latest forecast population")
    if set(forecasts.horizon) != {1, 2}:
        raise ValueError("Expected one- and two-year forecasts")
    if (
        quantiles.groupby(["entity_code", "horizon"])
        .state_coordinate.nunique()
        .ne(96)
        .any()
    ):
        raise ValueError("Incomplete state-coordinate quantiles")

    current_year = int(forecasts.current_state_year.unique().item())
    latest_states = states.loc[states.forecast_origin_year.eq(current_year)].copy()
    latest_information = information.loc[
        information.forecast_origin_year.eq(current_year)
    ].copy()
    if latest_states.entity_code.nunique() != 213:
        raise ValueError("Latest state population does not reconcile")

    production = pd.DataFrame(
        columns=[
            "country_code",
            "country_name",
            "promoted_score",
            "risk_category",
            "crisis_prob",
        ]
    )
    if production_reference is not None:
        production = pd.read_csv(production_reference)
        required = [
            "country_code",
            "country_name",
            "promoted_score",
            "risk_category",
            "crisis_prob",
        ]
        if not set(required) <= set(production):
            raise ValueError(
                f"Production reference missing {sorted(set(required)-set(production))}"
            )
        production = production[required].drop_duplicates("country_code")
    production_names = production.set_index("country_code").country_name.to_dict()

    availability = provider_availability(
        Path(provider_projections) if provider_projections else None
    )
    summary = (
        forecasts.merge(
            production.rename(
                columns={
                    "country_code": "entity_code",
                    "country_name": "production_country_name",
                    "promoted_score": "production_risk_score",
                    "risk_category": "production_risk_category",
                    "crisis_prob": "production_crisis_probability",
                }
            ),
            on="entity_code",
            how="left",
            validate="many_to_one",
        )
        .merge(
            availability,
            on=["entity_code", "forecast_year"],
            how="left",
            validate="one_to_one",
        )
    )
    summary["provider_projection_count"] = (
        summary.provider_projection_count.fillna(0).astype(int)
    )
    names = [_country_name(code, production_names) for code in summary.entity_code]
    summary["entity_name"] = [item[0] for item in names]
    summary["name_resolution"] = [item[1] for item in names]
    summary["production_reference_available"] = summary.production_risk_score.notna()
    summary["research_only_country"] = ~summary.production_reference_available
    summary["provider_scenario_available"] = summary.provider_projection_count.gt(0)
    summary["research_crisis_overlay_status"] = CRISIS_STATUS
    summary["research_crisis_probability"] = np.nan
    summary["research_output_status"] = "shadow_research_not_production"
    summary["source_cutoff"] = evidence["phase2"]["phase2a"]["source_cutoff"]
    summary["issued_at"] = issued_at
    summary["vintage_mode"] = VINTAGE_MODE
    summary["weo_2025_status_caveat"] = WEO_STATUS_CAVEAT
    summary["production_commit"] = production_commit
    summary["research_commit"] = research_commit

    input_identity = {
        "schema_version": SCHEMA_VERSION,
        "issued_at": issued_at,
        "source_cutoff": summary.source_cutoff.iloc[0],
        "current_state_year": current_year,
        "phase2_zip_sha256": evidence["phase2"]["source_zip_sha256"],
        "phase4_artifact_sha256": (
            "ab55f002f34fa992d3ade9dc27acdc32bdee379b4b586010dd1357d3b8844e6d"
        ),
        "research_commit": research_commit,
        "production_commit": production_commit,
    }
    batch_id = "shadow-" + stable_hash(input_identity)[:20]
    summary["forecast_batch_id"] = batch_id

    state_columns = [f"state_{index}" for index in range(1, 97)]
    current_state = latest_states.merge(
        latest_information,
        on=["entity_code", "forecast_origin_year"],
        how="left",
        validate="one_to_one",
    )
    current_state["state_vector_sha256"] = current_state[state_columns].apply(
        lambda row: stable_hash([float(value) for value in row]), axis=1
    )

    quantile_hashes = {}
    for key, group in quantiles.groupby(["entity_code", "horizon"], sort=True):
        ordered = group.sort_values("state_coordinate")
        quantile_hashes[key] = stable_hash(_clean_records(ordered))
    summary["forecast_coordinate_sha256"] = [
        quantile_hashes[(row.entity_code, row.horizon)]
        for row in summary.itertuples()
    ]
    summary["forecast_record_id"] = [
        stable_hash(
            {
                "batch": batch_id,
                "entity": row.entity_code,
                "horizon": int(row.horizon),
            }
        )[:24]
        for row in summary.itertuples()
    ]

    stable_keys = set(zip(stable_outcomes.predictor_id, stable_outcomes.horizon))
    observable_implications["stable_validated_outcome"] = [
        (predictor, horizon) in stable_keys
        for predictor, horizon in zip(
            observable_implications.predictor_id,
            observable_implications.horizon,
        )
    ]
    observable_implications["interpretation_status"] = np.where(
        observable_implications.stable_validated_outcome,
        "validated_directional_overlay",
        "directional_attribution_only",
    )

    production_codes = set(production.country_code.astype(str))
    research_codes = set(summary.entity_code.astype(str))
    population = pd.DataFrame(
        {"entity_code": sorted(research_codes | production_codes)}
    )
    population["in_research_forecast"] = population.entity_code.isin(research_codes)
    population["in_production_snapshot"] = population.entity_code.isin(production_codes)
    population["population_status"] = np.select(
        [
            population.in_research_forecast & population.in_production_snapshot,
            population.in_research_forecast & ~population.in_production_snapshot,
            ~population.in_research_forecast & population.in_production_snapshot,
        ],
        ["overlap", "research_only", "production_only"],
        default="neither",
    )
    population["entity_name"] = [
        _country_name(code, production_names)[0] for code in population.entity_code
    ]

    ledger = summary[
        [
            "forecast_batch_id",
            "forecast_record_id",
            "entity_code",
            "entity_name",
            "issued_at",
            "source_cutoff",
            "current_state_year",
            "forecast_year",
            "horizon",
            "forecast_quality",
            "observed_model_features",
            "observed_share",
            "state_uncertainty_proxy",
            "movement_radius_q50",
            "movement_radius_q80",
            "movement_radius_q95",
            "forecast_coordinate_sha256",
            "research_commit",
        ]
    ].copy()
    ledger = ledger.rename(
        columns={
            "horizon": "horizon_years",
            "current_state_year": "state_origin_year",
        }
    )
    ledger["realization_status"] = "pending"
    ledger["evaluation_not_before"] = (
        ledger.forecast_year.astype(str) + "-12-31"
    )
    ledger["realized_state_vintage"] = ""
    ledger["realized_state_sha256"] = ""
    ledger["state_rmse"] = np.nan
    ledger["state_mae"] = np.nan
    ledger["covered_50"] = pd.NA
    ledger["covered_80"] = pd.NA
    ledger["covered_95"] = pd.NA
    ledger["provider_projection_is_not_realization"] = True
    ledger["append_only_record"] = True

    summary = summary.sort_values(["entity_code", "horizon"]).reset_index(drop=True)
    current_state = current_state.sort_values("entity_code").reset_index(drop=True)
    quantiles = quantiles.sort_values(
        ["entity_code", "horizon", "state_coordinate"]
    ).reset_index(drop=True)
    peers = peers.sort_values(
        ["entity_code", "horizon", "probability", "peer_entity"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    analogues = analogues.sort_values(
        ["entity_code", "analogue_distance", "analogue_entity"]
    ).reset_index(drop=True)
    state_explanations = state_explanations.sort_values(
        ["entity_code", "rank"]
    ).reset_index(drop=True)
    movement_explanations = movement_explanations.sort_values(
        ["entity_code", "horizon", "rank"]
    ).reset_index(drop=True)
    observable_implications = observable_implications.sort_values(
        ["entity_code", "horizon", "predictor_id"]
    ).reset_index(drop=True)
    stable_outcomes = stable_outcomes.sort_values(
        ["horizon", "predictor_id"]
    ).reset_index(drop=True)
    population = population.sort_values("entity_code").reset_index(drop=True)
    availability = availability.sort_values(
        ["entity_code", "forecast_year"]
    ).reset_index(drop=True)
    ledger = ledger.sort_values(
        ["entity_code", "horizon_years"]
    ).reset_index(drop=True)

    write_csv(summary, output / "shadow-country-horizon-summary.csv")
    write_csv(current_state, output / "shadow-current-state-coordinates.csv.gz")
    write_csv(quantiles, output / "shadow-forecast-coordinate-quantiles.csv.gz")
    write_csv(peers, output / "shadow-future-peer-frequencies.csv.gz")
    write_csv(analogues, output / "shadow-historical-analogues.csv.gz")
    write_csv(
        state_explanations,
        output / "shadow-current-state-explanations.csv.gz",
    )
    write_csv(
        movement_explanations,
        output / "shadow-movement-explanations.csv.gz",
    )
    write_csv(
        observable_implications,
        output / "shadow-observable-implications.csv.gz",
    )
    write_csv(
        stable_outcomes,
        output / "shadow-stable-observable-catalog.csv",
    )
    write_csv(population, output / "shadow-population-reconciliation.csv")
    write_csv(
        availability,
        output / "shadow-provider-scenario-availability.csv",
    )
    write_csv(ledger, output / "prospective-forecast-ledger.csv")
    (output / "rejected-overlays.json").write_text(
        json.dumps(
            {
                "state_based_crisis_overlay": crisis_decision,
                "serving_rule": "omit_rejected_research_crisis_probability",
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )

    state_groups = {
        key: _clean_records(group.sort_values("rank").head(5))
        for key, group in state_explanations.groupby("entity_code", sort=False)
    }
    movement_groups = {
        (code, int(horizon)): _clean_records(group.sort_values("rank").head(5))
        for (code, horizon), group in movement_explanations.groupby(
            ["entity_code", "horizon"], sort=False
        )
    }
    peer_groups = {
        (code, int(horizon)): _clean_records(
            group.sort_values(
                ["probability", "peer_entity"], ascending=[False, True]
            ).head(5)
        )
        for (code, horizon), group in peers.groupby(
            ["entity_code", "horizon"], sort=False
        )
    }
    analogue_groups = {
        code: _clean_records(group.sort_values("analogue_distance").head(5))
        for code, group in analogues.groupby("entity_code", sort=False)
    }
    observable_groups = {
        (code, int(horizon)): _clean_records(
            group.assign(_abs=group.expected_standardized_change.abs())
            .sort_values("_abs", ascending=False)
            .drop(columns="_abs")
            .head(8)
        )
        for (code, horizon), group in observable_implications.groupby(
            ["entity_code", "horizon"], sort=False
        )
    }
    nested_records = []
    for row in summary.itertuples(index=False):
        code, horizon = row.entity_code, int(row.horizon)
        nested_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "forecast_batch_id": batch_id,
                "summary": {
                    key: (
                        None
                        if pd.isna(value)
                        else value.item()
                        if hasattr(value, "item")
                        else value
                    )
                    for key, value in row._asdict().items()
                },
                "current_state_explanations": state_groups.get(code, []),
                "movement_explanations": movement_groups.get((code, horizon), []),
                "future_peers": peer_groups.get((code, horizon), []),
                "historical_analogues": analogue_groups.get(code, []),
                "observable_implications": observable_groups.get(
                    (code, horizon), []
                ),
                "crisis_overlay": {
                    "research_status": CRISIS_STATUS,
                    "research_probability": None,
                    "production_probability": (
                        None
                        if pd.isna(row.production_crisis_probability)
                        else float(row.production_crisis_probability)
                    ),
                },
            }
        )
    write_jsonl_gzip(
        nested_records,
        output / "shadow-country-records.jsonl.gz",
    )

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_VERSION,
        "title": "Banking Copilot shadow country-horizon record",
        "type": "object",
        "required": [
            "schema_version",
            "forecast_batch_id",
            "summary",
            "current_state_explanations",
            "movement_explanations",
            "future_peers",
            "historical_analogues",
            "observable_implications",
            "crisis_overlay",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "forecast_batch_id": {"type": "string"},
            "summary": {"type": "object"},
            "current_state_explanations": {"type": "array"},
            "movement_explanations": {"type": "array"},
            "future_peers": {"type": "array"},
            "historical_analogues": {"type": "array"},
            "observable_implications": {"type": "array"},
            "crisis_overlay": {
                "type": "object",
                "properties": {
                    "research_status": {"const": CRISIS_STATUS},
                    "research_probability": {"type": "null"},
                    "production_probability": {"type": ["number", "null"]},
                },
                "required": [
                    "research_status",
                    "research_probability",
                    "production_probability",
                ],
            },
        },
    }
    (output / "shadow-country-record-schema.json").write_text(
        json.dumps(schema, indent=2, allow_nan=False) + "\n"
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "phase5_shadow_bundle_frozen_prospective_confirmation_open",
        "forecast_batch_id": batch_id,
        "issued_at": issued_at,
        "source_cutoff": summary.source_cutoff.iloc[0],
        "state_origin_year": current_year,
        "forecast_years": sorted(map(int, summary.forecast_year.unique())),
        "horizons": [1, 2],
        "vintage_mode": VINTAGE_MODE,
        "research_commit": research_commit,
        "production_commit": production_commit,
        "research_countries": int(summary.entity_code.nunique()),
        "country_horizon_rows": len(summary),
        "production_reference_countries": int(
            production.country_code.nunique()
        ),
        "research_production_overlap": len(research_codes & production_codes),
        "research_only_countries": sorted(research_codes - production_codes),
        "production_only_countries": sorted(production_codes - research_codes),
        "provider_projection_rows_read_by_baseline": 0,
        "provider_projection_policy": "separate_scenario_lane_only",
        "research_crisis_overlay_status": CRISIS_STATUS,
        "research_crisis_probabilities_served": 0,
        "production_classifier": {
            "bytes": 52580,
            "sha256": (
                "054811a0b12133592bd22de64e2141c6c969d473963170199cff441e8e689aee"
            ),
            "action": "preserve_unchanged",
        },
        "prospective_confirmation_status": "open_pending_future_realizations",
        "weo_2025_status_caveat": WEO_STATUS_CAVEAT,
        "input_identity": input_identity,
    }
    output_hashes = {
        path.name: sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file()
        and path.name not in {"shadow-manifest.json", "file-checksums.json"}
    }
    manifest["files"] = output_hashes
    (output / "shadow-manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    all_hashes = {
        path.name: sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "file-checksums.json"
    }
    (output / "file-checksums.json").write_text(
        json.dumps(all_hashes, indent=2, allow_nan=False) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2", required=True)
    parser.add_argument("--phase3", required=True)
    parser.add_argument("--phase4", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--production-reference")
    parser.add_argument("--provider-projections")
    parser.add_argument("--issued-at", default="2026-09-24")
    parser.add_argument(
        "--research-commit",
        default="31968c9596d928df9a12e29b5ea2ea442853213d",
    )
    parser.add_argument(
        "--production-commit",
        default="5957ca779dafa21f2e098c819bfb060f43243206",
    )
    args = parser.parse_args()
    result = build_bundle(
        args.phase2,
        args.phase3,
        args.phase4,
        args.output,
        production_reference=args.production_reference,
        provider_projections=args.provider_projections,
        issued_at=args.issued_at,
        research_commit=args.research_commit,
        production_commit=args.production_commit,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
