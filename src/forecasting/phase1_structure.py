"""Phase 1: target-independent structural and trajectory discovery.

No supervised target, crisis label, serving score, pillar score or production
classifier output is read. The phase discovers variance structure, redundancy,
temporal stability and entity trajectories from the full eligible information
library. See docs/prd/broad-feature-phase-plan-v0.5.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import svdvals

from .discovery import BroadSpectralSpace
from .discovery_execution import prepare_matrix, verify_snapshot, write_json
from .inventory import ForecastDataError
from .redundancy import WeightedReferenceGeometry
from src.scripts.audit_forecasting_sources import output_destination, sha256


SOURCES = ("FSIC", "FSIBSIS", "MFS", "WEO", "WGI")


@dataclass
class StructurePanel:
    predictors: pd.DataFrame
    context: pd.DataFrame
    summary: dict


def build_structure_panel(
    endpoints: pd.DataFrame,
    member_entities: set[str],
    *,
    information_lag_years: int = 1,
    max_origin_year: int | None = None,
) -> StructurePanel:
    """Build an uncapped annual structural panel with no target construction.

    Exact year-end observations only. Level, one-year lag and one-year change
    representations are retained when calculable. No supervised horizon
    determines the final origin year.
    """
    required = {
        "entity_code", "feature_id", "observation_period", "observation_year",
        "value", "break_in_year", "retrospective_eligible",
    }
    if not required <= set(endpoints):
        raise ForecastDataError(f"Incomplete annual endpoints: {sorted(required-set(endpoints))}")
    if not member_entities:
        raise ForecastDataError("Member-entity set cannot be empty")
    if not isinstance(information_lag_years, int) or isinstance(information_lag_years, bool) or information_lag_years < 1:
        raise ForecastDataError("Positive integer information lag required")

    e = endpoints.copy()
    e["entity_code"] = e.entity_code.astype(str).str.strip()
    e["observation_period"] = pd.to_datetime(e.observation_period, errors="raise")
    e["observation_year"] = pd.to_numeric(e.observation_year, errors="raise").astype(int)
    if e.duplicated(["entity_code", "feature_id", "observation_year"]).any():
        raise ForecastDataError("Nonunique annual endpoint")
    exact = pd.to_datetime(e.observation_year.astype(str) + "-12-31")
    if not e.observation_period.eq(exact).all():
        raise ForecastDataError("Phase 1 requires exact year-end endpoints")
    valid = e.loc[e.retrospective_eligible.astype(bool) & np.isfinite(pd.to_numeric(e.value, errors="coerce"))].copy()
    valid["value"] = pd.to_numeric(valid.value, errors="raise").astype(float)

    fields = [
        "entity_code", "feature_id", "observation_year", "observation_period",
        "value", "break_in_year",
    ]
    level = valid[fields].copy()
    level["forecast_origin_year"] = level.observation_year + information_lag_years
    level["representation"] = "level"

    lag = valid[fields].copy()
    lag["forecast_origin_year"] = lag.observation_year + information_lag_years + 1
    lag["representation"] = "lag_1y"

    older = valid[["entity_code", "feature_id", "observation_year", "value", "break_in_year"]].copy()
    older["observation_year"] += 1
    joined = valid.merge(
        older,
        on=["entity_code", "feature_id", "observation_year"],
        suffixes=("", "_prior"),
        validate="one_to_one",
    )
    delta = joined.loc[
        ~joined.break_in_year.astype(bool) & ~joined.break_in_year_prior.astype(bool),
        fields + ["value_prior"],
    ].copy()
    delta["value"] = delta.value - delta.pop("value_prior")
    delta["forecast_origin_year"] = delta.observation_year + information_lag_years
    delta["representation"] = "change_1y"

    values = pd.concat([level, lag, delta], ignore_index=True)
    if max_origin_year is not None:
        if not isinstance(max_origin_year, int) or isinstance(max_origin_year, bool):
            raise ForecastDataError("max_origin_year must be an integer or None")
        values = values.loc[values.forecast_origin_year <= max_origin_year].copy()
    values["predictor_id"] = values.feature_id + "::" + values.representation
    values["information_age_years"] = values.forecast_origin_year - values.observation_year
    values["experiment_mode"] = "retrospective_target_independent_structure"
    values["admission_state"] = "staged_not_supervised_selected"
    if (values.information_age_years < information_lag_years).any():
        raise ForecastDataError("Information-age invariant failed")
    values = values.sort_values(["entity_code", "forecast_origin_year", "predictor_id"]).reset_index(drop=True)
    if values.duplicated(["entity_code", "forecast_origin_year", "predictor_id"]).any():
        raise ForecastDataError("Duplicate Phase 1 predictor cell")

    predictors = values.loc[values.entity_code.isin(member_entities)].copy()
    context = values.loc[~values.entity_code.isin(member_entities)].copy()
    years = predictors.forecast_origin_year
    summary = {
        "status": "phase1_target_independent_structure_panel",
        "information_lag_years_assumed": information_lag_years,
        "maximum_origin_year": max_origin_year,
        "member_entities": len(member_entities),
        "entities_with_predictors": int(predictors.entity_code.nunique()),
        "context_entities": int(context.entity_code.nunique()),
        "earliest_origin": int(years.min()) if len(years) else None,
        "latest_origin": int(years.max()) if len(years) else None,
        "source_features": int(predictors.feature_id.nunique()),
        "predictor_representations": int(predictors.predictor_id.nunique()),
        "observed_predictor_cells": len(predictors),
        "targets_read": 0,
        "risk_labels_read": 0,
        "feature_count_cap": None,
    }
    return StructurePanel(predictors, context, summary)


def _common_variance_contribution(space: BroadSpectralSpace) -> pd.Series:
    loadings = space.loadings()
    k = space.n_components_
    values = (loadings[:, :k] ** 2) @ space.variance_ratios_[:k]
    return pd.Series(values, index=space.transformer_.active_features_, name="common_variance_share")


def _basis_on_common_features(space: BroadSpectralSpace, common: list[str], k: int) -> np.ndarray:
    idx = {name: i for i, name in enumerate(space.transformer_.active_features_)}
    rows = [idx[name] for name in common]
    a = space.loadings()[rows, :k]
    if a.size == 0:
        return np.empty((len(common), 0))
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    if not len(s):
        return np.empty((len(common), 0))
    tol = max(float(s[0]), 1.0) * np.finfo(float).eps * max(a.shape)
    return u[:, s > tol]


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a, b) / (na * nb))


def yearly_structure(x: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, BroadSpectralSpace], pd.DataFrame]:
    """Fit uncapped cross-sectional structure independently for every origin."""
    years = sorted(set(map(int, x.index.get_level_values("forecast_origin_year"))))
    rows = []
    spaces: dict[int, BroadSpectralSpace] = {}
    contributions = []
    for year in years:
        frame = x.xs(year, level="forecast_origin_year")
        if len(frame) < 2 or frame.notna().sum().sum() == 0:
            continue
        try:
            space = BroadSpectralSpace(.9).fit(frame)
        except ForecastDataError:
            continue
        spaces[year] = space
        rows.append({
            "forecast_origin_year": year,
            "entities": len(frame),
            "supplied_representations": len(frame.columns),
            "learnable_representations": len(space.transformer_.active_features_),
            "numerical_rank": len(space.eigenvalues_),
            "components_80": space.components_for(.8),
            "components_90": space.components_for(.9),
            "components_95": space.components_for(.95),
            "first_component_share": float(space.variance_ratios_[0]),
        })
        c = _common_variance_contribution(space).rename("common_variance_share").reset_index()
        c.columns = ["predictor_id", "common_variance_share"]
        c["forecast_origin_year"] = year
        contributions.append(c)
    if not rows:
        raise ForecastDataError("No annual structure could be fitted")
    return pd.DataFrame(rows), spaces, pd.concat(contributions, ignore_index=True)


def temporal_subspace_stability(
    spaces: dict[int, BroadSpectralSpace],
    contributions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare every pair of annual data-selected subspaces on common features."""
    years = sorted(spaces)
    by_year = {
        year: contributions.loc[contributions.forecast_origin_year.eq(year)]
        .set_index("predictor_id").common_variance_share
        for year in years
    }
    rows = []
    for i, ya in enumerate(years):
        a = spaces[ya]
        active_a = set(a.transformer_.active_features_)
        for yb in years[i + 1:]:
            b = spaces[yb]
            common = sorted(active_a & set(b.transformer_.active_features_))
            if len(common) < 2:
                continue
            k = min(a.n_components_, b.n_components_, len(common))
            qa = _basis_on_common_features(a, common, k)
            qb = _basis_on_common_features(b, common, k)
            r = min(qa.shape[1], qb.shape[1])
            if r == 0:
                continue
            singular = np.clip(svdvals(qa[:, :r].T @ qb[:, :r]), 0, 1)
            ca = by_year[ya].reindex(common).to_numpy()
            cb = by_year[yb].reindex(common).to_numpy()
            cc = _cosine(ca, cb)
            rows.append({
                "year_a": ya,
                "year_b": yb,
                "gap_years": yb - ya,
                "common_learnable_features": len(common),
                "compared_subspace_rank": r,
                "mean_squared_canonical_cosine": float(np.mean(singular ** 2)),
                "minimum_canonical_cosine": float(np.min(singular)),
                "mean_principal_angle_degrees": float(np.degrees(np.arccos(singular)).mean()),
                "feature_contribution_cosine": cc,
            })
    return pd.DataFrame(rows)


def feature_stability(contributions: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Summarize target-independent variance contribution persistence."""
    g = contributions.groupby("predictor_id", observed=True).common_variance_share
    out = g.agg(
        years_learnable="size",
        mean_common_variance_share="mean",
        median_common_variance_share="median",
        min_common_variance_share="min",
        max_common_variance_share="max",
        std_common_variance_share="std",
    ).reset_index()
    first = contributions.groupby("predictor_id").forecast_origin_year.min().rename("first_learnable_year")
    last = contributions.groupby("predictor_id").forecast_origin_year.max().rename("last_learnable_year")
    out = out.merge(first, on="predictor_id").merge(last, on="predictor_id")
    keep = [c for c in ["predictor_id", "feature_id", "source", "INDICATOR", "indicator_label", "UNIT", "unit_policy"] if c in ledger]
    return out.merge(ledger[keep].drop_duplicates("predictor_id"), on="predictor_id", how="left", validate="one_to_one")


def frozen_trajectories(x: pd.DataFrame, reference_year: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Map all origins through one frozen, target-independent reference space."""
    ref = x.xs(reference_year, level="forecast_origin_year")
    space = BroadSpectralSpace(.9).fit(ref)
    k = space.n_components_
    eigscale = np.sqrt(space.eigenvalues_[:k] / max(len(ref) - 1, 1))
    records = []
    coord_frames = []
    for year in sorted(set(map(int, x.index.get_level_values("forecast_origin_year")))):
        frame = x.xs(year, level="forecast_origin_year")
        z = space.transformer_.transform(frame)
        scores = z @ space.loadings()[:, :k]
        standardized = scores / eigscale
        observed = frame[space.transformer_.active_features_].notna().sum(axis=1).to_numpy()
        coords = pd.DataFrame(standardized, index=frame.index, columns=[f"state_{i+1}" for i in range(k)])
        coords["forecast_origin_year"] = year
        coords["observed_fitted_features"] = observed
        coords["observed_share"] = observed / len(space.transformer_.active_features_)
        coord_frames.append(coords.reset_index())
    coordinates = pd.concat(coord_frames, ignore_index=True)
    state_cols = [c for c in coordinates if c.startswith("state_")]
    coordinates = coordinates.sort_values(["entity_code", "forecast_origin_year"]).reset_index(drop=True)

    for entity, d in coordinates.groupby("entity_code", sort=True):
        d = d.sort_values("forecast_origin_year")
        prev_vec = None
        prev_year = None
        prev_velocity = None
        for _, row in d.iterrows():
            vec = row[state_cols].to_numpy(dtype=float)
            velocity = None
            displacement = acceleration = direction = None
            if prev_vec is not None and row.forecast_origin_year == prev_year + 1:
                velocity = vec - prev_vec
                displacement = float(np.sqrt(np.mean(velocity ** 2)))
                if prev_velocity is not None:
                    acceleration = float(np.sqrt(np.mean((velocity - prev_velocity) ** 2)))
                    direction = _cosine(velocity, prev_velocity)
            records.append({
                "entity_code": entity,
                "forecast_origin_year": int(row.forecast_origin_year),
                "observed_fitted_features": int(row.observed_fitted_features),
                "observed_share": float(row.observed_share),
                "reference_center_distance": float(np.sqrt(np.mean(vec ** 2))),
                "year_on_year_displacement": displacement,
                "acceleration": acceleration,
                "direction_persistence_cosine": direction,
            })
            prev_vec = vec
            prev_year = int(row.forecast_origin_year)
            prev_velocity = velocity

    trajectory = pd.DataFrame(records)

    # Nearest same-year peer in the frozen state space. No future outcome enters.
    peers = []
    for year, d in coordinates.groupby("forecast_origin_year", sort=True):
        if len(d) < 2:
            continue
        a = d[state_cols].to_numpy(dtype=float)
        dist2 = np.maximum(
            (a * a).sum(axis=1)[:, None] + (a * a).sum(axis=1)[None, :] - 2 * a @ a.T,
            0,
        )
        np.fill_diagonal(dist2, np.inf)
        idx = np.argmin(dist2, axis=1)
        distance = np.sqrt(dist2[np.arange(len(d)), idx] / len(state_cols))
        entities = d.entity_code.to_numpy()
        coverage = d.observed_share.to_numpy()
        for i in range(len(d)):
            peers.append({
                "entity_code": entities[i],
                "forecast_origin_year": int(year),
                "nearest_peer_entity": entities[idx[i]],
                "state_distance": float(distance[i]),
                "entity_observed_share": float(coverage[i]),
                "peer_observed_share": float(coverage[idx[i]]),
                "coverage_difference": float(abs(coverage[i] - coverage[idx[i]])),
            })

    summary = {
        "reference_year": reference_year,
        "reference_entities": len(ref),
        "reference_learnable_representations": len(space.transformer_.active_features_),
        "reference_components_90": k,
        "trajectory_rows": len(trajectory),
        "peer_rows": len(peers),
        "targets_read": 0,
        "risk_scores_read": 0,
    }
    return trajectory, pd.DataFrame(peers), summary


def redundancy_sensitivity(x: pd.DataFrame, ledger: pd.DataFrame, reference_year: int, output: Path) -> dict:
    """Compare unweighted/profile/source-balanced geometry without feature deletion."""
    ref = x.xs(reference_year, level="forecast_origin_year")
    base = BroadSpectralSpace(.9).fit(ref)
    cols = pd.Index(base.transformer_.active_features_, name="predictor_id")
    z = pd.DataFrame(base.transformer_.transform(ref), index=ref.index, columns=cols)
    mask = ref[cols].notna()
    sources = ledger.set_index("predictor_id").source.reindex(cols)
    if sources.isna().any():
        raise ForecastDataError("Missing source identity in redundancy sensitivity")
    rows = []
    for mode in ("unweighted", "exact_profile", "source_energy"):
        g = WeightedReferenceGeometry(mode).fit(z, mask, sources)
        g.weight_ledger_.to_csv(output / f"{mode}-weight-ledger.csv")
        g.spectrum().to_csv(output / f"{mode}-spectrum.csv", index=False)
        contrib = g.contributions()
        contrib.to_csv(output / f"{mode}-contributions.csv")
        rows.append({
            "mode": mode,
            "features_retained": g.features_retained_,
            "features_deleted": 0,
            "components_80": g.components_for(.8),
            "components_90": g.components_for(.9),
            "components_95": g.components_for(.95),
            "first_component_share": float(g.variance_ratios_[0]),
            "unique_exact_profiles": int(g.weight_ledger_.profile_sha256.nunique()),
        })
    return {"specifications": rows, "feature_count_cap": None}


def run(snapshot: Path, output: Path) -> dict:
    snapshot = Path(snapshot).resolve()
    output = output_destination(snapshot, Path(output))
    _, prior, verified = verify_snapshot(snapshot)
    output.mkdir(parents=True)

    registry = pd.read_csv(snapshot / "complete-registry.csv", low_memory=False)
    # Existing M2 predictors are used ONLY to recover the research member-entity
    # set. Phase 1 does not inherit the old end year or read any target file.
    existing = pd.read_parquet(snapshot / "panel" / "predictors.parquet", columns=["entity_code"])
    members = set(existing.entity_code.astype(str).unique())

    endpoint_frames = []
    for source in SOURCES:
        path = snapshot / source / "annual-endpoints.parquet"
        if not path.is_file():
            raise ForecastDataError(f"Missing annual endpoint store: {source}")
        frame = pd.read_parquet(path)
        frame["source"] = source
        endpoint_frames.append(frame)
    endpoints = pd.concat(endpoint_frames, ignore_index=True)

    cutoff_year = int(pd.Timestamp(prior["cutoff"]).year)
    panel = build_structure_panel(endpoints, members, max_origin_year=cutoff_year)
    panel.predictors.to_parquet(output / "phase1-predictors.parquet", index=False)
    panel.context.to_parquet(output / "phase1-context.parquet", index=False)
    write_json(output / "panel-summary.json", panel.summary)

    x, ledger, preprocessing = prepare_matrix(panel.predictors, registry)
    ledger.to_csv(output / "all-predictor-ledger.csv", index=False)

    annual, spaces, contributions = yearly_structure(x)
    annual.to_csv(output / "yearly-structure.csv", index=False)
    contributions.to_parquet(output / "yearly-feature-contributions.parquet", index=False)

    stability = temporal_subspace_stability(spaces, contributions)
    stability.to_csv(output / "subspace-stability.csv", index=False)

    feature = feature_stability(contributions, ledger)
    feature.to_csv(output / "feature-stability.csv", index=False)

    reference_year = int(annual.forecast_origin_year.max())
    trajectory, peers, trajectory_summary = frozen_trajectories(x, reference_year)
    trajectory.to_csv(output / "entity-trajectories.csv", index=False)
    peers.to_csv(output / "nearest-peer-geometry.csv", index=False)

    redundancy = redundancy_sensitivity(x, ledger, reference_year, output)

    report = {
        "status": "completed_phase1_target_independent_structure",
        "source_cutoff": prior["cutoff"],
        "source_artifact": 10463106988,
        "M2_files_verified": verified,
        "phase1_panel": panel.summary,
        "preprocessing": preprocessing,
        "reference_year": reference_year,
        "yearly_origins_fitted": int(len(annual)),
        "earliest_fitted_origin": int(annual.forecast_origin_year.min()),
        "latest_fitted_origin": int(annual.forecast_origin_year.max()),
        "feature_stability_rows": len(feature),
        "subspace_comparisons": len(stability),
        "trajectory": trajectory_summary,
        "redundancy": redundancy,
        "targets_read": 0,
        "crisis_labels_read": 0,
        "production_scores_read": 0,
        "production_modified": False,
        "crisis_classifier_retrained": False,
        "final_confirmation_evaluated": False,
        "limitations": [
            "Retrospective latest-vintage structural research; historical public-release vintages remain unverified.",
            "Variance, redundancy, distinctness and temporal stability are not predictive skill.",
            "The frozen reference geometry is descriptive and does not constitute a risk score.",
            "Context/global entities are retained separately and are not counted as country trajectories.",
        ],
    }
    write_json(output / "phase1-summary.json", report)
    checks = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    write_json(output / "output-checksums.json", checks)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.snapshot, args.output)
    print("PHASE1_COMPLETE", json.dumps({
        "latest_origin": report["latest_fitted_origin"],
        "yearly_origins": report["yearly_origins_fitted"],
        "feature_stability_rows": report["feature_stability_rows"],
        "trajectory_rows": report["trajectory"]["trajectory_rows"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
