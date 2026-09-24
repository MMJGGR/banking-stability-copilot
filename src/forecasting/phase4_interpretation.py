"""Phase 4A target-independent interpretation of the fixed Phase 2 state.

Applies an orthogonal varimax rotation to the fixed measurement loadings. The
rotation changes coordinates only: it preserves state distances, measurement
reconstruction and point-transition geometry when applied consistently.
"""
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd


STOPWORDS = {
    "and","of","to","the","in","for","on","from","total","other","all",
    "domestic","currency","percent","percentage","ratio","value","values",
    "assets","liabilities","financial","institutions","sector","sectors",
    "annual","index","current","net","gross","general","including",
    "excluding","basis","points","amount","claims","position","positions",
    "resident","residents","nonresident","nonresidents","depository",
    "corporations","corporation","bank","banks","banking","data",
}


def varimax(loadings: np.ndarray, gamma: float = 1.0, max_iter: int = 120,
            tol: float = 1e-5) -> tuple[np.ndarray, np.ndarray, dict]:
    """Orthogonal varimax rotation using SVD updates."""
    phi = np.asarray(loadings, dtype=float)
    if phi.ndim != 2 or not np.isfinite(phi).all():
        raise ValueError("finite 2D loadings required")
    p, k = phi.shape
    rotation = np.eye(k)
    objective_old = 0.0
    converged = False
    for iteration in range(1, max_iter + 1):
        rotated = phi @ rotation
        gradient = phi.T @ (
            rotated ** 3
            - (gamma / p) * rotated @ np.diag(np.sum(rotated ** 2, axis=0))
        )
        u, singular, vt = np.linalg.svd(gradient, full_matrices=False)
        rotation = u @ vt
        objective = float(np.sum(singular))
        if objective_old and objective - objective_old <= tol * max(1.0, objective_old):
            converged = True
            break
        objective_old = objective
    rotated = phi @ rotation
    return rotated, rotation, {
        "iterations": iteration,
        "converged": converged,
        "objective": objective_old,
        "orthogonality_max_abs_error": float(
            np.max(np.abs(rotation.T @ rotation - np.eye(k)))
        ),
    }


def _tokens(value: str) -> list[str]:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return [
        token for token in text.split()
        if len(token) > 2 and token not in STOPWORDS and not token.isdigit()
    ]


def candidate_labels(top: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    """Generate explicitly provisional labels from top-loading metadata."""
    docs: dict[str, set[str]] = {}
    per_dim: dict[str, list[str]] = {}
    for dim in dimensions:
        rows = top.loc[top.dimension.eq(dim)]
        tokens = []
        for label in rows.indicator_label.fillna(rows.INDICATOR).fillna(""):
            tokens.extend(_tokens(label))
        per_dim[dim] = tokens
        for token in set(tokens):
            docs.setdefault(token, set()).add(dim)
    n_docs = max(len(dimensions), 1)
    records = []
    for dim in dimensions:
        counts = pd.Series(per_dim[dim]).value_counts() if per_dim[dim] else pd.Series(dtype=float)
        scored = []
        for token, count in counts.items():
            idf = np.log((1 + n_docs) / (1 + len(docs.get(token, ())))) + 1
            scored.append((float(count * idf), token))
        words = [token for _, token in sorted(scored, reverse=True)[:3]]
        records.append({
            "dimension": dim,
            "descriptive_candidate_label": " / ".join(words) if words else "mixed observable structure",
            "label_status": "machine_generated_requires_analyst_review",
        })
    return pd.DataFrame(records)


def run_interpretation(loadings_path: str | Path, states_path: str | Path,
                       latest_quantiles_path: str | Path, output: str | Path,
                       top_n: int = 25) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    loadings = pd.read_csv(loadings_path)
    states = pd.read_csv(states_path)
    quantiles = pd.read_csv(latest_quantiles_path)
    state_cols = sorted(
        [c for c in loadings if c.startswith("state_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    if len(state_cols) != 96:
        raise ValueError(f"expected 96 state dimensions, found {len(state_cols)}")
    matrix = loadings[state_cols].to_numpy(dtype=float)
    rotated, rotation, diagnostics = varimax(matrix)
    dims = [f"interpreted_state_{i+1}" for i in range(len(state_cols))]

    rotated_loadings = pd.DataFrame(rotated, columns=dims)
    metadata = loadings.drop(columns=state_cols).reset_index(drop=True)
    rotated_loadings = pd.concat([metadata, rotated_loadings], axis=1)
    rotated_loadings.to_csv(output / "rotated-measurement-loadings.csv.gz", index=False)
    pd.DataFrame(rotation, index=state_cols, columns=dims).to_csv(
        output / "interpretation-rotation.csv"
    )

    top_records = []
    summary = []
    for j, dim in enumerate(dims):
        value = rotated[:, j]
        order = np.argsort(np.abs(value))[::-1]
        chosen = order[:top_n]
        for rank, idx in enumerate(chosen, start=1):
            row = metadata.iloc[idx]
            top_records.append({
                "dimension": dim,
                "rank_by_absolute_loading": rank,
                "loading": float(value[idx]),
                "loading_direction": "positive" if value[idx] >= 0 else "negative",
                **{c: row.get(c) for c in [
                    "predictor_id","feature_id","source","INDICATOR",
                    "indicator_label","UNIT","unit_policy","observed_cells",
                    "measurement_state",
                ]},
            })
        energy = value ** 2
        total = float(energy.sum())
        share = energy / total if total > 0 else np.zeros_like(energy)
        effective = 1.0 / float(np.sum(share ** 2)) if np.any(share) else 0.0
        source_share = (
            pd.DataFrame({"source": metadata.source.fillna("UNKNOWN"), "energy": energy})
            .groupby("source", observed=True).energy.sum()
            .div(total if total else 1.0)
            .sort_values(ascending=False)
        )
        representation = metadata.predictor_id.astype(str).str.rsplit("::", n=1).str[-1]
        representation_share = (
            pd.DataFrame({"representation": representation, "energy": energy})
            .groupby("representation", observed=True).energy.sum()
            .div(total if total else 1.0)
            .sort_values(ascending=False)
        )
        summary.append({
            "dimension": dim,
            "effective_contributing_representations": effective,
            "top_10_absolute_loading_energy_share": float(np.sort(share)[-10:].sum()),
            "dominant_source": source_share.index[0] if len(source_share) else None,
            "dominant_source_energy_share": float(source_share.iloc[0]) if len(source_share) else None,
            "source_energy_shares": ";".join(f"{k}:{v:.6f}" for k,v in source_share.items()),
            "representation_energy_shares": ";".join(f"{k}:{v:.6f}" for k,v in representation_share.items()),
        })
    top = pd.DataFrame(top_records)
    labels = candidate_labels(top, dims)
    dimension_summary = pd.DataFrame(summary).merge(labels, on="dimension", validate="one_to_one")
    top.to_csv(output / "dimension-top-loadings.csv.gz", index=False)
    dimension_summary.to_csv(output / "dimension-interpretation-summary.csv", index=False)

    z = states[state_cols].to_numpy(dtype=float)
    z_rot = z @ rotation
    rotated_states = states[["entity_code","forecast_origin_year"]].copy()
    rotated_states[dims] = z_rot
    rotated_states.to_csv(output / "rotated-country-year-states.csv.gz", index=False)
    rng = np.random.default_rng(17)
    sample_idx = np.sort(rng.choice(len(z), size=min(250, len(z)), replace=False))
    sample = z[sample_idx]
    sample_rot = z_rot[sample_idx]
    before = np.sqrt(np.maximum(
        (sample ** 2).sum(1)[:,None] + (sample ** 2).sum(1)[None,:] - 2*sample@sample.T,
        0,
    ))
    after = np.sqrt(np.maximum(
        (sample_rot ** 2).sum(1)[:,None] + (sample_rot ** 2).sum(1)[None,:] - 2*sample_rot@sample_rot.T,
        0,
    ))
    diagnostics["sample_pairwise_distance_max_abs_difference"] = float(np.max(np.abs(before-after)))
    diagnostics["measurement_term_max_abs_difference"] = float(
        np.max(np.abs((z[:100] @ matrix[:200].T) - (z_rot[:100] @ rotated[:200].T)))
    )

    latest_year = int(states.forecast_origin_year.max())
    latest = states.loc[states.forecast_origin_year.eq(latest_year), ["entity_code", *state_cols]].copy()
    latest_rot = latest[state_cols].to_numpy(dtype=float) @ rotation
    current_long = []
    for i, entity in enumerate(latest.entity_code):
        order = np.argsort(np.abs(latest_rot[i]))[::-1][:8]
        for rank, j in enumerate(order, start=1):
            current_long.append({
                "entity_code": entity,
                "current_state_year": latest_year,
                "rank": rank,
                "dimension": dims[j],
                "coordinate": float(latest_rot[i,j]),
                "contribution_direction": "positive" if latest_rot[i,j] >= 0 else "negative",
            })
    current_explanation = pd.DataFrame(current_long).merge(labels, on="dimension", how="left")
    current_explanation.to_csv(output / "latest-country-state-explanations.csv.gz", index=False)

    points = quantiles.pivot_table(
        index=["entity_code","current_state_year","forecast_year","horizon"],
        columns="state_coordinate", values="point", aggfunc="first"
    ).reset_index()
    points = points.merge(latest, on="entity_code", how="inner", validate="many_to_one", suffixes=("_future","_current"))
    future_cols = [f"state_{i}_future" for i in range(1,97)]
    current_cols = [f"state_{i}_current" for i in range(1,97)]
    if not all(c in points for c in [*future_cols, *current_cols]):
        raise ValueError("latest point/current state columns are incomplete")
    future_matrix = points[future_cols].to_numpy(dtype=float)
    current_matrix = points[current_cols].to_numpy(dtype=float)
    movement_rot = (future_matrix - current_matrix) @ rotation
    movement_records = []
    for i, row in points.iterrows():
        order = np.argsort(np.abs(movement_rot[i]))[::-1][:8]
        for rank, j in enumerate(order, start=1):
            movement_records.append({
                "entity_code": row.entity_code,
                "current_state_year": int(row.current_state_year),
                "forecast_year": int(row.forecast_year),
                "horizon": int(row.horizon),
                "rank": rank,
                "dimension": dims[j],
                "expected_coordinate_change": float(movement_rot[i,j]),
                "movement_direction": "increase" if movement_rot[i,j] >= 0 else "decrease",
            })
    movement_explanation = pd.DataFrame(movement_records).merge(labels, on="dimension", how="left")
    movement_explanation.to_csv(output / "latest-country-movement-explanations.csv.gz", index=False)

    report = {
        "status": "completed_phase4a_geometry_preserving_interpretation",
        "state_dimensions": len(state_cols),
        "features_in_rotation": len(loadings),
        "country_year_states": len(states),
        "latest_state_year": latest_year,
        "rotation": diagnostics,
        "labels_are_model_constraints": False,
        "labels_require_analyst_review": True,
        "production_modified": False,
    }
    pd.Series(report).to_json(output / "phase4a-summary.json", indent=2)
    return report


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--loadings", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--latest-quantiles", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run_interpretation(args.loadings,args.states,args.latest_quantiles,args.output),indent=2))
