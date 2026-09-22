import numpy as np
import pandas as pd

from src.forecasting.phase1_structure import (
    build_structure_panel,
    yearly_structure,
    temporal_subspace_stability,
    frozen_trajectories,
)


def endpoints():
    rows=[]
    for entity,shift in [("AAA",0.),("BBB",1.),("CTX",5.)]:
        for feature,mult in [("f1",1.),("f2",2.),("f3",.5)]:
            for year in [2022,2023,2024,2025]:
                rows.append({
                    "entity_code":entity,
                    "feature_id":feature,
                    "observation_period":pd.Timestamp(f"{year}-12-31"),
                    "observation_year":year,
                    "value":mult*(year-2020)+shift,
                    "break_in_year":False,
                    "retrospective_eligible":True,
                })
    return pd.DataFrame(rows)


def test_phase1_panel_has_no_supervised_boundary_and_caps_at_cutoff():
    result=build_structure_panel(
        endpoints(),
        {"AAA","BBB"},
        information_lag_years=1,
        max_origin_year=2026,
    )
    assert result.summary["targets_read"] == 0
    assert result.summary["risk_labels_read"] == 0
    assert result.summary["feature_count_cap"] is None
    assert result.summary["latest_origin"] == 2026
    assert result.predictors.forecast_origin_year.max() == 2026
    assert set(result.context.entity_code) == {"CTX"}
    assert set(result.predictors.representation) == {"level","lag_1y","change_1y"}


def test_phase1_panel_does_not_bridge_calendar_gap_for_change():
    d=endpoints()
    d=d[~((d.entity_code=="AAA")&(d.feature_id=="f1")&(d.observation_year==2023))]
    result=build_structure_panel(d,{"AAA","BBB"},max_origin_year=2026)
    bad=result.predictors[
        (result.predictors.entity_code=="AAA")&
        (result.predictors.feature_id=="f1")&
        (result.predictors.representation=="change_1y")&
        (result.predictors.observation_year==2024)
    ]
    assert bad.empty


def wide_panel():
    rng=np.random.default_rng(17)
    entities=["A","B","C","D","E","F"]
    rows=[]; idx=[]
    for year in [2023,2024]:
        base=rng.normal(size=(len(entities),5))
        if year==2024:
            base[:,0]+=0.1*base[:,1]
        for i,e in enumerate(entities):
            rows.append(base[i]);idx.append((e,year))
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx,names=["entity_code","forecast_origin_year"]),
        columns=[f"x{i}" for i in range(5)],
    )


def test_yearly_structure_and_stability_are_target_independent():
    x=wide_panel()
    annual,spaces,contrib=yearly_structure(x)
    assert set(annual.forecast_origin_year)=={2023,2024}
    assert annual.learnable_representations.eq(5).all()
    stability=temporal_subspace_stability(spaces,contrib)
    assert len(stability)==1
    assert stability.common_learnable_features.iloc[0]==5
    assert 0 <= stability.mean_squared_canonical_cosine.iloc[0] <= 1


def test_frozen_trajectory_outputs_peer_geometry_without_risk_score():
    x=wide_panel()
    trajectory,peers,summary=frozen_trajectories(x,2024)
    assert summary["targets_read"]==0
    assert summary["risk_scores_read"]==0
    assert len(trajectory)==12
    assert len(peers)==12
    assert "risk_score" not in trajectory.columns
    assert peers.nearest_peer_entity.notna().all()


def test_column_permutation_does_not_change_yearly_spectrum():
    x=wide_panel()
    a,_,_=yearly_structure(x)
    b,_,_=yearly_structure(x[x.columns[::-1]])
    cols=["forecast_origin_year","components_80","components_90","components_95","first_component_share"]
    pd.testing.assert_frame_equal(
        a[cols].reset_index(drop=True),
        b[cols].reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )


def test_phase1_source_has_no_target_file_dependency():
    import ast
    from pathlib import Path
    source=Path("src/forecasting/phase1_structure.py").read_text()
    tree=ast.parse(source)
    imported=[]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append((node.module or "") + ":" + ",".join(alias.name for alias in node.names))
    assert "target-pairs.csv" not in source
    assert all("TARGETS" not in item for item in imported)
    assert all("crisis_labels" not in item for item in imported)
    assert all("crisis_classifier" not in item for item in imported)
