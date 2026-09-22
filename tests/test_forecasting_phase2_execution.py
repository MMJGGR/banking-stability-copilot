from pathlib import Path


def test_phase2a_execution_is_target_independent_by_construction():
    source = Path(
        "src/forecasting/phase2_measurement_execution.py"
    ).read_text()
    assert "target-pairs.csv" not in source
    assert "crisis_classifier.pkl" not in source
    assert "risk_model.pkl" not in source
    assert "TARGETS" not in source


def test_phase2a_execution_stops_if_rank_grid_boundary_is_best():
    source = Path(
        "src/forecasting/phase2_measurement_execution.py"
    ).read_text()
    assert "phase2a_rank_search_incomplete_expand_boundary" in source
    assert "expand_rank_grid_before_fitting_final_measurement_state" in source


def test_phase2a_does_not_start_transition_modelling():
    source = Path(
        "src/forecasting/phase2_measurement_execution.py"
    ).read_text()
    assert '"phase2b_started": False' in source
