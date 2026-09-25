import numpy as np
import pandas as pd

from src.forecasting.phase6_score import (
    historical_percentile,
    latest_components,
    select_weights,
)
from src.forecasting.phase6_sovereign import (
    EXCLUDED_NONCURRENT_ENTITIES,
    _derive_episodes,
    country_name_to_iso3,
    find_data_header,
)
from src.forecasting.phase6_execution import replacement_gate


def test_sovereign_rectangular_header_ignores_metadata_rows():
    raw = pd.DataFrame([
        ["SERIES", None, None, None, None],
        ["DEBT_COUNTRY", "COUNTRY", "COUNTRY", None, None],
        ["DEBT_YEAR", "YEAR", "YEAR", None, None],
        ["DEBT_COUNTRY", "DEBT_YEAR", "DEBT_TOTAL_2025", "DEBT_FISCAL_ARREARS_2025", "DEBT_TOTAL_DEBT_2025"],
        ["Kenya", 2020, 4.0, 1.0, 50.0],
    ])
    row, mapping = find_data_header(raw)
    assert row == 3
    assert mapping["country_name"] == 0
    assert mapping["year"] == 1
    assert mapping["primary_default_stock_usd_m"] == 2
    assert mapping["domestic_arrears_usd_m"] == 3


def test_country_name_mapping_and_episode_derivation():
    assert country_name_to_iso3("Kenya") == "KEN"
    assert country_name_to_iso3("Côte d’Ivoire") == "CIV"
    assert country_name_to_iso3("Bosnia & Herzegovina") == "BIH"
    assert country_name_to_iso3("Dem. Rep. of Congo (Kinshasa)") == "COD"
    assert country_name_to_iso3("Kosovo") == "XKX"
    assert country_name_to_iso3("São Tomé and Príncipe") == "STP"
    assert country_name_to_iso3("eSwatini (Swaziland)") == "SWZ"
    assert country_name_to_iso3("Czechoslovakia") is None
    assert "CZECHOSLOVAKIA" in EXCLUDED_NONCURRENT_ENTITIES
    status = pd.DataFrame({
        "country_code": ["AAA"] * 7 + ["BBB"] * 3,
        "year": [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2000, 2001, 2002],
        "active": [False, True, True, False, True, True, False, False, False, False],
    })
    episodes = _derive_episodes(status, "active")
    assert episodes[["start_year", "end_year"]].to_records(index=False).tolist() == [
        (2001, 2002),
        (2004, 2005),
    ]


def test_own_history_percentile_uses_only_prior_origins():
    signal = pd.Series([0.8, 0.2, 0.9, 0.1])
    countries = pd.Series(["A", "A", "A", "A"])
    origins = pd.Series([2000, 2001, 2002, 2003])
    result = historical_percentile(
        signal, countries, origins, shrinkage_observations=0
    )
    assert result.iloc[0] == 0.5
    assert result.iloc[1] == 0.0
    assert result.iloc[2] == 1.0
    assert result.iloc[3] == 0.0


def test_score_weight_selection_keeps_all_owner_dimensions():
    frame = pd.DataFrame({
        "peer": [0.1, 0.2, 0.8, 0.9],
        "history": [0.2, 0.1, 0.7, 0.8],
        "absolute": [0.05, 0.10, 0.60, 0.70],
        "y": [0, 0, 1, 1],
    })
    selected = select_weights(frame)
    assert np.isclose(
        selected["peer_weight"]
        + selected["history_weight"]
        + selected["absolute_weight"],
        1.0,
    )
    assert min(
        selected["peer_weight"],
        selected["history_weight"],
        selected["absolute_weight"],
    ) >= 0.10


def test_latest_components_keep_peer_history_and_absolute_visible():
    history = pd.DataFrame({
        "country": ["A", "A", "B", "B"],
        "origin": [2000, 2001, 2000, 2001],
        "proba": [0.2, 0.4, 0.6, 0.5],
    })
    result = latest_components(
        pd.Series([0.7, 0.3]),
        pd.Series(["A", "B"]),
        2002,
        history,
        {"peer_weight": 0.3, "history_weight": 0.3, "absolute_weight": 0.4},
    )
    assert set(["peer", "history", "absolute", "replacement_score"]) <= set(result)
    assert result.replacement_score.between(1, 10).all()


def test_replacement_gate_never_authorizes_production_without_confirmation():
    rows = []
    for target in ("banking", "sovereign", "either"):
        rows.extend([
            {
                "target": target,
                "model": "event_rate",
                "brier": 0.10,
                "log_loss": 0.35,
            },
            {
                "target": target,
                "model": "combined",
                "brier": 0.08,
                "log_loss": 0.30,
            },
        ])
    result = replacement_gate(
        pd.DataFrame(rows),
        {"score_event_rate_monotonic": True},
    )
    assert result["development_gate_passed"] is True
    assert result["replacement_authorized"] is False
    assert result["prospective_confirmation"].startswith("pending")
