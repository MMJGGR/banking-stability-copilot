from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.phase6_features import add_state_trajectory_features
from src.forecasting.phase6_labels import (
    build_horizon_targets,
    parse_boc_boe_default_source,
    parse_world_bank_debt_distress,
    status_to_episodes,
)
from src.forecasting.phase6_models import (
    build_oof_replacement_ratings,
    candidate_feature_groups,
    category_diagnostics,
    evaluate_event_family,
    rating_components,
    select_development_model,
)


def test_boc_wide_workbook_parser(tmp_path):
    path = tmp_path / "boc.xlsx"
    frame = pd.DataFrame(
        {
            "Country": ["Kenya", "Ghana", "Argentina"],
            "1960": [0.0, 0.0, 10.0],
            "1961": [0.0, 2.0, 5.0],
            "1962": [0.0, 0.0, 0.0],
        }
    )
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name="Country data", index=False)
    parsed, report = parse_boc_boe_default_source(path)
    assert set(parsed.country_code) == {"KEN", "GHA", "ARG"}
    assert parsed.loc[
        parsed.country_code.eq("GHA") & parsed.year.eq(1961),
        "any_default_status",
    ].iloc[0] == 1
    assert report["positive_country_years"] == 3


def test_world_bank_public_distress_parser_prefers_t12345(tmp_path):
    path = tmp_path / "non_confidential.csv"
    pd.DataFrame(
        {
            "country_code": ["KEN", "KEN", "GHA", "GHA"],
            "year": [2000, 2001, 2000, 2001],
            "t12345": [0, 1, 0, 0],
            "other_default_flag": [1, 1, 1, 1],
        }
    ).to_csv(path, index=False)
    parsed, report = parse_world_bank_debt_distress(path)
    assert report["target_column"] == "t12345"
    assert parsed.distress_status.sum() == 1


def test_status_to_episodes_and_horizon_exclusions():
    status = pd.DataFrame(
        {
            "country_code": ["AAA"] * 6 + ["BBB"] * 6,
            "year": list(range(2000, 2006)) * 2,
            "any_default_status": [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    episodes = status_to_episodes(
        status,
        status_column="any_default_status",
        maximum_internal_gap_years=1,
        event_type="sovereign_distress",
    )
    episode = episodes.loc[episodes.country_code.eq("AAA")].iloc[0]
    assert episode.start_year == 2001
    assert episode.end_year == 2003

    base = pd.DataFrame(
        {
            "country_code": ["AAA"] * 8,
            "forecast_origin_year": list(range(1999, 2007)),
        }
    )
    panel, exclusions = build_horizon_targets(
        base,
        episodes,
        horizons=(1, 2, 3),
        cooldown_years=2,
        coverage_end_year=2006,
        target_prefix="sovereign_distress",
    )
    assert panel.loc[
        panel.forecast_origin_year.eq(2000), "sovereign_distress_1y"
    ].iloc[0] == 1
    assert not panel.loc[
        panel.forecast_origin_year.eq(2001), "sovereign_distress_1y_eligible"
    ].iloc[0]
    assert (exclusions.exclusion_reason == "right_censored").any()


def test_state_trajectory_uses_only_prior_exact_calendar_rows():
    states = pd.DataFrame(
        {
            "entity_code": ["AAA", "AAA", "AAA", "AAA"],
            "forecast_origin_year": [2018, 2019, 2021, 2022],
            "state_1": [0.0, 1.0, 10.0, 12.0],
            "state_2": [0.0, 0.0, 1.0, 1.0],
        }
    )
    info = pd.DataFrame(
        {
            "entity_code": states.entity_code,
            "forecast_origin_year": states.forecast_origin_year,
            "state_uncertainty_proxy": 0.2,
            "observed_share": 0.7,
            "effective_information": 3.0,
        }
    )
    result, summary = add_state_trajectory_features(states, info)
    assert result.loc[result.forecast_origin_year.eq(2019), "velocity_state_1"].iloc[0] == 1.0
    assert np.isnan(
        result.loc[result.forecast_origin_year.eq(2021), "velocity_state_1"].iloc[0]
    )
    assert summary["provider_projection_rows_read"] == 0


def test_rating_uses_imminence_as_floor_not_average():
    reference = pd.DataFrame(
        {
            "country_code": ["AAA", "AAA", "BBB", "BBB"],
            "forecast_origin_year": [2018, 2019, 2018, 2019],
            "probability": [0.01, 0.02, 0.03, 0.04],
        }
    )
    current = pd.DataFrame(
        {
            "country_code": ["AAA", "BBB"],
            "forecast_origin_year": [2020, 2020],
            "probability": [0.99, 0.01],
        }
    )
    scored = rating_components(reference, current, peer_weight=0.5)
    high = scored.loc[scored.country_code.eq("AAA")].iloc[0]
    assert high.risk_intensity == 1.0
    assert high.replacement_risk_score == 10.0


def _synthetic_event_panel(seed=17):
    rng = np.random.default_rng(seed)
    rows = []
    for country_index in range(50):
        country = f"E{country_index:02d}"
        country_effect = rng.normal(scale=0.7)
        previous = None
        for year in range(1982, 2025):
            state = np.array(
                [
                    country_effect + 0.4 * np.sin(year / 4) + rng.normal(scale=0.25),
                    rng.normal(scale=0.8),
                    rng.normal(scale=0.6),
                    rng.normal(scale=0.5),
                ]
            )
            velocity = np.full(4, np.nan) if previous is None else state - previous
            signal = state[0] + 0.5 * (velocity[0] if previous is not None else 0)
            probability = 1 / (1 + np.exp(-(1.2 * signal - 2.4)))
            event = int(rng.random() < 0.45 * probability)
            row = {
                "country_code": country,
                "forecast_origin_year": year,
                "state_norm": float(np.sqrt(np.mean(state**2))),
                "velocity_norm": float(np.sqrt(np.nanmean(velocity**2))) if previous is not None else np.nan,
                "acceleration_norm": 0.1,
                "own_state_anomaly": abs(signal),
                "own_state_distance": abs(signal) / 2,
                "own_history_years": year - 1982,
                "own_history_supported": year >= 1985,
                "state_uncertainty_proxy": 0.2,
                "observed_share": 0.7,
                "effective_information": 3.0,
                "velocity_observed_share": float(np.isfinite(velocity).mean()),
                "acceleration_observed_share": 1.0,
                "raw_warning": signal + rng.normal(scale=0.15),
                "raw_noise": rng.normal(),
            }
            for index, value in enumerate(state, 1):
                row[f"state_{index}"] = value
            for index, value in enumerate(velocity, 1):
                row[f"velocity_state_{index}"] = value
            for horizon in (1, 2, 3):
                row[f"any_systemic_event_{horizon}y"] = event
                row[f"any_systemic_event_{horizon}y_eligible"] = True
            rows.append(row)
            previous = state
    return pd.DataFrame(rows)


def test_time_ordered_challenger_and_replacement_score_run():
    panel = _synthetic_event_panel()
    raw_groups = {
        "banking_raw": ["raw_warning", "raw_noise"],
        "sovereign_raw": ["raw_warning", "raw_noise"],
        "all_raw": ["raw_warning", "raw_noise"],
    }
    groups = candidate_feature_groups(panel, raw_groups, "combined")
    assert "state_1" not in groups["raw_targeted"]
    result = evaluate_event_family(
        panel,
        event_family="combined",
        target_prefix="any_systemic_event",
        raw_groups=raw_groups,
        horizons=(3,),
        minimum_train_positives=5,
        minimum_test_positives=1,
    )
    selection = select_development_model(result["aggregate_metrics"], 3)
    assert selection["model"] != "event_rate"
    ratings = build_oof_replacement_ratings(
        result, selected_model=selection["model"], horizon=3
    )
    assert len(ratings) > 100
    assert ratings.replacement_risk_score.between(1, 10).all()
    category, summary = category_diagnostics(ratings)
    assert not category.empty
    assert summary["high_risk_event_rate"] > summary["low_risk_event_rate"]


def test_phase6_source_firewall():
    root = Path("src/forecasting")
    labels = (root / "phase6_labels.py").read_text()
    features = (root / "phase6_features.py").read_text()
    models = (root / "phase6_models.py").read_text()
    execution = (root / "phase6_execution.py").read_text()

    # Labels are banking-crisis / sovereign-default files only; the decision
    # feature frame and final audit explicitly report zero provider projections.
    assert "WEO provider projections" in labels
    assert "provider_projection_rows_read" in features
    assert "provider_projection_rows_read" in execution

    for source in (labels, features, models, execution):
        assert "train_model.py" not in source
        assert "crisis_classifier.pkl" not in source
