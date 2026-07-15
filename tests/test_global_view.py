from pathlib import Path

import pandas as pd
import pytest

from src.dashboard.global_view import (
    build_systemic_watchlist,
    calculate_weighted_metrics,
    regional_risk_summary,
    weaker_pillar_label,
)


def test_weaker_pillar_uses_lower_strength_score_without_calling_it_attribution():
    assert weaker_pillar_label(
        pd.Series({"economic_pillar": 8.0, "industry_pillar": 3.0})
    ) == "Banking system"
    assert weaker_pillar_label(
        pd.Series({"economic_pillar": 2.0, "industry_pillar": 7.0})
    ) == "Operating environment"
    assert weaker_pillar_label(
        pd.Series({"economic_pillar": 5.0, "industry_pillar": 5.0})
    ) == "Similar strength"
    assert weaker_pillar_label(
        pd.Series({"economic_pillar": 5.0, "industry_pillar": None})
    ) == "Not comparable"


def test_watchlist_enforces_and_reports_its_exact_universe():
    frame = pd.DataFrame(
        {
            "country_code": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
            "country_name": ["A", "B", "C", "D", "E", "F"],
            "risk_score": [9.0, 8.0, 7.0, 6.5, 7.5, 9.5],
            "nominal_gdp": [100.0, 200.0, 300.0, 400.0, 500.0, None],
            "economic_pillar": [5.0, 5.0, 5.0, 2.0, 8.0, 5.0],
            "industry_pillar": [5.0, 5.0, 5.0, 7.0, 3.0, 5.0],
        }
    )

    watchlist, metadata = build_systemic_watchlist(frame)

    assert watchlist["country_code"].tolist() == ["EEE", "DDD"]
    assert watchlist["weaker_pillar"].tolist() == [
        "Banking system",
        "Operating environment",
    ]
    assert metadata == {
        "available": True,
        "risk_threshold": 6.0,
        "gdp_median": 300.0,
        "eligible_countries": 5,
        "excluded_missing": 1,
        "matched_total": 2,
        "omitted_by_limit": 0,
        "limit": 10,
    }


def test_watchlist_reports_rows_omitted_by_the_display_limit():
    frame = pd.DataFrame(
        {
            "risk_score": [9.0, 8.0, 7.0, 6.5],
            "nominal_gdp": [100.0, 200.0, 300.0, 400.0],
            "economic_pillar": [4.0] * 4,
            "industry_pillar": [5.0] * 4,
        }
    )

    watchlist, metadata = build_systemic_watchlist(frame, limit=1)

    assert len(watchlist) == 1
    assert metadata["matched_total"] == 2
    assert metadata["omitted_by_limit"] == 1


def test_weighted_metrics_do_not_turn_unavailable_data_into_zero():
    assert calculate_weighted_metrics(pd.DataFrame({"risk_score": [2.0, 8.0]})) == {}

    metrics = calculate_weighted_metrics(
        pd.DataFrame(
            {
                "risk_score": [2.0, 8.0],
                "nominal_gdp": [1.0, 3.0],
            }
        )
    )
    assert metrics["global_risk_score"] == pytest.approx(6.5)
    assert pd.isna(metrics["global_economic_pillar"])


def test_region_summary_discloses_weighted_and_fallback_bases():
    frame = pd.DataFrame(
        {
            "Region": ["Africa", "Africa", "Europe"],
            "risk_score": [8.0, 4.0, 3.0],
            "nominal_gdp": [3.0, 1.0, None],
        }
    )

    summary = regional_risk_summary(frame).set_index("Region")

    assert summary.loc["Africa", "Weighted Risk"] == pytest.approx(7.0)
    assert summary.loc["Africa", "Basis"] == "GDP weighted"
    assert summary.loc["Europe", "Weighted Risk"] == pytest.approx(3.0)
    assert summary.loc["Europe", "Basis"] == "Unweighted; GDP unavailable"


def test_global_view_source_has_no_misleading_delta_or_main_driver_copy():
    source = Path("src/dashboard/global_view.py").read_text(encoding="utf-8")

    assert "delta=" not in source
    assert "Main Driver" not in source
    assert "✓" not in source
    assert "âœ“" not in source

