import pandas as pd

from src.crisis_panel import (
    CrisisPanelConfig,
    FeatureSpec,
    build_crisis_panel,
    build_crisis_panel_result,
)


class StubLabels:
    SOURCE_COVERAGE_END_YEAR = 2010

    def __init__(self, crises=None):
        self.crises = crises or {}


GDP = FeatureSpec(
    name="gdp_growth",
    source="WEO",
    family="macro",
    indicator_code="NGDP_RPCH",
    max_age_years=2,
)


def _weo(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "country_code",
            "indicator_code",
            "indicator_name",
            "period",
            "value",
            "observation_status",
        ],
    )


def test_feature_asof_join_never_selects_after_lagged_cutoff():
    weo = _weo(
        [
            ("USA", "NGDP_RPCH", "GDP growth", "2000-12-31", 1.0, "actual"),
            ("USA", "NGDP_RPCH", "GDP growth", "2001-12-31", 2.0, "actual"),
            ("USA", "NGDP_RPCH", "GDP growth", "2002-12-31", 99.0, "actual"),
        ]
    )
    panel = build_crisis_panel(
        weo,
        None,
        StubLabels(),
        ["USA"],
        [GDP],
        CrisisPanelConfig(start_year=2002, end_year=2002, feature_lag_years=1),
    )

    row = panel.iloc[0]
    assert row["feature_cutoff_year"] == 2001
    assert row["gdp_growth"] == 2.0
    assert row["gdp_growth__observation_year"] == 2001
    assert row["gdp_growth__observation_year"] <= row["feature_cutoff_year"]


def test_crisis_targets_use_start_year_and_stable_event_id_for_one_to_three_year_horizon():
    labels = StubLabels({"USA": [(2005, 2006)]})
    panel = build_crisis_panel(
        _weo([("USA", "NGDP_RPCH", "GDP growth", "2000", 2.0, "actual")]),
        None,
        labels,
        ["USA"],
        [GDP],
        CrisisPanelConfig(
            start_year=2001,
            end_year=2004,
            post_crisis_cooldown_years=0,
        ),
    ).set_index("forecast_origin_year")

    assert panel.loc[2001, "crisis_target"] == 0
    for origin, years_to_crisis in ((2002, 3), (2003, 2), (2004, 1)):
        assert panel.loc[origin, "crisis_target"] == 1
        assert panel.loc[origin, "crisis_event_id"] == "USA-2005-2006"
        assert panel.loc[origin, "years_to_crisis"] == years_to_crisis


def test_active_cooldown_and_right_censored_rows_are_excluded_and_audited():
    labels = StubLabels({"USA": [(2000, 2001)]})
    result = build_crisis_panel_result(
        _weo([("USA", "NGDP_RPCH", "GDP growth", "1995", 2.0, "actual")]),
        None,
        labels,
        ["USA"],
        [GDP],
        CrisisPanelConfig(
            start_year=2000,
            end_year=2009,
            post_crisis_cooldown_years=2,
            label_coverage_end_year=2010,
        ),
    )

    assert result.panel["forecast_origin_year"].tolist() == [2004, 2005, 2006, 2007]
    reasons = result.exclusions.set_index("forecast_origin_year")["exclusion_reason"]
    assert reasons.loc[2000] == "active_crisis"
    assert reasons.loc[2001] == "active_crisis"
    assert reasons.loc[2002] == "post_crisis_cooldown"
    assert reasons.loc[2003] == "post_crisis_cooldown"
    assert reasons.loc[2008] == "right_censored"
    assert reasons.loc[2009] == "right_censored"


def test_staleness_missingness_directness_and_family_coverage_are_explicit():
    inflation = FeatureSpec(
        name="inflation",
        source="WEO",
        family="macro",
        indicator_code="PCPIPCH",
        max_age_years=2,
    )
    weo = _weo(
        [("USA", "NGDP_RPCH", "GDP growth", "2000", 2.0, "actual")]
    )
    panel = build_crisis_panel(
        weo,
        None,
        StubLabels(),
        ["USA"],
        [GDP, inflation],
        CrisisPanelConfig(
            start_year=2003,
            end_year=2003,
            family_min_coverage=0.5,
        ),
    )
    row = panel.iloc[0]

    assert row["gdp_growth__age_years"] == 3
    assert row["gdp_growth__direct"]
    assert not row["gdp_growth__missing"]
    assert row["gdp_growth__stale"]
    assert not row["gdp_growth__available"]
    assert row["inflation__missing"]
    assert not row["inflation__available"]
    assert row["family_macro__available_count"] == 0
    assert row["family_macro__coverage_ratio"] == 0.0
    assert not row["family_macro__available"]


def test_duplicate_source_and_universe_rows_do_not_duplicate_country_origin():
    weo = _weo(
        [
            ("USA", "NGDP_RPCH", "GDP growth", "2001-03-31", 1.0, "actual"),
            ("USA", "NGDP_RPCH", "GDP growth", "2001-12-31", 2.0, "actual"),
            ("USA", "NGDP_RPCH", "GDP growth", "2001-12-31", 2.0, "actual"),
        ]
    )
    panel = build_crisis_panel(
        weo,
        None,
        StubLabels(),
        ["USA", "USA"],
        [GDP],
        CrisisPanelConfig(start_year=2002, end_year=2004),
    )

    assert not panel.duplicated(["country_code", "forecast_origin_year"]).any()
    assert len(panel) == 3
    assert panel["gdp_growth"].eq(2.0).all()


def test_additional_world_bank_source_respects_the_same_cutoff_contract():
    wb_credit = FeatureSpec(
        name="bank_credit_gdp",
        source="WB",
        family="credit_cycle",
        indicator_code="bank_credit_gdp",
        max_age_years=2,
    )
    wb = _weo(
        [
            ("USA", "bank_credit_gdp", "Bank credit", "2000", 40.0, "actual"),
            ("USA", "bank_credit_gdp", "Bank credit", "2001", 45.0, "actual"),
            ("USA", "bank_credit_gdp", "Bank credit", "2002", 999.0, "actual"),
        ]
    )
    panel = build_crisis_panel(
        None,
        None,
        StubLabels(),
        ["USA"],
        [wb_credit],
        CrisisPanelConfig(start_year=2002, end_year=2002, feature_lag_years=1),
        additional_sources={"WB": wb},
    )
    assert panel.loc[0, "bank_credit_gdp"] == 45.0
    assert panel.loc[0, "bank_credit_gdp__observation_year"] == 2001
