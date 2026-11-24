"""Adversarial foundation tests; synthetic data is not forecast validation."""
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from src.forecasting.inventory import ForecastDataError, inventory_source, source_frames
from src.forecasting.temporal import select_as_of, purged_forward_split
from src.forecasting.baseline import BroadRidgeRegressor, compare_development_fold


def observation(**changes):
    row = dict(country_code="AAA", indicator_code="TEST", indicator_name="Capital",
               unit="PT", frequency="A", period=pd.Timestamp("2020-12-31"),
               value=12., observation_status="actual")
    row.update(changes)
    return row


def inventory(rows):
    return inventory_source(pd.DataFrame(rows), "FSIC", "2025-12-31")


def test_registry_expands_without_allowlist():
    result = inventory([observation(), observation(indicator_code="NEW_NEVER_SEEN", value=7)])
    assert len(result.registry) == 2
    assert result.registry.model_admission.eq("not_assessed_inventory_only").all()


@pytest.mark.parametrize("field,value", [("unit", "USD"), ("frequency", "Q"), ("SECTOR", "Other")])
def test_economic_dimensions_remain_distinct(field, value):
    row = observation()
    row["SECTOR"] = "Deposit takers"
    result = inventory([row, {**row, field: value}])
    assert len(result.registry) == 2
    assert result.summary["conflicting_observation_cells"] == 0


def test_missing_numeric_dimension_is_json_null():
    result = inventory([observation(unit_multiplier=np.nan), observation(unit_multiplier=6.)])
    assert len(result.registry) == 2
    assert any(json.loads(s)["unit_multiplier"] is None for s in result.registry.identity_json)


def test_duplicate_rows_do_not_create_independent_observations():
    result = inventory([observation(), observation()])
    assert result.summary["equal_duplicate_rows_coalesced"] == 1
    assert result.summary["unambiguous_retrospective_cells"] == 1


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_conflicts_quarantined_and_permutations_invariant(seed):
    frame = pd.DataFrame([observation(), observation(value=15), observation(country_code="BBB"), observation()])
    original = frame.copy(deep=True)
    a = inventory_source(frame, "FSIC", "2025-12-31")
    b = inventory_source(frame.sample(frac=1, random_state=seed), "FSIC", "2025-12-31")
    for field in ("registry", "conflicts", "country_coverage", "year_coverage"):
        pd.testing.assert_frame_equal(getattr(a, field), getattr(b, field), check_exact=True)
    assert a.summary == b.summary
    assert a.summary["conflicting_observation_cells"] == 1
    assert a.summary["unambiguous_retrospective_cells"] == 1
    pd.testing.assert_frame_equal(frame, original)


def test_future_and_projection_rows_are_inventoried_not_used_as_realizations():
    result = inventory([observation(), observation(period="2026-12-31"),
                        observation(country_code="CCC", observation_status="projection")])
    assert result.summary["after_cutoff_rows"] == 1
    assert result.summary["non_realized_status_rows"] == 1
    assert result.summary["unambiguous_retrospective_cells"] == 1


def test_missing_units_and_release_dates_are_reported_not_invented():
    result = inventory([observation(unit="")])
    assert result.summary["features_with_missing_units"] == 1
    assert result.summary["public_availability_rows"] == 0
    assert result.summary["vintage_date_rows"] == 0


@pytest.mark.parametrize("source", ["MFS", "WEO", "FSIC"])
def test_all_long_core_sources_supported(source):
    result = inventory_source(pd.DataFrame([observation()]), source, "2025-12-31")
    assert result.summary["source"] == source


def test_bsis_preserves_sector_frequency_and_labels():
    frame = pd.DataFrame({"country_code": ["AAA", "AAA"], "COUNTRY": ["AAA", "AAA"],
                          "SECTOR": ["DT", "OFC"], "INDICATOR": ["Loans, Domestic currency"] * 2,
                          "2020": [5., 6.], "2020-Q4": [5., 6.], "2020-M12": [5., 6.]})
    result = inventory_source(frame, "FSIBSIS", "2025-12-31")
    assert len(result.registry) == 6
    assert set(result.registry.frequency) == {"A", "Q", "M"}
    assert result.registry.identity_kind.eq("label_only_no_verified_source_code").all()
    assert result.summary["features_with_missing_units"] == 6


def test_bsis_all_missing_frequency_is_not_fabricated():
    frame = pd.DataFrame({"country_code": ["AAA"], "SECTOR": ["DT"], "INDICATOR": ["Loans"],
                          "2020": [5.], "2020-Q4": [np.nan]})
    result = inventory_source(frame, "FSIBSIS", "2025-12-31")
    assert len(result.registry) == 1


def test_wgi_keeps_new_numeric_measures_but_excludes_year_metadata():
    frame = pd.DataFrame({"country_code": ["AAA"], "year": pd.array([2020], dtype="Int64"),
                          "old_measure": [1.], "new_measure": [2.]})
    result = inventory_source(frame, "WGI", "2025-12-31")
    assert set(result.registry.indicator_code) == {"old_measure", "new_measure"}
    assert result.summary["features_with_missing_units"] == 2


def test_wgi_unknown_nonnumeric_metadata_requires_review():
    with pytest.raises(ForecastDataError, match="schema"):
        inventory_source(pd.DataFrame({"country_code": ["AAA"], "year": [2020], "notes": ["hi"]}), "WGI", "2025-12-31")


@pytest.mark.parametrize("field,value", [("value", np.inf), ("value", "not numeric"), ("period", "nonsense")])
def test_invalid_observations_are_audited(field, value):
    result = inventory([observation(**{field: value})])
    assert result.summary["unambiguous_retrospective_cells"] == 0
    assert result.summary["invalid_numeric_rows"] + result.summary["invalid_period_rows"] == 1


def test_missing_identity_rejected():
    with pytest.raises(ForecastDataError, match="identity"):
        inventory([observation(indicator_code=" ")])


def test_unknown_source_is_not_silent_empty_success():
    with pytest.raises(ForecastDataError, match="adapter"):
        list(source_frames(pd.DataFrame({"country_code": ["AAA"]}), "NEW"))


def dated(**changes):
    row = dict(country_code="AAA", feature_id="x", observation_period="2020-12-31",
               available_at="2021-03-01", vintage_at="2021-03-01", value=12., status="actual")
    row.update(changes)
    return row


def test_revision_after_origin_cannot_change_verified_features():
    original = dated()
    revision = dated(available_at="2023-01-01", vintage_at="2023-01-01", value=90.)
    frame = pd.DataFrame([original, revision])
    selected = select_as_of(frame, "2021-12-31")
    assert selected.observations.value.tolist() == [12.]
    assert selected.audit["timestamp_rules_passed"]
    assert select_as_of(frame, "2024-12-31").observations.value.tolist() == [90.]


def test_unknown_availability_cannot_be_called_verified():
    frame = pd.DataFrame([dated(available_at=None)])
    with pytest.raises(ForecastDataError, match="Unknown release"):
        select_as_of(frame, "2021-12-31")


def test_retrospective_requires_explicit_lag_and_labels_assumptions():
    frame = pd.DataFrame([dated(available_at=None, vintage_at=None)])
    with pytest.raises(ForecastDataError, match="explicit"):
        select_as_of(frame, "2021-12-31", mode="retrospective_latest_vintage")
    result = select_as_of(frame, "2021-12-31", mode="retrospective_latest_vintage", assumed_lag_days=180)
    assert result.observations.value.tolist() == [12.]
    assert result.audit["selected_unknown_release_dates"] == 1
    assert not result.audit["timestamp_rules_passed"]


def test_latest_missing_value_is_not_replaced_by_older_value():
    frame = pd.DataFrame([dated(), dated(observation_period="2021-12-31", available_at="2022-03-01",
                                        vintage_at="2022-03-01", value=np.nan)])
    result = select_as_of(frame, "2023-12-31")
    assert np.isnan(result.observations.value.iloc[0])


def test_forecasts_are_not_silently_used_as_realized_features():
    frame = pd.DataFrame([dated(status="projection")])
    assert select_as_of(frame, "2021-12-31").observations.empty


def test_same_vintage_conflict_fails_even_after_reordering():
    frame = pd.DataFrame([dated(), dated(value=13.)])
    for value in [frame, frame.iloc[::-1]]:
        with pytest.raises(ForecastDataError, match="Conflicting"):
            select_as_of(value, "2021-12-31")


def test_equal_vintage_duplicates_coalesce_and_do_not_mutate():
    frame = pd.DataFrame([dated(), dated()]); original = frame.copy(deep=True)
    assert len(select_as_of(frame, "2021-12-31").observations) == 1
    pd.testing.assert_frame_equal(frame, original)


def test_mixed_unknown_and_known_vintages_fail_closed():
    frame = pd.DataFrame([dated(), dated(vintage_at=None)])
    with pytest.raises(ForecastDataError, match="Mixed"):
        select_as_of(frame, "2021-12-31", mode="retrospective_latest_vintage", assumed_lag_days=0)


def metadata():
    return pd.DataFrame({"country_code": ["AAA"] * 6,
                         "forecast_origin": [f"{y}-01-01" for y in range(2010, 2016)],
                         "target_end": [f"{y+1}-12-31" for y in range(2010, 2016)],
                         "target_available_at": [f"{y+2}-06-30" for y in range(2010, 2016)]})


def test_purge_unresolved_targets_at_fold_boundary():
    result = purged_forward_split(metadata(), "2014-01-01", "2015-12-31")
    assert result.train_positions.tolist() == [0, 1]
    assert result.validation_positions.tolist() == [4, 5]
    assert result.audit["purged_unresolved_or_embargoed_rows"] == 2


def test_embargo_and_observation_publication_lag_both_respected():
    result = purged_forward_split(metadata(), "2014-01-01", "2015-12-31", embargo_days=200)
    assert result.train_positions.tolist() == [0]


@pytest.mark.parametrize("bad", ["duplicate", "missing", "backwards", "missing_country"])
def test_invalid_fold_metadata_fails(bad):
    frame = metadata()
    if bad == "duplicate": frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    if bad == "missing": frame.loc[0, "target_available_at"] = None
    if bad == "backwards": frame.loc[0, "target_end"] = "2000-01-01"
    if bad == "missing_country": frame.loc[0, "country_code"] = None
    with pytest.raises(ForecastDataError):
        purged_forward_split(frame, "2014-01-01", "2015-12-31")


def matrix(n=35, p=80):
    rng = np.random.default_rng(251)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = X.f0 * 3 - X.f1 * 2 + .05 * rng.normal(size=n)
    X.loc[::4, "f3"] = np.nan
    X["not_observed_in_training"] = np.nan
    return X, y


def test_broad_ridge_accepts_more_features_than_rows_and_keeps_contract():
    X, y = matrix()
    model = BroadRidgeRegressor().fit(X, y)
    assert model.n_features_in_ == 81
    assert len(model.active_features_) == 80
    assert model.unlearnable_features_ == ["not_observed_in_training"]
    assert model.training_audit_["feature_cap"] is None
    assert np.isfinite(model.predict(X)).all()


def test_preprocessing_does_not_learn_from_validation_values():
    X, y = matrix()
    model = BroadRidgeRegressor().fit(X.iloc[:25], y.iloc[:25])
    medians = model.medians_.copy(); scales = model.scaler_.scale_.copy()
    valid = X.iloc[25:].copy(); valid["f0"] = 1e9
    model.predict(valid)
    pd.testing.assert_series_equal(medians, model.medians_)
    np.testing.assert_array_equal(scales, model.scaler_.scale_)


def test_future_only_feature_stays_unlearnable_in_the_existing_fit():
    X, y = matrix(); model = BroadRidgeRegressor().fit(X, y)
    before = model.predict(X)
    revised = X.copy(); revised["not_observed_in_training"] = 10000
    np.testing.assert_array_equal(before, model.predict(revised))


def test_column_order_and_pickle_roundtrip():
    X, y = matrix(); model = BroadRidgeRegressor().fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(model.predict(X), restored.predict(X[X.columns[::-1]]))


@pytest.mark.parametrize("bad", ["metadata", "infinity", "duplicate_columns", "non_numeric", "all_missing"])
def test_bad_model_inputs_fail(bad):
    X, y = matrix()
    if bad == "metadata": X["forecast_origin_year"] = 2020
    if bad == "infinity": X.loc[0, "f0"] = np.inf
    if bad == "duplicate_columns": X.columns = ["same"] * len(X.columns)
    if bad == "non_numeric": X["notes"] = "text"
    if bad == "all_missing": X[:] = np.nan
    with pytest.raises(ForecastDataError): BroadRidgeRegressor().fit(X, y)


def test_target_alignment_checked_not_silently_reordered():
    X, y = matrix()
    with pytest.raises(ForecastDataError, match="index"):
        BroadRidgeRegressor().fit(X, y.iloc[::-1])


def test_prediction_schema_changes_fail_closed():
    X, y = matrix(); model = BroadRidgeRegressor().fit(X, y)
    with pytest.raises(ForecastDataError, match="schema"):
        model.predict(X.drop(columns="f0"))


def test_matched_development_comparison_runs_without_promotion_claim():
    X, y = matrix(n=6, p=5)
    result = compare_development_fold(X, y, metadata(), X.f0, ["f0", "f1"],
                                      "2014-01-01", "2015-12-31")
    assert result["status"] == "development_only_not_promotion_evidence"
    assert {m["rows"] for m in result["metrics"].values()} == {2}
    assert result["model_audits"]["broad_ridge"]["candidate_features"] == 6
    altered = y.copy(); altered.iloc[4:] = 1000
    other = compare_development_fold(X, altered, metadata(), X.f0, ["f0", "f1"],
                                     "2014-01-01", "2015-12-31")
    pd.testing.assert_frame_equal(result["predictions"].drop(columns="actual"),
                                  other["predictions"].drop(columns="actual"))


@pytest.mark.parametrize("folder", ["cache", "artifacts", "data", "src", "tests", ".git", ".github"])
def test_research_output_cannot_overwrite_serving_or_source_paths(tmp_path, folder):
    from src.scripts.audit_forecasting_sources import output_destination
    snapshot = tmp_path / "snapshot"; snapshot.mkdir()
    with pytest.raises(ForecastDataError, match="Protected"):
        output_destination(snapshot, snapshot / folder / "new-research")


def test_research_output_must_be_new(tmp_path):
    from src.scripts.audit_forecasting_sources import output_destination
    snapshot = tmp_path / "snapshot"; snapshot.mkdir()
    existing = tmp_path / "research"; existing.mkdir()
    with pytest.raises(ForecastDataError, match="already exists"):
        output_destination(snapshot, existing)
    assert output_destination(snapshot, tmp_path / "new") == tmp_path / "new"


def test_source_checksum_failure_precedes_output_writes(tmp_path):
    from src.scripts.audit_forecasting_sources import run
    snapshot = tmp_path / "snapshot"
    (snapshot / "cache").mkdir(parents=True); (snapshot / "artifacts").mkdir()
    (snapshot / "cache/FSIC_cache.parquet").write_bytes(b"not a source")
    (snapshot / "artifacts/data_manifest.json").write_text(json.dumps({
        "artifacts": {"cache/FSIC_cache.parquet": {"sha256": "wrong", "bytes": 12}}}))
    output = tmp_path / "new"
    with pytest.raises(ForecastDataError, match="checksum"):
        run(snapshot, output)
    assert not output.exists()
