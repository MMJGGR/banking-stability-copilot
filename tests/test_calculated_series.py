import json

import pandas as pd
import pytest

from src.dashboard.calculated_series import (
    build_query_metadata,
    check_cross_sectional_additivity,
    check_recipe_units,
    check_unit_compatibility,
    compute_cross_sectional_share,
    compute_expression_formula,
    compute_ratio,
    compute_temporal_change,
    diagnose_alignment,
    FormulaValidationError,
    get_calculation_recipe,
    get_calculation_recipes,
    normalize_observation_frame,
    validate_formula,
)


def _frame():
    return pd.DataFrame(
        {
            "country_code": ["KEN", "KEN", "KEN", "MOZ", "MOZ", "MOZ"],
            "indicator_code": ["A", "A", "B", "A", "A", "B"],
            "period": [
                "2024-12-31",
                "2025-12-31",
                "2025-12-31",
                "2025-12-31",
                "2026-12-31",
                "2025-12-31",
            ],
            "frequency": ["A", "A", "A", "A", "A", "A"],
            "value": [100.0, 120.0, 60.0, 80.0, 100.0, 40.0],
        }
    )


def test_normalize_observation_frame_aligns_one_indicator():
    normalized = normalize_observation_frame(
        _frame(),
        "A",
        "indicator_code",
        "Indicator A",
    )

    assert set(normalized["indicator_label"]) == {"Indicator A"}
    assert set(normalized["country_code"]) == {"KEN", "MOZ"}
    assert normalized["date"].dtype.kind == "M"


def test_compute_ratio_aligns_country_date_frequency_only():
    numerator = normalize_observation_frame(_frame(), "A", "indicator_code", "A")
    denominator = normalize_observation_frame(_frame(), "B", "indicator_code", "B")

    ratio = compute_ratio(numerator, denominator, scale=100)

    assert set(ratio["country_code"]) == {"KEN", "MOZ"}
    ken = ratio.loc[ratio["country_code"] == "KEN"].iloc[0]
    moz = ratio.loc[ratio["country_code"] == "MOZ"].iloc[0]
    assert ken["value"] == pytest.approx(200.0)
    assert moz["value"] == pytest.approx(200.0)
    assert ratio["date"].nunique() == 1


def test_compute_ratio_drops_zero_denominator():
    numerator = pd.DataFrame(
        {
            "country_code": ["KEN"],
            "date": [pd.Timestamp("2025-12-31")],
            "frequency": ["A"],
            "value": [10.0],
        }
    )
    denominator = numerator.copy()
    denominator["value"] = 0.0

    ratio = compute_ratio(numerator, denominator)

    assert ratio.empty


def test_compute_expression_formula_supports_parentheses_and_reused_operands():
    frame = pd.DataFrame(
        {
            "country_code": ["KEN", "KEN", "KEN"],
            "indicator_code": ["A", "B", "D"],
            "period": ["2025-12-31", "2025-12-31", "2025-12-31"],
            "frequency": ["A", "A", "A"],
            "value": [120.0, 70.0, 20.0],
        }
    )
    operands = {
        key: normalize_observation_frame(frame, key, "indicator_code", key)
        for key in ("A", "B", "D")
    }

    result, plan = compute_expression_formula("(A-B)/(B-D)", operands, scale=100)

    assert plan.normalized_formula == "(A - B) / (B - D)"
    assert plan.used_operands == ("A", "B", "D")
    assert result["value"].iloc[0] == pytest.approx(100.0)


def test_compute_expression_formula_supports_numeric_constants():
    frame = pd.DataFrame(
        {
            "country_code": ["KEN"],
            "indicator_code": ["A"],
            "period": ["2025-12-31"],
            "frequency": ["A"],
            "value": [10.0],
        }
    )
    operands = {"A": normalize_observation_frame(frame, "A", "indicator_code", "A")}

    result, _ = compute_expression_formula("(A * 2) + 5", operands)

    assert result["value"].iloc[0] == pytest.approx(25.0)


def test_compute_expression_formula_drops_zero_denominator():
    frame = pd.DataFrame(
        {
            "country_code": ["KEN", "KEN", "KEN"],
            "indicator_code": ["A", "B", "D"],
            "period": ["2025-12-31", "2025-12-31", "2025-12-31"],
            "frequency": ["A", "A", "A"],
            "value": [10.0, 20.0, 20.0],
        }
    )
    operands = {
        key: normalize_observation_frame(frame, key, "indicator_code", key)
        for key in ("A", "B", "D")
    }

    result, _ = compute_expression_formula("A / (B - D)", operands)

    assert result.empty


def test_validate_formula_rejects_code_injection_syntax():
    with pytest.raises(FormulaValidationError):
        validate_formula("__import__('os').system('echo bad')", ["A", "B", "C", "D"])


def test_validate_formula_rejects_unknown_operands():
    with pytest.raises(FormulaValidationError):
        validate_formula("A + Z", ["A", "B", "C", "D"])


def test_compute_cross_sectional_share_sums_to_100_by_period():
    data = normalize_observation_frame(_frame(), "A", "indicator_code", "A")
    data = data.loc[data["date"] == pd.Timestamp("2025-12-31")]

    shares = compute_cross_sectional_share(data)

    assert shares["value"].sum() == pytest.approx(100.0)
    assert shares.loc[shares["country_code"] == "KEN", "value"].iloc[0] == pytest.approx(60.0)


def test_compute_temporal_change_modes():
    data = normalize_observation_frame(_frame(), "A", "indicator_code", "A")
    ken = data.loc[data["country_code"] == "KEN"]

    period = compute_temporal_change(ken, "period_pct")
    base = compute_temporal_change(ken, "base_pct")
    index = compute_temporal_change(ken, "index_100")

    assert period["value"].iloc[0] == pytest.approx(20.0)
    assert base["value"].iloc[-1] == pytest.approx(20.0)
    assert index["value"].iloc[0] == pytest.approx(100.0)
    assert index["value"].iloc[-1] == pytest.approx(120.0)


def test_compute_temporal_change_rejects_unknown_mode():
    data = normalize_observation_frame(_frame(), "A", "indicator_code", "A")

    with pytest.raises(ValueError):
        compute_temporal_change(data, "bad_mode")


def test_cross_sectional_share_gate_requires_additive_levels():
    currency = check_cross_sectional_additivity("USD billions")
    percentage = check_cross_sectional_additivity("Percent of GDP")
    unknown = check_cross_sectional_additivity("source unit unavailable")

    assert currency.valid is True
    assert currency.output_unit == "percent"
    assert percentage.valid is False
    assert "must not be summed" in percentage.reason
    assert unknown.valid is False
    assert "Additivity is unknown" in unknown.reason
    assert check_cross_sectional_additivity(
        "source unit unavailable", additive=True
    ).valid is True


def test_unit_compatibility_catches_scale_mismatch_and_unsafe_addition():
    same_currency = check_unit_compatibility(
        "ratio", ["USD billions", "USD billions"]
    )
    mixed_currency_scale = check_unit_compatibility(
        "ratio", ["USD billions", "USD millions"]
    )
    incompatible_add = check_unit_compatibility(
        "add", ["Percent of GDP", "USD billions"]
    )

    assert same_currency.valid is True
    assert same_currency.output_unit == "ratio"
    assert mixed_currency_scale.valid is False
    assert "convert" in mixed_currency_scale.reason.lower()
    assert incompatible_add.valid is False


def test_change_unit_check_allows_one_series_but_discloses_unknown_unit():
    check = check_unit_compatibility("change", [None])

    assert check.valid is True
    assert check.output_unit == "percent or index"
    assert check.warnings


def test_alignment_diagnostics_report_exact_matches_and_losses():
    numerator = normalize_observation_frame(_frame(), "A", "indicator_code", "A")
    denominator = normalize_observation_frame(_frame(), "B", "indicator_code", "B")
    ratio = compute_ratio(numerator, denominator, scale=100)

    diagnostics = diagnose_alignment(
        {"numerator": numerator, "denominator": denominator},
        result=ratio,
    )

    assert diagnostics.matched_observations == 2
    assert diagnostics.output_observations == 2
    assert diagnostics.dropped_after_calculation == 0
    assert diagnostics.input_observations == {"numerator": 4, "denominator": 2}
    assert diagnostics.dropped_observations == {"numerator": 2, "denominator": 0}
    assert diagnostics.matched_frequencies == ("A",)
    assert diagnostics.matched_period_start == "2025-12-31"
    assert diagnostics.matched_period_end == "2025-12-31"
    assert diagnostics.matched_countries == 2


def test_alignment_diagnostics_separate_invalid_result_losses():
    numerator = pd.DataFrame(
        {
            "country_code": ["KEN", "MOZ"],
            "date": [pd.Timestamp("2025-12-31")] * 2,
            "frequency": ["A", "A"],
            "value": [10.0, 20.0],
        }
    )
    denominator = numerator.copy()
    denominator["value"] = [2.0, 0.0]
    ratio = compute_ratio(numerator, denominator)

    diagnostics = diagnose_alignment(
        {"numerator": numerator, "denominator": denominator}, result=ratio
    )

    assert diagnostics.matched_observations == 2
    assert diagnostics.output_observations == 1
    assert diagnostics.dropped_after_calculation == 1


def test_task_recipes_cover_ratio_share_and_change_presets():
    assert {recipe.operation for recipe in get_calculation_recipes()} == {
        "ratio",
        "share",
        "change",
    }
    ratio = get_calculation_recipe("ratio_percent")
    share = get_calculation_recipe("cross_sectional_share_percent")
    changes = get_calculation_recipes("change")

    assert ratio.formula_template == "A / B"
    assert ratio.scale == 100.0
    assert share.requires_additive_input is True
    assert {recipe.temporal_mode for recipe in changes} == {
        "period_pct",
        "base_pct",
        "index_100",
    }
    assert check_recipe_units(
        "cross_sectional_share_percent", ["Percent of GDP"]
    ).valid is False


def test_query_metadata_is_stable_json_ready_and_carries_alignment():
    numerator = normalize_observation_frame(_frame(), "A", "indicator_code", "A")
    denominator = normalize_observation_frame(_frame(), "B", "indicator_code", "B")
    diagnostics = diagnose_alignment({"A": numerator, "B": denominator})
    unit_check = check_recipe_units(
        "ratio_percent", ["Percent of GDP", "Percent of GDP"]
    )

    metadata = build_query_metadata(
        operation="ratio",
        recipe_key="ratio_percent",
        dataset="WEO",
        source_version="2026-04",
        indicators={"B": "Revenue", "A": "Interest"},
        countries=["moz", "KEN", "KEN"],
        requested_frequency="A",
        requested_range="10 Years",
        formula="A / B",
        scale=100,
        units={"B": "Percent of GDP", "A": "Percent of GDP"},
        observation_statuses=["projection", "actual", "actual"],
        alignment=diagnostics,
        unit_compatibility=unit_check,
    )

    assert metadata["schema_version"] == "bankenv.calculation-query.v1"
    assert metadata["query"]["countries"] == ["KEN", "MOZ"]
    assert list(metadata["query"]["indicators"]) == ["A", "B"]
    assert metadata["query"]["observation_statuses"] == ["actual", "projection"]
    assert metadata["calculation"]["recipe"]["key"] == "ratio_percent"
    assert metadata["calculation"]["unit_compatibility"]["valid"] is True
    assert metadata["alignment"]["matched_observations"] == 2
    assert json.loads(json.dumps(metadata)) == metadata


def test_query_metadata_rejects_recipe_operation_mismatch():
    with pytest.raises(ValueError, match="is for 'share', not 'ratio'"):
        build_query_metadata(
            operation="ratio",
            recipe_key="cross_sectional_share_percent",
        )
