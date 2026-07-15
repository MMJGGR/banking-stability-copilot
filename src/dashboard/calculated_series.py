"""Calculated time-series helpers for the Data Explorer.

These utilities keep Data Explorer calculations explicit and auditable. They
only align observations with the same country, date, and frequency; callers can
decide how to present missing observations instead of silently filling gaps.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalculationSpec:
    """Human-readable description of a calculated series."""

    formula: str
    mode: str
    notes: tuple[str, ...] = ()


ALIGN_KEYS = ["country_code", "date", "frequency"]


class FormulaValidationError(ValueError):
    """Raised when a user formula includes unsupported syntax."""


@dataclass(frozen=True)
class FormulaPlan:
    """Validated formula representation for display and execution."""

    expression: ast.Expression
    normalized_formula: str
    used_operands: tuple[str, ...]


@dataclass(frozen=True)
class UnitProfile:
    """Conservative unit semantics used by calculation validity gates."""

    raw_unit: str
    canonical_unit: str
    dimension: str
    additive: bool | None


@dataclass(frozen=True)
class UnitCompatibility:
    """Result of checking whether an operation is meaningful for its units."""

    valid: bool
    operation: str
    units: tuple[str, ...]
    output_unit: str | None
    reason: str
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "operation": self.operation,
            "units": list(self.units),
            "output_unit": self.output_unit,
            "reason": self.reason,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class AlignmentDiagnostics:
    """Exact-key alignment coverage for one or more calculation operands."""

    matched_observations: int
    output_observations: int | None
    dropped_after_calculation: int | None
    input_observations: Mapping[str, int]
    dropped_observations: Mapping[str, int]
    input_frequencies: Mapping[str, tuple[str, ...]]
    input_periods: Mapping[str, tuple[str | None, str | None]]
    matched_frequencies: tuple[str, ...]
    matched_period_start: str | None
    matched_period_end: str | None
    matched_countries: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched_observations": self.matched_observations,
            "output_observations": self.output_observations,
            "dropped_after_calculation": self.dropped_after_calculation,
            "input_observations": dict(self.input_observations),
            "dropped_observations": dict(self.dropped_observations),
            "input_frequencies": {
                name: list(values) for name, values in self.input_frequencies.items()
            },
            "input_periods": {
                name: {"start": bounds[0], "end": bounds[1]}
                for name, bounds in self.input_periods.items()
            },
            "matched_frequencies": list(self.matched_frequencies),
            "matched_period_start": self.matched_period_start,
            "matched_period_end": self.matched_period_end,
            "matched_countries": self.matched_countries,
        }


@dataclass(frozen=True)
class CalculationRecipe:
    """Task-oriented preset that can drive an Explorer recipe selector."""

    key: str
    title: str
    operation: str
    description: str
    example: str
    scale: float
    output_unit: str
    temporal_mode: str | None = None
    requires_additive_input: bool = False
    formula_template: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "operation": self.operation,
            "description": self.description,
            "example": self.example,
            "scale": self.scale,
            "output_unit": self.output_unit,
            "temporal_mode": self.temporal_mode,
            "requires_additive_input": self.requires_additive_input,
            "formula_template": self.formula_template,
        }


CALCULATION_RECIPES = (
    CalculationRecipe(
        key="ratio_percent",
        title="Ratio as percent",
        operation="ratio",
        description="Divide one aligned series by another and multiply by 100.",
        example="Interest expense / government revenue",
        scale=100.0,
        output_unit="percent",
        formula_template="A / B",
    ),
    CalculationRecipe(
        key="cross_sectional_share_percent",
        title="Country share of selected total",
        operation="share",
        description="Divide each selected country's additive level by the group total.",
        example="Country reserves / selected-country reserves",
        scale=100.0,
        output_unit="percent",
        requires_additive_input=True,
    ),
    CalculationRecipe(
        key="period_change_percent",
        title="Change from prior period",
        operation="change",
        description="Calculate period-over-period percent change at native frequency.",
        example="Annual change in nominal GDP",
        scale=100.0,
        output_unit="percent",
        temporal_mode="period_pct",
    ),
    CalculationRecipe(
        key="base_change_percent",
        title="Change from first selected period",
        operation="change",
        description="Calculate percent change from the first observation in range.",
        example="Change in reserves since 2020",
        scale=100.0,
        output_unit="percent",
        temporal_mode="base_pct",
    ),
    CalculationRecipe(
        key="rebase_index_100",
        title="Rebase first period to 100",
        operation="change",
        description="Express each observation relative to a first-period index of 100.",
        example="Comparable deposit-growth paths across countries",
        scale=100.0,
        output_unit="index (first period = 100)",
        temporal_mode="index_100",
    ),
)


def get_calculation_recipes(operation: str | None = None) -> tuple[CalculationRecipe, ...]:
    """Return task presets in display order, optionally filtered by operation."""
    if operation is None:
        return CALCULATION_RECIPES
    normalized = str(operation).strip().lower()
    return tuple(recipe for recipe in CALCULATION_RECIPES if recipe.operation == normalized)


def get_calculation_recipe(key: str) -> CalculationRecipe:
    """Return one named recipe or fail with a UI-safe list of valid keys."""
    normalized = str(key or "").strip().lower()
    for recipe in CALCULATION_RECIPES:
        if recipe.key == normalized:
            return recipe
    valid = ", ".join(recipe.key for recipe in CALCULATION_RECIPES)
    raise ValueError(f"Unknown calculation recipe {key!r}. Use one of: {valid}.")


def profile_unit(unit: str | None, *, additive: bool | None = None) -> UnitProfile:
    """Infer conservative semantics from a source-supplied unit label.

    Unknown units are deliberately not assumed additive. Callers with trusted
    metadata can provide an explicit ``additive`` declaration.
    """
    raw = str(unit or "").strip()
    canonical = " ".join(raw.lower().replace("_", " ").split())
    if not canonical:
        return UnitProfile(raw, "unknown", "unknown", additive)

    percent_markers = ("%", "percent", "percentage", "pct", "percentage point")
    ratio_markers = (
        "ratio",
        "per capita",
        " of gdp",
        "/gdp",
        " of revenue",
        "/revenue",
        "times",
    )
    index_markers = ("index", "base year", "=100")
    currency_markers = (
        "usd",
        "eur",
        "gbp",
        "kes",
        "lcu",
        "currency",
        "local currency",
    )
    count_markers = ("count", "number", "persons", "people", "units")

    if any(marker in canonical for marker in percent_markers):
        dimension = "percentage"
        inferred_additive = False
    elif any(marker in canonical for marker in index_markers):
        dimension = "index"
        inferred_additive = False
    elif any(marker in canonical for marker in ratio_markers) or canonical in {"x", "multiple"}:
        dimension = "ratio"
        inferred_additive = False
    elif any(marker in canonical for marker in currency_markers):
        dimension = "currency"
        inferred_additive = "per capita" not in canonical
    elif any(marker in canonical for marker in count_markers):
        dimension = "count"
        inferred_additive = True
    else:
        dimension = "unknown"
        inferred_additive = None

    return UnitProfile(
        raw_unit=raw,
        canonical_unit=canonical,
        dimension=dimension,
        additive=inferred_additive if additive is None else bool(additive),
    )


def check_cross_sectional_additivity(
    unit: str | None,
    *,
    additive: bool | None = None,
) -> UnitCompatibility:
    """Require an additive level before calculating country shares."""
    profile = profile_unit(unit, additive=additive)
    if profile.additive is True:
        return UnitCompatibility(
            valid=True,
            operation="share",
            units=(profile.canonical_unit,),
            output_unit="percent",
            reason="The input is declared or inferred to be an additive level.",
        )
    if profile.additive is False:
        reason = (
            "Cross-sectional shares require additive levels; percentages, ratios, "
            "per-capita series, and indexes must not be summed across countries."
        )
    else:
        reason = (
            "Additivity is unknown. Supply source metadata that explicitly marks "
            "the series as additive before calculating a cross-sectional share."
        )
    return UnitCompatibility(
        valid=False,
        operation="share",
        units=(profile.canonical_unit,),
        output_unit=None,
        reason=reason,
    )


def check_unit_compatibility(
    operation: str,
    units: Iterable[str | None],
    *,
    additive: bool | None = None,
) -> UnitCompatibility:
    """Validate unit semantics for ratio, share, change, add, or subtract.

    The result is advisory and explicit: existing compute helpers retain their
    current arithmetic behavior, while UI callers can gate execution on
    ``result.valid``.
    """
    normalized_operation = str(operation or "").strip().lower()
    profiles = tuple(profile_unit(unit) for unit in units)
    canonical_units = tuple(profile.canonical_unit for profile in profiles)

    if normalized_operation == "share":
        if len(profiles) != 1:
            return UnitCompatibility(
                False,
                "share",
                canonical_units,
                None,
                "A share recipe requires exactly one input series.",
            )
        return check_cross_sectional_additivity(
            profiles[0].raw_unit,
            additive=additive,
        )

    if normalized_operation == "ratio":
        if len(profiles) != 2:
            return UnitCompatibility(
                False,
                "ratio",
                canonical_units,
                None,
                "A ratio requires numerator and denominator units.",
            )
        numerator, denominator = profiles
        if "unknown" in canonical_units:
            return UnitCompatibility(
                False,
                "ratio",
                canonical_units,
                None,
                "Both source units must be known before a ratio is presented.",
            )
        if (
            numerator.dimension == denominator.dimension == "currency"
            and numerator.canonical_unit != denominator.canonical_unit
        ):
            return UnitCompatibility(
                False,
                "ratio",
                canonical_units,
                None,
                "Currency units or scales differ; convert them before division.",
            )
        output_unit = (
            "ratio"
            if numerator.canonical_unit == denominator.canonical_unit
            else f"{numerator.canonical_unit} per {denominator.canonical_unit}"
        )
        warnings: tuple[str, ...] = ()
        if numerator.dimension in {"percentage", "ratio", "index"}:
            warnings = (
                "The numerator is already normalized; review whether a ratio of "
                "normalized series has an interpretable economic meaning.",
            )
        return UnitCompatibility(
            True,
            "ratio",
            canonical_units,
            output_unit,
            "Units can be divided without an unhandled currency-scale conversion.",
            warnings,
        )

    if normalized_operation in {"add", "subtract"}:
        if len(profiles) < 2:
            return UnitCompatibility(
                False,
                normalized_operation,
                canonical_units,
                None,
                "Addition and subtraction require at least two units.",
            )
        if "unknown" in canonical_units or len(set(canonical_units)) != 1:
            return UnitCompatibility(
                False,
                normalized_operation,
                canonical_units,
                None,
                "Addition and subtraction require identical declared units and scales.",
            )
        return UnitCompatibility(
            True,
            normalized_operation,
            canonical_units,
            canonical_units[0],
            "All operands use the same declared unit and scale.",
        )

    if normalized_operation == "change":
        if len(profiles) != 1:
            return UnitCompatibility(
                False,
                "change",
                canonical_units,
                None,
                "A temporal-change recipe requires exactly one input series.",
            )
        warnings = (
            ("The source unit is not supplied; label the exported source value accordingly.",)
            if profiles[0].dimension == "unknown"
            else ()
        )
        return UnitCompatibility(
            True,
            "change",
            canonical_units,
            "percent or index",
            "Temporal change compares one series with itself at different periods.",
            warnings,
        )

    raise ValueError(f"Unsupported unit-check operation: {operation}")


def check_recipe_units(
    recipe_key: str,
    units: Iterable[str | None],
    *,
    additive: bool | None = None,
) -> UnitCompatibility:
    """Apply the correct unit gate for a named task recipe."""
    recipe = get_calculation_recipe(recipe_key)
    return check_unit_compatibility(
        recipe.operation,
        units,
        additive=additive if recipe.requires_additive_input else None,
    )


def _period_bounds(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    dates = pd.to_datetime(frame.get("date"), errors="coerce")
    if not isinstance(dates, pd.Series) or not dates.notna().any():
        return None, None
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def _ordered_frequencies(values: Iterable[Any]) -> tuple[str, ...]:
    present = {str(value) for value in values if pd.notna(value) and str(value)}
    preferred = [frequency for frequency in ("M", "Q", "A") if frequency in present]
    return tuple(preferred + sorted(present.difference(preferred)))


def diagnose_alignment(
    operands: Mapping[str, pd.DataFrame],
    result: pd.DataFrame | None = None,
) -> AlignmentDiagnostics:
    """Report exact country/date/frequency matches and losses by operand."""
    if not operands:
        return AlignmentDiagnostics(
            matched_observations=0,
            output_observations=0 if result is not None else None,
            dropped_after_calculation=0 if result is not None else None,
            input_observations={},
            dropped_observations={},
            input_frequencies={},
            input_periods={},
            matched_frequencies=(),
            matched_period_start=None,
            matched_period_end=None,
            matched_countries=0,
        )

    keyed_frames: dict[str, pd.DataFrame] = {}
    input_observations: dict[str, int] = {}
    input_frequencies: dict[str, tuple[str, ...]] = {}
    input_periods: dict[str, tuple[str | None, str | None]] = {}
    key_indexes: list[pd.MultiIndex] = []
    for raw_name, frame in operands.items():
        name = str(raw_name)
        missing = set(ALIGN_KEYS).difference(frame.columns)
        if missing:
            raise ValueError(
                f"Operand {name!r} is missing alignment columns: {sorted(missing)}"
            )
        keyed = frame[ALIGN_KEYS].copy()
        keyed["date"] = pd.to_datetime(keyed["date"], errors="coerce")
        keyed = keyed.dropna(subset=ALIGN_KEYS).drop_duplicates(ALIGN_KEYS)
        keyed_frames[name] = keyed
        input_observations[name] = len(keyed)
        input_frequencies[name] = _ordered_frequencies(keyed["frequency"])
        input_periods[name] = _period_bounds(keyed)
        key_indexes.append(pd.MultiIndex.from_frame(keyed[ALIGN_KEYS]))

    intersection = key_indexes[0]
    for index in key_indexes[1:]:
        intersection = intersection.intersection(index)
    matched = intersection.to_frame(index=False, name=ALIGN_KEYS)
    matched_count = len(matched)
    dropped = {
        name: max(count - matched_count, 0)
        for name, count in input_observations.items()
    }
    matched_frequencies = _ordered_frequencies(
        matched.get("frequency", pd.Series(dtype=str))
    )
    matched_start, matched_end = _period_bounds(matched)
    output_count = None if result is None else int(len(result))
    dropped_after = (
        None if output_count is None else max(matched_count - output_count, 0)
    )
    return AlignmentDiagnostics(
        matched_observations=matched_count,
        output_observations=output_count,
        dropped_after_calculation=dropped_after,
        input_observations=input_observations,
        dropped_observations=dropped,
        input_frequencies=input_frequencies,
        input_periods=input_periods,
        matched_frequencies=matched_frequencies,
        matched_period_start=matched_start,
        matched_period_end=matched_end,
        matched_countries=(
            int(matched["country_code"].nunique()) if not matched.empty else 0
        ),
    )


def build_query_metadata(
    *,
    operation: str,
    recipe_key: str | None = None,
    dataset: str | None = None,
    source_version: str | None = None,
    indicators: Mapping[str, str] | None = None,
    countries: Iterable[str] = (),
    requested_frequency: str | None = None,
    requested_range: str | None = None,
    formula: str | None = None,
    scale: float | None = None,
    units: Mapping[str, str] | None = None,
    observation_statuses: Iterable[str] = (),
    alignment: AlignmentDiagnostics | None = None,
    unit_compatibility: UnitCompatibility | None = None,
) -> dict[str, Any]:
    """Build stable, JSON-ready metadata for chart-data or CSV exports."""
    recipe = get_calculation_recipe(recipe_key) if recipe_key else None
    normalized_operation = str(operation).strip().lower()
    if recipe is not None and recipe.operation != normalized_operation:
        raise ValueError(
            f"Recipe {recipe.key!r} is for {recipe.operation!r}, not "
            f"{normalized_operation!r}."
        )
    return {
        "schema_version": "bankenv.calculation-query.v1",
        "query": {
            "dataset": None if dataset is None else str(dataset),
            "source_version": None if source_version is None else str(source_version),
            "indicators": dict(
                sorted((str(key), str(value)) for key, value in (indicators or {}).items())
            ),
            "countries": sorted({str(country).upper() for country in countries}),
            "requested_frequency": (
                None if requested_frequency is None else str(requested_frequency)
            ),
            "requested_range": None if requested_range is None else str(requested_range),
            "observation_statuses": sorted(
                {str(status) for status in observation_statuses}
            ),
        },
        "calculation": {
            "operation": normalized_operation,
            "recipe": recipe.as_dict() if recipe else None,
            "formula": None if formula is None else str(formula),
            "scale": None if scale is None else float(scale),
            "units": dict(
                sorted((str(key), str(value)) for key, value in (units or {}).items())
            ),
            "unit_compatibility": (
                unit_compatibility.as_dict() if unit_compatibility else None
            ),
        },
        "alignment": alignment.as_dict() if alignment else None,
    }


def normalize_observation_frame(
    frame: pd.DataFrame,
    indicator_value,
    indicator_col: str,
    label: str,
) -> pd.DataFrame:
    """Return a normalized single-indicator observation frame.

    Required input columns are ``country_code``, ``period`` and ``value``.
    ``frequency`` is preserved when available; otherwise annual frequency is
    assumed to prevent mixed-frequency alignment.
    """
    required = {"country_code", "period", "value", indicator_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Observation frame is missing columns: {sorted(missing)}")

    data = frame.loc[frame[indicator_col] == indicator_value].copy()
    if data.empty:
        return pd.DataFrame(columns=ALIGN_KEYS + ["value", "indicator_label"])

    data["date"] = pd.to_datetime(data["period"].astype(str), errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    if "frequency" not in data.columns:
        data["frequency"] = "A"
    data["frequency"] = data["frequency"].fillna("A").astype(str)
    data = data.dropna(subset=["date", "value"])
    data = data.sort_values("date").drop_duplicates(
        subset=ALIGN_KEYS,
        keep="last",
    )
    data["indicator_label"] = label
    keep = ALIGN_KEYS + ["value", "indicator_label"]
    optional = [
        column
        for column in ("observation_status", "period")
        if column in data.columns
    ]
    return data[keep + optional].reset_index(drop=True)


def restrict_frequency(frame: pd.DataFrame, frequency: str | None) -> pd.DataFrame:
    """Restrict a normalized frame to one frequency when requested."""
    if frequency is None or "frequency" not in frame.columns:
        return frame.copy()
    return frame.loc[frame["frequency"] == frequency].copy()


def filter_time_range(frame: pd.DataFrame, time_range: str) -> pd.DataFrame:
    """Filter a normalized frame to a display range."""
    if frame.empty or time_range == "All Data":
        return frame.copy()
    max_date = frame["date"].max()
    years = {
        "5 Years": 5,
        "10 Years": 10,
        "20 Years": 20,
    }.get(time_range)
    if years is None:
        return frame.copy()
    return frame.loc[frame["date"] >= max_date - pd.DateOffset(years=years)].copy()


def compute_ratio(
    numerator: pd.DataFrame,
    denominator: pd.DataFrame,
    scale: float = 1.0,
) -> pd.DataFrame:
    """Compute ``numerator / denominator * scale`` on exact aligned periods."""
    merged = numerator.merge(
        denominator,
        on=ALIGN_KEYS,
        how="inner",
        suffixes=("_numerator", "_denominator"),
    )
    if merged.empty:
        return pd.DataFrame(
            columns=ALIGN_KEYS
            + [
                "value",
                "numerator_value",
                "denominator_value",
                "calculation_flag",
            ]
        )

    merged["numerator_value"] = pd.to_numeric(
        merged["value_numerator"],
        errors="coerce",
    )
    merged["denominator_value"] = pd.to_numeric(
        merged["value_denominator"],
        errors="coerce",
    )
    valid = (
        merged["numerator_value"].notna()
        & merged["denominator_value"].notna()
        & (merged["denominator_value"] != 0)
    )
    merged["value"] = np.where(
        valid,
        merged["numerator_value"] / merged["denominator_value"] * scale,
        np.nan,
    )
    merged["calculation_flag"] = np.where(valid, "calculated", "invalid_denominator")
    return merged[
        ALIGN_KEYS
        + [
            "value",
            "numerator_value",
            "denominator_value",
            "calculation_flag",
        ]
    ].dropna(subset=["value"]).reset_index(drop=True)


def validate_formula(formula: str, allowed_operands: Iterable[str]) -> FormulaPlan:
    """Validate a user formula and return a safe execution plan.

    Only operand names from ``allowed_operands``, numeric constants, arithmetic
    operators, unary signs, and parentheses are accepted. Function calls,
    attributes, indexing, comparisons and other Python syntax are rejected.
    """
    normalized = str(formula or "").strip().upper()
    allowed = {str(name).upper() for name in allowed_operands}
    allowed_text = ", ".join(sorted(allowed))
    if not normalized:
        raise FormulaValidationError(f"Enter a formula using declared operands: {allowed_text}.")
    try:
        expression = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise FormulaValidationError("Formula syntax is invalid.") from exc

    used: set[str] = set()

    def visit(node):
        if isinstance(node, ast.Expression):
            visit(node.body)
            return
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                raise FormulaValidationError("Only +, -, * and / operators are supported.")
            visit(node.left)
            visit(node.right)
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                raise FormulaValidationError("Only unary + and - signs are supported.")
            visit(node.operand)
            return
        if isinstance(node, ast.Name):
            name = node.id.upper()
            if name not in allowed:
                raise FormulaValidationError(
                    f"Unknown operand {node.id!r}. Use only: {allowed_text}."
                )
            used.add(name)
            return
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise FormulaValidationError("Only numeric constants are supported.")
            return
        raise FormulaValidationError(
            "Unsupported formula syntax. Use operands, numbers, +, -, *, / and parentheses only."
        )

    visit(expression)
    if not used:
        raise FormulaValidationError("Formula must reference at least one operand.")
    return FormulaPlan(
        expression=expression,
        normalized_formula=ast.unparse(expression),
        used_operands=tuple(name for name in sorted(used)),
    )


def compute_expression_formula(
    formula: str,
    operands: dict[str, pd.DataFrame],
    scale: float = 1.0,
) -> tuple[pd.DataFrame, FormulaPlan]:
    """Compute a validated arithmetic expression over aligned operand frames."""
    normalized_operands = {str(key).upper(): value for key, value in operands.items()}
    plan = validate_formula(formula, normalized_operands.keys())
    value_columns = [f"{name.lower()}_value" for name in plan.used_operands]

    merged = None
    for name in plan.used_operands:
        frame = normalized_operands[name].rename(columns={"value": f"{name.lower()}_value"})
        keep = ALIGN_KEYS + [f"{name.lower()}_value"]
        frame = frame[keep].copy()
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=ALIGN_KEYS, how="inner")

    if merged is None or merged.empty:
        return (
            pd.DataFrame(columns=ALIGN_KEYS + ["value", *value_columns, "calculation_flag"]),
            plan,
        )

    for column in value_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    base_valid = merged[value_columns].notna().all(axis=1)

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Name):
            return merged[f"{node.id.lower()}_value"], base_valid.copy()
        if isinstance(node, ast.Constant):
            return float(node.value), pd.Series(True, index=merged.index)
        if isinstance(node, ast.UnaryOp):
            values, valid = evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                return -values, valid
            return values, valid
        if isinstance(node, ast.BinOp):
            left, left_valid = evaluate(node.left)
            right, right_valid = evaluate(node.right)
            valid = left_valid & right_valid
            if isinstance(node.op, ast.Add):
                return left + right, valid
            if isinstance(node.op, ast.Sub):
                return left - right, valid
            if isinstance(node.op, ast.Mult):
                return left * right, valid
            denominator_ok = right != 0
            valid = valid & denominator_ok
            return left / right, valid
        raise FormulaValidationError("Unsupported formula syntax.")

    values, valid = evaluate(plan.expression)
    merged["value"] = np.where(valid, values * scale, np.nan)
    merged["calculation_flag"] = np.where(valid, "calculated", "invalid_formula_input")
    result_columns = ALIGN_KEYS + ["value", *value_columns, "calculation_flag"]
    return merged[result_columns].dropna(subset=["value"]).reset_index(drop=True), plan


def compute_cross_sectional_share(
    frame: pd.DataFrame,
    group_keys: Iterable[str] = ("date", "frequency"),
    scale: float = 100.0,
) -> pd.DataFrame:
    """Compute country share of the selected-country group total by period."""
    data = frame.copy()
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["value"])
    group_keys = list(group_keys)
    totals = (
        data.groupby(group_keys, dropna=False)["value"]
        .sum()
        .rename("group_total")
        .reset_index()
    )
    data = data.merge(totals, on=group_keys, how="left")
    valid = data["group_total"].notna() & (data["group_total"] != 0)
    data["raw_value"] = data["value"]
    data["value"] = np.where(valid, data["raw_value"] / data["group_total"] * scale, np.nan)
    data["calculation_flag"] = np.where(valid, "calculated", "invalid_group_total")
    return data.dropna(subset=["value"]).reset_index(drop=True)


def compute_temporal_change(
    frame: pd.DataFrame,
    mode: str,
    group_cols: Iterable[str] = ("country_code",),
) -> pd.DataFrame:
    """Compute period change, base-period change, or rebased index.

    ``mode`` values:
    - ``period_pct``: period-over-period percent change
    - ``base_pct``: percent change from first observation in each group
    - ``index_100``: first observation in each group equals 100
    """
    if mode not in {"period_pct", "base_pct", "index_100"}:
        raise ValueError(f"Unsupported temporal mode: {mode}")

    data = frame.copy()
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["date", "value"]).sort_values(list(group_cols) + ["date"])
    group_cols = list(group_cols)

    if mode == "period_pct":
        data["raw_value"] = data["value"]
        data["value"] = data.groupby(group_cols)["raw_value"].pct_change() * 100
        data["calculation_flag"] = "period_pct_change"
    else:
        first = data.groupby(group_cols)["value"].transform("first")
        valid = first.notna() & (first != 0)
        data["raw_value"] = data["value"]
        if mode == "base_pct":
            data["value"] = np.where(valid, (data["raw_value"] / first - 1) * 100, np.nan)
            data["calculation_flag"] = "base_pct_change"
        else:
            data["value"] = np.where(valid, data["raw_value"] / first * 100, np.nan)
            data["calculation_flag"] = "index_100"

    return data.dropna(subset=["value"]).reset_index(drop=True)


def available_frequencies(frame: pd.DataFrame) -> list[str]:
    """Return supported frequencies in display order."""
    if "frequency" not in frame.columns:
        return []
    present = set(frame["frequency"].dropna().astype(str))
    return [frequency for frequency in ("M", "Q", "A") if frequency in present]
