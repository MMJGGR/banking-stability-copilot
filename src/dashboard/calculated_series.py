"""Calculated time-series helpers for the Data Explorer.

These utilities keep Data Explorer calculations explicit and auditable. They
only align observations with the same country, date, and frequency; callers can
decide how to present missing observations instead of silently filling gaps.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable

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
