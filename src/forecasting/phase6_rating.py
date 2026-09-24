"""Transparent three-axis replacement rating.

The rating answers three different questions and keeps their evidence separate:
1) risk relative to same-year peers;
2) stress relative to the country's strictly prior history;
3) imminence of a banking or sovereign event.

Risk and evidence confidence are deliberately separate. Event risk is an
upward-only overlay so an imminent event cannot be averaged away by a benign
peer or historical position.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


CATEGORY_LABELS = (
    "1-2: Very Low Risk",
    "3-4: Low Risk",
    "5-6: Moderate Risk",
    "7-8: High Risk",
    "9-10: Very High Risk",
)


def risk_category(score: float) -> str:
    if not np.isfinite(score):
        return "Unavailable"
    if score <= 2:
        return CATEGORY_LABELS[0]
    if score <= 4:
        return CATEGORY_LABELS[1]
    if score <= 6:
        return CATEGORY_LABELS[2]
    if score <= 8:
        return CATEGORY_LABELS[3]
    return CATEGORY_LABELS[4]


def _midrank_percentile(values: pd.Series) -> pd.Series:
    """Return deterministic mid-rank percentiles in [0, 1]."""
    values = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna() & np.isfinite(values)
    n = int(valid.sum())
    if n == 0:
        return result
    rank = values.loc[valid].rank(method="average")
    result.loc[valid] = (rank - 0.5) / n
    return result.clip(0, 1)


def add_peer_relative_index(
    frame: pd.DataFrame,
    *,
    signal_col: str,
    year_col: str = "forecast_origin_year",
) -> pd.DataFrame:
    required = {signal_col, year_col}
    if not required <= set(frame):
        raise ValueError(f"Missing peer columns: {sorted(required-set(frame))}")
    result = frame.copy()
    result["peer_relative_risk_index"] = (
        result.groupby(year_col, observed=True, sort=False)[signal_col]
        .transform(_midrank_percentile)
    )
    result["peer_reference_count"] = (
        result.groupby(year_col, observed=True)[signal_col]
        .transform(lambda values: int(pd.to_numeric(values, errors="coerce").notna().sum()))
        .astype(int)
    )
    return result


def add_strict_history_index(
    frame: pd.DataFrame,
    *,
    signal_col: str,
    entity_col: str = "entity_code",
    year_col: str = "forecast_origin_year",
    minimum_prior_years: int = 5,
) -> pd.DataFrame:
    """Compare each signal only with strictly prior observations for that entity."""
    if minimum_prior_years < 1:
        raise ValueError("minimum_prior_years must be positive")
    required = {signal_col, entity_col, year_col}
    if not required <= set(frame):
        raise ValueError(f"Missing history columns: {sorted(required-set(frame))}")
    result = frame.copy()
    result[year_col] = pd.to_numeric(result[year_col], errors="raise").astype(int)
    if result.duplicated([entity_col, year_col]).any():
        raise ValueError("Entity-year rows must be unique")
    result = result.sort_values([entity_col, year_col]).copy()
    index = pd.Series(np.nan, index=result.index, dtype=float)
    prior_count = pd.Series(0, index=result.index, dtype=int)
    first_supported = pd.Series(pd.NA, index=result.index, dtype="Int64")

    for _, group in result.groupby(entity_col, observed=True, sort=False):
        prior: list[float] = []
        support_year = None
        for row_index, row in group.iterrows():
            value = pd.to_numeric(pd.Series([row[signal_col]]), errors="coerce").iloc[0]
            finite_prior = np.asarray(prior, dtype=float)
            finite_prior = finite_prior[np.isfinite(finite_prior)]
            prior_count.loc[row_index] = len(finite_prior)
            if len(finite_prior) >= minimum_prior_years and np.isfinite(value):
                less = np.sum(finite_prior < value)
                equal = np.sum(finite_prior == value)
                index.loc[row_index] = (less + 0.5 * equal) / len(finite_prior)
                if support_year is None:
                    support_year = int(row[year_col])
            if support_year is not None:
                first_supported.loc[row_index] = support_year
            if np.isfinite(value):
                # Append only after scoring, which guarantees strict past use.
                prior.append(float(value))

    result["own_history_stress_index"] = index.clip(0, 1)
    result["own_history_prior_years"] = prior_count
    result["own_history_first_supported_year"] = first_supported
    result["own_history_supported"] = prior_count.ge(minimum_prior_years)
    return result.sort_index()


def combine_event_probabilities(
    banking_probability: pd.Series,
    sovereign_probability: pd.Series,
    *,
    direct_composite_probability: pd.Series | None = None,
    preferred_method: str = "direct_if_available_else_max",
) -> pd.DataFrame:
    """Combine two governed event heads without silently assuming independence."""
    bank = pd.to_numeric(banking_probability, errors="coerce").clip(0, 1)
    sovereign = pd.to_numeric(sovereign_probability, errors="coerce").clip(0, 1)
    if not bank.index.equals(sovereign.index):
        raise ValueError("Banking and sovereign probability indices must match")
    direct = (
        pd.Series(np.nan, index=bank.index, dtype=float)
        if direct_composite_probability is None
        else pd.to_numeric(direct_composite_probability, errors="coerce").reindex(bank.index).clip(0, 1)
    )
    both = bank.notna() & sovereign.notna()
    either = bank.notna() | sovereign.notna()
    max_probability = pd.concat([bank, sovereign], axis=1).max(axis=1, skipna=True)
    max_probability.loc[~either] = np.nan
    independence = pd.Series(np.nan, index=bank.index, dtype=float)
    independence.loc[both] = 1 - (1 - bank.loc[both]) * (1 - sovereign.loc[both])

    selected = pd.Series(np.nan, index=bank.index, dtype=float)
    method = pd.Series("unavailable", index=bank.index, dtype=object)
    if preferred_method == "direct_if_available_else_max":
        has_direct = direct.notna()
        selected.loc[has_direct] = direct.loc[has_direct]
        method.loc[has_direct] = "direct_composite"
        fallback = ~has_direct & either
        selected.loc[fallback] = max_probability.loc[fallback]
        method.loc[fallback & both] = "max_lower_bound_fallback"
        method.loc[fallback & ~both] = "single_available_head"
    elif preferred_method == "max":
        selected = max_probability
        method.loc[either & both] = "max_lower_bound"
        method.loc[either & ~both] = "single_available_head"
    elif preferred_method == "independence":
        selected = independence
        method.loc[both] = "conditional_independence_sensitivity"
    else:
        raise ValueError(f"Unknown event combination method: {preferred_method}")

    coverage = pd.Series("no_event_head", index=bank.index, dtype=object)
    coverage.loc[bank.notna() & sovereign.isna()] = "banking_only"
    coverage.loc[bank.isna() & sovereign.notna()] = "sovereign_only"
    coverage.loc[both] = "banking_and_sovereign"
    return pd.DataFrame(
        {
            "banking_crisis_probability_1_3y": bank,
            "sovereign_crisis_probability_1_3y": sovereign,
            "direct_composite_event_probability_1_3y": direct,
            "event_probability_max_lower_bound": max_probability,
            "event_probability_independence_sensitivity": independence,
            "banking_or_sovereign_event_probability_1_3y": selected,
            "event_probability_method": method,
            "event_head_coverage": coverage,
        }
    )


class EmpiricalPercentileMap:
    """Training-only empirical CDF used to put event probabilities on score scale."""

    def fit(self, values: Iterable[float]):
        array = np.asarray(list(values), dtype=float)
        array = array[np.isfinite(array)]
        if not len(array):
            raise ValueError("At least one finite reference value is required")
        self.reference_ = np.sort(array)
        return self

    def transform(self, values: Iterable[float]) -> np.ndarray:
        if not hasattr(self, "reference_"):
            raise ValueError("Percentile map is not fitted")
        array = np.asarray(list(values), dtype=float)
        output = np.full(len(array), np.nan, dtype=float)
        valid = np.isfinite(array)
        if valid.any():
            left = np.searchsorted(self.reference_, array[valid], side="left")
            right = np.searchsorted(self.reference_, array[valid], side="right")
            output[valid] = ((left + right) / 2) / len(self.reference_)
        return np.clip(output, 0, 1)


@dataclass
class ThreeAxisRatingPolicy:
    """Registered monotone combination of peer, history and event axes."""

    peer_weight: float = 0.5
    event_overlay_strength: float = 1.0
    minimum_prior_years: int = 5
    observed_share_floor: float | None = None
    uncertainty_ceiling: float | None = None

    def __post_init__(self):
        if not 0 <= self.peer_weight <= 1:
            raise ValueError("peer_weight must lie in [0, 1]")
        if not 0 <= self.event_overlay_strength <= 1:
            raise ValueError("event_overlay_strength must lie in [0, 1]")
        if self.minimum_prior_years < 1:
            raise ValueError("minimum_prior_years must be positive")

    def fit_event_reference(self, event_probabilities: Iterable[float]):
        self.event_map_ = EmpiricalPercentileMap().fit(event_probabilities)
        return self

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        structural_signal_col: str = "structural_stress_probability_2y",
        event_probability_col: str = "banking_or_sovereign_event_probability_1_3y",
        observed_share_col: str = "observed_share",
        uncertainty_col: str = "state_uncertainty_proxy",
    ) -> pd.DataFrame:
        if not hasattr(self, "event_map_"):
            raise ValueError("fit_event_reference must be called first")
        result = add_peer_relative_index(frame, signal_col=structural_signal_col)
        result = add_strict_history_index(
            result,
            signal_col=structural_signal_col,
            minimum_prior_years=self.minimum_prior_years,
        )
        result["event_imminence_risk_index"] = self.event_map_.transform(
            result[event_probability_col]
        )

        peer = result["peer_relative_risk_index"]
        history = result["own_history_stress_index"]
        history_for_score = history.where(history.notna(), peer)
        base = self.peer_weight * peer + (1 - self.peer_weight) * history_for_score
        event = result["event_imminence_risk_index"]
        overlay = self.event_overlay_strength * (event - base).clip(lower=0)
        overlay = overlay.where(event.notna(), 0.0)
        final = (base + overlay).clip(0, 1)

        result["structural_risk_index"] = base
        result["event_overlay_delta"] = overlay
        result["replacement_risk_index"] = final
        result["peer_relative_risk_score"] = 1 + 9 * peer
        result["own_history_risk_score"] = 1 + 9 * history
        result["event_imminence_risk_score"] = 1 + 9 * event
        result["replacement_risk_score"] = (1 + 9 * final).clip(1, 10)
        result["replacement_risk_category"] = result.replacement_risk_score.map(risk_category)

        confidence = pd.Series("supported", index=result.index, dtype=object)
        confidence.loc[~result.own_history_supported] = "provisional_short_history"
        if observed_share_col in result and self.observed_share_floor is not None:
            low_coverage = pd.to_numeric(result[observed_share_col], errors="coerce").lt(
                self.observed_share_floor
            )
            confidence.loc[low_coverage] = "provisional_low_coverage"
        if uncertainty_col in result and self.uncertainty_ceiling is not None:
            high_uncertainty = pd.to_numeric(result[uncertainty_col], errors="coerce").gt(
                self.uncertainty_ceiling
            )
            confidence.loc[high_uncertainty] = "provisional_high_uncertainty"
        missing_structural = pd.to_numeric(result[structural_signal_col], errors="coerce").isna()
        confidence.loc[missing_structural] = "unavailable_missing_structural_signal"
        result["rating_support_status"] = confidence
        result["rating_confidence_is_separate_from_risk"] = True
        result["peer_weight"] = self.peer_weight
        result["event_overlay_strength"] = self.event_overlay_strength
        return result
