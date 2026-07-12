"""Leakage-aware annual panel construction for banking-crisis models.

The production scoring model is cross-sectional, while crisis-model validation
needs country-year observations.  This module creates those observations from
normalised long-form WEO and FSIC data without filling unavailable features.

The panel is deliberately explicit about the two time axes:

* ``forecast_origin_year`` is when a prediction is notionally made.
* ``feature_cutoff_year`` is the latest observation year that may be selected.

Targets refer to crisis *starts* in the configured forward horizon.  Active
crisis years and the configurable post-crisis cooldown are removed, and their
reasons are retained in the audit result.  The caller can therefore distinguish
genuine negatives from contaminated or right-censored observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


_STANDARD_OBSERVATION_COLUMNS = (
    "country_code",
    "value",
    "observation_year",
    "_period_sort",
    "indicator_code",
    "indicator_name",
    "observation_status",
    "_input_direct",
)


@dataclass(frozen=True)
class FeatureSpec:
    """Declare one source feature and its availability policy.

    ``indicator_code`` and ``indicator_name_pattern`` are combined when both
    are supplied.  The latter is a case-insensitive regular expression.
    ``max_age_years`` controls the availability flag; stale values remain in
    the panel for auditability but are never described as available.
    """

    name: str
    source: str
    family: str
    indicator_code: str | None = None
    indicator_name_pattern: str | None = None
    max_age_years: int | None = None
    allowed_statuses: tuple[str, ...] = ("actual", "estimate", "unknown")
    direct: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("FeatureSpec.name must be non-empty")
        if not self.family or not self.family.strip():
            raise ValueError(f"Feature {self.name!r} must have a family")
        if self.source.upper() not in {"WEO", "FSIC", "WB"}:
            raise ValueError(
                f"Feature {self.name!r} has unsupported source {self.source!r}"
            )
        if self.indicator_code is None and self.indicator_name_pattern is None:
            raise ValueError(
                f"Feature {self.name!r} needs an indicator code or name pattern"
            )
        if self.max_age_years is not None and self.max_age_years < 0:
            raise ValueError("max_age_years must be non-negative or None")


@dataclass(frozen=True)
class CrisisPanelConfig:
    """Time, target and contamination policy for an annual crisis panel."""

    start_year: int | None = None
    end_year: int | None = None
    horizon_start_years: int = 1
    horizon_end_years: int = 3
    feature_lag_years: int = 1
    exclude_active_crisis: bool = True
    post_crisis_cooldown_years: int = 3
    label_coverage_end_year: int | None = None
    drop_right_censored: bool = True
    family_min_coverage: float = 0.5

    def __post_init__(self) -> None:
        if self.horizon_start_years < 1:
            raise ValueError("horizon_start_years must be at least one")
        if self.horizon_end_years < self.horizon_start_years:
            raise ValueError("horizon_end_years must not precede horizon_start_years")
        if self.feature_lag_years < 0:
            raise ValueError("feature_lag_years must be non-negative")
        if self.post_crisis_cooldown_years < 0:
            raise ValueError("post_crisis_cooldown_years must be non-negative")
        if not 0.0 <= self.family_min_coverage <= 1.0:
            raise ValueError("family_min_coverage must be between zero and one")
        if (
            self.start_year is not None
            and self.end_year is not None
            and self.start_year > self.end_year
        ):
            raise ValueError("start_year must not exceed end_year")


@dataclass
class CrisisPanelResult:
    """Panel plus the records and manifests needed to audit its construction."""

    panel: pd.DataFrame
    exclusions: pd.DataFrame
    events: pd.DataFrame
    feature_manifest: pd.DataFrame
    config: CrisisPanelConfig
    diagnostics: Mapping[str, int | float | str] = field(default_factory=dict)


IMF_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("gdp_growth", "WEO", "macro", indicator_code="NGDP_RPCH", max_age_years=2),
    FeatureSpec("inflation", "WEO", "macro", indicator_code="PCPIPCH", max_age_years=2),
    FeatureSpec(
        "current_account_gdp",
        "WEO",
        "external",
        indicator_code="BCA_NGDPD",
        max_age_years=2,
    ),
    FeatureSpec(
        "govt_debt_gdp",
        "WEO",
        "government_liquidity",
        indicator_code="GGXWDG_NGDP",
        max_age_years=2,
    ),
    FeatureSpec(
        "fiscal_balance_gdp",
        "WEO",
        "government_liquidity",
        indicator_code="GGXCNL_NGDP",
        max_age_years=2,
    ),
    FeatureSpec(
        "primary_balance_gdp",
        "WEO",
        "government_liquidity",
        indicator_code="GGXONLB_NGDP",
        max_age_years=2,
    ),
    FeatureSpec("unemployment", "WEO", "macro", indicator_code="LUR", max_age_years=2),
    FeatureSpec(
        "gdp_per_capita",
        "WEO",
        "structural",
        indicator_code="NGDPDPC",
        max_age_years=3,
    ),
    FeatureSpec(
        "govt_revenue_gdp",
        "WEO",
        "government_liquidity",
        indicator_code="GGR_NGDP",
        max_age_years=2,
    ),
    FeatureSpec(
        "capital_adequacy",
        "FSIC",
        "bank_capital",
        indicator_name_pattern=r"Regulatory capital to risk-weighted assets.*Core FSI",
        max_age_years=2,
    ),
    FeatureSpec(
        "npl_ratio",
        "FSIC",
        "asset_quality",
        indicator_name_pattern=r"Nonperforming loans to total gross loans.*Core FSI",
        max_age_years=2,
    ),
    FeatureSpec(
        "roe",
        "FSIC",
        "bank_earnings",
        indicator_name_pattern=r"Return on equity.*Core FSI",
        max_age_years=2,
    ),
    FeatureSpec(
        "liquid_assets_total",
        "FSIC",
        "bank_liquidity",
        indicator_name_pattern=r"Liquid assets to total assets.*Percent",
        max_age_years=2,
    ),
    FeatureSpec(
        "npl_provisions",
        "FSIC",
        "asset_quality",
        indicator_name_pattern=r"Provisions to nonperforming loans.*Percent",
        max_age_years=2,
    ),
    FeatureSpec(
        "real_estate_loans",
        "FSIC",
        "credit_concentration",
        indicator_name_pattern=(
            r"Residential real estate loans to total gross loans.*Core FSI"
        ),
        max_age_years=2,
    ),
)


WORLD_BANK_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "bank_credit_gdp",
        "WB",
        "credit_cycle",
        indicator_code="bank_credit_gdp",
        max_age_years=2,
    ),
    FeatureSpec(
        "bank_credit_gdp_change_3y",
        "WB",
        "credit_cycle",
        indicator_code="bank_credit_gdp_change_3y",
        max_age_years=2,
        direct=False,
    ),
    FeatureSpec(
        "bank_credit_gdp_gap_10y",
        "WB",
        "credit_cycle",
        indicator_code="bank_credit_gdp_gap_10y",
        max_age_years=2,
        direct=False,
    ),
    FeatureSpec(
        "bank_credit_to_deposits",
        "WB",
        "funding",
        indicator_code="bank_credit_to_deposits",
        max_age_years=3,
    ),
    FeatureSpec(
        "bank_deposits_gdp",
        "WB",
        "funding",
        indicator_code="bank_deposits_gdp",
        max_age_years=3,
    ),
    FeatureSpec(
        "bank_zscore",
        "WB",
        "bank_resilience",
        indicator_code="bank_zscore",
        max_age_years=3,
    ),
    FeatureSpec(
        "wb_bank_npl_ratio",
        "WB",
        "asset_quality",
        indicator_code="wb_bank_npl_ratio",
        max_age_years=2,
    ),
    FeatureSpec(
        "wb_bank_capital_assets",
        "WB",
        "bank_capital",
        indicator_code="wb_bank_capital_assets",
        max_age_years=2,
    ),
    FeatureSpec(
        "wb_bank_liquid_reserves_assets",
        "WB",
        "bank_liquidity",
        indicator_code="wb_bank_liquid_reserves_assets",
        max_age_years=2,
    ),
    FeatureSpec(
        "broad_money_to_reserves",
        "WB",
        "external_liquidity",
        indicator_code="broad_money_to_reserves",
        max_age_years=2,
    ),
    FeatureSpec(
        "reserves_months_imports",
        "WB",
        "external_liquidity",
        indicator_code="reserves_months_imports",
        max_age_years=2,
    ),
    FeatureSpec(
        "lending_interest_rate",
        "WB",
        "debt_service_pressure",
        indicator_code="lending_interest_rate",
        max_age_years=2,
    ),
    FeatureSpec(
        "lending_interest_rate_change_3y",
        "WB",
        "debt_service_pressure",
        indicator_code="lending_interest_rate_change_3y",
        max_age_years=2,
        direct=False,
    ),
    FeatureSpec(
        "real_interest_rate",
        "WB",
        "debt_service_pressure",
        indicator_code="real_interest_rate",
        max_age_years=2,
    ),
    FeatureSpec(
        "commodity_export_concentration",
        "WB",
        "external_vulnerability",
        indicator_code="commodity_export_concentration",
        max_age_years=3,
        direct=False,
    ),
    FeatureSpec(
        "natural_resource_rents_gdp",
        "WB",
        "external_vulnerability",
        indicator_code="natural_resource_rents_gdp",
        max_age_years=3,
    ),
    FeatureSpec(
        "terms_of_trade_deterioration_3y",
        "WB",
        "external_vulnerability",
        indicator_code="terms_of_trade_deterioration_3y",
        max_age_years=3,
        direct=False,
    ),
    FeatureSpec(
        "commodity_shock_exposure",
        "WB",
        "external_vulnerability",
        indicator_code="commodity_shock_exposure",
        max_age_years=3,
        direct=False,
    ),
)


DEFAULT_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    *IMF_FEATURE_SPECS,
    *WORLD_BANK_FEATURE_SPECS,
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean")
    truthy = {"1", "true", "yes", "y", "direct"}
    falsy = {"0", "false", "no", "n", "derived", "imputed"}
    text = series.astype("string").str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result.loc[text.isin(truthy)] = True
    result.loc[text.isin(falsy)] = False
    return result


def _normalise_observations(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=_STANDARD_OBSERVATION_COLUMNS)
    if "country_code" not in frame.columns or "value" not in frame.columns:
        raise ValueError("Normalised observations require country_code and value columns")

    data = frame.copy()
    data["country_code"] = (
        data["country_code"].astype("string").str.strip().str.upper()
    )
    data["value"] = pd.to_numeric(data["value"], errors="coerce")

    if "period" in data.columns:
        if pd.api.types.is_numeric_dtype(data["period"]):
            numeric_year = pd.to_numeric(data["period"], errors="coerce").astype(
                "Int64"
            )
            period = pd.to_datetime(
                numeric_year.astype("string"), format="%Y", errors="coerce"
            )
        else:
            period = pd.to_datetime(data["period"], errors="coerce")
    else:
        year_col = "observation_year" if "observation_year" in data.columns else "year"
        if year_col not in data.columns:
            raise ValueError(
                "Normalised observations require period, observation_year, or year"
            )
        numeric_year = pd.to_numeric(data[year_col], errors="coerce").astype("Int64")
        period = pd.to_datetime(numeric_year.astype("string"), format="%Y", errors="coerce")

    data["_period_sort"] = period
    data["observation_year"] = period.dt.year.astype("Int64")
    data["indicator_code"] = data.get(
        "indicator_code", pd.Series(pd.NA, index=data.index, dtype="string")
    ).astype("string")
    data["indicator_name"] = data.get(
        "indicator_name", pd.Series(pd.NA, index=data.index, dtype="string")
    ).astype("string")
    data["observation_status"] = (
        data.get(
            "observation_status",
            pd.Series("unknown", index=data.index, dtype="string"),
        )
        .astype("string")
        .fillna("unknown")
        .str.strip()
        .str.lower()
    )
    direct_col = "is_direct" if "is_direct" in data.columns else "direct"
    if direct_col in data.columns:
        data["_input_direct"] = _coerce_bool(data[direct_col])
    else:
        data["_input_direct"] = pd.Series(
            pd.NA, index=data.index, dtype="boolean"
        )

    data = data.dropna(
        subset=["country_code", "value", "observation_year", "_period_sort"]
    )
    return data.loc[:, _STANDARD_OBSERVATION_COLUMNS].reset_index(drop=True)


def _country_frame(country_universe: Iterable[str] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(country_universe, pd.DataFrame):
        if "country_code" not in country_universe.columns:
            raise ValueError("country_universe DataFrame requires country_code")
        keep = ["country_code"]
        if "country_name" in country_universe.columns:
            keep.append("country_name")
        countries = country_universe.loc[:, keep].copy()
    else:
        countries = pd.DataFrame({"country_code": list(country_universe)})
    countries["country_code"] = (
        countries["country_code"].astype("string").str.strip().str.upper()
    )
    countries = countries[countries["country_code"].str.len().eq(3)]
    return countries.drop_duplicates("country_code", keep="first").reset_index(drop=True)


def crisis_event_frame(labels: object) -> pd.DataFrame:
    """Normalise a ``CrisisLabels``-like object's episodes into event rows."""

    crises = getattr(labels, "crises", None)
    if crises is None:
        crises = getattr(labels, "SYSTEMIC_CRISES", None)
    if not isinstance(crises, Mapping):
        raise ValueError("labels must expose a crises or SYSTEMIC_CRISES mapping")

    records: list[dict[str, int | str]] = []
    for country, periods in crises.items():
        code = str(country).strip().upper()
        for start, end in periods:
            start_year, end_year = int(start), int(end)
            if end_year < start_year:
                raise ValueError(f"Invalid crisis interval for {code}: {start}-{end}")
            records.append(
                {
                    "country_code": code,
                    "crisis_start_year": start_year,
                    "crisis_end_year": end_year,
                    "crisis_event_id": f"{code}-{start_year}-{end_year}",
                }
            )
    if not records:
        return pd.DataFrame(
            columns=[
                "country_code",
                "crisis_start_year",
                "crisis_end_year",
                "crisis_event_id",
            ]
        )
    return (
        pd.DataFrame.from_records(records)
        .drop_duplicates(["country_code", "crisis_start_year", "crisis_end_year"])
        .sort_values(["country_code", "crisis_start_year", "crisis_end_year"])
        .reset_index(drop=True)
    )


def _resolve_origin_bounds(
    config: CrisisPanelConfig,
    observations: Sequence[pd.DataFrame],
    label_coverage_end_year: int | None,
) -> tuple[int, int]:
    years = pd.concat(
        [frame["observation_year"] for frame in observations if not frame.empty],
        ignore_index=True,
    ) if any(not frame.empty for frame in observations) else pd.Series(dtype="Int64")
    if config.start_year is None:
        if years.empty:
            raise ValueError("start_year is required when no observations are available")
        start = int(years.min()) + config.feature_lag_years
    else:
        start = int(config.start_year)

    if config.end_year is None:
        if years.empty:
            raise ValueError("end_year is required when no observations are available")
        end = int(years.max()) + config.feature_lag_years
        if config.drop_right_censored and label_coverage_end_year is not None:
            end = min(end, label_coverage_end_year - config.horizon_end_years)
    else:
        end = int(config.end_year)
    if end < start:
        raise ValueError(
            f"Resolved end year {end} precedes resolved start year {start}"
        )
    return start, end


def _label_and_filter_grid(
    grid: pd.DataFrame,
    events: pd.DataFrame,
    config: CrisisPanelConfig,
    coverage_end_year: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = grid.copy()
    grid["target_window_start_year"] = (
        grid["forecast_origin_year"] + config.horizon_start_years
    )
    grid["target_window_end_year"] = (
        grid["forecast_origin_year"] + config.horizon_end_years
    )

    pairs = grid[["_row_id", "country_code", "forecast_origin_year"]].merge(
        events, on="country_code", how="left"
    )
    has_event = pairs["crisis_start_year"].notna()
    active = pd.Series(False, index=pairs.index)
    cooldown = pd.Series(False, index=pairs.index)
    target = pd.Series(False, index=pairs.index)
    event_rows = pairs.loc[has_event]
    active.loc[has_event] = event_rows["forecast_origin_year"].between(
        event_rows["crisis_start_year"],
        event_rows["crisis_end_year"],
        inclusive="both",
    )
    cooldown.loc[has_event] = (
        event_rows["forecast_origin_year"].gt(event_rows["crisis_end_year"])
        & event_rows["forecast_origin_year"].le(
            event_rows["crisis_end_year"] + config.post_crisis_cooldown_years
        )
    )
    target.loc[has_event] = event_rows["crisis_start_year"].between(
        event_rows["forecast_origin_year"] + config.horizon_start_years,
        event_rows["forecast_origin_year"] + config.horizon_end_years,
        inclusive="both",
    )

    contaminated = pairs.assign(_active=active, _cooldown=cooldown).groupby(
        "_row_id", sort=False
    )[["_active", "_cooldown"]].any()
    grid = grid.merge(contaminated, left_on="_row_id", right_index=True, how="left")
    grid[["_active", "_cooldown"]] = grid[["_active", "_cooldown"]].fillna(False)

    target_events = pairs.loc[
        target,
        [
            "_row_id",
            "crisis_event_id",
            "crisis_start_year",
            "crisis_end_year",
        ],
    ].sort_values(["_row_id", "crisis_start_year", "crisis_end_year"])
    event_counts = target_events.groupby("_row_id").size().rename("target_event_count")
    first_events = target_events.drop_duplicates("_row_id", keep="first")
    grid = grid.merge(first_events, on="_row_id", how="left").merge(
        event_counts, left_on="_row_id", right_index=True, how="left"
    )
    grid["target_event_count"] = grid["target_event_count"].fillna(0).astype("int16")
    grid["crisis_target"] = grid["crisis_event_id"].notna().astype("int8")
    grid["years_to_crisis"] = (
        grid["crisis_start_year"] - grid["forecast_origin_year"]
    ).astype("Int64")

    grid["right_censored"] = False
    if coverage_end_year is not None:
        grid["right_censored"] = grid["target_window_end_year"].gt(
            coverage_end_year
        )
    grid["active_crisis"] = grid["_active"].astype(bool)
    grid["post_crisis_cooldown"] = grid["_cooldown"].astype(bool)

    exclude = pd.Series(False, index=grid.index)
    if config.exclude_active_crisis:
        exclude |= grid["active_crisis"]
    exclude |= grid["post_crisis_cooldown"]
    if config.drop_right_censored:
        exclude |= grid["right_censored"]

    reasons = np.select(
        [
            grid["active_crisis"],
            grid["post_crisis_cooldown"],
            grid["right_censored"],
        ],
        ["active_crisis", "post_crisis_cooldown", "right_censored"],
        default="",
    )
    grid["exclusion_reason"] = reasons
    exclusions = grid.loc[exclude].copy()
    panel = grid.loc[~exclude].copy()
    internal = ["_active", "_cooldown"]
    return panel.drop(columns=internal), exclusions.drop(columns=internal)


def _select_feature(
    grid: pd.DataFrame,
    observations: pd.DataFrame,
    spec: FeatureSpec,
) -> pd.DataFrame:
    metadata_columns = [
        spec.name,
        f"{spec.name}__observation_year",
        f"{spec.name}__age_years",
        f"{spec.name}__direct",
        f"{spec.name}__missing",
        f"{spec.name}__stale",
        f"{spec.name}__available",
        f"{spec.name}__status",
        f"{spec.name}__indicator_code",
        f"{spec.name}__indicator_name",
    ]
    if observations.empty:
        result = grid[["_row_id"]].copy()
        result[spec.name] = np.nan
        result[f"{spec.name}__observation_year"] = pd.Series(
            pd.NA, index=result.index, dtype="Int64"
        )
        result[f"{spec.name}__age_years"] = pd.Series(
            pd.NA, index=result.index, dtype="Int64"
        )
        result[f"{spec.name}__direct"] = False
        result[f"{spec.name}__missing"] = True
        result[f"{spec.name}__stale"] = False
        result[f"{spec.name}__available"] = False
        result[f"{spec.name}__status"] = pd.NA
        result[f"{spec.name}__indicator_code"] = pd.NA
        result[f"{spec.name}__indicator_name"] = pd.NA
        return result[["_row_id", *metadata_columns]]

    selected = observations
    if spec.indicator_code is not None:
        selected = selected[
            selected["indicator_code"].str.upper().eq(spec.indicator_code.upper())
        ]
    if spec.indicator_name_pattern is not None:
        selected = selected[
            selected["indicator_name"].str.contains(
                spec.indicator_name_pattern, case=False, na=False, regex=True
            )
        ]
    allowed = {status.lower() for status in spec.allowed_statuses}
    selected = selected[selected["observation_status"].isin(allowed)].copy()
    selected = selected[
        selected["country_code"].isin(grid["country_code"].unique())
    ]
    if selected.empty:
        return _select_feature(grid, pd.DataFrame(), spec)

    # Select one deterministic source record per country/year first (the latest
    # period within the year), then an as-of record for each forecast cutoff.
    selected = selected.sort_values(
        [
            "country_code",
            "observation_year",
            "_period_sort",
            "indicator_code",
            "indicator_name",
        ],
        na_position="first",
    ).drop_duplicates(["country_code", "observation_year"], keep="last")

    left = grid[["_row_id", "country_code", "forecast_origin_year", "feature_cutoff_year"]]
    left = left.sort_values(["feature_cutoff_year", "country_code"])
    # ``merge_asof`` requires the exact same physical dtype on both time keys;
    # selected observations are non-null here, so a NumPy int is appropriate.
    right = selected.copy()
    right["observation_year"] = right["observation_year"].astype("int64")
    right = right.sort_values(["observation_year", "country_code"])
    matched = pd.merge_asof(
        left,
        right,
        left_on="feature_cutoff_year",
        right_on="observation_year",
        by="country_code",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_row_id")

    observation_year = matched["observation_year"].astype("Int64")
    if (
        observation_year.notna()
        & observation_year.gt(matched["feature_cutoff_year"])
    ).any():
        raise AssertionError(f"Future observation selected for feature {spec.name}")

    missing = matched["value"].isna()
    age = (matched["forecast_origin_year"] - observation_year).astype("Int64")
    if spec.max_age_years is None:
        stale = pd.Series(False, index=matched.index)
    else:
        stale = age.gt(spec.max_age_years).fillna(False)
    explicit_direct = matched["_input_direct"].astype("boolean")
    direct = explicit_direct.fillna(spec.direct).fillna(False).astype(bool) & ~missing

    result = matched[["_row_id"]].copy()
    result[spec.name] = matched["value"]
    result[f"{spec.name}__observation_year"] = observation_year
    result[f"{spec.name}__age_years"] = age
    result[f"{spec.name}__direct"] = direct
    result[f"{spec.name}__missing"] = missing.astype(bool)
    result[f"{spec.name}__stale"] = stale.astype(bool)
    result[f"{spec.name}__available"] = (~missing & ~stale).astype(bool)
    result[f"{spec.name}__status"] = matched["observation_status"]
    result[f"{spec.name}__indicator_code"] = matched["indicator_code"]
    result[f"{spec.name}__indicator_name"] = matched["indicator_name"]
    return result[["_row_id", *metadata_columns]]


def _add_family_availability(
    panel: pd.DataFrame,
    specs: Sequence[FeatureSpec],
    minimum_coverage: float,
) -> pd.DataFrame:
    panel = panel.copy()
    families: dict[str, list[str]] = {}
    for spec in specs:
        families.setdefault(spec.family, []).append(spec.name)
    available_family_columns: list[str] = []
    for family, features in families.items():
        prefix = f"family_{_safe_name(family)}"
        flags = panel[[f"{name}__available" for name in features]].astype(bool)
        count = flags.sum(axis=1).astype("int16")
        coverage = count / len(features)
        panel[f"{prefix}__available_count"] = count
        panel[f"{prefix}__feature_count"] = len(features)
        panel[f"{prefix}__coverage_ratio"] = coverage
        panel[f"{prefix}__available"] = coverage.ge(minimum_coverage)
        panel[f"{prefix}__complete"] = count.eq(len(features))
        available_family_columns.append(f"{prefix}__available")
    if available_family_columns:
        panel["available_family_count"] = panel[available_family_columns].sum(axis=1).astype("int16")
        panel["total_family_count"] = len(available_family_columns)
    else:
        panel["available_family_count"] = 0
        panel["total_family_count"] = 0
    return panel


def build_crisis_panel_result(
    weo_df: pd.DataFrame | None,
    fsic_df: pd.DataFrame | None,
    labels: object,
    country_universe: Iterable[str] | pd.DataFrame,
    feature_specs: Sequence[FeatureSpec] = DEFAULT_FEATURE_SPECS,
    config: CrisisPanelConfig | None = None,
    additional_sources: Mapping[str, pd.DataFrame] | None = None,
) -> CrisisPanelResult:
    """Build the annual panel and return its full construction audit.

    Values are selected with an as-of join at ``feature_cutoff_year``.  Missing
    and stale observations remain explicit; this function performs no median,
    mean, forward-across-origin, or model-based imputation.
    """

    config = config or CrisisPanelConfig()
    specs = tuple(feature_specs)
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Feature names must be unique")
    countries = _country_frame(country_universe)
    if countries.empty:
        raise ValueError("country_universe contains no valid ISO3 country codes")

    observations = {
        "WEO": _normalise_observations(weo_df),
        "FSIC": _normalise_observations(fsic_df),
    }
    for source_name, source_frame in (additional_sources or {}).items():
        observations[str(source_name).upper()] = _normalise_observations(source_frame)
    for source_name in {spec.source.upper() for spec in specs}:
        observations.setdefault(
            source_name,
            pd.DataFrame(columns=_STANDARD_OBSERVATION_COLUMNS),
        )
    events = crisis_event_frame(labels)
    inferred_coverage = getattr(labels, "SOURCE_COVERAGE_END_YEAR", None)
    coverage_end = (
        config.label_coverage_end_year
        if config.label_coverage_end_year is not None
        else int(inferred_coverage) if inferred_coverage is not None else None
    )
    start, end = _resolve_origin_bounds(
        config, tuple(observations.values()), coverage_end
    )

    origins = pd.DataFrame({"forecast_origin_year": range(start, end + 1)})
    grid = countries.merge(origins, how="cross")
    grid["feature_cutoff_year"] = (
        grid["forecast_origin_year"] - config.feature_lag_years
    )
    grid = grid.sort_values(["country_code", "forecast_origin_year"]).reset_index(drop=True)
    grid["_row_id"] = np.arange(len(grid), dtype=np.int64)
    grid, exclusions = _label_and_filter_grid(grid, events, config, coverage_end)

    for spec in specs:
        selected = _select_feature(grid, observations[spec.source.upper()], spec)
        grid = grid.merge(selected, on="_row_id", how="left", validate="one_to_one")
    grid = _add_family_availability(grid, specs, config.family_min_coverage)

    if grid.duplicated(["country_code", "forecast_origin_year"]).any():
        raise AssertionError("Duplicate country-origin observations in crisis panel")
    for spec in specs:
        obs_year = grid[f"{spec.name}__observation_year"]
        if (obs_year.notna() & obs_year.gt(grid["feature_cutoff_year"])).any():
            raise AssertionError(f"Feature cutoff violated for {spec.name}")

    feature_manifest = pd.DataFrame(
        [
            {
                "feature": spec.name,
                "source": spec.source.upper(),
                "family": spec.family,
                "indicator_code": spec.indicator_code,
                "indicator_name_pattern": spec.indicator_name_pattern,
                "max_age_years": spec.max_age_years,
                "allowed_statuses": ",".join(spec.allowed_statuses),
                "direct_source_feature": spec.direct,
            }
            for spec in specs
        ]
    )
    diagnostics: dict[str, int | float | str] = {
        "panel_rows": len(grid),
        "countries": int(grid["country_code"].nunique()),
        "forecast_origins": int(grid["forecast_origin_year"].nunique()),
        "positive_rows": int(grid["crisis_target"].sum()),
        "unique_positive_events": int(
            grid.loc[grid["crisis_target"].eq(1), "crisis_event_id"].nunique()
        ),
        "excluded_rows": len(exclusions),
        "start_year": start,
        "end_year": end,
        "label_coverage_end_year": coverage_end if coverage_end is not None else "unknown",
    }
    grid = grid.drop(columns=["_row_id", "exclusion_reason"], errors="ignore")
    exclusions = exclusions.drop(columns=["_row_id"], errors="ignore")
    grid.attrs["feature_families"] = {
        family: [spec.name for spec in specs if spec.family == family]
        for family in dict.fromkeys(spec.family for spec in specs)
    }
    grid.attrs["diagnostics"] = diagnostics
    return CrisisPanelResult(
        panel=grid.reset_index(drop=True),
        exclusions=exclusions.reset_index(drop=True),
        events=events,
        feature_manifest=feature_manifest,
        config=config,
        diagnostics=diagnostics,
    )


def build_crisis_panel(
    weo_df: pd.DataFrame | None,
    fsic_df: pd.DataFrame | None,
    labels: object,
    country_universe: Iterable[str] | pd.DataFrame,
    feature_specs: Sequence[FeatureSpec] = DEFAULT_FEATURE_SPECS,
    config: CrisisPanelConfig | None = None,
    additional_sources: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Convenience API returning only the model-ready annual panel."""

    return build_crisis_panel_result(
        weo_df=weo_df,
        fsic_df=fsic_df,
        labels=labels,
        country_universe=country_universe,
        feature_specs=feature_specs,
        config=config,
        additional_sources=additional_sources,
    ).panel


__all__ = [
    "CrisisPanelConfig",
    "CrisisPanelResult",
    "DEFAULT_FEATURE_SPECS",
    "IMF_FEATURE_SPECS",
    "WORLD_BANK_FEATURE_SPECS",
    "FeatureSpec",
    "build_crisis_panel",
    "build_crisis_panel_result",
    "crisis_event_frame",
]
