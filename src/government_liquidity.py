"""General-government (sovereign fiscal) liquidity feature construction.

The existing staged block in :mod:`src.external_liquidity` covers *external*
liquidity (IMF BOP/IIP plus World Bank external-debt-service). It does not
model the *government's own* fiscal liquidity: gross public debt, the primary
and structural balance, the interest burden relative to revenue, and the
overall financing requirement. Rating-agency style sovereign analysis
(Fitch/Moody's/S&P) leans heavily on interest-to-revenue and debt-to-revenue
affordability, not on external ratios alone.

Those general-government series already live in the WEO cache that the model
pipeline builds, so this block is derivable and testable from local data
without the BOP/IIP/World-Bank API calls that the external block depends on.

The features here are **staged challenger inputs**. They are not wired into
production scoring; a separate challenger comparison must be reviewed before
any promoted model consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd

from src.config import BASE_DIR, CACHE_DIR
from src.lfs_resolver import ensure_lfs_file
from src.model_store import load_model_artifact


WEO_CACHE_PATH = Path(CACHE_DIR) / "WEO_cache.parquet"

GOVT_FEATURE_OBSERVATIONS = Path(CACHE_DIR) / "government" / "government_liquidity_observations.parquet"
GOVT_FEATURE_VALUES = Path(CACHE_DIR) / "government" / "government_liquidity_features.parquet"
GOVT_FEATURE_REPORT = Path(BASE_DIR) / "artifacts" / "government_liquidity_features_report.json"


@dataclass(frozen=True)
class FiscalSeriesSpec:
    """A general-government WEO series mapped to a raw observation key."""

    feature_key: str
    weo_code: str
    label: str
    quality: str = "observed"


# Raw general-government series pulled from the WEO cache. All are percent of
# GDP except where noted. These map to the native WEO indicator codes; the
# derived affordability ratios are computed in
# :func:`build_government_liquidity_features`.
FISCAL_SERIES_SPECS: tuple[FiscalSeriesSpec, ...] = (
    FiscalSeriesSpec("gross_debt_gdp", "GGXWDG_NGDP", "General government gross debt, % of GDP"),
    FiscalSeriesSpec("fiscal_balance_gdp", "GGXCNL_NGDP", "General government net lending/borrowing, % of GDP"),
    FiscalSeriesSpec("primary_balance_gdp", "GGXONLB_NGDP", "General government primary balance, % of GDP"),
    FiscalSeriesSpec("revenue_gdp", "GGR_NGDP", "General government revenue, % of GDP"),
    FiscalSeriesSpec("expenditure_gdp", "GGX_NGDP", "General government total expenditure, % of GDP"),
    FiscalSeriesSpec("structural_balance_potential_gdp", "GGSB_NPGDP", "General government structural balance, % of potential GDP"),
)

_WEO_CODE_BY_KEY = {spec.weo_code: spec for spec in FISCAL_SERIES_SPECS}


def load_weo_fiscal_observations(
    weo_df: pd.DataFrame | None = None,
    as_of_date=None,
    model_countries: list[str] | None = None,
    include_estimates: bool = True,
    include_projections: bool = False,
    specs: tuple[FiscalSeriesSpec, ...] = FISCAL_SERIES_SPECS,
) -> pd.DataFrame:
    """Return long-format general-government observations from the WEO cache.

    The cutoff and observation-status selection mirror
    ``CrisisFeatureEngineer.extract_weo_features`` so this staged block cannot
    silently admit projections that the production feature path excludes.
    """
    columns = [
        "source", "feature_key", "feature_label", "quality", "dataset_version",
        "country_code", "period_label", "period", "observation_status", "value",
    ]
    if weo_df is None:
        ensure_lfs_file(WEO_CACHE_PATH)
        weo_df = pd.read_parquet(WEO_CACHE_PATH)
    if weo_df is None or weo_df.empty:
        return pd.DataFrame(columns=columns)

    cutoff = pd.Timestamp(as_of_date or f"{pd.Timestamp.today().year - 1}-12-31")
    frame = weo_df.copy()
    frame = frame[frame["indicator_code"].astype(str).isin(_WEO_CODE_BY_KEY.keys())]
    frame = frame[pd.to_datetime(frame["period"], errors="coerce") <= cutoff]

    if "observation_status" in frame.columns:
        allowed = {"actual", "unknown"}
        if include_estimates:
            allowed.add("estimate")
        if include_projections:
            allowed.add("projection")
        frame = frame[frame["observation_status"].isin(allowed)]
    else:
        frame["observation_status"] = "unknown"

    frame = frame.copy()
    frame["country_code"] = frame["country_code"].astype(str).str.strip().str.upper()
    if model_countries is not None:
        keep = {str(code).upper() for code in model_countries}
        frame = frame[frame["country_code"].isin(keep)]

    frame = frame[frame["country_code"].str.fullmatch(r"[A-Z]{3}")]
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[frame["value"].notna()]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    spec_map = _WEO_CODE_BY_KEY
    normalized = pd.DataFrame(
        {
            "source": "WEO",
            "feature_key": frame["indicator_code"].map(lambda c: spec_map[c].feature_key),
            "feature_label": frame["indicator_code"].map(lambda c: spec_map[c].label),
            "quality": frame["indicator_code"].map(lambda c: spec_map[c].quality),
            "dataset_version": frame.get("dataset", "WEO"),
            "country_code": frame["country_code"],
            "period_label": pd.to_datetime(frame["period"]).dt.year.astype(str),
            "period": pd.to_datetime(frame["period"]),
            "observation_status": frame["observation_status"],
            "value": frame["value"],
        }
    )
    return normalized.reset_index(drop=True)


def latest_fiscal_matrix(observations: pd.DataFrame) -> pd.DataFrame:
    """Latest value and period per country-feature (wide)."""
    if observations is None or observations.empty:
        return pd.DataFrame(columns=["country_code"])
    latest = (
        observations.sort_values("period")
        .groupby(["country_code", "feature_key"], as_index=False)
        .last()
    )
    values = latest.pivot(index="country_code", columns="feature_key", values="value")
    periods = latest.pivot(index="country_code", columns="feature_key", values="period")
    periods = periods.rename(columns={col: f"{col}_period" for col in periods.columns})
    result = values.join(periods).reset_index()
    result.columns.name = None
    return result


def safe_ratio(numerator: pd.Series, denominator: pd.Series, scale: float = 100.0) -> pd.Series:
    if numerator is None:
        return pd.Series(pd.NA, index=denominator.index, dtype="Float64")
    denominator = denominator.replace({0: pd.NA})
    return numerator / denominator * scale


def build_government_liquidity_features(
    observations: pd.DataFrame,
    model_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Compute staged general-government liquidity features and a coverage report."""
    model_country_set = None
    if model_features is None:
        model = load_model_artifact()
        model_features = model["feature_values"]
        model_country_set = set(
            model["country_scores"]["country_code"].dropna().astype(str).str.upper()
        )

    base = model_features[["country_code"]].copy()
    base["country_code"] = base["country_code"].astype(str).str.upper()
    if model_country_set is not None:
        base = base[base["country_code"].isin(model_country_set)]
    base = base.drop_duplicates("country_code")

    raw = latest_fiscal_matrix(observations)
    features = base.merge(raw, on="country_code", how="left")

    def col(name: str) -> pd.Series:
        if name in features:
            return pd.to_numeric(features[name], errors="coerce")
        return pd.Series(pd.NA, index=features.index, dtype="Float64")

    gross_debt = col("gross_debt_gdp")
    fiscal_balance = col("fiscal_balance_gdp")
    primary_balance = col("primary_balance_gdp")
    revenue = col("revenue_gdp")

    # Interest burden as % of GDP. Overall balance already nets out interest,
    # so interest expense = primary balance - overall balance.
    interest_gdp = primary_balance - fiscal_balance

    derived = pd.DataFrame({"country_code": features["country_code"]})
    derived["govt_gross_debt_gdp"] = gross_debt
    derived["govt_fiscal_balance_gdp"] = fiscal_balance
    derived["govt_primary_balance_gdp"] = primary_balance
    derived["govt_revenue_gdp"] = revenue
    derived["govt_expenditure_gdp"] = col("expenditure_gdp")
    derived["govt_structural_balance_gdp"] = col("structural_balance_potential_gdp")
    derived["govt_interest_gdp"] = interest_gdp
    # Core sovereign-liquidity affordability ratios.
    derived["govt_interest_to_revenue"] = safe_ratio(interest_gdp, revenue)
    derived["govt_debt_to_revenue"] = safe_ratio(gross_debt, revenue)
    # Financing requirement flow signals (deficit floored at zero).
    overall_deficit = (-fiscal_balance).where(fiscal_balance < 0, 0)
    primary_deficit = (-primary_balance).where(primary_balance < 0, 0)
    derived["govt_overall_deficit_gdp"] = overall_deficit
    derived["govt_primary_deficit_gdp"] = primary_deficit

    derived["government_liquidity_feature_count"] = (
        derived.drop(columns=["country_code"]).notna().sum(axis=1)
    )

    feature_cols = [c for c in derived.columns if c != "country_code"]
    model_country_count = int(base["country_code"].nunique())
    coverage_counts = {}
    for feature in feature_cols:
        if feature == "government_liquidity_feature_count":
            coverage_counts[feature] = int((derived[feature] > 0).sum())
        else:
            coverage_counts[feature] = int(derived[feature].notna().sum())

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_countries": model_country_count,
        "observation_rows": int(len(observations)) if observations is not None else 0,
        "observation_countries": (
            int(observations["country_code"].nunique())
            if observations is not None and not observations.empty
            else 0
        ),
        "feature_coverage": {
            feature: {
                "countries": coverage_counts[feature],
                "pct_model_countries": (
                    round(float(coverage_counts[feature]) / model_country_count * 100, 1)
                    if model_country_count
                    else 0.0
                ),
            }
            for feature in feature_cols
        },
        "notes": [
            "Features are staged challenger inputs and are not wired into production scoring.",
            "Source: IMF WEO general-government fiscal series (GGXWDG_NGDP, GGXCNL_NGDP, "
            "GGXONLB_NGDP, GGR_NGDP, GGX_NGDP, GGSB_NPGDP). The cutoff and observation-status "
            "selection match the production WEO feature path (actuals and estimates only; "
            "projections excluded).",
            "govt_interest_gdp is derived as primary balance minus overall balance because "
            "the overall balance already nets out interest; it is an implied interest bill, "
            "not a reported interest-expense series.",
            "govt_interest_to_revenue and govt_debt_to_revenue are the rating-agency-style "
            "affordability ratios (interest burden and debt stock relative to revenue capacity).",
            "govt_overall_deficit_gdp and govt_primary_deficit_gdp are financing-requirement "
            "flow signals; a full gross financing need also needs debt amortization/rollover, "
            "which WEO does not carry and remains an IMF Fiscal Monitor / GFS source gap.",
            "govt_structural_balance_gdp (GGSB_NPGDP) has limited country coverage in WEO.",
        ],
    }
    return derived, report


def model_country_codes() -> list[str]:
    scores = load_model_artifact()["country_scores"]
    return sorted(scores["country_code"].dropna().astype(str).str.upper().unique().tolist())


def write_government_liquidity_outputs(
    observations: pd.DataFrame,
    features: pd.DataFrame,
    report: dict,
    observations_path: Path = GOVT_FEATURE_OBSERVATIONS,
    features_path: Path = GOVT_FEATURE_VALUES,
    report_path: Path = GOVT_FEATURE_REPORT,
) -> None:
    observations_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    observations.to_parquet(observations_path, index=False)
    features.to_parquet(features_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
