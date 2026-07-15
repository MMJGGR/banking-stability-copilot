import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import json
import re
import logging
from pathlib import Path

from src.data_loader import (
    FSIBSISLoader,
    IMFDataLoader,
    WGILoader,
    is_time_period_column,
    parse_period_label,
)
from src.country_names import fill_missing_country_names
from src.health import build_health_report
from src.model_evidence import (
    active_model_feature_codes,
    crisis_validation_display_state,
    liquidity_feature_role,
    validated_confusion_matrix_path,
)
from src import model_store

SNAPSHOT_ARCHIVE = getattr(model_store, "SNAPSHOT_ARCHIVE", None)
load_data_manifest = model_store.load_data_manifest

if hasattr(model_store, "load_model_artifact_with_fallback"):
    load_model_artifact_with_fallback = model_store.load_model_artifact_with_fallback
else:
    def load_model_artifact_with_fallback():
        """Backward-compatible active-model loader for stale deployments."""
        return model_store.load_model_artifact(), model_store.load_data_manifest(), {
            "mode": "active",
            "fallback_reason": "archive-aware loader unavailable",
        }


if hasattr(model_store, "list_archived_snapshots"):
    list_archived_snapshots = model_store.list_archived_snapshots
else:
    def list_archived_snapshots() -> list:
        return []


if hasattr(model_store, "load_archived_snapshot"):
    load_archived_snapshot = model_store.load_archived_snapshot
else:
    def load_archived_snapshot(name: str):
        raise FileNotFoundError("Archived snapshots are unavailable in this deployment")
from src.dashboard.styles import score_to_tier
from src.dashboard.presentation import (
    accessible_plotly_config,
    apply_responsive_chart_layout,
    format_identifier,
    format_pillar_label,
    render_dashboard_styles,
    render_full_label,
)
from src.dashboard.components import (
    render_time_series_deep_dive,
)
from src.dashboard.calculated_series import (
    available_frequencies,
    check_cross_sectional_additivity,
    check_unit_compatibility,
    compute_cross_sectional_share,
    compute_expression_formula,
    compute_ratio,
    compute_temporal_change,
    diagnose_alignment,
    filter_time_range,
    FormulaValidationError,
    normalize_observation_frame,
    restrict_frequency,
    validate_formula,
)
from src.dashboard.global_view import render_global_summary
from src.dashboard.evidence import (
    build_active_feature_registry,
    build_active_input_inventory,
)
from src import utils as dashboard_utils


LOGGER = logging.getLogger(__name__)


def _fallback_driver_metric_value(
    summary: dict,
    score_row: pd.Series,
    column: str,
    drivers: list[dict] | None = None,
    pipeline=None,
) -> float | None:
    """Fallback for mixed Streamlit deployments with an older ``src.utils``.

    Streamlit Cloud can briefly run a refreshed ``app.py`` against cached helper
    modules during redeploys. Keeping this small compatibility shim in the app
    prevents a full startup crash if ``src.utils.driver_metric_value`` has not
    landed yet.
    """
    value = summary.get(column) if isinstance(summary, dict) else None
    try:
        missing = value is None or pd.isna(value)
    except (TypeError, ValueError):
        missing = value is None
    if missing and column in score_row.index:
        value = score_row.get(column)
    try:
        if value is None or pd.isna(value):
            raise ValueError("missing")
        return float(value)
    except (TypeError, ValueError):
        pass

    drivers = drivers or []
    if column in {"critical_missing_share", "critical_penalty"}:
        critical_drivers = [
            driver for driver in drivers
            if driver.get("is_critical")
        ]
        if critical_drivers:
            missing_share = sum(
                1 for driver in critical_drivers
                if driver.get("is_imputed")
            ) / len(critical_drivers)
        else:
            missing_share = 0.0
        if column == "critical_missing_share":
            return float(missing_share)
        max_penalty = getattr(pipeline, "critical_missing_max_penalty", 0.0)
        return float(missing_share * max_penalty)

    if column == "crisis_uplift":
        return 0.0

    return None


find_peers = dashboard_utils.find_peers
driver_metric_value = getattr(
    dashboard_utils,
    "driver_metric_value",
    _fallback_driver_metric_value,
)

APP_PEER_FEATURE_WEIGHTS = getattr(
    dashboard_utils,
    "PEER_FEATURE_WEIGHTS",
    {
        "risk_score": 1.00,
        "economic_pillar": 0.90,
        "industry_pillar": 0.90,
        "data_coverage": 0.50,
        "nominal_gdp": 2.00,
        "gdp_per_capita": 2.00,
        "capital_adequacy": 0.45,
        "npl_ratio": 0.55,
        "liquid_assets_st_liab": 0.35,
        "customer_deposits_loans": 0.35,
        "govt_interest_to_revenue": 0.75,
        "govt_debt_to_revenue": 0.75,
        "govt_revenue_gdp": 0.40,
        "govt_primary_deficit_gdp": 0.45,
        "govt_interest_to_revenue_change_3y": 0.35,
        "govt_debt_to_revenue_change_3y": 0.35,
        "govt_primary_deficit_gdp_change_3y": 0.30,
        "govt_revenue_gdp_change_3y": 0.30,
        "net_iip_gdp": 0.65,
        "external_liabilities_gdp": 0.55,
        "reserves_to_goods_services_imports": 0.65,
        "reserves_to_current_account_payments": 0.55,
        "gross_external_financing_need_proxy_gdp": 0.75,
        "portfolio_liabilities_gdp": 0.45,
        "commodity_export_share_pct": 0.35,
        "wb_total_external_debt_service_gni_pct": 0.35,
        "wb_ppg_external_debt_service_gdp": 0.35,
        "wb_public_financing_need_ext_debt_service_proxy_gdp": 0.35,
        "current_account_gdp": 0.35,
        "govt_debt_gdp": 0.35,
    },
)


def _numeric_scalar(value) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric <= 0:
        return None
    return numeric


def _app_merge_peer_features(
    scores: pd.DataFrame,
    feature_values: pd.DataFrame | None,
) -> pd.DataFrame:
    if feature_values is None or feature_values.empty or "country_code" not in feature_values:
        return scores

    extra_columns = [
        column for column in APP_PEER_FEATURE_WEIGHTS
        if column in feature_values.columns and column not in scores.columns
    ]
    if not extra_columns:
        return scores

    features = feature_values[["country_code", *extra_columns]].copy()
    features["country_code"] = features["country_code"].astype(str).str.upper()
    features = features.drop_duplicates("country_code")
    return scores.merge(features, on="country_code", how="left")


def _app_apply_structural_peer_filters(
    target_row: pd.Series,
    candidates: pd.DataFrame,
    n_peers: int,
) -> tuple[pd.DataFrame, str]:
    filtered = candidates.copy()
    applied = []

    target_income = _numeric_scalar(target_row.get("gdp_per_capita"))
    if target_income is not None and "gdp_per_capita" in filtered.columns:
        income = pd.to_numeric(filtered["gdp_per_capita"], errors="coerce")
        income_filtered = filtered[income.between(target_income * 0.35, target_income * 2.50)]
        if len(income_filtered) >= max(4, min(n_peers, 6)):
            filtered = income_filtered
            applied.append("income")

    target_size = _numeric_scalar(target_row.get("nominal_gdp"))
    if target_size is not None and "nominal_gdp" in filtered.columns:
        size = pd.to_numeric(filtered["nominal_gdp"], errors="coerce")
        scale_filtered = filtered[size.between(target_size / 15.0, target_size * 15.0)]
        if len(scale_filtered) >= max(4, min(n_peers, 6)):
            filtered = scale_filtered
            applied.append("scale")

    basis = " + ".join(applied) if applied else "global model-feature distance"
    return filtered, basis


def _app_weighted_robust_distances(
    frame: pd.DataFrame,
    columns: list[str],
    target_index: int = 0,
) -> np.ndarray:
    numeric = pd.DataFrame(index=frame.index)
    missing = pd.DataFrame(index=frame.index)

    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        if column in {"nominal_gdp", "gdp_per_capita"}:
            series = np.log10(series.clip(lower=1))
        lower, upper = series.quantile([0.02, 0.98])
        if pd.notna(lower) and pd.notna(upper) and lower < upper:
            series = series.clip(lower, upper)
        numeric[column] = series
        missing[column] = series.isna()

    medians = numeric.median(numeric_only=True)
    iqr = (numeric.quantile(0.75) - numeric.quantile(0.25)).replace(0, np.nan)
    scaled = ((numeric - medians) / iqr).fillna(0.0)

    target = scaled.iloc[target_index]
    weights = pd.Series(
        {column: APP_PEER_FEATURE_WEIGHTS[column] for column in columns},
        dtype=float,
    )
    distances = np.sqrt(((scaled.iloc[1:] - target) ** 2).mul(weights).sum(axis=1))

    target_missing = missing.iloc[target_index]
    missing_penalty = (
        missing.iloc[1:].ne(target_missing, axis=1).sum(axis=1).astype(float) * 0.03
    )
    return (distances + missing_penalty).to_numpy()


def _app_find_peers(
    target_country: str,
    scores_df: pd.DataFrame,
    n_peers: int = 4,
    feature_values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """App-local robust peer selector for stale helper-module deploys."""
    if scores_df is None or scores_df.empty or "country_code" not in scores_df:
        return pd.DataFrame()

    target_country = str(target_country).upper()
    scores = scores_df.copy()
    scores["country_code"] = scores["country_code"].astype(str).str.upper()
    if target_country not in set(scores["country_code"]):
        return pd.DataFrame()

    peer_frame = _app_merge_peer_features(scores, feature_values)
    target_row = peer_frame[peer_frame["country_code"] == target_country].iloc[0]
    candidates = peer_frame[peer_frame["country_code"] != target_country].copy()
    if candidates.empty:
        return pd.DataFrame()

    if "data_coverage" in candidates:
        high_coverage = candidates[
            pd.to_numeric(candidates["data_coverage"], errors="coerce").fillna(0) >= 0.75
        ]
        if not high_coverage.empty:
            candidates = high_coverage

    candidates, peer_basis = _app_apply_structural_peer_filters(
        target_row,
        candidates,
        n_peers=n_peers,
    )

    distance_columns = [
        column for column in APP_PEER_FEATURE_WEIGHTS
        if column in peer_frame.columns
        and column in candidates.columns
        and (
            pd.notna(target_row.get(column))
            or pd.to_numeric(candidates[column], errors="coerce").notna().any()
        )
    ]
    if not distance_columns:
        return pd.DataFrame()

    distance_frame = pd.concat(
        [target_row.to_frame().T, candidates],
        ignore_index=True,
    )
    distances = _app_weighted_robust_distances(
        distance_frame,
        distance_columns,
        target_index=0,
    )

    peers = candidates.copy()
    peers["distance"] = distances
    peers["peer_basis"] = peer_basis
    return peers.sort_values(["distance", "country_name"]).head(n_peers)


BASE_DIR = Path(__file__).resolve().parent
EXTERNAL_REFERENCE_DIR = BASE_DIR / "data" / "reference"
EXTERNAL_FEATURES_PATHS = (
    EXTERNAL_REFERENCE_DIR / "external_liquidity_features.parquet",
    BASE_DIR / "cache" / "external" / "external_liquidity_features.parquet",
    BASE_DIR / "artifacts" / "wb_debt_service_features.parquet",
)
EXTERNAL_OBSERVATIONS_PATHS = (
    EXTERNAL_REFERENCE_DIR / "external_feature_observations.parquet",
    BASE_DIR / "cache" / "external" / "external_feature_observations.parquet",
    BASE_DIR / "artifacts" / "wb_debt_service_observations.parquet",
)
EXTERNAL_REPORT_PATHS = (
    EXTERNAL_REFERENCE_DIR / "external_liquidity_features_report.json",
    BASE_DIR / "artifacts" / "external_liquidity_features_report.json",
    BASE_DIR / "artifacts" / "wb_debt_service_report.json",
)
MODEL_MONITORING_REPORT_PATHS = (
    BASE_DIR / "artifacts" / "liquidity_candidate_score_movement.json",
    BASE_DIR / "artifacts" / "liquidity_challenger_comparison.json",
)
CRISIS_VALIDATION_SUMMARY_PATH = BASE_DIR / "artifacts" / "crisis_validation_summary.json"
CANDIDATE_RISK_OVERLAY_SCORE_PATHS = {
    "liquidity": (
        BASE_DIR / "artifacts" / "snapshots" / "2026-06-30-challenger-liquidity" / "challenger_scores.parquet"
    ),
    "commodity": (
        BASE_DIR / "artifacts" / "snapshots" / "2026-06-30-challenger-commodity" / "challenger_scores.parquet"
    ),
    "combined": (
        BASE_DIR / "artifacts" / "snapshots" / "2026-06-30-challenger-combined" / "challenger_scores.parquet"
    ),
}
EXTERNAL_SOURCE_LABEL = "External liquidity"
GOVT_FEATURES_PATHS = (
    EXTERNAL_REFERENCE_DIR / "government_liquidity_features.parquet",
    BASE_DIR / "cache" / "government" / "government_liquidity_features.parquet",
)
GOVT_OBSERVATIONS_PATHS = (
    EXTERNAL_REFERENCE_DIR / "government_liquidity_observations.parquet",
    BASE_DIR / "cache" / "government" / "government_liquidity_observations.parquet",
)
GOVT_REPORT_PATHS = (
    EXTERNAL_REFERENCE_DIR / "government_liquidity_features_report.json",
    BASE_DIR / "artifacts" / "government_liquidity_features_report.json",
)
GOVT_SOURCE_LABEL = "Government liquidity"
GOVT_FEATURE_COUNT_COL = "government_liquidity_feature_count"
GOVT_FEATURE_LABELS = {
    "govt_gross_debt_gdp": "General government gross debt / GDP",
    "govt_fiscal_balance_gdp": "Fiscal balance / GDP",
    "govt_primary_balance_gdp": "Primary balance / GDP",
    "govt_revenue_gdp": "General government revenue / GDP",
    "govt_expenditure_gdp": "General government expenditure / GDP",
    "govt_structural_balance_gdp": "Structural balance / potential GDP",
    "govt_interest_gdp": "Interest burden / GDP (implied)",
    "govt_interest_to_revenue": "Interest burden / revenue",
    "govt_debt_to_revenue": "Gross debt / revenue",
    "govt_overall_deficit_gdp": "Overall deficit / GDP",
    "govt_primary_deficit_gdp": "Primary deficit / GDP",
    "govt_interest_to_revenue_change_3y": "Interest / revenue change, 3Y",
    "govt_debt_to_revenue_change_3y": "Debt / revenue change, 3Y",
    "govt_primary_deficit_gdp_change_3y": "Primary deficit / GDP change, 3Y",
    "govt_revenue_gdp_change_3y": "Revenue / GDP change, 3Y",
    "gross_debt_gdp": "General government gross debt / GDP",
    "fiscal_balance_gdp": "Fiscal balance / GDP",
    "primary_balance_gdp": "Primary balance / GDP",
    "revenue_gdp": "General government revenue / GDP",
    "expenditure_gdp": "General government expenditure / GDP",
    "structural_balance_potential_gdp": "Structural balance / potential GDP",
}
EXTERNAL_FEATURE_COUNT_COL = "external_liquidity_feature_count"
EXTERNAL_FEATURE_LABELS = {
    "current_account_receipts_gdp": "Current-account receipts / GDP",
    "current_account_payments_gdp": "Current-account payments / GDP",
    "current_account_balance_gdp_bop": "Current-account balance / GDP (BOP)",
    "goods_services_exports_gdp": "Goods and services exports / GDP",
    "goods_services_imports_gdp": "Goods and services imports / GDP",
    "reserves_gdp_iip": "Reserve assets / GDP (IIP)",
    "reserves_to_current_account_payments": "Reserves / current-account payments",
    "reserves_to_goods_services_imports": "Reserves / goods and services imports",
    "net_iip_gdp": "Net IIP / GDP",
    "external_liabilities_gdp": "External liabilities / GDP",
    "portfolio_liabilities_gdp": "Portfolio liabilities / GDP",
    "portfolio_liability_flows_gdp": "Portfolio liability flows / GDP",
    "portfolio_net_flows_gdp": "Portfolio net flows / GDP",
    "fdi_liability_flows_gdp": "FDI liability flows / GDP",
    "fdi_net_flows_gdp": "FDI net flows / GDP",
    "stable_financing_share": "Stable financing share (FDI / inward flows)",
    "terms_of_trade_index": "Terms of trade index (2015=100)",
    "commodity_export_share_pct": "Commodity export share of merchandise",
    "reer_index": "Real effective exchange rate (2010=100)",
    "reer_appreciation_5y_pct": "REER appreciation vs 5-year average",
    "investment_income_debits_to_cxr": "Investment-income debits / current-account receipts",
    "gross_external_financing_need_proxy_gdp": "Gross external financing need proxy / GDP",
    "wb_total_external_debt_service_exports_pct": "Total external debt service / exports",
    "wb_total_external_debt_service_gni_pct": "Total external debt service / GNI",
    "wb_ppg_external_debt_service_exports_pct": "PPG external debt service / exports",
    "wb_ppg_external_debt_service_gni_pct": "PPG external debt service / GNI",
    "wb_total_external_debt_service_gdp": "Total external debt service / GDP",
    "wb_ppg_external_debt_service_gdp": "PPG external debt service / GDP",
    "wb_total_external_debt_service_revenue_proxy": "Total external debt service / revenue proxy",
    "wb_ppg_external_debt_service_revenue_proxy": "PPG external debt service / revenue proxy",
    "wb_government_interest_payments_revenue_pct": "Government interest payments / revenue",
    "wb_government_revenue_ex_grants_gdp_pct": "Government revenue excluding grants / GDP",
    "wb_public_financing_need_ext_debt_service_proxy_gdp": "Public financing pressure proxy / GDP",
}


# Page Config
st.set_page_config(
    page_title="BankEnv",
    page_icon="assets/bankenv-favicon.svg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply the style-only payload without adding a focusable Markdown artifact.
render_dashboard_styles()


PRIMARY_VIEWS = ("Global", "Country", "Explorer", "Methodology")


def _query_param_value(name: str, default: str = "") -> str:
    """Return one normalized query-parameter value across Streamlit versions."""
    try:
        value = st.query_params.get(name, default)
    except Exception:
        return default
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    return str(value or default)


def _sync_public_query_state() -> None:
    """Keep shareable page/country state in the URL without exposing internals."""
    try:
        st.query_params["view"] = str(
            st.session_state.get("primary_view", "Global")
        ).lower()
        country_code = st.session_state.get("profile_country_code")
        if country_code:
            st.query_params["country"] = str(country_code).upper()
        explorer_country = st.session_state.get("explorer_focus_country")
        if explorer_country:
            st.query_params["explorer_country"] = str(explorer_country).upper()
        explorer_tool = st.session_state.get("explorer_tool")
        if explorer_tool:
            st.query_params["tool"] = str(explorer_tool).lower()
    except Exception:
        # URL state is a convenience; it must never prevent the app from loading.
        return


def _render_brand_shell() -> None:
    """Render the compact product identity before expensive data work starts."""
    st.markdown(
        """
        <div class="bankenv-brand" aria-label="BankEnv">
            <span class="bankenv-brand-mark" aria-hidden="true">
                <svg viewBox="0 0 64 64" focusable="false">
                    <path class="bankenv-main-stroke" d="M17 47H48" stroke-width="4" stroke-linecap="round"/>
                    <path class="bankenv-muted-stroke" d="M17 18V47" stroke-width="4" stroke-linecap="round"/>
                    <path class="bankenv-main-stroke" d="M25 41V31" stroke-width="6" stroke-linecap="round"/>
                    <path class="bankenv-accent-stroke" d="M34 41V23" stroke-width="6" stroke-linecap="round"/>
                    <path class="bankenv-main-stroke" d="M43 41V27" stroke-width="6" stroke-linecap="round"/>
                    <path class="bankenv-accent-stroke" d="M23 28L31 22L39 25L47 17" fill="none" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </span>
            <span class="bankenv-brand-name">BankEnv</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _segmented_navigation(
    label: str,
    options,
    *,
    key: str,
    on_change=None,
):
    """Use segmented controls when available with a horizontal-radio fallback."""
    segmented = getattr(st, "segmented_control", None)
    if callable(segmented):
        return segmented(
            label,
            options=options,
            key=key,
            selection_mode="single",
            on_change=on_change,
            label_visibility="collapsed",
        )
    return st.radio(
        label,
        options=options,
        key=key,
        horizontal=True,
        on_change=on_change,
        label_visibility="collapsed",
    )


_render_brand_shell()
_requested_view = _query_param_value("view", "global").strip().lower()
_view_lookup = {view.lower(): view for view in PRIMARY_VIEWS}
st.session_state.setdefault(
    "primary_view",
    _view_lookup.get(_requested_view, "Global"),
)
if st.session_state.get("primary_view") not in PRIMARY_VIEWS:
    st.session_state["primary_view"] = _view_lookup.get(_requested_view, "Global")
primary_view = _segmented_navigation(
    "Primary navigation",
    options=PRIMARY_VIEWS,
    key="primary_view",
    on_change=_sync_public_query_state,
)
if primary_view not in PRIMARY_VIEWS:
    primary_view = "Global"

# ==============================================================================
# DATA LOADING (Cached)
# ==============================================================================
def _serving_version() -> str:
    """Identity of the active serving artifact, read fresh each run.

    Threaded into the cached loaders below as a cache key so that republishing
    the model (which changes its checksum in the git-tracked manifest) busts
    every dependent cache. Without this, `@st.cache_*` results keyed only on
    static values ("Active", country_code) would serve a stale computation from
    before the update until the process restarts.
    """
    try:
        manifest = model_store.load_data_manifest()
        sha = (
            manifest.get("artifacts", {})
            .get("cache/risk_model.pkl", {})
            .get("sha256")
        )
        if sha:
            return str(sha)
        return str(
            manifest.get("model", {}).get("training_date")
            or manifest.get("snapshot_id")
            or ""
        )
    except Exception:
        return ""


@st.cache_resource
def load_all_data(version: str = ""):
    """Load model artifacts and lightweight reference data.

    Loads the checksum-verified active artifact when possible; otherwise
    degrades to the newest archived last-known-good snapshot bundle so a bad
    refresh cannot take the application down. ``version`` is the active
    artifact identity; changing it busts this cache on republish.
    """
    try:
        model, served_manifest, serving_status = (
            load_model_artifact_with_fallback()
        )
        scores_df = model['country_scores'].copy()
        # Artifacts built from official SDMX feeds may lack display names.
        fill_missing_country_names(scores_df, fallback_to_code=True)
        model_features = model.get('feature_values')
        pca_info = dict(model.get('pca_info', {}))
        pca_info.setdefault('training_date', model['training_date'])
    except Exception as e:
        LOGGER.exception("Active and fallback model loading failed")
        st.error(
            "The served model artifact could not be loaded or verified. Retry "
            "after the deployment finishes; technical details are in application logs."
        )
        return None, None, None, None, None, {}, {"mode": "error", "active_error": str(e)}

    loader = IMFDataLoader()

    try:
        wgi_loader = WGILoader()
        wgi_data = wgi_loader.load()
    except Exception:
        LOGGER.exception("WGI source loading failed")
        wgi_data = None
        serving_status = dict(serving_status or {})
        serving_status.setdefault("source_failures", {})["WGI"] = (
            "The governance source could not be loaded."
        )

    return (
        scores_df, loader, wgi_data, model_features, pca_info,
        served_manifest, serving_status,
    )


@st.cache_resource(max_entries=4)
def load_archived_snapshot_cached(name: str):
    """Checksum-verified, read-only load of one archived snapshot bundle."""
    return load_archived_snapshot(name)


@st.cache_data(show_spinner=False, max_entries=64)
def find_prior_comparable_score(
    country_code: str,
    current_snapshot: str,
    economic_features: tuple[str, ...],
    banking_features: tuple[str, ...],
    economic_weight: float,
    banking_weight: float,
) -> dict:
    """Return the newest earlier score only when the fitted contract matches."""
    candidates = [
        name for name in list_archived_snapshots()
        if "challenger" not in name.lower()
        and str(name) < str(current_snapshot)
    ]
    for name in sorted(candidates, reverse=True):
        try:
            artifact, archive_manifest = load_archived_snapshot(name)
            archive_pca = artifact.get("pca_info", {})
            compatible = (
                set(archive_pca.get("economic_loadings", {})) == set(economic_features)
                and set(archive_pca.get("industry_loadings", {})) == set(banking_features)
                and float(archive_pca.get("economic_weight", 0.5)) == float(economic_weight)
                and float(archive_pca.get("industry_weight", 0.5)) == float(banking_weight)
            )
            if not compatible:
                continue
            country_scores = artifact.get("country_scores", pd.DataFrame())
            row = country_scores[
                country_scores["country_code"].astype(str).str.upper()
                == str(country_code).upper()
            ]
            if row.empty:
                continue
            return {
                "available": True,
                "snapshot": str(
                    (archive_manifest or {}).get("snapshot_id") or name
                ),
                "risk_score": float(row.iloc[0]["risk_score"]),
            }
        except Exception:
            LOGGER.exception("Could not inspect archived snapshot %s", name)
    return {
        "available": False,
        "reason": "No earlier reviewed snapshot uses the same active feature and weighting contract.",
    }


@st.cache_resource(max_entries=8)
def load_inference_pipeline(snapshot: str, version: str = ""):
    """Load the fitted pillar pipeline for driver-table attribution.

    ``version`` is the active artifact identity; changing it busts this cache
    when the model is republished.
    """
    import pickle
    from pathlib import Path

    from src.config import CACHE_DIR
    from src.lfs_resolver import ensure_lfs_file

    if snapshot == "Active":
        path = Path(CACHE_DIR) / "inference_pipeline.pkl"
    else:
        path = SNAPSHOT_ARCHIVE / snapshot / "inference_pipeline.pkl"
    ensure_lfs_file(path)
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    pipeline = artifact.get("pillar_pipeline")
    if pipeline is None or not pipeline.fitted_:
        raise ValueError("No fitted pillar pipeline available for this snapshot")
    return pipeline


@st.cache_data(show_spinner=False, max_entries=64)
def compute_country_score_bridge(
    snapshot: str,
    country_code: str,
    version: str,
    _features: pd.DataFrame,
    _pipeline,
) -> dict:
    """Recompute the persisted structural score stages for one country.

    Older served artifacts predate the explicit bridge columns. Reusing their
    checksum-matched fitted pipeline exposes the corrected floor/penalty
    semantics without changing the served score or rebuilding the snapshot.
    """
    del snapshot, version
    if _features is None or _features.empty:
        return {}
    country = _features[
        _features["country_code"].astype(str).str.upper()
        == str(country_code).upper()
    ]
    if country.empty:
        return {}
    result = _pipeline.transform(country)
    if result.empty:
        return {}
    return result.iloc[0].to_dict()


@st.cache_data(show_spinner=False)
def load_imputed_feature_values() -> pd.DataFrame:
    """Load the score-time values used after model imputation, when packaged."""
    from src.config import CACHE_DIR

    path = Path(CACHE_DIR) / "imputed_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        LOGGER.exception("Could not load imputed feature values")
        return pd.DataFrame()


@st.cache_data(show_spinner=False, max_entries=64)
def compute_country_drivers(snapshot: str, country_code: str,
                            version: str, _model: dict, _pipeline) -> dict:
    """Per-feature score attribution for one country (rank 22).

    ``version`` is the active artifact identity so a republished model busts
    this cache (``_model``/``_pipeline`` are excluded from the cache key).
    """
    from src.scripts.explain_country_scores import build_driver_table

    report = build_driver_table([country_code], model=_model, pipeline=_pipeline)
    return report["countries"][country_code]


@st.cache_data(show_spinner=False, max_entries=12)
def compute_peer_dominant_drivers(
    snapshot: str,
    country_codes: tuple[str, ...],
    version: str,
    _model: dict,
    _pipeline,
) -> dict:
    """Return the dominant live-score driver for each displayed peer country."""
    from src.scripts.explain_country_scores import build_driver_table

    report = build_driver_table(list(country_codes), model=_model, pipeline=_pipeline)
    output = {}
    for country_code in country_codes:
        payload = report.get("countries", {}).get(country_code, {})
        drivers = payload.get("drivers") or []
        if not drivers:
            output[country_code] = "Unavailable"
            continue
        risk_raising = [
            item for item in drivers
            if float(item.get("risk_contribution", 0.0) or 0.0) > 0
        ]
        dominant = max(
            risk_raising or drivers,
            key=lambda item: (
                float(item.get("risk_contribution", 0.0) or 0.0)
                if risk_raising
                else abs(float(item.get("risk_contribution", 0.0) or 0.0))
            ),
        )
        contribution = dominant.get("risk_contribution")
        direction = "raises risk" if contribution is not None and contribution >= 0 else "lowers risk"
        output[country_code] = (
            f"{format_identifier(dominant.get('feature', 'Unavailable'))} "
            f"({direction})"
        )
    return output


def candidate_overlay_scenario(
    show_liquidity: bool,
    show_commodity: bool,
) -> tuple[str | None, str, list[str]]:
    """Return overlay scenario key, display label and candidate groups."""
    if show_liquidity and show_commodity:
        return "combined", "Combined", [
            "government_liquidity",
            "external_liquidity",
            "external_vulnerability",
        ]
    if show_liquidity:
        return "liquidity", "Liquidity", [
            "government_liquidity",
            "external_liquidity",
        ]
    if show_commodity:
        return "commodity", "Commodity", ["external_vulnerability"]
    return None, "Off", []


@st.cache_data(show_spinner=False)
def load_candidate_overlay_scores(scenario: str) -> pd.DataFrame:
    """Load optional candidate risk overlay scores for non-production analysis."""
    path = CANDIDATE_RISK_OVERLAY_SCORE_PATHS.get(scenario)
    if path is None or not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "country_code" not in frame.columns or "risk_score" not in frame.columns:
        return pd.DataFrame()
    frame = frame[["country_code", "risk_score"]].copy()
    frame["country_code"] = frame["country_code"].astype(str).str.upper()
    frame = frame.rename(columns={"risk_score": "candidate_risk_score"})
    return frame.drop_duplicates("country_code")


@st.cache_data(show_spinner=False, max_entries=24)
def load_multi_country_history(country_codes: tuple[str, ...], dataset: str) -> pd.DataFrame:
    """Load selected-country history slices for cross-country comparison."""
    loader = IMFDataLoader()
    frames = []
    for country_code in country_codes:
        country_data = loader.get_country_data(country_code, dataset)
        if country_data is not None and len(country_data) > 0:
            frames.append(country_data)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_external_insight_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load compact staged external-liquidity data for app-wide insights.

    Raw upstream IMF/WB downloads are intentionally not loaded by Streamlit.
    The hosted app reads a compact derived reference dataset when present and
    falls back to local workflow artifacts during development.
    """
    features = pd.DataFrame()
    observations = pd.DataFrame()
    report = {}

    for path in EXTERNAL_FEATURES_PATHS:
        if path.exists():
            features = pd.read_parquet(path)
            break

    for path in EXTERNAL_OBSERVATIONS_PATHS:
        if path.exists():
            observations = pd.read_parquet(path)
            break

    for path in EXTERNAL_REPORT_PATHS:
        if path.exists():
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                report = {}
            break

    if "country_code" in features.columns:
        features["country_code"] = features["country_code"].astype(str).str.upper()

    if not observations.empty:
        observations = observations.copy()
        observations["country_code"] = observations["country_code"].astype(str).str.upper()
        observations["indicator_code"] = observations["feature_key"]
        observations["indicator_name"] = observations["feature_label"].fillna(
            observations["feature_key"].map(EXTERNAL_FEATURE_LABELS)
        )
        observations["period"] = pd.to_datetime(observations["period"], errors="coerce")
        observations["frequency"] = "A"
        observations = observations.dropna(subset=["period", "value"])

    observations = _append_external_derived_observations(observations)

    return features, observations, report


def _append_external_derived_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Append derived annual external-liquidity series for Explorer charts."""
    if observations is None or observations.empty:
        return observations

    required = {
        "country_code",
        "period",
        "period_label",
        "frequency",
        "feature_key",
        "value",
    }
    if not required.issubset(observations.columns):
        return observations

    base = observations.copy()
    base["value"] = pd.to_numeric(base["value"], errors="coerce")
    pivot = (
        base.dropna(subset=["value"])
        .sort_values("period")
        .pivot_table(
            index=["country_code", "period", "period_label", "frequency"],
            columns="feature_key",
            values="value",
            aggfunc="last",
        )
        .reset_index()
    )
    if pivot.empty:
        return observations

    def col(name: str) -> pd.Series:
        if name in pivot.columns:
            return pd.to_numeric(pivot[name], errors="coerce")
        return pd.Series(pd.NA, index=pivot.index, dtype="Float64")

    derived = pd.DataFrame(
        {
            "country_code": pivot["country_code"],
            "period": pivot["period"],
            "period_label": pivot["period_label"],
            "frequency": pivot["frequency"],
        }
    )

    commodity_components = pd.concat(
        [
            col("wb_fuel_exports_pct"),
            col("wb_ores_metals_exports_pct"),
            col("wb_agri_raw_exports_pct"),
            col("wb_food_exports_pct"),
        ],
        axis=1,
    )
    derived["commodity_export_share_pct"] = commodity_components.sum(
        axis=1,
        min_count=1,
    ).clip(upper=100)

    if {"fdi_liability_flows_usd", "portfolio_liability_flows_usd"}.issubset(pivot.columns):
        fdi = col("fdi_liability_flows_usd").abs()
        portfolio = col("portfolio_liability_flows_usd").abs()
        denominator = (fdi + portfolio).replace({0: pd.NA})
        derived["stable_financing_share"] = fdi / denominator * 100.0

    if "wb_reer_index" in pivot.columns:
        ordered = pivot[["country_code", "period", "wb_reer_index"]].copy()
        ordered["wb_reer_index"] = pd.to_numeric(
            ordered["wb_reer_index"],
            errors="coerce",
        )
        ordered = ordered.sort_values(["country_code", "period"])
        baseline = (
            ordered.groupby("country_code")["wb_reer_index"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
        )
        gap = (ordered["wb_reer_index"] / baseline - 1.0) * 100.0
        derived["reer_appreciation_5y_pct"] = gap.to_numpy()

    value_columns = [
        column for column in derived.columns
        if column not in {"country_code", "period", "period_label", "frequency"}
    ]
    rows = []
    for feature_key in value_columns:
        part = derived[
            ["country_code", "period", "period_label", "frequency", feature_key]
        ].rename(columns={feature_key: "value"})
        part = part.dropna(subset=["value"])
        if part.empty:
            continue
        part["feature_key"] = feature_key
        part["indicator_code"] = feature_key
        part["feature_label"] = EXTERNAL_FEATURE_LABELS.get(
            feature_key,
            feature_key.replace("_", " ").title(),
        )
        part["indicator_name"] = part["feature_label"]
        part["source"] = "World Bank inputs / BankEnv derivation"
        part["quality"] = "derived_series"
        part["dataset_version"] = "BankEnv external formulas v1"
        part["observation_status"] = "derived"
        rows.append(part)

    if not rows:
        return observations

    additions = pd.concat(rows, ignore_index=True)
    combined = pd.concat([observations, additions], ignore_index=True, sort=False)
    return combined.drop_duplicates(
        ["country_code", "period", "feature_key"],
        keep="last",
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_government_insight_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load compact staged general-government (fiscal) liquidity data.

    Mirrors ``load_external_insight_data``: the hosted app reads a compact
    derived reference dataset when present and falls back to local workflow
    artifacts during development. The large WEO cache is never loaded here.
    """
    features = pd.DataFrame()
    observations = pd.DataFrame()
    report = {}

    for path in GOVT_FEATURES_PATHS:
        if path.exists():
            features = pd.read_parquet(path)
            break

    for path in GOVT_OBSERVATIONS_PATHS:
        if path.exists():
            observations = pd.read_parquet(path)
            break

    for path in GOVT_REPORT_PATHS:
        if path.exists():
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                report = {}
            break

    if "country_code" in features.columns:
        features["country_code"] = features["country_code"].astype(str).str.upper()

    if not observations.empty:
        observations = observations.copy()
        observations["country_code"] = observations["country_code"].astype(str).str.upper()
        observations["indicator_code"] = observations["feature_key"]
        observations["indicator_name"] = observations["feature_label"].fillna(
            observations["feature_key"].map(GOVT_FEATURE_LABELS)
        )
        observations["period"] = pd.to_datetime(observations["period"], errors="coerce")
        observations["frequency"] = "A"
        observations = observations.dropna(subset=["period", "value"])

    observations = _append_government_derived_observations(observations)

    return features, observations, report


def _append_government_derived_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Append historical fiscal-liquidity ratios derived from raw WEO series.

    Explorer charts need source-like time series, not latest feature snapshots.
    The derived government features below are therefore calculated for every
    country/period where the underlying WEO fiscal observations align.
    """
    if observations is None or observations.empty:
        return observations

    required = {
        "country_code",
        "period",
        "period_label",
        "frequency",
        "feature_key",
        "value",
    }
    if not required.issubset(observations.columns):
        return observations

    base = observations.copy()
    base["value"] = pd.to_numeric(base["value"], errors="coerce")
    pivot = (
        base.dropna(subset=["value"])
        .sort_values("period")
        .pivot_table(
            index=["country_code", "period", "period_label", "frequency"],
            columns="feature_key",
            values="value",
            aggfunc="last",
        )
        .reset_index()
    )
    if pivot.empty:
        return observations

    def col(name: str) -> pd.Series:
        if name in pivot.columns:
            return pd.to_numeric(pivot[name], errors="coerce")
        return pd.Series(pd.NA, index=pivot.index, dtype="Float64")

    def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denominator = denominator.replace({0: pd.NA})
        return numerator / denominator * 100

    fiscal_balance = col("fiscal_balance_gdp")
    primary_balance = col("primary_balance_gdp")
    revenue = col("revenue_gdp")
    gross_debt = col("gross_debt_gdp")
    interest_gdp = primary_balance - fiscal_balance
    id_columns = ["country_code", "period", "period_label", "frequency"]

    derived_values = {
        "govt_interest_gdp": interest_gdp,
        "govt_interest_to_revenue": safe_ratio(interest_gdp, revenue),
        "govt_debt_to_revenue": safe_ratio(gross_debt, revenue),
        "govt_revenue_gdp": revenue,
        "govt_overall_deficit_gdp": (
            (-fiscal_balance).where(fiscal_balance < 0, 0).where(fiscal_balance.notna())
        ),
        "govt_primary_deficit_gdp": (
            (-primary_balance).where(primary_balance < 0, 0).where(primary_balance.notna())
        ),
    }
    derived_frame = pivot[id_columns].copy().sort_values(["country_code", "period"])
    for feature_key, values in derived_values.items():
        derived_frame[feature_key] = pd.to_numeric(values, errors="coerce")
    for feature_key in [
        "govt_interest_to_revenue",
        "govt_debt_to_revenue",
        "govt_primary_deficit_gdp",
        "govt_revenue_gdp",
    ]:
        derived_values[f"{feature_key}_change_3y"] = (
            derived_frame.groupby("country_code")[feature_key]
            .transform(lambda s: s - s.shift(3))
        )

    rows = []
    for feature_key, values in derived_values.items():
        series = pivot[id_columns].copy()
        series["value"] = pd.to_numeric(values, errors="coerce")
        series = series.dropna(subset=["value"])
        if series.empty:
            continue
        series["source"] = "Government liquidity features"
        series["feature_key"] = feature_key
        series["feature_label"] = GOVT_FEATURE_LABELS.get(feature_key, feature_key)
        series["quality"] = "derived_series"
        series["dataset_version"] = "BankEnv government formulas v1"
        series["indicator_code"] = feature_key
        series["indicator_name"] = series["feature_label"]
        if "observation_status" in observations.columns:
            series["observation_status"] = "derived"
        rows.append(series)

    if not rows:
        return observations

    derived = pd.concat(rows, ignore_index=True)
    output_columns = list(observations.columns)
    for column in output_columns:
        if column not in derived.columns:
            derived[column] = pd.NA
    derived = derived[output_columns]
    return pd.concat([observations, derived], ignore_index=True)


def _fsibsis_frequency(label) -> str:
    text = str(label).upper()
    if 'M' in text:
        return 'M'
    if 'Q' in text:
        return 'Q'
    return 'A'


def _fsibsis_wide_to_long(fsibsis_wide: pd.DataFrame, country_code: str) -> pd.DataFrame:
    """Convert FSIBSIS country data from loader wide shape into app history shape."""
    if fsibsis_wide is None or len(fsibsis_wide) == 0:
        return pd.DataFrame()

    time_cols = [
        col for col in fsibsis_wide.columns
        if is_time_period_column(str(col))
    ]
    if not time_cols or 'INDICATOR' not in fsibsis_wide.columns:
        return pd.DataFrame()

    long_df = fsibsis_wide.melt(
        id_vars=['INDICATOR'],
        value_vars=time_cols,
        var_name='period_label',
        value_name='value',
    )
    long_df = long_df.dropna(subset=['value'])
    if len(long_df) == 0:
        return pd.DataFrame()

    long_df['indicator_name'] = long_df['INDICATOR']
    long_df['indicator_code'] = long_df['INDICATOR']
    long_df['period'] = long_df['period_label'].map(parse_period_label)
    long_df['country_code'] = country_code
    long_df['frequency'] = long_df['period_label'].map(_fsibsis_frequency)
    return long_df.dropna(subset=['period'])


@st.cache_data(show_spinner=False, max_entries=12)
def load_multi_country_fsibsis_history(country_codes: tuple[str, ...]) -> pd.DataFrame:
    """Load FSIBSIS history slices for selected countries only."""
    frames = []
    fsibsis_loader = FSIBSISLoader()
    fsibsis_loader.load()
    for country_code in country_codes:
        fsibsis_wide = fsibsis_loader.get_country_data(country_code)
        fsibsis_long = _fsibsis_wide_to_long(fsibsis_wide, country_code)
        if fsibsis_long is not None and len(fsibsis_long) > 0:
            frames.append(fsibsis_long)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def render_indicator_comparison(
    scores: pd.DataFrame,
    selected_country: str,
    default_peer_codes: list[str],
    country_formatter,
    wgi_panel: pd.DataFrame | None,
):
    """Render one indicator across multiple countries at source periodicity."""
    available_codes = scores.sort_values('country_name')['country_code'].tolist()
    default_countries = []
    for code in [selected_country] + default_peer_codes:
        if code in available_codes and code not in default_countries:
            default_countries.append(code)
    default_countries = default_countries[:5]
    source_options, source_to_dataset = _explorer_source_options()

    control_col1, control_col2 = st.columns([1, 3])
    with control_col1:
        source_choice = st.selectbox(
            "Source",
            source_options,
            key="compare_source",
        )
    with control_col2:
        compare_countries = st.multiselect(
            "Countries",
            options=available_codes,
            default=default_countries,
            format_func=country_formatter,
            key=f"compare_countries_{selected_country}",
            max_selections=8,
            help="Compare one indicator across the selected country and peers. Keep the set small for hosted performance.",
        )

    if not compare_countries:
        st.info("Select at least one country to compare.")
        return

    dataset = source_to_dataset[source_choice]

    with st.spinner(f"Loading {dataset} history for selected countries..."):
        source_df = _safe_load_comparison_source(dataset, compare_countries, wgi_panel)
    if source_df is None or len(source_df) == 0:
        _render_source_empty_state(dataset, "the selected countries")
        return

    source_df = source_df.copy()
    source_df['country_name'] = source_df['country_code'].map(country_formatter)

    use_name_as_key = dataset in ("FSIC", "FSIBSIS") and 'indicator_name' in source_df.columns
    if use_name_as_key:
        indicator_options = source_df['indicator_name'].dropna().unique().tolist()
        indicator_options = sorted(indicator_options, key=lambda x: x.lower())
        display_map = {name: name for name in indicator_options}
        indicator_col = 'indicator_name'
    else:
        mapping = (
            source_df[['indicator_code', 'indicator_name']]
            .dropna()
            .drop_duplicates('indicator_code')
            if 'indicator_name' in source_df.columns
            else pd.DataFrame(columns=['indicator_code', 'indicator_name'])
        )
        name_map = dict(zip(mapping['indicator_code'], mapping['indicator_name']))

        def display_indicator(code):
            name = name_map.get(code)
            if pd.notna(name) and str(name).strip() and str(name) != str(code):
                return f"{name} ({code})"
            return str(code).replace('_', ' ').title()

        indicator_options = sorted(
            source_df['indicator_code'].dropna().unique().tolist(),
            key=display_indicator,
        )
        display_map = {code: display_indicator(code) for code in indicator_options}
        indicator_col = 'indicator_code'

    if not indicator_options:
        st.info("No comparable indicators were found for the selected source.")
        return

    indicator_col1, indicator_col2, indicator_col3 = st.columns([3, 1, 1])
    with indicator_col1:
        selected_indicator = st.selectbox(
            "Indicator",
            options=indicator_options,
            format_func=lambda x: display_map[x],
            key=f"compare_indicator_{dataset}",
        )
        render_full_label(display_map[selected_indicator])
    with indicator_col2:
        time_range = st.selectbox(
            "Range",
            ["5 Years", "10 Years", "20 Years", "All Data"],
            index=1,
            key=f"compare_range_{dataset}",
        )
    comparison_metadata = _indicator_metadata_row(
                source_df,
                dataset,
                source_choice,
                selected_indicator,
                indicator_col,
                display_map[selected_indicator],
                "Compare item",
            )
    _render_indicator_metadata([comparison_metadata])

    chart_df = source_df[source_df[indicator_col] == selected_indicator].copy()
    chart_df['date'] = pd.to_datetime(chart_df['period'].astype(str), errors='coerce')
    chart_df = chart_df.dropna(subset=['date', 'value']).sort_values('date')

    with indicator_col3:
        selected_freq = None
        if 'frequency' in chart_df.columns:
            freq_labels = {'M': 'Monthly', 'Q': 'Quarterly', 'A': 'Annual'}
            available_freqs = [
                f for f in ('M', 'Q', 'A')
                if f in set(chart_df['frequency'].dropna())
            ]
            if len(available_freqs) > 1:
                selected_freq = st.selectbox(
                    "Periodicity",
                    available_freqs,
                    format_func=lambda f: freq_labels.get(f, f),
                    key=f"compare_frequency_{dataset}_{selected_indicator}",
                )
            elif available_freqs:
                selected_freq = available_freqs[0]
            if selected_freq:
                chart_df = chart_df[chart_df['frequency'] == selected_freq]

    comparison_unit = str(comparison_metadata.get("Unit") or "Not specified")
    if "index" in comparison_unit.lower() and len(compare_countries) > 1:
        st.error(
            "Cross-country comparison is blocked because this source index may "
            "use country-specific base values. Use Calculate → Change over time "
            "and rebase each country to 100."
        )
        return
    if not st.button(
        "Apply Comparison",
        type="primary",
        key=f"apply_compare_{dataset}_{selected_indicator}",
    ):
        st.caption("Review the source, countries, unit, and range, then press Apply Comparison.")
        return

    if len(chart_df) == 0:
        st.info("No observations found for that indicator/country set.")
        return

    chart_df = chart_df.drop_duplicates(
        subset=['country_code', 'date'],
        keep='last',
    )
    max_date = chart_df['date'].max()
    if time_range == "5 Years":
        chart_df = chart_df[chart_df['date'] >= max_date - pd.DateOffset(years=5)]
    elif time_range == "10 Years":
        chart_df = chart_df[chart_df['date'] >= max_date - pd.DateOffset(years=10)]
    elif time_range == "20 Years":
        chart_df = chart_df[chart_df['date'] >= max_date - pd.DateOffset(years=20)]

    title = display_map[selected_indicator]
    status_dash = (
        "observation_status"
        if "observation_status" in chart_df.columns
        and chart_df["observation_status"].nunique(dropna=True) > 1
        else None
    )
    fig = px.line(
        chart_df,
        x='date',
        y='value',
        color='country_name',
        line_dash=status_dash,
        markers=True,
        title=title,
        labels={"country_name": "Country", "value": comparison_unit},
        hover_data=[
            column for column in ("period", "observation_status")
            if column in chart_df.columns
        ],
    )
    apply_responsive_chart_layout(
        fig,
        title=title,
        showlegend=chart_df["country_code"].nunique() > 1 or status_dash is not None,
        yaxis_title=comparison_unit,
    )
    fig.update_layout(
        height=390,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    if "observation_status" in chart_df.columns:
        projected = chart_df[
            chart_df["observation_status"].astype(str).str.lower().str.contains(
                "forecast|projected|projection",
                regex=True,
            )
        ]
        if not projected.empty:
            fig.add_vline(
                x=projected["date"].min(),
                line_dash="dot",
                line_color="#7C8798",
                annotation_text="Projection begins",
                annotation_position="top right",
            )
    focus_label = country_formatter(selected_country)
    for trace in fig.data:
        if str(trace.name).split(",")[0].strip() == focus_label:
            trace.update(line={"width": 4})
        else:
            trace.update(line={"width": 2}, opacity=0.75)
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme="streamlit",
        key=f"compare_chart_{dataset}_{selected_indicator}",
        config=accessible_plotly_config(),
    )

    latest_table = (
        chart_df.sort_values('date')
        .groupby(['country_code', 'country_name'], as_index=False)
        .last()[['country_name', 'date', 'value']]
        .sort_values('country_name')
    )
    latest_table['Latest Period'] = latest_table['date'].dt.strftime('%Y-%m-%d')
    latest_table['Latest Value'] = latest_table['value'].map(lambda x: f"{x:,.2f}")
    latest_table = latest_table.rename(columns={'country_name': 'Country'})
    st.dataframe(
        latest_table[['Country', 'Latest Period', 'Latest Value']],
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("View Full Comparison Data", expanded=False):
        export_columns = [
            column for column in (
                "country_code", "country_name", "date", "frequency", "value",
                "observation_status", "indicator_code", "indicator_name", "unit",
                "source", "dataset_version",
            )
            if column in chart_df.columns
        ]
        export_df = chart_df[export_columns].copy()
        export_df["date"] = pd.to_datetime(
            export_df["date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        st.dataframe(export_df, use_container_width=True, hide_index=True)
        safe_indicator = re.sub(r"[^A-Za-z0-9_-]+", "_", str(selected_indicator))
        st.download_button(
            "Download Full Comparison",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"bankenv_{dataset.lower()}_{safe_indicator}.csv",
            mime="text/csv",
            key=f"download_compare_{dataset}_{safe_indicator}",
        )


def _load_comparison_source(
    dataset: str,
    countries: list[str],
    wgi_panel: pd.DataFrame | None,
) -> pd.DataFrame:
    """Load a source panel for selected countries for Data Explorer tools."""
    if dataset == "WGI":
        if wgi_panel is None or len(wgi_panel) == 0:
            return pd.DataFrame()
        governance_cols = [
            'voice_accountability', 'political_stability', 'govt_effectiveness',
            'regulatory_quality', 'rule_of_law', 'control_corruption'
        ]
        available_cols = [c for c in governance_cols if c in wgi_panel.columns]
        source_df = wgi_panel[wgi_panel['country_code'].isin(countries)].copy()
        if len(source_df) == 0 or not available_cols:
            return pd.DataFrame()
        source_df = source_df.melt(
            id_vars=['country_code', 'year'],
            value_vars=available_cols,
            var_name='indicator_code',
            value_name='value',
        )
        source_df['indicator_name'] = source_df['indicator_code'].str.replace('_', ' ').str.title()
        source_df['period'] = pd.to_datetime(source_df['year'].astype(str) + '-12-31')
        source_df['frequency'] = 'A'
        return source_df

    if dataset == "FSIBSIS":
        return load_multi_country_fsibsis_history(tuple(countries))

    if dataset == "EXTERNAL":
        _, observations, _ = load_external_insight_data()
        if observations.empty:
            return pd.DataFrame()
        return observations[observations["country_code"].isin(countries)].copy()

    if dataset == "GOVT":
        _, observations, _ = load_government_insight_data()
        if observations.empty:
            return pd.DataFrame()
        return observations[observations["country_code"].isin(countries)].copy()

    return load_multi_country_history(tuple(countries), dataset)


def _render_source_empty_state(dataset: str, context: str) -> None:
    """Distinguish source failure, missing package, and no matching observations."""
    failures = dict(
        (globals().get("serving_status") or {}).get("source_failures", {})
    )
    failures.update(st.session_state.get("_runtime_source_failures", {}))
    if dataset in failures:
        st.error(
            f"{dataset} failed to load, so {context} cannot be displayed. "
            "Retry this view after the source cache is available."
        )
    elif dataset in {"EXTERNAL", "GOVT"}:
        st.info(
            f"The {dataset} derived feature series is not packaged for {context}. "
            "The served score remains available from its verified artifact."
        )
    else:
        st.info(
            f"No {dataset} observations match {context}. Try another country, "
            "indicator, frequency, or range."
        )


def _safe_load_comparison_source(
    dataset: str,
    countries: list[str],
    wgi_panel: pd.DataFrame | None,
) -> pd.DataFrame:
    """Load an Explorer source without exposing technical exceptions to users."""
    try:
        frame = _load_comparison_source(dataset, countries, wgi_panel)
        st.session_state.setdefault("_runtime_source_failures", {}).pop(
            dataset,
            None,
        )
        return frame
    except Exception:
        LOGGER.exception(
            "Explorer source load failed for %s (%s)",
            dataset,
            ",".join(countries),
        )
        st.session_state.setdefault("_runtime_source_failures", {})[dataset] = True
        return pd.DataFrame()


def _explorer_source_options() -> tuple[list[str], dict[str, str]]:
    """Return the canonical Data Explorer source choices."""
    source_options = [
        "Official · Economic (WEO)",
        "Official · Banking ratios (FSIC)",
        "Official · Bank balance sheet (FSIBSIS)",
        "Official · Monetary (MFS)",
        "Official · Governance (WGI)",
        f"BankEnv · {EXTERNAL_SOURCE_LABEL}",
        f"BankEnv · {GOVT_SOURCE_LABEL}",
    ]
    source_to_dataset = {
        "Official · Economic (WEO)": "WEO",
        "Official · Banking ratios (FSIC)": "FSIC",
        "Official · Bank balance sheet (FSIBSIS)": "FSIBSIS",
        "Official · Monetary (MFS)": "MFS",
        "Official · Governance (WGI)": "WGI",
        f"BankEnv · {EXTERNAL_SOURCE_LABEL}": "EXTERNAL",
        f"BankEnv · {GOVT_SOURCE_LABEL}": "GOVT",
    }
    return source_options, source_to_dataset


def render_source_inspector(
    selected_country: str,
    country_formatter,
    wgi_panel: pd.DataFrame | None,
):
    """Render one-country source inspection using the shared Explorer sources."""
    st.caption(
        "Inspect one source for the focus country. This uses the same source "
        "choices as Compare and Calculate."
    )

    source_options, source_to_dataset = _explorer_source_options()
    source_col, action_col = st.columns([2, 3])
    with source_col:
        source_choice = st.selectbox(
            "Source",
            source_options,
            key="inspect_source",
        )
    dataset = source_to_dataset[source_choice]
    inspection_key = f"{dataset}:{str(selected_country).upper()}"
    with action_col:
        load_source = st.button(
            "Load Source",
            key=f"inspect_load_{dataset}_{selected_country}",
            help="Loads only this selected source for the focus country.",
            type="primary",
            use_container_width=True,
        )
    if load_source:
        st.session_state["loaded_source_inspection"] = inspection_key

    if st.session_state.get("loaded_source_inspection") != inspection_key:
        st.caption("Select a source and press Load Source to inspect its history.")
        return

    with st.spinner(f"Loading {dataset} history for {selected_country}..."):
        source_df = _safe_load_comparison_source(dataset, [selected_country], wgi_panel)

    if source_df is None or len(source_df) == 0:
        _render_source_empty_state(
            dataset,
            country_formatter(selected_country),
        )
        return

    indicator_options, _, _ = _indicator_selector_metadata(source_df, dataset)
    st.caption(
        f"{len(indicator_options)} indicators available for "
        f"{country_formatter(selected_country)} from {source_choice}."
    )
    try:
        render_time_series_deep_dive(source_df, dataset, selected_country)
    except Exception:
        LOGGER.exception(
            "Source Inspector chart failed for %s/%s",
            dataset,
            selected_country,
        )
        st.error(
            "This source loaded but its chart could not be rendered. Retry the "
            "source; technical details are available in application logs."
        )


def _indicator_selector_metadata(source_df: pd.DataFrame, dataset: str):
    """Return indicator options, labels and key column for a source frame."""
    use_name_as_key = dataset in ("FSIC", "FSIBSIS") and 'indicator_name' in source_df.columns
    if use_name_as_key:
        indicator_options = source_df['indicator_name'].dropna().unique().tolist()
        indicator_options = sorted(indicator_options, key=lambda x: x.lower())
        display_map = {name: name for name in indicator_options}
        indicator_col = 'indicator_name'
    else:
        mapping = (
            source_df[['indicator_code', 'indicator_name']]
            .dropna()
            .drop_duplicates('indicator_code')
            if 'indicator_name' in source_df.columns
            else pd.DataFrame(columns=['indicator_code', 'indicator_name'])
        )
        name_map = dict(zip(mapping['indicator_code'], mapping['indicator_name']))

        def display_indicator(code):
            name = name_map.get(code)
            if pd.notna(name) and str(name).strip() and str(name) != str(code):
                return f"{name} ({code})"
            return str(code).replace('_', ' ').title()

        indicator_options = sorted(
            source_df['indicator_code'].dropna().unique().tolist(),
            key=display_indicator,
        )
        display_map = {code: display_indicator(code) for code in indicator_options}
        indicator_col = 'indicator_code'
    return indicator_options, display_map, indicator_col


def _unit_display(value) -> str:
    """Return a readable unit label from source unit metadata when available."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return {
        "PT": "Percent",
        "USD": "US dollar",
        "EUR": "Euro",
        "XDC": "Domestic currency",
    }.get(text, text)


SOURCE_CONTEXT = {
    "WEO": {
        "dataflow": "IMF World Economic Outlook (WEO)",
        "scope": "Country macroeconomic, fiscal and external-sector series.",
        "caution": "WEO may include IMF estimates/projections; use observation status where shown.",
    },
    "FSIC": {
        "dataflow": "IMF Financial Soundness Indicators, core set (FSIC)",
        "scope": "Banking-sector soundness ratios and related numerators/denominators.",
        "caution": "Many FSIC items are ratios; avoid mixing percent and currency-valued items without checking units.",
    },
    "FSIBSIS": {
        "dataflow": "IMF Financial Soundness Indicators balance sheets (FSIBSIS)",
        "scope": "Deposit-taker balance-sheet and income-statement items.",
        "caution": "Closest source for bank balance-sheet levels; some exposure concepts may be narrower than MFS claims.",
    },
    "MFS": {
        "dataflow": "IMF Monetary and Financial Statistics, depository corporations (MFS_DC)",
        "scope": "Monetary-sector balance sheets split across central bank, other depository corporations and aggregate depository corporations.",
        "caution": "MFS is not deposit-takers-only. Use ODCORP for other depository corporations; DCORP includes the central bank.",
    },
    "WGI": {
        "dataflow": "World Bank Worldwide Governance Indicators (WGI)",
        "scope": "Annual country governance percentile-style scores.",
        "caution": "Governance scores are not balance-sheet values and should not be mixed with monetary levels.",
    },
    "EXTERNAL": {
        "dataflow": "BankEnv-derived external-liquidity feature series",
        "scope": "Derived external-sector and liquidity indicators assembled from official upstream sources.",
        "caution": "Review formula, upstream source, vintage, and coverage before treating a derived ratio as observed data.",
    },
    "GOVT": {
        "dataflow": "BankEnv-derived government-liquidity feature series",
        "scope": "Derived fiscal and government-liquidity indicators assembled from official upstream sources.",
        "caution": "Implied ratios inherit upstream estimate/projection status and are not reported government cash data.",
    },
}


MFS_PREFIX_CONTEXT = {
    "DCORP": "Depository corporations aggregate: central bank plus other depository corporations; not deposit-takers-only.",
    "ODCORP": "Other depository corporations: closest MFS proxy for deposit-taking banks/commercial banks; excludes central bank.",
    "S121": "Central bank sector; not commercial banks/deposit takers.",
}


def _first_indicator_code(subset: pd.DataFrame, fallback) -> str:
    if "indicator_code" in subset.columns:
        codes = subset["indicator_code"].dropna().astype(str)
        codes = codes[codes.str.strip() != ""]
        if len(codes) > 0:
            return codes.iloc[0]
    return str(fallback)


def _source_context(dataset: str, code: str, label: str) -> dict[str, str]:
    """Return source-specific context for a selected indicator."""
    context = SOURCE_CONTEXT.get(
        dataset,
        {
            "dataflow": dataset,
            "scope": "Source-specific observations.",
            "caution": "Check units, frequency and coverage before combining with other series.",
        },
    ).copy()
    text = f"{code} {label}".lower()
    prefix = str(code).split("_", 1)[0].upper()

    if dataset == "MFS":
        context["institutional sector"] = MFS_PREFIX_CONTEXT.get(
            prefix,
            "MFS institutional sector could not be inferred from the code prefix.",
        )
        if str(code).endswith("_EAWR"):
            context["caution"] += " This code uses euro-area-wide residency treatment."
    elif dataset == "FSIBSIS":
        context["institutional sector"] = "Deposit takers / banking balance-sheet sector."
    elif dataset == "FSIC":
        context["institutional sector"] = "Financial soundness indicator reporting sector; often deposit-taker focused."
    else:
        context["institutional sector"] = "Country-level source series."

    if "claims on" in text:
        context["instrument / side"] = "Asset-side claim or exposure."
    elif "assets" in text:
        context["instrument / side"] = "Asset-side balance-sheet item."
    elif "liabilities to" in text:
        context["instrument / side"] = "Liability owed to the named counterparty."
    elif "liabilities" in text:
        context["instrument / side"] = "Liability-side balance-sheet item."
    elif "capital" in text or "equity" in text or "reserves" in text:
        context["instrument / side"] = "Capital, reserves or equity denominator candidate."
    elif "percent" in text:
        context["instrument / side"] = "Ratio or percentage series."
    else:
        context["instrument / side"] = "Source-defined indicator."

    counterpart_checks = [
        ("state and local government", "State and local government"),
        ("state and local governments", "State and local government"),
        ("central government", "Central government"),
        ("general government", "General government"),
        ("public non-financial corporations", "Public non-financial corporations"),
        ("private sector", "Private sector"),
        ("other financial corporations", "Other financial corporations"),
        ("central bank", "Central bank"),
        ("nonresidents", "Nonresidents"),
    ]
    counterpart = "Not specified in label"
    for pattern, value in counterpart_checks:
        if pattern in text:
            counterpart = value
            break
    context["counterparty / sector"] = counterpart
    return context


def _join_distinct(values, limit: int = 4) -> str:
    cleaned = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan" and text not in cleaned:
            cleaned.append(text)
    if not cleaned:
        return "Not specified"
    if len(cleaned) > limit:
        return ", ".join(cleaned[:limit]) + f" +{len(cleaned) - limit} more"
    return ", ".join(cleaned)


def _indicator_label_components(label: str) -> str:
    parts = [
        part.strip()
        for part in str(label).split(",")
        if part.strip() and part.strip().lower() != "nan"
    ]
    if not parts:
        return "Not specified"
    return " | ".join(parts)


def _indicator_plain_language_note(dataset: str, label: str) -> str:
    """Provide a short source-label-derived explanation for common item types."""
    text = str(label or "").lower()
    notes = []
    if "assets" in text:
        notes.append("asset-side position")
    if "liabilities" in text:
        notes.append("liability-side position")
    if "claims on" in text:
        notes.append("claim/exposure to the named counterpart")
    if "state and local government" in text:
        notes.append("counterparty is state and local government")
    if "general government" in text:
        notes.append("counterparty/sector is general government")
    if "percent" in text:
        notes.append("ratio or percentage series")
    if dataset == "MFS" and not notes:
        notes.append("monetary and financial statistics item")
    if dataset == "FSIBSIS" and not notes:
        notes.append("bank balance-sheet or income-statement item")
    return "; ".join(notes) if notes else "Source label only"


def _indicator_metadata_row(
    source_df: pd.DataFrame,
    dataset: str,
    source_label: str,
    indicator_value,
    indicator_col: str,
    display_label: str,
    role: str,
) -> dict:
    """Build one compact metadata row for a selected source item."""
    subset = source_df.loc[source_df[indicator_col] == indicator_value].copy()
    if subset.empty:
        context = _source_context(dataset, str(indicator_value), display_label)
        return {
            "Role": role,
            "Source": source_label,
            "Upstream source": "Not specified",
            "Dataset version": "Not specified",
            "Dataflow": context["dataflow"],
            "Source scope": context["scope"],
            "Code": str(indicator_value),
            "Source label": display_label,
            "Source dimensions": _indicator_label_components(display_label),
            "Institutional scope": context["institutional sector"],
            "Instrument / side": context["instrument / side"],
            "Counterparty / sector": context["counterparty / sector"],
            "Unit": "Not specified",
            "Frequency": "Not specified",
            "Observation status": "Not specified",
            "Quality / lineage": "Not specified",
            "Latest actual year": "Not specified",
            "Coverage": "No selected-country observations",
            "Plain-language note": _indicator_plain_language_note(dataset, display_label),
            "Use caution": context["caution"],
        }

    source_name = display_label
    if "indicator_name" in subset.columns:
        names = subset["indicator_name"].dropna().astype(str)
        names = names[names.str.strip() != ""]
        if len(names) > 0:
            source_name = names.iloc[0]

    representative_code = _first_indicator_code(subset, indicator_value)
    code = str(indicator_value)
    if "indicator_code" in subset.columns:
        code = _join_distinct(subset["indicator_code"].dropna().astype(str).unique(), limit=3)
    context = _source_context(dataset, representative_code, source_name)

    if "unit" in subset.columns:
        unit = _join_distinct([_unit_display(value) for value in subset["unit"].unique()], limit=3)
    else:
        unit = "Not specified"

    if "frequency" in subset.columns:
        frequency = _join_distinct(
            [
                {"M": "Monthly", "Q": "Quarterly", "A": "Annual"}.get(str(value), str(value))
                for value in subset["frequency"].dropna().unique()
            ],
            limit=3,
        )
    else:
        frequency = "Not specified"

    observation_status = (
        _join_distinct(subset["observation_status"].dropna().astype(str).unique(), limit=4)
        if "observation_status" in subset.columns
        else "Not specified"
    )
    latest_actual_year = (
        _join_distinct(
            [
                str(int(value)) if pd.notna(value) and float(value).is_integer() else str(value)
                for value in pd.to_numeric(subset["latest_actual_year"], errors="coerce").dropna().unique()
            ],
            limit=3,
        )
        if "latest_actual_year" in subset.columns
        else "Not specified"
    )
    upstream_source = (
        _join_distinct(subset["source"].dropna().astype(str).unique(), limit=4)
        if "source" in subset.columns else "Not specified"
    )
    dataset_version = (
        _join_distinct(subset["dataset_version"].dropna().astype(str).unique(), limit=4)
        if "dataset_version" in subset.columns else "Not specified"
    )
    quality_lineage = (
        _join_distinct(subset["quality"].dropna().astype(str).unique(), limit=4)
        if "quality" in subset.columns else "Not specified"
    )

    period = pd.to_datetime(subset.get("period"), errors="coerce")
    valid_period = period.dropna()
    if valid_period.empty:
        coverage = f"{subset['country_code'].nunique()} selected countries"
    else:
        coverage = (
            f"{subset['country_code'].nunique()} selected countries; "
            f"{valid_period.min().date()} to {valid_period.max().date()}"
        )

    return {
        "Role": role,
        "Source": source_label,
        "Upstream source": upstream_source,
        "Dataset version": dataset_version,
        "Dataflow": context["dataflow"],
        "Source scope": context["scope"],
        "Code": code,
        "Source label": source_name,
        "Source dimensions": _indicator_label_components(source_name),
        "Institutional scope": context["institutional sector"],
        "Instrument / side": context["instrument / side"],
        "Counterparty / sector": context["counterparty / sector"],
        "Unit": unit,
        "Frequency": frequency,
        "Observation status": observation_status,
        "Quality / lineage": quality_lineage,
        "Latest actual year": latest_actual_year,
        "Coverage": coverage,
        "Plain-language note": _indicator_plain_language_note(dataset, source_name),
        "Use caution": context["caution"],
    }


def _render_indicator_metadata(rows: list[dict]):
    """Render selected item metadata beneath calculation controls."""
    rows = [row for row in rows if row]
    if not rows:
        return
    with st.expander("Selected item metadata", expanded=False):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "Metadata is taken from the loaded source cache and source codelist "
            "labels. The plain-language note is derived from the source label."
        )


def _render_calculated_chart(
    chart_df: pd.DataFrame,
    title: str,
    country_formatter,
    y_title: str = None,
    chart_key: str = None,
):
    """Render a standard calculated-series line chart and latest table."""
    if chart_df is None or len(chart_df) == 0:
        st.info("No aligned observations are available for that calculation.")
        return

    chart_df = chart_df.copy()
    chart_df['country_name'] = chart_df['country_code'].map(country_formatter)
    chart_df = chart_df.sort_values('date')
    status_dash = (
        "observation_status"
        if "observation_status" in chart_df.columns
        and chart_df["observation_status"].nunique(dropna=True) > 1
        else None
    )
    fig = px.line(
        chart_df,
        x='date',
        y='value',
        color='country_name',
        line_dash=status_dash,
        markers=True,
        title=title,
        hover_data=["period"] if "period" in chart_df.columns else None,
    )
    apply_responsive_chart_layout(
        fig,
        title=title,
        showlegend=chart_df["country_code"].nunique() > 1 or status_dash is not None,
        yaxis_title=y_title or "Source-defined value",
    )
    fig.update_layout(
        height=390,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    if "observation_status" in chart_df.columns:
        projected = chart_df[
            chart_df["observation_status"].astype(str).str.lower().str.contains(
                "forecast|projected|projection",
                regex=True,
            )
        ]
        if not projected.empty:
            fig.add_vline(
                x=projected["date"].min(),
                line_dash="dot",
                line_color="#7C8798",
                annotation_text="Projection begins",
                annotation_position="top right",
            )
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme="streamlit",
        key=chart_key,
        config=accessible_plotly_config(),
    )

    latest = (
        chart_df.sort_values('date')
        .groupby(['country_code', 'country_name'], as_index=False)
        .last()[['country_name', 'date', 'value']]
        .sort_values('country_name')
    )
    latest['Latest Period'] = latest['date'].dt.strftime('%Y-%m-%d')
    latest['Latest Value'] = latest['value'].map(lambda x: f"{x:,.2f}")
    latest = latest.rename(columns={'country_name': 'Country'})
    st.dataframe(
        latest[['Country', 'Latest Period', 'Latest Value']],
        use_container_width=True,
        hide_index=True,
    )
    if "observation_status" in chart_df.columns:
        statuses = sorted(
            chart_df["observation_status"].dropna().astype(str).unique()
        )
        if statuses:
            st.caption("Observation status: " + ", ".join(statuses) + ".")
    with st.expander("View Full Chart Data", expanded=False):
        export_columns = [
            column for column in (
                "country_code", "country_name", "date", "frequency", "value",
                "indicator_label", "observation_status",
            )
            if column in chart_df.columns
        ]
        full_data = chart_df[export_columns].copy()
        full_data["date"] = pd.to_datetime(
            full_data["date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        st.dataframe(full_data, use_container_width=True, hide_index=True)
        safe_key = re.sub(r"[^A-Za-z0-9_-]+", "_", chart_key or "calculated_series")
        st.download_button(
            "Download Full History",
            data=full_data.to_csv(index=False).encode("utf-8"),
            file_name=f"bankenv_{safe_key}.csv",
            mime="text/csv",
            key=f"download_{safe_key}",
        )


def _render_custom_formula_builder(
    source_options: list[str],
    source_to_dataset: dict[str, str],
    calc_countries: list[str],
    wgi_panel: pd.DataFrame | None,
    time_range: str,
    country_formatter,
):
    """Render a safe expression-based formula builder with per-operand sources."""
    setup_col, formula_col, scale_col = st.columns([1, 3, 1])
    with setup_col:
        operand_count = st.number_input(
            "Operands",
            min_value=2,
            max_value=12,
            value=2,
            step=1,
            key="calc_formula_operand_count",
            help="Creates operand slots A, B, C... dynamically.",
        )
    allowed_operands = [chr(ord("A") + idx) for idx in range(int(operand_count))]
    with formula_col:
        default_formula = "A / B"
        formula_text = st.text_input(
            "Formula",
            value=default_formula,
            key="calc_formula_expression",
            help=(
                "Use declared operand slots with +, -, *, / and parentheses. "
                "Function calls, attributes, indexing and comparisons are rejected."
            ),
        )
    with scale_col:
        scale_label = st.selectbox(
            "Scale",
            ["As calculated", "Percent / x100"],
            key="calc_formula_scale",
        )
    scale = 100.0 if scale_label == "Percent / x100" else 1.0

    try:
        formula_plan = validate_formula(formula_text, allowed_operands)
    except FormulaValidationError as exc:
        st.error(str(exc))
        return

    operand_sources = {}
    used_operands = list(formula_plan.used_operands)
    source_cols = st.columns(min(4, max(1, len(used_operands))))
    for idx, operand in enumerate(used_operands):
        with source_cols[idx % len(source_cols)]:
            operand_sources[operand] = st.selectbox(
                f"{operand} source",
                source_options,
                key=f"calc_formula_{operand.lower()}_source",
            )

    loaded_sources = {}

    def load_formula_source(source_label: str):
        source_dataset = source_to_dataset[source_label]
        if source_dataset not in loaded_sources:
            with st.spinner(f"Loading {source_dataset} formula history..."):
                frame = _safe_load_comparison_source(source_dataset, calc_countries, wgi_panel)
            if frame is None or len(frame) == 0:
                return source_dataset, pd.DataFrame(), [], {}, "indicator_code"
            options, labels, key_col = _indicator_selector_metadata(frame, source_dataset)
            loaded_sources[source_dataset] = (frame, options, labels, key_col)
        frame, options, labels, key_col = loaded_sources[source_dataset]
        return source_dataset, frame, options, labels, key_col

    operand_data = {}
    for operand in used_operands:
        operand_dataset, operand_df, options, labels, key_col = load_formula_source(
            operand_sources[operand]
        )
        if operand_df is None or len(operand_df) == 0 or not options:
            st.info(f"No {operand_dataset} data is available for operand {operand}.")
            return
        operand_data[operand] = (operand_dataset, operand_df, options, labels, key_col)

    operand_keys = {}
    operand_cols = st.columns(min(4, max(1, len(used_operands))))
    for idx, operand in enumerate(used_operands):
        with operand_cols[idx % len(operand_cols)]:
            operand_dataset, _, options, labels, _ = operand_data[operand]
            operand_keys[operand] = st.selectbox(
                f"{operand} item",
                options,
                format_func=lambda value, label_map=labels: label_map[value],
                key=f"calc_formula_{operand.lower()}_item_{operand_dataset}",
            )
            render_full_label(labels[operand_keys[operand]], f"{operand} item")

    metadata_rows = []
    operand_frames = {}
    operand_labels = {}
    for operand in used_operands:
        operand_dataset, operand_df, _, labels, key_col = operand_data[operand]
        operand_key = operand_keys[operand]
        operand_labels[operand] = f"{labels[operand_key]} [{operand_dataset}]"
        metadata_rows.append(
            _indicator_metadata_row(
                operand_df,
                operand_dataset,
                operand_sources[operand],
                operand_key,
                key_col,
                labels[operand_key],
                operand,
            )
        )
        operand_frames[operand] = normalize_observation_frame(
            operand_df,
            operand_key,
            key_col,
            labels[operand_key],
        )
    _render_indicator_metadata(metadata_rows)

    with st.expander("Formula Audit", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Check": "Accepted syntax",
                        "Result": "Operands, numbers, +, -, *, / and parentheses only",
                    },
                    {
                        "Check": "Rejected syntax",
                        "Result": "No function calls, attributes, indexing, comparisons or eval",
                    },
                    {
                        "Check": "Parsed formula",
                        "Result": formula_plan.normalized_formula,
                    },
                    {
                        "Check": "Operands used",
                        "Result": ", ".join(formula_plan.used_operands),
                    },
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    frequency_sets = [set(available_frequencies(frame)) for frame in operand_frames.values()]
    common_freqs = [
        frequency
        for frequency in ("M", "Q", "A")
        if all(frequency in present for present in frequency_sets)
    ]
    selected_freq = None
    if len(common_freqs) > 1:
        selected_freq = st.selectbox(
            "Periodicity",
            common_freqs,
            format_func=lambda f: {'M': 'Monthly', 'Q': 'Quarterly', 'A': 'Annual'}.get(f, f),
            key=f"calc_formula_frequency_{formula_plan.normalized_formula}",
        )
    elif common_freqs:
        selected_freq = common_freqs[0]

    restricted_operands = {
        operand: restrict_frequency(frame, selected_freq)
        for operand, frame in operand_frames.items()
    }
    if not st.button(
        "Apply Custom Formula",
        type="primary",
        key=f"apply_formula_{formula_plan.normalized_formula}",
    ):
        st.caption("Review the operands and formula, then press Apply Custom Formula.")
        return
    formula, formula_plan = compute_expression_formula(
        formula_text,
        restricted_operands,
        scale=scale,
    )
    formula = filter_time_range(formula, time_range)

    title = formula_plan.normalized_formula
    for operand in sorted(operand_labels, key=len, reverse=True):
        title = re.sub(rf"\b{operand}\b", operand_labels[operand], title)
    _render_calculated_chart(
        formula,
        title=title,
        country_formatter=country_formatter,
        y_title=scale_label,
        chart_key=f"calc_formula_chart_{formula_plan.normalized_formula}",
    )
    st.caption(
        f"Formula: {title}{' x 100' if scale != 1.0 else ''}. "
        "Only exact country/date/frequency matches are used; invalid or zero-denominator rows are excluded."
    )


def render_calculated_series_builder(
    scores: pd.DataFrame,
    selected_country: str,
    default_peer_codes: list[str],
    country_formatter,
    wgi_panel: pd.DataFrame | None,
):
    """Render bounded multi-indicator, ratio, share and temporal calculations."""
    st.caption(
        "Build lightweight exploratory calculations from source histories. "
        "Calculations align observations by country, date, and reporting frequency."
    )

    available_codes = scores.sort_values('country_name')['country_code'].tolist()
    default_countries = []
    for code in [selected_country] + default_peer_codes:
        if code in available_codes and code not in default_countries:
            default_countries.append(code)
    default_countries = default_countries[:5]

    source_options, source_to_dataset = _explorer_source_options()

    mode_col, country_col, range_col = st.columns([2, 3, 1])
    with mode_col:
        calc_mode = st.selectbox(
            "Calculation",
            [
                "Raw multi-indicator panels",
                "Ratio",
                "Custom formula",
                "Cross-sectional share",
                "Temporal change / index",
            ],
            key="calc_mode",
        )
    with country_col:
        calc_countries = st.multiselect(
            "Countries",
            options=available_codes,
            default=default_countries,
            format_func=country_formatter,
            key=f"calc_countries_{selected_country}",
            max_selections=8,
            help="Choose up to eight countries for hosted performance.",
        )
    with range_col:
        time_range = st.selectbox(
            "Range",
            ["5 Years", "10 Years", "20 Years", "All Data"],
            index=1,
            key="calc_range",
        )
    if not calc_countries:
        st.info("Select at least one country.")
        return
    if calc_mode == "Custom formula":
        _render_custom_formula_builder(
            source_options=source_options,
            source_to_dataset=source_to_dataset,
            calc_countries=calc_countries,
            wgi_panel=wgi_panel,
            time_range=time_range,
            country_formatter=country_formatter,
        )
        return

    source_choice = st.selectbox(
        "Source",
        source_options,
        key="calc_source",
    )
    dataset = source_to_dataset[source_choice]

    with st.spinner(f"Loading {dataset} history for calculated series..."):
        source_df = _safe_load_comparison_source(dataset, calc_countries, wgi_panel)
    if source_df is None or len(source_df) == 0:
        _render_source_empty_state(dataset, "the selected calculation countries")
        return

    indicator_options, display_map, indicator_col = _indicator_selector_metadata(source_df, dataset)
    if not indicator_options:
        st.info("No indicators are available for this source/country selection.")
        return

    if calc_mode == "Raw multi-indicator panels":
        selected_indicators = st.multiselect(
            "Indicators",
            options=indicator_options,
            default=[],
            format_func=lambda x: display_map[x],
            key=f"calc_raw_indicators_{dataset}",
            max_selections=5,
            help="Choose up to five indicators; each renders separately to preserve units.",
        )
        render_full_label(
            "; ".join(display_map[indicator] for indicator in selected_indicators),
            "Selected indicators",
        )
        if not selected_indicators:
            st.info("Select at least one indicator.")
            return
        _render_indicator_metadata(
            [
                _indicator_metadata_row(
                    source_df,
                    dataset,
                    source_choice,
                    indicator,
                    indicator_col,
                    display_map[indicator],
                    f"Panel {idx + 1}",
                )
                for idx, indicator in enumerate(selected_indicators)
            ],
        )
        freq_options = available_frequencies(source_df)
        selected_freq = None
        if len(freq_options) > 1:
            selected_freq = st.selectbox(
                "Periodicity",
                freq_options,
                format_func=lambda f: {'M': 'Monthly', 'Q': 'Quarterly', 'A': 'Annual'}.get(f, f),
                key=f"calc_raw_frequency_{dataset}",
            )
        elif freq_options:
            selected_freq = freq_options[0]
        if not st.button(
            "Apply Indicator Panels",
            type="primary",
            key=f"apply_raw_panels_{dataset}",
        ):
            st.caption("Press Apply Indicator Panels to render the selected histories.")
            return
        for idx, indicator in enumerate(selected_indicators):
            panel = normalize_observation_frame(
                source_df,
                indicator,
                indicator_col,
                display_map[indicator],
            )
            panel = filter_time_range(restrict_frequency(panel, selected_freq), time_range)
            _render_calculated_chart(
                panel,
                title=display_map[indicator],
                country_formatter=country_formatter,
                chart_key=f"calc_raw_chart_{dataset}_{idx}",
            )
        st.caption("Formula: raw source value. Missing periods are not filled.")
        return

    if calc_mode == "Ratio":
        den_source_col, scale_col = st.columns([3, 1])
        with den_source_col:
            denominator_source = st.selectbox(
                "Denominator source",
                source_options,
                index=source_options.index(source_choice),
                key=f"calc_ratio_den_source_{dataset}",
            )
        with scale_col:
            scale_label = st.selectbox(
                "Scale",
                ["Ratio", "Percent"],
                key=f"calc_ratio_scale_{dataset}",
            )

        denominator_dataset = source_to_dataset[denominator_source]
        if denominator_dataset == dataset:
            denominator_source_df = source_df
        else:
            with st.spinner(f"Loading {denominator_dataset} denominator history..."):
                denominator_source_df = _safe_load_comparison_source(
                    denominator_dataset,
                    calc_countries,
                    wgi_panel,
                )
        if denominator_source_df is None or len(denominator_source_df) == 0:
            st.info(f"No {denominator_dataset} data is available for the selected countries.")
            return

        denominator_options, denominator_display, denominator_col = _indicator_selector_metadata(
            denominator_source_df,
            denominator_dataset,
        )
        if not denominator_options:
            st.info("No denominator indicators are available for this source/country selection.")
            return

        num_col, den_col = st.columns([2, 2])
        with num_col:
            numerator_key = st.selectbox(
                "Numerator item",
                indicator_options,
                format_func=lambda x: display_map[x],
                key=f"calc_ratio_num_{dataset}",
            )
            render_full_label(display_map[numerator_key], "Numerator")
        with den_col:
            denominator_key = st.selectbox(
                "Denominator item",
                denominator_options,
                format_func=lambda x: denominator_display[x],
                key=f"calc_ratio_den_{denominator_dataset}",
            )
            render_full_label(denominator_display[denominator_key], "Denominator")
        ratio_metadata = [
                _indicator_metadata_row(
                    source_df,
                    dataset,
                    source_choice,
                    numerator_key,
                    indicator_col,
                    display_map[numerator_key],
                    "Numerator",
                ),
                _indicator_metadata_row(
                    denominator_source_df,
                    denominator_dataset,
                    denominator_source,
                    denominator_key,
                    denominator_col,
                    denominator_display[denominator_key],
                    "Denominator",
                ),
            ]
        _render_indicator_metadata(ratio_metadata)
        scale = 100.0 if scale_label == "Percent" else 1.0
        numerator = normalize_observation_frame(
            source_df,
            numerator_key,
            indicator_col,
            display_map[numerator_key],
        )
        denominator = normalize_observation_frame(
            denominator_source_df,
            denominator_key,
            denominator_col,
            denominator_display[denominator_key],
        )
        common_freqs = [
            f for f in available_frequencies(numerator)
            if f in set(denominator.get("frequency", pd.Series(dtype=str)))
        ]
        selected_freq = None
        if len(common_freqs) > 1:
            selected_freq = st.selectbox(
                "Periodicity",
                common_freqs,
                format_func=lambda f: {'M': 'Monthly', 'Q': 'Quarterly', 'A': 'Annual'}.get(f, f),
                key=(
                    f"calc_ratio_frequency_{dataset}_{denominator_dataset}_"
                    f"{numerator_key}_{denominator_key}"
                ),
            )
        elif common_freqs:
            selected_freq = common_freqs[0]
        unit_check = check_unit_compatibility(
            "ratio",
            [ratio_metadata[0].get("Unit"), ratio_metadata[1].get("Unit")],
        )
        if not unit_check.valid:
            st.error(f"Ratio blocked: {unit_check.reason}")
            return
        for unit_warning in unit_check.warnings:
            st.warning(unit_warning)
        if not st.button(
            "Apply Ratio",
            type="primary",
            key=(
                f"apply_ratio_{dataset}_{denominator_dataset}_"
                f"{numerator_key}_{denominator_key}"
            ),
        ):
            st.caption("Review the selected units and press Apply Ratio to calculate.")
            return
        numerator_aligned = restrict_frequency(numerator, selected_freq)
        denominator_aligned = restrict_frequency(denominator, selected_freq)
        ratio = compute_ratio(
            numerator_aligned,
            denominator_aligned,
            scale=scale,
        )
        alignment = diagnose_alignment(
            {"Numerator": numerator_aligned, "Denominator": denominator_aligned},
            result=ratio,
        )
        ratio = filter_time_range(ratio, time_range)
        numerator_label = f"{display_map[numerator_key]} [{dataset}]"
        denominator_label = f"{denominator_display[denominator_key]} [{denominator_dataset}]"
        _render_calculated_chart(
            ratio,
            title=f"{numerator_label} / {denominator_label}",
            country_formatter=country_formatter,
            y_title=scale_label,
            chart_key=f"calc_ratio_chart_{dataset}_{denominator_dataset}",
        )
        st.caption(
            f"Formula: {numerator_label} ÷ {denominator_label}"
            f"{' × 100' if scale_label == 'Percent' else ''}. "
            "Only exact country/date/frequency matches are used; zero denominators are excluded."
        )
        st.caption(
            f"Alignment: {alignment.matched_observations:,} exact matches; "
            f"{alignment.dropped_after_calculation or 0:,} dropped after division; "
            f"{alignment.matched_countries:,} countries."
        )
        return

    if calc_mode == "Cross-sectional share":
        indicator_key = st.selectbox(
            "Indicator",
            indicator_options,
            format_func=lambda x: display_map[x],
            key=f"calc_share_indicator_{dataset}",
        )
        render_full_label(display_map[indicator_key])
        share_metadata = _indicator_metadata_row(
            source_df,
            dataset,
            source_choice,
            indicator_key,
            indicator_col,
            display_map[indicator_key],
            "Share item",
        )
        _render_indicator_metadata([share_metadata])
        base = normalize_observation_frame(
            source_df,
            indicator_key,
            indicator_col,
            display_map[indicator_key],
        )
        freq_options = available_frequencies(base)
        selected_freq = None
        if len(freq_options) > 1:
            selected_freq = st.selectbox(
                "Periodicity",
                freq_options,
                format_func=lambda f: {'M': 'Monthly', 'Q': 'Quarterly', 'A': 'Annual'}.get(f, f),
                key=f"calc_share_frequency_{dataset}_{indicator_key}",
            )
        elif freq_options:
            selected_freq = freq_options[0]
        additivity_check = check_cross_sectional_additivity(
            share_metadata.get("Unit")
        )
        if not additivity_check.valid:
            st.error(f"Share blocked: {additivity_check.reason}")
            return
        if not st.button(
            "Apply Cross-Sectional Share",
            type="primary",
            key=f"apply_share_{dataset}_{indicator_key}",
        ):
            st.caption(
                "Confirm the selected countries define the intended denominator, "
                "then press Apply Cross-Sectional Share."
            )
            return
        share = compute_cross_sectional_share(restrict_frequency(base, selected_freq))
        share = filter_time_range(share, time_range)
        _render_calculated_chart(
            share,
            title=f"Share of selected-country total: {display_map[indicator_key]}",
            country_formatter=country_formatter,
            y_title="Percent of selected group",
            chart_key=f"calc_share_chart_{dataset}_{indicator_key}",
        )
        st.caption(
            "Formula: country value ÷ sum of selected countries for the same period × 100. "
            "The selected country set defines the denominator."
        )
        return

    indicator_key = st.selectbox(
        "Indicator",
        indicator_options,
        format_func=lambda x: display_map[x],
        key=f"calc_temporal_indicator_{dataset}",
    )
    render_full_label(display_map[indicator_key])
    _render_indicator_metadata(
        [
            _indicator_metadata_row(
                source_df,
                dataset,
                source_choice,
                indicator_key,
                indicator_col,
                display_map[indicator_key],
                "Temporal item",
            )
        ],
    )
    temporal_mode = st.radio(
        "Temporal calculation",
        ["period_pct", "base_pct", "index_100"],
        format_func=lambda x: {
            "period_pct": "Period-over-period % change",
            "base_pct": "Change from first period %",
            "index_100": "Rebased index: first period = 100",
        }[x],
        horizontal=True,
        key=f"calc_temporal_mode_{dataset}_{indicator_key}",
    )
    base = normalize_observation_frame(
        source_df,
        indicator_key,
        indicator_col,
        display_map[indicator_key],
    )
    freq_options = available_frequencies(base)
    selected_freq = None
    if len(freq_options) > 1:
        selected_freq = st.selectbox(
            "Periodicity",
            freq_options,
            format_func=lambda f: {'M': 'Monthly', 'Q': 'Quarterly', 'A': 'Annual'}.get(f, f),
            key=f"calc_temporal_frequency_{dataset}_{indicator_key}",
        )
    elif freq_options:
        selected_freq = freq_options[0]
    if not st.button(
        "Apply Temporal Calculation",
        type="primary",
        key=f"apply_temporal_{dataset}_{indicator_key}_{temporal_mode}",
    ):
        st.caption("Review the range and periodicity, then press Apply Temporal Calculation.")
        return
    ranged = filter_time_range(restrict_frequency(base, selected_freq), time_range)
    temporal = compute_temporal_change(ranged, temporal_mode)
    _render_calculated_chart(
        temporal,
        title=f"{display_map[indicator_key]} — {temporal_mode.replace('_', ' ')}",
        country_formatter=country_formatter,
        y_title="Percent" if temporal_mode != "index_100" else "Index",
        chart_key=f"calc_temporal_chart_{dataset}_{indicator_key}_{temporal_mode}",
    )
    st.caption(
        "Formula: period change, first-period change, or rebased index calculated "
        "separately for each country after frequency and time-range selection."
    )


def render_candidate_country_evidence(
    country_code: str,
    model_features: pd.DataFrame | None,
    overlay_enabled: bool = False,
    selected_groups: list[str] | None = None,
    active_features: set[str] | None = None,
):
    """Show monitored candidate liquidity evidence for the selected country."""
    report = _load_first_json(MODEL_MONITORING_REPORT_PATHS)
    candidate_groups = report.get("candidate_groups") or {}
    if selected_groups:
        candidate_features = []
        for group in selected_groups:
            candidate_features.extend(candidate_groups.get(group, []))
    else:
        candidate_features = report.get("candidate_features") or report.get("added_features") or []
    if active_features is not None:
        candidate_features = [
            feature for feature in candidate_features
            if feature not in active_features
        ]
    if not candidate_features:
        return

    if not overlay_enabled:
        show_additional_evidence = st.checkbox(
            "Show additional insight-only evidence",
            value=False,
            key=f"show_additional_evidence_{country_code}",
            help=(
                "Loads candidate fields that are available for analysis but are "
                "not included in the served score."
            ),
        )
        if not show_additional_evidence:
            return

    external_features, _, external_report = load_external_insight_data()
    government_features, _, government_report = load_government_insight_data()
    frames = []
    if not government_features.empty:
        frames.append(government_features)
    if not external_features.empty:
        frames.append(external_features)
    if not frames:
        return

    feature_frame = frames[0]
    for frame in frames[1:]:
        feature_frame = feature_frame.merge(frame, on="country_code", how="outer")
    feature_frame["country_code"] = feature_frame["country_code"].astype(str).str.upper()
    country = feature_frame[feature_frame["country_code"] == str(country_code).upper()]
    if country.empty:
        return
    country_row = country.iloc[0]

    govt_coverage = government_report.get("feature_coverage", {})
    external_coverage = external_report.get("feature_coverage", {})
    active_columns = (
        active_features
        if active_features is not None
        else set(model_features.columns) if model_features is not None else set()
    )
    rows = []
    for feature in candidate_features:
        if feature not in feature_frame.columns:
            continue
        raw_value = pd.to_numeric(pd.Series([country_row.get(feature)]), errors="coerce").iloc[0]
        value = "—" if pd.isna(raw_value) else f"{float(raw_value):,.2f}"
        is_govt = feature.startswith("govt_")
        is_commodity = feature == "commodity_export_share_pct"
        coverage = (govt_coverage if is_govt else external_coverage).get(feature, {})
        labels = GOVT_FEATURE_LABELS if is_govt else EXTERNAL_FEATURE_LABELS
        rows.append(
            {
                "Feature": labels.get(feature, feature.replace("_", " ").title()),
                "Value": value,
                "Group": (
                    "Government liquidity"
                    if is_govt
                    else "External vulnerability"
                    if is_commodity
                    else "External liquidity"
                ),
                "Coverage": (
                    f"{coverage.get('countries')} countries"
                    if coverage.get("countries") is not None
                    else "—"
                ),
                "Role": (
                    "Active score input"
                    if feature in active_columns
                    else "Candidate / monitoring"
                ),
            }
        )
    if not rows:
        return

    if overlay_enabled:
        st.markdown("### Scenario Evidence")
        st.caption(
            "These fields are packaged for review and analysis. Commodity exposure is "
            "an external vulnerability factor, not a liquidity metric. They do not explain "
            "the served score unless marked as active score inputs."
        )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        return

    st.markdown("### Additional Insight-Only Evidence")
    st.caption(
        "These fields are packaged for review and analysis. Commodity exposure is "
        "an external vulnerability factor, not a liquidity metric. They do not explain "
        "the served score unless marked as active score inputs."
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _display_value(value, integer: bool = False) -> str:
    """Format card values without leaking None/NaN into the UI."""
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    if integer:
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _source_role(source_name: str) -> str:
    roles = {
        "WEO": "Macro, fiscal, GDP and external-balance baseline",
        "FSIC": "Core banking soundness ratios",
        "MFS": "Monetary, credit and banking balance-sheet aggregates",
        "FSIBSIS": "Detailed bank balance-sheet and income-statement measures",
        "WGI": "Governance and institutional-quality scores",
    }
    return roles.get(source_name, "Supporting source")


def _load_first_json(paths: tuple[Path, ...]) -> dict:
    for path in paths:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {}


def _comparison_block(report: dict) -> dict:
    for key in (
        "candidate_effect_vs_active_retrain",
        "feature_effect_challenger_vs_control",
        "headline_candidate_vs_production",
        "headline_challenger_vs_production",
    ):
        value = report.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _liquidity_feature_status_rows(
    report: dict,
    active_features: set[str] | None = None,
) -> list[dict]:
    govt_report = _load_first_json(GOVT_REPORT_PATHS)
    external_report = _load_first_json(EXTERNAL_REPORT_PATHS)
    govt_coverage = govt_report.get("feature_coverage", {})
    external_coverage = external_report.get("feature_coverage", {})

    def row(feature: str, group: str) -> dict:
        source = "Government liquidity" if feature.startswith("govt_") else "External liquidity"
        coverage = govt_coverage.get(feature) if feature.startswith("govt_") else external_coverage.get(feature)
        coverage = coverage or {}
        labels = GOVT_FEATURE_LABELS if feature.startswith("govt_") else EXTERNAL_FEATURE_LABELS
        return {
            "Feature": labels.get(feature, feature.replace("_", " ").title()),
            "Code": feature,
            "Group": group,
            "Role": liquidity_feature_role(feature, report, active_features),
            "Coverage": _display_value(coverage.get("countries"), integer=True),
            "% Scored Countries": (
                f"{coverage.get('pct_model_countries'):.1f}%"
                if coverage.get("pct_model_countries") is not None
                else _display_value(None)
            ),
            "Source": source,
        }

    rows: list[dict] = []
    seen: set[str] = set()

    def append_feature(feature: str, group: str) -> None:
        if feature in seen:
            return
        rows.append(row(feature, group))
        seen.add(feature)

    for feature in report.get("active_features") or []:
        group = "Government" if feature.startswith("govt_") else "External"
        append_feature(feature, group)

    candidate_groups = report.get("candidate_groups") or {}
    if candidate_groups:
        for group_name, features in candidate_groups.items():
            group = {
                "government_liquidity": "Government liquidity",
                "external_liquidity": "External liquidity",
                "external_vulnerability": "External vulnerability",
            }.get(group_name, str(group_name).replace("_", " ").title())
            for feature in features:
                append_feature(feature, group)
    else:
        for feature in report.get("candidate_features") or report.get("added_features") or []:
            group = "Government" if feature.startswith("govt_") else "External"
            append_feature(feature, group)
    return rows


def render_model_monitoring_summary(pca_info: dict | None = None):
    report = _load_first_json(MODEL_MONITORING_REPORT_PATHS)
    active_features = active_model_feature_codes(pca_info)

    st.markdown("#### Model Monitoring")
    st.caption(
        "Candidate features are reviewed through score movement before they are "
        "approved for inclusion in the served score."
    )
    st.caption(
        "The Country Profile can optionally display saved candidate risk overlays. "
        "Liquidity and commodity overlays are independent; selecting both uses the "
        "combined scenario. Overlays are analytical only and do not change the served "
        "score, ranking, or score drivers."
    )

    if report:
        effect = _comparison_block(report)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cutoff", _display_value(report.get("cutoff")))
        c2.metric(
            "Countries Compared",
            _display_value(effect.get("countries_compared"), integer=True),
        )
        c3.metric(
            "Mean |Δ Score|",
            _display_value(effect.get("mean_absolute_score_change")),
        )
        c4.metric(
            "Tier Changes",
            _display_value(effect.get("risk_tier_changes"), integer=True),
        )

        review_basis = str(report.get("review_basis") or "candidate comparison").replace("_", " ")
        st.caption(f"Review basis: {review_basis}. Served scores are unchanged by this report.")

        status_rows = _liquidity_feature_status_rows(
            report,
            active_features=active_features or None,
        )
        if status_rows:
            with st.expander("Active and candidate liquidity features", expanded=True):
                reported_candidates = set(report.get("candidate_features") or [])
                reclassified = sorted(reported_candidates.intersection(active_features))
                if reclassified:
                    st.info(
                        "This comparison report predates the current serving artifact. "
                        "Feature roles below are derived from the active model loadings; "
                        f"{len(reclassified)} formerly monitored field(s) are now active."
                    )
                st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

        movements = effect.get("largest_movements") or []
        if movements:
            with st.expander("Largest candidate score movements", expanded=False):
                movement_df = pd.DataFrame(movements).rename(
                    columns={
                        "country_code": "Country",
                        "base": "Baseline",
                        "challenger": "Candidate",
                        "delta": "Δ Score",
                    }
                )
                st.dataframe(movement_df, use_container_width=True, hide_index=True)

        if report.get("promotion_requires_owner_review"):
            st.warning(
                "Monitoring only: inclusion in the served score requires owner review "
                "under the score-movement gate."
            )
        else:
            st.success("Candidate score movement is within the configured review gate.")
    else:
        st.info("No liquidity candidate score-movement report is packaged with this deployment.")

def render_methodology_workspace(
    scores: pd.DataFrame,
    features: pd.DataFrame | None,
    manifest: dict,
    pca: dict | None,
) -> None:
    """Render the single authoritative methodology, data, and release view."""
    st.markdown("# Methodology")
    snapshot_id = str(manifest.get("snapshot_id") or "Unversioned")
    snapshot_status = str(manifest.get("snapshot_status") or "Not recorded")
    integrity_verified = snapshot_status.lower() == "verified"
    registry = build_active_feature_registry(pca)
    operating_count = int((registry["pillar"] == "economic").sum()) if not registry.empty else 0
    banking_count = int((registry["pillar"] == "industry").sum()) if not registry.empty else 0
    st.caption(
        f"Snapshot {snapshot_id} · {len(scores):,} countries · "
        + (
            "Artifact integrity verified; model approval not recorded."
            if integrity_verified
            else "Artifact integrity is not verified."
        )
    )
    methodology_sections = (
        "How the Score Works",
        "Data and Coverage",
        "Validation and Release",
    )
    if st.session_state.get("methodology_section") not in {
        None,
        *methodology_sections,
    }:
        st.session_state["methodology_section"] = methodology_sections[0]
    section = _segmented_navigation(
        "Methodology section",
        options=methodology_sections,
        key="methodology_section",
    )

    if section == "How the Score Works":
        st.markdown("## Model Card")
        st.caption(
            "BankEnv is a country-level banking-system risk screener for analyst "
            "triage and peer comparison. It is not a bank rating, crisis-timing "
            "forecast, or automated lending decision."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Scale", "1–10", help="1 lower relative risk; 10 higher relative risk")
        m2.metric("Operating Inputs", f"{operating_count}")
        m3.metric("Banking Inputs", f"{banking_count}")
        m4.metric("Served Countries", f"{len(scores):,}")
        score_flow = pd.DataFrame(
            [
                ("1. Active inputs", "Dated macro, fiscal, governance, external, liquidity, and banking evidence.", "Country evidence shows value, unit, source, period, and status."),
                ("2. Two fitted pillars", "Operating Environment and Banking System components are estimated separately.", "Both are displayed in risk orientation: 1 lower; 10 higher."),
                ("3. Coverage policy", "Confidence adjustment and a minimum risk floor limit false precision.", "Any floor effect is a separate Country score-bridge step."),
                ("4. Critical-data penalty", "A bounded penalty applies when defined core banking fields are imputed.", "Country view names the exact fields and penalty."),
                ("5. Legacy crisis adjustment", "An upward-only overlay remains in the served artifact.", "It is not presented as a validated crisis probability."),
                ("6. Served score", "Rounded 1–10 relative-risk output for the active snapshot.", "The Country Score Bridge reconciles every adjustment."),
            ],
            columns=["Stage", "Purpose", "Public Interpretation"],
        )
        st.dataframe(score_flow, use_container_width=True, hide_index=True)
        use_col, limit_col = st.columns(2)
        with use_col:
            st.markdown("### Appropriate Uses")
            st.markdown(
                "- Cross-country screening and watchlist prioritization.\n"
                "- Peer comparison and evidence-gap review.\n"
                "- Structured input to analyst judgment."
            )
        with limit_col:
            st.markdown("### Important Limits")
            st.markdown(
                "- Relative scores can move with the country universe or coverage.\n"
                "- Public-source lags and imputation can be material.\n"
                "- The legacy crisis overlay is not decision-grade on its own."
            )
        with st.expander("Technical Diagnostics", expanded=False):
            st.info(
                "Combined Pillar is a separate PCA diagnostic and is not the "
                "arithmetic precursor to the served score. It is excluded from "
                "the public Country score block."
            )
            if not registry.empty:
                technical = registry[
                    ["label", "feature", "pillar_label", "loading", "unit", "source_family"]
                ].rename(columns={
                    "label": "Input", "feature": "Technical Code",
                    "pillar_label": "Pillar", "loading": "Loading",
                    "unit": "Unit", "source_family": "Source / Lineage",
                })
                st.dataframe(
                    technical,
                    use_container_width=True,
                    hide_index=True,
                    column_config={"Loading": st.column_config.NumberColumn(format="%+.4f")},
                )

    elif section == "Data and Coverage":
        st.markdown("## Data Card")
        st.caption(
            "Official upstream sources are separated from BankEnv-derived feature "
            "packages. Active membership always comes from fitted loading maps."
        )
        source_rows: list[dict] = []
        for source_name, details in sorted((manifest.get("sources") or {}).items()):
            source_rows.append({
                "Dataset": source_name,
                "Type": "Official upstream source",
                "Countries": _display_value(details.get("countries"), integer=True),
                "Latest Observation": _display_value(details.get("latest_observation")),
                "Role": _source_role(source_name),
                "Availability": "Active source family",
            })
        external_features, external_observations, external_report = load_external_insight_data()
        government_features, government_observations, government_report = load_government_insight_data()
        for label, package, observations, report in (
            ("External liquidity", external_features, external_observations, external_report),
            ("Government liquidity", government_features, government_observations, government_report),
        ):
            active_in_package = [
                feature for feature in registry["feature"].tolist()
                if feature in package.columns
            ] if not registry.empty and not package.empty else []
            source_rows.append({
                "Dataset": label,
                "Type": "BankEnv-derived feature package",
                "Countries": _display_value(
                    int(package["country_code"].nunique())
                    if not package.empty and "country_code" in package
                    else 0,
                    integer=True,
                ),
                "Latest Observation": _display_value(report.get("cutoff") or report.get("as_of_date")),
                "Role": f"{len(active_in_package)} active inputs; remaining fields insight-only",
                "Availability": "Available" if not observations.empty else "Not packaged",
            })
        st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)
        if features is None or features.empty or registry.empty:
            st.info(
                "Feature-level coverage is unavailable because the served feature "
                "artifact or loading maps are not packaged."
            )
        else:
            country_total = int(features["country_code"].nunique())
            coverage_rows = []
            for item in registry.to_dict(orient="records"):
                feature = item["feature"]
                direct = int(features[feature].notna().sum()) if feature in features else 0
                coverage_rows.append({
                    "Input": item["label"],
                    "Pillar": item["pillar_label"],
                    "Direct Countries": direct,
                    "Direct Coverage": direct / country_total if country_total else np.nan,
                    "Source / Lineage": item["source_family"],
                    "Role": item["model_role"],
                })
            coverage_table = pd.DataFrame(coverage_rows).sort_values(["Direct Coverage", "Input"])
            st.markdown("### Active-Input Coverage")
            filter_a, filter_b, filter_c, filter_d = st.columns([2, 1, 1, 1])
            with filter_a:
                coverage_search = st.text_input(
                    "Search active inputs",
                    key="methodology_coverage_search",
                    placeholder="Input, source, or role",
                )
            with filter_b:
                coverage_pillar = st.selectbox(
                    "Pillar",
                    ["All", *sorted(coverage_table["Pillar"].dropna().unique())],
                    key="methodology_coverage_pillar",
                )
            with filter_c:
                coverage_source = st.selectbox(
                    "Source",
                    ["All", *sorted(coverage_table["Source / Lineage"].dropna().unique())],
                    key="methodology_coverage_source",
                )
            with filter_d:
                coverage_role = st.selectbox(
                    "Role",
                    ["All", *sorted(coverage_table["Role"].dropna().unique())],
                    key="methodology_coverage_role",
                )
            filtered_coverage = coverage_table.copy()
            if coverage_search.strip():
                needle = coverage_search.strip().lower()
                search_mask = pd.Series(False, index=filtered_coverage.index)
                for column in ("Input", "Source / Lineage", "Role"):
                    search_mask |= filtered_coverage[column].astype(str).str.lower().str.contains(
                        needle,
                        regex=False,
                    )
                filtered_coverage = filtered_coverage.loc[search_mask]
            if coverage_pillar != "All":
                filtered_coverage = filtered_coverage[
                    filtered_coverage["Pillar"] == coverage_pillar
                ]
            if coverage_source != "All":
                filtered_coverage = filtered_coverage[
                    filtered_coverage["Source / Lineage"] == coverage_source
                ]
            if coverage_role != "All":
                filtered_coverage = filtered_coverage[
                    filtered_coverage["Role"] == coverage_role
                ]
            st.dataframe(
                filtered_coverage,
                use_container_width=True,
                hide_index=True,
                column_config={"Direct Coverage": st.column_config.NumberColumn(format="percent")},
            )
            st.download_button(
                "Download Data Card Inventory",
                data=coverage_table.to_csv(index=False).encode("utf-8"),
                file_name=f"bankenv_data_card_{snapshot_id}.csv",
                mime="text/csv",
                key="download_data_card_inventory",
            )
        st.caption(
            "Explorer preserves source-native periodicity. The served model uses "
            "one latest-allowed country cross-section at the snapshot cutoff."
        )

    elif section == "Validation and Release":
        st.markdown("## Validation and Release")
        validation = _load_first_json((CRISIS_VALIDATION_SUMMARY_PATH,))
        validation_state = crisis_validation_display_state(validation)
        lifecycle = pd.DataFrame([
            {
                "Control": "Artifact integrity",
                "Status": "Verified" if integrity_verified else "Not verified",
                "Meaning": "Manifest checks the served bundle; this is not model approval.",
            },
            {
                "Control": "Named model approval",
                "Status": "Not recorded",
                "Meaning": "No approval transition is recorded in the served manifest.",
            },
            {
                "Control": "Crisis-classifier validation",
                "Status": "Displayable" if validation_state["display_metrics"] else "Superseded / withheld",
                "Meaning": str(validation_state["reason"]),
            },
            {
                "Control": "Rollback",
                "Status": "Archived snapshots available" if SNAPSHOT_ARCHIVE else "Not recorded",
                "Meaning": "Fallback is restricted to checksum-verified bundles.",
            },
        ])
        st.dataframe(lifecycle, use_container_width=True, hide_index=True)
        st.markdown("### Confusion Matrix Status")
        confusion_matrix_path = validated_confusion_matrix_path(validation, BASE_DIR)
        if confusion_matrix_path is not None:
            st.image(
                str(confusion_matrix_path),
                caption="Governed crisis-classifier confusion matrix",
            )
        else:
            st.warning(
                "No schema-versioned, checksum-linked confusion matrix is approved "
                "for the served classifier. Superseded threshold results are not "
                "shown as current validation evidence."
            )
        st.caption(
            "The structural country score remains usable as an analyst screener; "
            "the crisis overlay must not be used as a stand-alone prediction model."
        )
        st.markdown("### Release Gate")
        release_rows = pd.DataFrame([
            ("Artifact checksums and schema", "Passed for served bundle" if integrity_verified else "Open"),
            ("Material score-movement review", "Required for every candidate"),
            ("Out-of-time and grouped validation", "Required before classifier promotion"),
            ("Named owner approval", "Not recorded"),
            ("Rollback bundle and smoke test", "Required before promotion"),
        ], columns=["Requirement", "Current State"])
        st.dataframe(release_rows, use_container_width=True, hide_index=True)
        show_candidate_appendix = st.checkbox(
            "Show candidate monitoring appendix",
            value=False,
            key="show_candidate_monitoring_appendix",
            help=(
                "Loads archived challenger movement and validation evidence. "
                "These diagnostics do not change the served score."
            ),
        )
        if show_candidate_appendix:
            render_model_monitoring_summary(pca)


_serving_ver = _serving_version()
(
    scores_df, loader, wgi_data, model_features, pca_info,
    served_manifest, serving_status,
) = load_all_data(_serving_ver)
# In fallback mode the manifest describing what is actually being served is
# the archived bundle's manifest, not the active one on disk.
data_manifest = served_manifest or load_data_manifest()
health_report = build_health_report(data_manifest, serving_status)

if health_report.get("overall") in {"stale", "degraded", "unknown"}:
    st.warning(
        "Some serving or source-freshness checks need attention. Scores remain "
        "available where the verified fallback permits; see Methodology → Data "
        "and Coverage for source vintages."
    )

if scores_df is None:
    st.error("Application cannot start without model data.")
    st.stop()

# Prepare data for Global View (Merge GDP for weighting)
if scores_df is not None and model_features is not None:
    if 'nominal_gdp' not in scores_df.columns and 'nominal_gdp' in model_features.columns:
        scores_df = scores_df.merge(model_features[['country_code', 'nominal_gdp']], on='country_code', how='left')

# ==============================================================================
# Header state and optional diagnostics
# ==============================================================================
SHOW_ADMIN_DIAGNOSTICS = os.getenv("SHOW_ADMIN_DIAGNOSTICS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

ACTIVE_SNAPSHOT_OPTION = "Active"
selected_snapshot = ACTIVE_SNAPSHOT_OPTION

if SHOW_ADMIN_DIAGNOSTICS:
    archived_names = list_archived_snapshots()

    def format_snapshot_option(name: str) -> str:
        if name == ACTIVE_SNAPSHOT_OPTION:
            return f"Active ({data_manifest.get('snapshot_id', 'unversioned')})"
        if "challenger" in name:
            return f"{name} — UNAPPROVED"
        return name

    selected_snapshot = st.selectbox(
        "Snapshot",
        options=[ACTIVE_SNAPSHOT_OPTION] + archived_names,
        format_func=format_snapshot_option,
        key="snapshot_select",
        label_visibility="collapsed",
        help=(
            "Inspect an archived snapshot bundle read-only. Challenger "
            "bundles are unapproved review candidates."
        ),
    )

viewing_archived = selected_snapshot != ACTIVE_SNAPSHOT_OPTION
if viewing_archived:
    try:
        archived_artifact, archived_manifest = load_archived_snapshot_cached(
            selected_snapshot
        )
        scores_df = archived_artifact['country_scores'].copy()
        fill_missing_country_names(scores_df, fallback_to_code=True)
        model_features = archived_artifact.get('feature_values')
        pca_info = dict(archived_artifact.get('pca_info', {}))
        pca_info.setdefault('training_date', archived_artifact['training_date'])
        if archived_manifest:
            data_manifest = archived_manifest
        if (
            model_features is not None
            and 'nominal_gdp' not in scores_df.columns
            and 'nominal_gdp' in model_features.columns
        ):
            scores_df = scores_df.merge(
                model_features[['country_code', 'nominal_gdp']],
                on='country_code', how='left',
            )
    except Exception as snapshot_error:
        st.error(f"Could not load archived snapshot: {snapshot_error}")
        viewing_archived = False

available_countries = scores_df.sort_values('country_name')[['country_code', 'country_name']].drop_duplicates()
available_country_codes = available_countries['country_code'].tolist()
country_name_lookup = dict(
    zip(available_countries['country_code'], available_countries['country_name'])
)


def format_country_option(country_code: str) -> str:
    name = country_name_lookup.get(country_code, country_code)
    return f"{name} ({country_code})"


def safe_find_peers(
    target_country: str,
    scores: pd.DataFrame,
    n_peers: int,
    features: pd.DataFrame | None,
) -> pd.DataFrame:
    """Call the current peer engine, tolerating stale Streamlit imports.

    Streamlit Cloud can briefly serve a mixed redeploy where ``app.py`` is new
    but ``src.utils.find_peers`` is still the old three-argument function.
    Falling back to the app-local robust selector prevents both crashes and
    poor two-pillar peer sets such as USA -> Cyprus/Dominica/Fiji.
    """
    try:
        return find_peers(
            target_country,
            scores,
            n_peers=n_peers,
            feature_values=features,
        )
    except TypeError as exc:
        if "feature_values" not in str(exc):
            raise
        return _app_find_peers(
            target_country,
            scores,
            n_peers=n_peers,
            feature_values=features,
        )

HEALTH_LABELS = {
    "ok": "Healthy",
    "stale": "Stale data",
    "degraded": "Fallback mode",
    "unknown": "Unknown",
}


def render_system_health_panel():
    """Render internal serving diagnostics for admin use only."""
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("Serving Mode", health_report["serving_mode"].title())
    hc2.metric("Snapshot", str(health_report.get("snapshot_id") or "—"))
    hc3.metric(
        "Snapshot Status",
        str(health_report.get("snapshot_status") or "—").replace("_", " ").title(),
    )
    generated_age = health_report.get("generated_age_days")
    hc4.metric(
        "Snapshot Age",
        "—" if generated_age is None else f"{generated_age} days",
    )
    for note in health_report["notes"]:
        st.warning(note)
    if health_report["sources"]:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Source": source["source"],
                        "Latest Observation": source["latest_observation"],
                        "Age (days)": source["age_days"],
                        "Freshness SLA (days)": source["sla_days"],
                        "Status": source["status"],
                    }
                    for source in health_report["sources"]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "Freshness SLAs are the approved thresholds from docs/GOVERNANCE.md. "
        "Fallback mode means the app is serving the last verified archived "
        "snapshot because the active artifact failed validation."
    )

if viewing_archived:
    st.warning(
        f"Viewing archived snapshot **{selected_snapshot}** read-only"
        + (" — this is an **unapproved challenger** candidate, not the served model"
           if "challenger" in selected_snapshot else "")
        + ". Diagnostics continue to describe the active serving state."
    )

default_country_code = 'USA' if 'USA' in available_country_codes else available_country_codes[0]
_requested_country = _query_param_value("country", default_country_code).upper()
if _requested_country not in available_country_codes:
    _requested_country = default_country_code
if st.session_state.get("profile_country_code") not in available_country_codes:
    st.session_state["profile_country_code"] = _requested_country
_requested_explorer_country = _query_param_value(
    "explorer_country",
    st.session_state.get("profile_country_code", default_country_code),
).upper()
if _requested_explorer_country not in available_country_codes:
    _requested_explorer_country = st.session_state.get(
        "profile_country_code",
        default_country_code,
    )
if st.session_state.get("explorer_focus_country") not in available_country_codes:
    st.session_state["explorer_focus_country"] = _requested_explorer_country
_sync_public_query_state()

# ==============================================================================
# VIEW: Global Summary
# ==============================================================================
if primary_view == "Global":
    render_global_summary(scores_df, model_features, loader)

# ==============================================================================
# VIEW: Country Profile
# ==============================================================================
if primary_view == "Country":
    selected_country_code = st.selectbox(
        "Country",
        options=available_country_codes,
        format_func=format_country_option,
        key="profile_country_code",
        on_change=_sync_public_query_state,
        help="This selector controls the Country tab only.",
    )

    country_score_row = scores_df[scores_df['country_code'] == selected_country_code].iloc[0]
    selected_country_name = country_score_row['country_name']

    risk_score = float(country_score_row['risk_score'])
    tier = score_to_tier(risk_score)
    percentile = (scores_df['risk_score'] < risk_score).mean()
    imputed_feature_values = load_imputed_feature_values()
    input_inventory = build_active_input_inventory(
        selected_country_code,
        model_features,
        pca_info,
        imputed_feature_values,
    )
    active_feature_codes = set(input_inventory.rows["feature"].tolist())
    direct_count = input_inventory.coverage.numerator
    direct_total = input_inventory.coverage.denominator
    direct_coverage = input_inventory.coverage.ratio or 0.0
    prior_score_evidence = find_prior_comparable_score(
        selected_country_code,
        str(data_manifest.get("snapshot_id") or ""),
        tuple(sorted((pca_info or {}).get("economic_loadings", {}))),
        tuple(sorted((pca_info or {}).get("industry_loadings", {}))),
        float((pca_info or {}).get("economic_weight", 0.5)),
        float((pca_info or {}).get("industry_weight", 0.5)),
    )

    driver_version = (
        _serving_ver if selected_snapshot == ACTIVE_SNAPSHOT_OPTION
        else selected_snapshot
    )
    driver_pipeline_for_profile = None
    score_bridge: dict = {}
    try:
        driver_pipeline_for_profile = load_inference_pipeline(
            selected_snapshot,
            driver_version,
        )
        score_bridge = compute_country_score_bridge(
            selected_snapshot,
            selected_country_code,
            driver_version,
            model_features,
            driver_pipeline_for_profile,
        )
    except Exception:
        LOGGER.exception(
            "Score bridge unavailable for %s in %s",
            selected_country_code,
            selected_snapshot,
        )

    st.markdown(f"# {selected_country_name}")
    snapshot_label = str(data_manifest.get("snapshot_id") or "Unversioned")
    integrity_verified = (
        str(data_manifest.get("snapshot_status", "")).lower() == "verified"
    )
    lifecycle_text = (
        "Artifact integrity verified; model approval not recorded."
        if integrity_verified
        else "Artifact integrity status is not verified; interpret cautiously."
    )
    st.caption(f"Snapshot {snapshot_label} · Served score · {lifecycle_text}")

    with st.container(border=True):
        tier_labels = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Risk Score",
            f"{risk_score:.1f}/10",
            help="1 is lower relative risk; 10 is higher relative risk.",
        )
        m2.metric("Risk Tier", tier_labels.get(tier, "N/A"))
        m3.metric(
            "Risk Percentile",
            f"{percentile:.0%}",
            help="Share of scored countries with a lower risk score.",
        )
        m4.metric(
            "Direct Active-Input Coverage",
            f"{direct_count}/{direct_total}",
            help=(
                "Reported or derived values present before scoring imputation, "
                "divided by all features active in the fitted pillar loadings."
            ),
        )
        st.caption(f"Direct active-input coverage: {direct_coverage:.0%}.")
        if prior_score_evidence.get("available"):
            prior_score = float(prior_score_evidence["risk_score"])
            movement = risk_score - prior_score
            direction = (
                "higher relative risk" if movement > 0
                else "lower relative risk" if movement < 0
                else "unchanged"
            )
            st.caption(
                f"Comparable score history: {movement:+.1f} points ({direction}) "
                f"versus {prior_score_evidence['snapshot']}."
            )
        else:
            st.caption(str(prior_score_evidence.get("reason")))

    with st.container(border=True):
        st.markdown("## Risk Components")
        st.caption("All values below use the same direction: 1 lower risk; 10 higher risk.")
        econ_strength = float(country_score_row['economic_pillar'])
        banking_strength = float(country_score_row['industry_pillar'])
        econ_risk = 1 + 9 * (1 - econ_strength / 10)
        banking_risk = 1 + 9 * (1 - banking_strength / 10)
        bd1, bd2 = st.columns(2)
        bd1.metric("Operating Environment Risk", f"{econ_risk:.1f}/10")
        bd2.metric("Banking System Risk", f"{banking_risk:.1f}/10")
        st.caption(
            "These are risk-oriented views of the two strength percentiles. "
            "They are not averaged to produce the served score; the fitted raw "
            "pillar components are combined before percentile mapping."
        )

    with st.container(border=True):
        st.markdown("## Score Bridge")
        st.caption("Each disclosed adjustment is shown separately and reconciles to the served score.")
        if score_bridge:
            pillar_score = float(score_bridge.get("pillar_risk_score", np.nan))
            confidence_delta = float(score_bridge.get("confidence_adjustment", 0.0))
            confidence_score = float(
                score_bridge.get("confidence_adjusted_risk_score", pillar_score)
            )
            floor_delta = float(score_bridge.get("risk_floor_delta", 0.0))
            post_floor = float(score_bridge.get("score_after_risk_floor", confidence_score))
            penalty = float(score_bridge.get("critical_penalty_applied", 0.0))
            pre_round = float(
                score_bridge.get("pre_round_structural_risk_score", post_floor + penalty)
            )
            structural = float(score_bridge.get("structural_risk_score", round(pre_round, 1)))
            crisis_uplift = float(country_score_row.get("crisis_uplift", 0.0) or 0.0)
            bridge_rows = pd.DataFrame(
                [
                    {"Stage": "Pillar-only relative risk", "Change": np.nan, "Result": pillar_score},
                    {"Stage": "Coverage confidence adjustment", "Change": confidence_delta, "Result": confidence_score},
                    {"Stage": "Minimum risk floor", "Change": floor_delta, "Result": post_floor},
                    {"Stage": "Critical-data penalty", "Change": penalty, "Result": pre_round},
                    {"Stage": "Structural score after rounding", "Change": structural - pre_round, "Result": structural},
                    {"Stage": "Legacy crisis adjustment", "Change": crisis_uplift, "Result": risk_score},
                ]
            )
            st.dataframe(
                bridge_rows,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Stage": st.column_config.TextColumn("Stage", width="large"),
                    "Change": st.column_config.NumberColumn("Change", format="%+.2f"),
                    "Result": st.column_config.NumberColumn("Result", format="%.2f"),
                },
            )
            missing_fields = tuple(score_bridge.get("critical_missing_fields") or ())
            if floor_delta > 1e-12:
                st.warning(
                    f"Incomplete coverage raised the score by {floor_delta:.2f} "
                    "to the model's minimum risk floor."
                )
            if missing_fields:
                field_labels = ", ".join(format_identifier(field) for field in missing_fields)
                st.warning(
                    f"Critical inputs imputed: {field_labels}. The related risk "
                    f"penalty added {penalty:.2f} points."
                )
            st.caption(
                "Legacy crisis adjustment is an upward-only served-model overlay; "
                "it is not presented as a validated crisis probability."
            )
        else:
            st.info(
                "Score-stage evidence is unavailable for this snapshot. The served "
                "score remains visible, but its adjustment bridge cannot be verified here."
            )

    with st.container(border=True):
        st.markdown("## Score Drivers")
        driver_state_key = f"score_drivers_loaded_{selected_snapshot}_{selected_country_code}"
        if not st.session_state.get(driver_state_key, False):
            st.caption(
                "Feature attribution is calculated on demand because it is slower "
                "than loading the served country profile."
            )
            if st.button("Load score drivers", key=f"load_{driver_state_key}"):
                st.session_state[driver_state_key] = True
                st.rerun()
        else:
            try:
                if driver_pipeline_for_profile is None:
                    raise ValueError("checksum-matched inference pipeline is unavailable")
                driver_model = {
                    'country_scores': scores_df,
                    'feature_values': model_features,
                    'training_date': pca_info.get('training_date'),
                    'pca_info': pca_info,
                    'trained': True,
                    'countries_trained': len(scores_df),
                }
                payload = compute_country_drivers(
                    selected_snapshot, selected_country_code, driver_version,
                    driver_model, driver_pipeline_for_profile,
                )
                if 'error' in payload:
                    st.info(
                        "Feature attribution is unavailable for this country and snapshot."
                    )
                else:
                    drivers = payload.get("drivers", [])
                    evidence_by_feature = (
                        input_inventory.rows.set_index("feature").to_dict(orient="index")
                        if not input_inventory.rows.empty
                        else {}
                    )
                    driver_rows = [
                        {
                            "Feature": format_identifier(driver["feature"]),
                            "Pillar": format_pillar_label(driver["pillar"]),
                            "Raw Value": (
                                np.nan if driver["raw_value"] is None
                                else float(driver["raw_value"])
                            ),
                            "Value Used": float(driver["used_value"]),
                            "Status": (
                                "Imputed for scoring" if driver["is_imputed"]
                                else "Reported or derived"
                            ),
                            "Evidence Type": evidence_by_feature.get(
                                driver["feature"], {}
                            ).get("evidence_type", "Not recorded"),
                            "Unit": evidence_by_feature.get(
                                driver["feature"], {}
                            ).get("unit", "Source-defined"),
                            "Period": evidence_by_feature.get(
                                driver["feature"], {}
                            ).get("period"),
                            "Source / Lineage": evidence_by_feature.get(
                                driver["feature"], {}
                            ).get("source_family", "Not recorded"),
                            "Risk Contribution": float(driver["risk_contribution"]),
                            "Peer Percentile": (
                                np.nan if driver["peer_percentile_raw"] is None
                                else float(driver["peer_percentile_raw"])
                            ),
                        }
                        for driver in drivers
                    ]
                    driver_df = pd.DataFrame(
                        driver_rows,
                        columns=[
                            "Feature", "Pillar", "Raw Value", "Value Used",
                            "Status", "Evidence Type", "Unit", "Period",
                            "Source / Lineage", "Risk Contribution",
                            "Peer Percentile",
                        ],
                    )
                    raising = driver_df[driver_df["Risk Contribution"] > 0].nlargest(
                        5, "Risk Contribution"
                    )
                    mitigating = driver_df[driver_df["Risk Contribution"] < 0].nsmallest(
                        5, "Risk Contribution"
                    )
                    driver_col1, driver_col2 = st.columns(2)
                    contribution_chart = pd.concat([raising, mitigating]).copy()
                    if not contribution_chart.empty:
                        contribution_chart["Direction"] = np.where(
                            contribution_chart["Risk Contribution"] > 0,
                            "Raises risk",
                            "Mitigates risk",
                        )
                        contribution_chart = contribution_chart.sort_values(
                            "Risk Contribution"
                        )
                        driver_figure = px.bar(
                            contribution_chart,
                            x="Risk Contribution",
                            y="Feature",
                            color="Direction",
                            orientation="h",
                            color_discrete_map={
                                "Raises risk": "#C84A50",
                                "Mitigates risk": "#2F80ED",
                            },
                            labels={"Risk Contribution": "Contribution to raw pillar risk"},
                        )
                        apply_responsive_chart_layout(
                            driver_figure,
                            title="Largest feature contributions",
                            showlegend=True,
                            yaxis_title=None,
                        )
                        driver_figure.update_layout(
                            height=max(360, 38 * len(contribution_chart)),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            legend_title_text=None,
                        )
                        st.plotly_chart(
                            driver_figure,
                            use_container_width=True,
                            theme="streamlit",
                            key=f"driver_contributions_{selected_country_code}",
                            config=accessible_plotly_config(),
                        )
                    with driver_col1:
                        st.markdown("### Main Risk-Raising Drivers")
                        st.dataframe(
                            raising[
                                [
                                    "Feature", "Value Used", "Unit", "Period",
                                    "Source / Lineage", "Risk Contribution", "Status",
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Risk Contribution": st.column_config.NumberColumn(
                                    "Contribution", format="%+.3f"
                                )
                            },
                        )
                    with driver_col2:
                        st.markdown("### Main Risk-Mitigating Drivers")
                        st.dataframe(
                            mitigating[
                                [
                                    "Feature", "Value Used", "Unit", "Period",
                                    "Source / Lineage", "Risk Contribution", "Status",
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Risk Contribution": st.column_config.NumberColumn(
                                    "Contribution", format="%+.3f"
                                )
                            },
                        )
                    with st.expander("Technical attribution table", expanded=False):
                        st.dataframe(
                            driver_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Raw Value": st.column_config.NumberColumn(format="%.3f"),
                                "Value Used": st.column_config.NumberColumn(format="%.3f"),
                                "Risk Contribution": st.column_config.NumberColumn(format="%+.4f"),
                                "Peer Percentile": st.column_config.NumberColumn(format="%.1f"),
                            },
                        )
                        st.download_button(
                            "Download Attribution Audit",
                            data=driver_df.to_csv(index=False).encode("utf-8"),
                            file_name=(
                                f"bankenv_{selected_country_code.lower()}_"
                                f"score_attribution_{snapshot_label}.csv"
                            ),
                            mime="text/csv",
                            key=f"download_attribution_{selected_country_code}",
                        )
                    st.caption(
                        "Positive risk contributions push the country toward higher "
                        "risk relative to the training mean; contributions sum to the "
                        "raw pillar components before percentile mapping, confidence "
                        "weighting, floors, penalties, and the legacy crisis adjustment. "
                        "Imputed rows use model-filled values, not reported data."
                    )
            except Exception:
                LOGGER.exception(
                    "Feature attribution failed for %s",
                    selected_country_code,
                )
                st.info(
                    "Feature attribution could not be loaded. Retry this panel; "
                    "the served score and evidence inventory remain available."
                )

    with st.container(border=True):
        st.markdown("## Model Evidence")
        st.caption(
            "The inventory is derived from the fitted loading maps. Every active "
            "input shows the value used for scoring, unit, source family, period, "
            "and whether the value was direct or imputed."
        )
        inventory_df = input_inventory.rows.copy()
        if inventory_df.empty:
            st.info(
                "The active loading-map inventory is unavailable for this snapshot. "
                "The served score can be viewed, but its input evidence cannot be listed."
            )
        else:
            strongest = (
                inventory_df.assign(_importance=inventory_df["loading"].abs())
                .sort_values(["pillar", "_importance"], ascending=[True, False])
                .groupby("pillar", sort=False)
                .head(6)
                .drop(columns="_importance")
            )
            compact = strongest[
                [
                    "label", "pillar_label", "value", "unit", "period",
                    "status", "evidence_type",
                ]
            ].rename(
                columns={
                    "label": "Input",
                    "pillar_label": "Pillar",
                    "value": "Value Used",
                    "unit": "Unit",
                    "period": "Period",
                    "status": "Status",
                    "evidence_type": "Evidence Type",
                }
            )
            st.markdown("### Key Active Inputs")
            st.dataframe(
                compact,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Input": st.column_config.TextColumn(width="large"),
                    "Value Used": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            with st.expander(
                f"All {len(inventory_df)} Active Inputs",
                expanded=False,
            ):
                filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
                with filter_col1:
                    evidence_search = st.text_input(
                        "Search inputs",
                        key=f"evidence_search_{selected_country_code}",
                        placeholder="Feature, source, or unit",
                    )
                with filter_col2:
                    pillar_filter = st.selectbox(
                        "Pillar",
                        options=["All", *sorted(inventory_df["pillar_label"].unique())],
                        key=f"evidence_pillar_{selected_country_code}",
                    )
                with filter_col3:
                    status_filter = st.selectbox(
                        "Status",
                        options=["All", *sorted(inventory_df["status"].unique())],
                        key=f"evidence_status_{selected_country_code}",
                    )
                filtered_inventory = inventory_df.copy()
                if evidence_search.strip():
                    needle = evidence_search.strip().lower()
                    search_columns = [
                        "label", "feature", "source_family", "unit", "pillar_label"
                    ]
                    mask = pd.Series(False, index=filtered_inventory.index)
                    for column in search_columns:
                        mask |= filtered_inventory[column].astype(str).str.lower().str.contains(
                            needle,
                            regex=False,
                        )
                    filtered_inventory = filtered_inventory.loc[mask]
                if pillar_filter != "All":
                    filtered_inventory = filtered_inventory[
                        filtered_inventory["pillar_label"] == pillar_filter
                    ]
                if status_filter != "All":
                    filtered_inventory = filtered_inventory[
                        filtered_inventory["status"] == status_filter
                    ]
                authoritative = filtered_inventory[
                    [
                        "label", "feature", "pillar_label", "value", "unit",
                        "period", "status", "evidence_type", "source_family",
                        "model_role",
                    ]
                ].rename(
                    columns={
                        "label": "Input",
                        "feature": "Technical Code",
                        "pillar_label": "Pillar",
                        "value": "Value Used",
                        "unit": "Unit",
                        "period": "Period",
                        "status": "Status",
                        "evidence_type": "Evidence Type",
                        "source_family": "Source / Lineage",
                        "model_role": "Score Role",
                    }
                )
                st.dataframe(
                    authoritative,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Input": st.column_config.TextColumn(width="large"),
                        "Value Used": st.column_config.NumberColumn(format="%.3f"),
                    },
                )
                st.download_button(
                    "Download Active-Input Evidence",
                    data=authoritative.to_csv(index=False).encode("utf-8"),
                    file_name=(
                        f"bankenv_{selected_country_code.lower()}_active_inputs_"
                        f"{snapshot_label}.csv"
                    ),
                    mime="text/csv",
                    key=f"download_inputs_{selected_country_code}",
                )

        render_candidate_country_evidence(
            selected_country_code,
            model_features,
            overlay_enabled=False,
            selected_groups=None,
            active_features=active_feature_codes,
        )

    with st.container(border=True):
        st.markdown("## Peer Comparison")

        peers_df = safe_find_peers(
            selected_country_code,
            scores_df,
            6,
            model_features,
        )
        nearest_peer_codes = (
            peers_df['country_code'].tolist()
            if peers_df is not None and len(peers_df) > 0
            else []
        )
        peer_options = [
            code for code in available_country_codes
            if code != selected_country_code
        ]
        recommended_peer_codes = nearest_peer_codes[:4]
        peer_widget_key = f"custom_peer_codes_{selected_country_code}"
        if peer_widget_key in st.session_state:
            cleaned_peers = list(
                dict.fromkeys(
                    code for code in st.session_state.get(peer_widget_key, [])
                    if code in peer_options
                )
            )[:8]
            if cleaned_peers != list(st.session_state.get(peer_widget_key, [])):
                st.session_state[peer_widget_key] = cleaned_peers

        def _reset_recommended_peers() -> None:
            st.session_state[peer_widget_key] = list(recommended_peer_codes)

        custom_peer_codes = st.multiselect(
            "Peer set",
            options=peer_options,
            default=recommended_peer_codes,
            format_func=format_country_option,
            key=peer_widget_key,
            max_selections=8,
            help=(
                "Defaults to nearest-neighbor peers from the model feature space. "
                "Choose up to eight countries; the same set carries into Explorer."
            ),
        )
        peer_codes = list(custom_peer_codes)
        peer_mode = (
            "Recommended peers"
            if peer_codes == recommended_peer_codes
            else "Custom peers"
            if peer_codes
            else "No peers selected"
        )
        peer_status_col, peer_reset_col = st.columns([3, 1])
        peer_status_col.caption(
            f"{peer_mode}. Recommended peers use economic scale, development "
            "level, banking structure, liquidity, and score proximity."
        )
        peer_reset_col.button(
            "Reset Recommended",
            key=f"reset_peers_{selected_country_code}",
            use_container_width=True,
            on_click=_reset_recommended_peers,
            disabled=not recommended_peer_codes,
        )
        st.session_state[f"shared_peer_codes_{selected_country_code}"] = peer_codes

        if peer_codes:
            comparison_cols = [
                'country_code', 'country_name', 'risk_score',
                'economic_pillar', 'industry_pillar', 'data_coverage'
            ]
            selected_row = country_score_row[comparison_cols].to_frame().T
            peer_rows = scores_df[scores_df['country_code'].isin(peer_codes)].copy()
            peer_rows['_peer_order'] = pd.Categorical(
                peer_rows['country_code'],
                categories=peer_codes,
                ordered=True,
            )
            peer_rows = peer_rows.sort_values('_peer_order')
            peers_comparison = pd.concat([selected_row, peer_rows[comparison_cols]], ignore_index=True)

            displayed_codes = tuple(peers_comparison["country_code"].astype(str).str.upper().tolist())
            try:
                if driver_pipeline_for_profile is None:
                    raise ValueError("inference pipeline unavailable")
                peer_driver_model = {
                    "country_scores": scores_df,
                    "feature_values": model_features,
                    "training_date": pca_info.get("training_date"),
                    "pca_info": pca_info,
                    "trained": True,
                    "countries_trained": len(scores_df),
                }
                dominant_drivers = compute_peer_dominant_drivers(
                    selected_snapshot,
                    displayed_codes,
                    driver_version,
                    peer_driver_model,
                    driver_pipeline_for_profile,
                )
            except Exception:
                LOGGER.exception("Peer driver attribution unavailable")
                dominant_drivers = {code: "Unavailable" for code in displayed_codes}

            peers_comparison["Main Risk Driver"] = (
                peers_comparison["country_code"].astype(str).str.upper()
                .map(dominant_drivers).fillna("Unavailable")
            )

            peers_comparison["risk_score"] = pd.to_numeric(
                peers_comparison["risk_score"], errors="coerce"
            )
            peers_comparison["Delta vs Focus"] = (
                peers_comparison["risk_score"] - risk_score
            )
            peers_comparison["Tier"] = peers_comparison["risk_score"].map(
                lambda score: tier_labels.get(score_to_tier(score), "Unavailable")
                if pd.notna(score) else "Unavailable"
            )

            def _country_direct_coverage(code: str) -> float:
                try:
                    inventory = build_active_input_inventory(
                        code,
                        model_features,
                        pca_info,
                        imputed_feature_values,
                    )
                    return float(inventory.coverage.ratio or 0.0)
                except Exception:
                    LOGGER.exception("Coverage inventory unavailable for %s", code)
                    return np.nan

            peers_comparison["Direct Coverage"] = peers_comparison[
                "country_code"
            ].astype(str).str.upper().map(_country_direct_coverage)
            nearest_metadata = (
                peers_df.set_index("country_code")
                if peers_df is not None and not peers_df.empty
                else pd.DataFrame()
            )
            peers_comparison["Peer Distance"] = peers_comparison["country_code"].map(
                nearest_metadata["distance"]
                if not nearest_metadata.empty and "distance" in nearest_metadata
                else {}
            )
            peers_comparison["Why This Peer"] = peers_comparison["country_code"].map(
                nearest_metadata["peer_basis"]
                if not nearest_metadata.empty and "peer_basis" in nearest_metadata
                else {}
            )
            peers_comparison.loc[
                peers_comparison["country_code"] == selected_country_code,
                ["Peer Distance", "Why This Peer"],
            ] = [0.0, "Focus country"]
            peers_comparison.loc[
                peers_comparison["Why This Peer"].isna(),
                "Why This Peer",
            ] = "User selected"
            peers_comparison["Operating Environment Risk"] = 1 + 9 * (
                1 - pd.to_numeric(
                    peers_comparison["economic_pillar"], errors="coerce"
                ) / 10
            )
            peers_comparison["Banking System Risk"] = 1 + 9 * (
                1 - pd.to_numeric(
                    peers_comparison["industry_pillar"], errors="coerce"
                ) / 10
            )
            peers_comparison.insert(
                0,
                'Role',
                ['Focus'] + [
                    'Recommended' if code in recommended_peer_codes else 'Custom'
                    for code in peer_rows['country_code'].tolist()
                ],
            )

            compact_peer_table = peers_comparison[
                [
                    "Role", "country_name", "risk_score", "Delta vs Focus",
                    "Tier", "Direct Coverage", "Main Risk Driver",
                ]
            ].rename(
                columns={"country_name": "Country", "risk_score": "Risk Score"}
            )
            st.dataframe(
                compact_peer_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Country": st.column_config.TextColumn(width="medium"),
                    "Risk Score": st.column_config.NumberColumn(format="%.1f"),
                    "Delta vs Focus": st.column_config.NumberColumn(format="%+.1f"),
                    "Direct Coverage": st.column_config.NumberColumn(format="percent"),
                    "Main Risk Driver": st.column_config.TextColumn(width="large"),
                },
            )
            with st.expander("Peer Selection Evidence", expanded=False):
                detailed_peer_table = peers_comparison[
                    [
                        "Role", "country_name", "risk_score", "Delta vs Focus",
                        "Operating Environment Risk", "Banking System Risk",
                        "Direct Coverage", "Peer Distance", "Why This Peer",
                        "Main Risk Driver",
                    ]
                ].rename(
                    columns={"country_name": "Country", "risk_score": "Risk Score"}
                )
                st.dataframe(
                    detailed_peer_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Risk Score": st.column_config.NumberColumn(format="%.1f"),
                        "Delta vs Focus": st.column_config.NumberColumn(format="%+.1f"),
                        "Operating Environment Risk": st.column_config.NumberColumn(format="%.1f"),
                        "Banking System Risk": st.column_config.NumberColumn(format="%.1f"),
                        "Direct Coverage": st.column_config.NumberColumn(format="percent"),
                        "Peer Distance": st.column_config.NumberColumn(format="%.2f"),
                    },
                )
            st.caption(
                "Main Risk Driver is the largest positive feature-level contribution "
                "to the served score. Peer distance is lower for closer matches."
            )
        else:
            st.info(
                "No peer countries are selected. Choose one or reset to the "
                "recommended set to restore the comparison."
            )

    with st.expander("Saved Scenario Preview", expanded=False):
        st.caption(
            "Optional saved challenger artifacts show a separate analytical score. "
            "They do not change the served tier, percentile, drivers, ranking, or peers."
        )
        overlay_col1, overlay_col2 = st.columns(2)
        with overlay_col1:
            show_liquidity_overlay = st.toggle(
                "Liquidity scenario",
                value=False,
                key="profile_liquidity_challenger_overlay",
                help="Government and external-liquidity challenger features.",
            )
        with overlay_col2:
            show_commodity_overlay = st.toggle(
                "Commodity scenario",
                value=False,
                key="profile_commodity_challenger_overlay",
                help="Commodity export concentration; independent of liquidity.",
            )
        overlay_scenario, overlay_label, overlay_groups = candidate_overlay_scenario(
            show_liquidity_overlay,
            show_commodity_overlay,
        )
        challenger_scores_for_profile = (
            load_candidate_overlay_scores(overlay_scenario)
            if overlay_scenario else pd.DataFrame()
        )
        challenger_score = None
        if not challenger_scores_for_profile.empty:
            challenger_row = challenger_scores_for_profile[
                challenger_scores_for_profile["country_code"] == selected_country_code
            ]
            if not challenger_row.empty:
                challenger_score = float(challenger_row.iloc[0]["candidate_risk_score"])
        show_challenger_overlay = overlay_scenario is not None
        if show_challenger_overlay and challenger_score is not None:
            scenario_delta = challenger_score - risk_score
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Served Score", f"{risk_score:.1f}/10")
            sc2.metric(f"{overlay_label} Scenario", f"{challenger_score:.1f}/10")
            sc3.metric("Difference", f"{scenario_delta:+.1f}")
        elif show_challenger_overlay:
            st.info(
                "The selected saved scenario is not packaged for this country. "
                "Turn it off or inspect the candidate inventory below."
            )


        if show_challenger_overlay:
            render_candidate_country_evidence(
                selected_country_code,
                model_features,
                overlay_enabled=True,
                selected_groups=overlay_groups,
                active_features=active_feature_codes,
            )

# ==============================================================================
# VIEW: Data Explorer
# ==============================================================================
if primary_view == "Explorer":
    st.markdown("# Data Explorer")
    st.caption(
        "Compare source histories, calculate auditable ratios and changes, or "
        "inspect one source. Nothing here changes the served model score."
    )
    st.session_state.setdefault(
        "explorer_tool",
        {
            "compare": "Compare",
            "calculate": "Calculate",
            "source inspector": "Source Inspector",
        }.get(_query_param_value("tool", "compare").strip().lower(), "Compare"),
    )
    explorer_tools = ("Compare", "Calculate", "Source Inspector")
    if st.session_state.get("explorer_tool") not in explorer_tools:
        st.session_state["explorer_tool"] = "Compare"
    explorer_tool = _segmented_navigation(
        "Explorer task",
        options=explorer_tools,
        key="explorer_tool",
        on_change=_sync_public_query_state,
    )

    with st.container(border=True):
        explorer_col1, explorer_col2 = st.columns([2, 3])
        with explorer_col1:
            explorer_focus_country = st.selectbox(
                "Focus country",
                options=available_country_codes,
                format_func=format_country_option,
                key="explorer_focus_country",
                on_change=_sync_public_query_state,
                help=(
                    "Used for source history and to seed comparison "
                    "country defaults."
                ),
            )
        explorer_default_peers = []
        with explorer_col2:
            if explorer_tool in {"Compare", "Calculate"}:
                explorer_peers_df = safe_find_peers(
                    explorer_focus_country,
                    scores_df,
                    4,
                    model_features,
                )
                explorer_nearest_peer_codes = (
                    explorer_peers_df['country_code'].tolist()
                    if explorer_peers_df is not None and len(explorer_peers_df) > 0
                    else []
                )
                shared_peer_codes = st.session_state.get(
                    f"shared_peer_codes_{explorer_focus_country}",
                    [],
                )
                explorer_default_peers = (
                    [
                        code for code in shared_peer_codes
                        if code in available_country_codes
                        and code != explorer_focus_country
                    ][:8]
                    or explorer_nearest_peer_codes[:4]
                )
                if explorer_default_peers:
                    st.caption(
                        "Comparison peers: "
                        + ", ".join(
                            format_country_option(code)
                            for code in explorer_default_peers
                        )
                    )
                else:
                    st.caption(
                        "No nearest-neighbor peers are available for this country."
                    )
            else:
                st.caption(
                    "Only the selected source and focus-country history are loaded."
                )

    if explorer_tool == "Compare":
        render_indicator_comparison(
            scores=scores_df,
            selected_country=explorer_focus_country,
            default_peer_codes=explorer_default_peers,
            country_formatter=format_country_option,
            wgi_panel=wgi_data,
        )
    elif explorer_tool == "Calculate":
        render_calculated_series_builder(
            scores=scores_df,
            selected_country=explorer_focus_country,
            default_peer_codes=explorer_default_peers,
            country_formatter=format_country_option,
            wgi_panel=wgi_data,
        )
    elif explorer_tool == "Source Inspector":
        render_source_inspector(
            selected_country=explorer_focus_country,
            country_formatter=format_country_option,
            wgi_panel=wgi_data,
        )


# ==============================================================================
# VIEW: Methodology
# ==============================================================================
if primary_view == "Methodology":
    render_methodology_workspace(
        scores=scores_df,
        features=model_features,
        manifest=data_manifest,
        pca=pca_info,
    )
    if SHOW_ADMIN_DIAGNOSTICS:
        with st.expander(
            f"Admin diagnostics: {HEALTH_LABELS.get(health_report['overall'], health_report['overall'])}",
            expanded=health_report["overall"] in ("degraded", "unknown"),
        ):
            render_system_health_panel()
