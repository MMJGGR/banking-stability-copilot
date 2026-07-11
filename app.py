import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os

from src.data_loader import (
    FSIBSISLoader,
    IMFDataLoader,
    WGILoader,
    is_time_period_column,
    parse_period_label,
)
from src.country_names import fill_missing_country_names
from src.health import build_health_report
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
from src.dashboard.styles import STYLES, score_to_tier
from src.dashboard.components import (
    render_summary_card, 
    render_data_snapshot,
    render_time_series_deep_dive,
    WEO_INDICATORS,
    FSIC_NAME_PATTERNS
)
from src.dashboard.calculated_series import (
    available_frequencies,
    compute_cross_sectional_share,
    compute_ratio,
    compute_temporal_change,
    filter_time_range,
    normalize_observation_frame,
    restrict_frequency,
)
from src.dashboard.global_view import render_global_summary
from src.utils import find_peers


def extract_mermaid_code(markdown_text: str) -> str:
    """Extract mermaid code block from markdown."""
    import re
    match = re.search(r'```mermaid\n(.*?)\n```', markdown_text, re.DOTALL)
    if match:
        return match.group(1)
    return None

def render_markdown_with_images(markdown_text: str):
    """
    Render markdown text, identifying and displaying local images separately.
    Standard st.markdown cannot render local file paths provided in ![alt](path).
    """
    import re
    import os
    
    # Get the directory where app.py is located (project root)
    app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Pattern to find images: ![alt text](path)
    # capturing groups: 1=alt, 2=path
    pattern = r'!\[(.*?)\]\((.*?)\)'
    
    # Split text by images
    parts = re.split(pattern, markdown_text)
    
    # re.split returns [text, alt, path, text, alt, path, ...]
    # We iterate and render
    
    i = 0
    while i < len(parts):
        text_segment = parts[i]
        if text_segment.strip():
            st.markdown(text_segment)
        
        # If there are more parts, next are alt and path
        if i + 2 < len(parts):
            alt_text = parts[i+1]
            image_path = parts[i+2]
            
            # Resolve relative paths from app directory
            if not os.path.isabs(image_path):
                image_path = os.path.join(app_dir, image_path)
            
            # Check if file exists to prevent errors
            if os.path.exists(image_path):
                try:
                    # Read image as bytes for reliable Streamlit rendering
                    with open(image_path, 'rb') as img_file:
                        st.image(img_file.read(), caption=alt_text)
                except Exception as e:
                    st.warning(f"Could not load image: {image_path} - {e}")
            else:
                # Image not found - show placeholder message
                st.info(f"{alt_text} (image will appear after model training)")
                
            i += 3 # skip (text, alt, path)
        else:
            i += 1


# Page Config
st.set_page_config(
    page_title="BankEnv",
    page_icon="assets/bankenv-favicon.svg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply Custom Styles
st.markdown(STYLES, unsafe_allow_html=True)

# ==============================================================================
# DATA LOADING (Cached)
# ==============================================================================
@st.cache_resource
def load_all_data():
    """Load model artifacts and lightweight reference data.

    Loads the checksum-verified active artifact when possible; otherwise
    degrades to the newest archived last-known-good snapshot bundle so a bad
    refresh cannot take the application down.
    """
    # timestamp: force_reload_2026_01_12_v2
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
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None, {}, {"mode": "error", "active_error": str(e)}

    loader = IMFDataLoader()

    try:
        wgi_loader = WGILoader()
        wgi_data = wgi_loader.load()
    except Exception as e:
        wgi_data = None

    return (
        scores_df, loader, wgi_data, model_features, pca_info,
        served_manifest, serving_status,
    )


@st.cache_resource(max_entries=4)
def load_archived_snapshot_cached(name: str):
    """Checksum-verified, read-only load of one archived snapshot bundle."""
    return load_archived_snapshot(name)


@st.cache_resource(max_entries=4)
def load_inference_pipeline(snapshot: str):
    """Load the fitted pillar pipeline for driver-table attribution."""
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


@st.cache_data(show_spinner=False, max_entries=32)
def compute_country_drivers(snapshot: str, country_code: str,
                            _model: dict, _pipeline) -> dict:
    """Per-feature score attribution for one country (rank 22)."""
    from src.scripts.explain_country_scores import build_driver_table

    report = build_driver_table([country_code], model=_model, pipeline=_pipeline)
    return report["countries"][country_code]


@st.cache_data(show_spinner=False, max_entries=48)
def load_country_history(country_code: str, dataset: str) -> pd.DataFrame:
    """Load one selected-country history slice for the Data Explorer."""
    return IMFDataLoader().get_country_data(country_code, dataset)


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


@st.cache_data(show_spinner=False, max_entries=24)
def load_fsibsis_country_history(country_code: str) -> pd.DataFrame:
    """Load one FSIBSIS selected-country history slice on demand."""
    fsibsis_loader = FSIBSISLoader()
    fsibsis_loader.load()
    fsibsis_wide = fsibsis_loader.get_country_data(country_code)
    return _fsibsis_wide_to_long(fsibsis_wide, country_code)


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
    st.markdown("### Cross-Country Indicator Comparison")

    available_codes = scores.sort_values('country_name')['country_code'].tolist()
    default_countries = []
    for code in [selected_country] + default_peer_codes:
        if code in available_codes and code not in default_countries:
            default_countries.append(code)
    default_countries = default_countries[:5]

    control_col1, control_col2 = st.columns([1, 3])
    with control_col1:
        source_choice = st.selectbox(
            "Source",
            [
                "Economic (WEO)",
                "Banking ratios (FSIC)",
                "Bank balance sheet (FSIBSIS)",
                "Monetary (MFS)",
                "Governance (WGI)",
            ],
            key="compare_source",
        )
    with control_col2:
        compare_countries = st.multiselect(
            "Countries",
            options=available_codes,
            default=default_countries,
            format_func=country_formatter,
            key=f"compare_countries_{selected_country}",
            help="Compare one indicator across the selected country and peers. Keep the set small for hosted performance.",
        )

    if not compare_countries:
        st.info("Select at least one country to compare.")
        return

    if len(compare_countries) > 8:
        st.warning("Showing the first 8 selected countries to keep the hosted app responsive.")
        compare_countries = compare_countries[:8]

    source_to_dataset = {
        "Economic (WEO)": "WEO",
        "Banking ratios (FSIC)": "FSIC",
        "Bank balance sheet (FSIBSIS)": "FSIBSIS",
        "Monetary (MFS)": "MFS",
        "Governance (WGI)": "WGI",
    }
    dataset = source_to_dataset[source_choice]

    with st.spinner(f"Loading {dataset} history for selected countries..."):
        source_df = _load_comparison_source(dataset, compare_countries, wgi_panel)
    if source_df is None or len(source_df) == 0:
        st.info(f"No {dataset} data is available for the selected countries.")
        return

    source_df = source_df.copy()
    source_df['country_name'] = source_df['country_code'].map(country_formatter)

    use_name_as_key = dataset in ("FSIC", "FSIBSIS") and 'indicator_name' in source_df.columns
    if use_name_as_key:
        indicator_options = source_df['indicator_name'].dropna().unique().tolist()
        indicator_options = sorted(indicator_options, key=lambda x: x.lower())
        display_map = {name: name[:90] + "..." if len(name) > 90 else name for name in indicator_options}
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
    with indicator_col2:
        time_range = st.selectbox(
            "Range",
            ["5 Years", "10 Years", "20 Years", "All Data"],
            index=1,
            key=f"compare_range_{dataset}",
        )

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
    fig = px.line(
        chart_df,
        x='date',
        y='value',
        color='country_name',
        markers=True,
        title=title,
    )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=48, b=20),
        xaxis_title=None,
        yaxis_title=None,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme="streamlit",
        key=f"compare_chart_{dataset}_{selected_indicator}",
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

    return load_multi_country_history(tuple(countries), dataset)


def _indicator_selector_metadata(source_df: pd.DataFrame, dataset: str):
    """Return indicator options, labels and key column for a source frame."""
    use_name_as_key = dataset in ("FSIC", "FSIBSIS") and 'indicator_name' in source_df.columns
    if use_name_as_key:
        indicator_options = source_df['indicator_name'].dropna().unique().tolist()
        indicator_options = sorted(indicator_options, key=lambda x: x.lower())
        display_map = {name: name[:90] + "..." if len(name) > 90 else name for name in indicator_options}
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
    fig = px.line(
        chart_df,
        x='date',
        y='value',
        color='country_name',
        markers=True,
        title=title,
    )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=48, b=20),
        xaxis_title=None,
        yaxis_title=y_title,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme="streamlit",
        key=chart_key,
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


def render_calculated_series_builder(
    scores: pd.DataFrame,
    selected_country: str,
    default_peer_codes: list[str],
    country_formatter,
    wgi_panel: pd.DataFrame | None,
):
    """Render bounded multi-indicator, ratio, share and temporal calculations."""
    st.markdown("### Calculated Series")
    st.caption(
        "Build lightweight exploratory calculations from one source at a time. "
        "Calculations align observations by country, date, and reporting frequency."
    )

    available_codes = scores.sort_values('country_name')['country_code'].tolist()
    default_countries = []
    for code in [selected_country] + default_peer_codes:
        if code in available_codes and code not in default_countries:
            default_countries.append(code)
    default_countries = default_countries[:5]

    source_col, country_col = st.columns([1, 3])
    with source_col:
        source_choice = st.selectbox(
            "Source",
            [
                "Economic (WEO)",
                "Banking ratios (FSIC)",
                "Bank balance sheet (FSIBSIS)",
                "Monetary (MFS)",
                "Governance (WGI)",
            ],
            key="calc_source",
        )
    with country_col:
        calc_countries = st.multiselect(
            "Countries",
            options=available_codes,
            default=default_countries,
            format_func=country_formatter,
            key=f"calc_countries_{selected_country}",
            help="Selected countries are capped at 8 for hosted performance.",
        )
    if not calc_countries:
        st.info("Select at least one country.")
        return
    if len(calc_countries) > 8:
        st.warning("Using the first 8 selected countries to keep the hosted app responsive.")
        calc_countries = calc_countries[:8]

    dataset = {
        "Economic (WEO)": "WEO",
        "Banking ratios (FSIC)": "FSIC",
        "Bank balance sheet (FSIBSIS)": "FSIBSIS",
        "Monetary (MFS)": "MFS",
        "Governance (WGI)": "WGI",
    }[source_choice]

    with st.spinner(f"Loading {dataset} history for calculated series..."):
        source_df = _load_comparison_source(dataset, calc_countries, wgi_panel)
    if source_df is None or len(source_df) == 0:
        st.info(f"No {dataset} data is available for the selected countries.")
        return

    indicator_options, display_map, indicator_col = _indicator_selector_metadata(source_df, dataset)
    if not indicator_options:
        st.info("No indicators are available for this source/country selection.")
        return

    mode_col, range_col = st.columns([2, 1])
    with mode_col:
        calc_mode = st.selectbox(
            "Calculation",
            [
                "Raw multi-indicator panels",
                "Ratio",
                "Cross-sectional share",
                "Temporal change / index",
            ],
            key=f"calc_mode_{dataset}",
        )
    with range_col:
        time_range = st.selectbox(
            "Range",
            ["5 Years", "10 Years", "20 Years", "All Data"],
            index=1,
            key=f"calc_range_{dataset}",
        )

    if calc_mode == "Raw multi-indicator panels":
        selected_indicators = st.multiselect(
            "Indicators",
            options=indicator_options,
            default=indicator_options[: min(3, len(indicator_options))],
            format_func=lambda x: display_map[x],
            key=f"calc_raw_indicators_{dataset}",
            help="Shows up to 5 indicators as separate panels to avoid mixing units.",
        )
        selected_indicators = selected_indicators[:5]
        if not selected_indicators:
            st.info("Select at least one indicator.")
            return
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
        num_col, den_col, scale_col = st.columns([2, 2, 1])
        with num_col:
            numerator_key = st.selectbox(
                "Numerator",
                indicator_options,
                format_func=lambda x: display_map[x],
                key=f"calc_ratio_num_{dataset}",
            )
        with den_col:
            denominator_key = st.selectbox(
                "Denominator",
                indicator_options,
                format_func=lambda x: display_map[x],
                key=f"calc_ratio_den_{dataset}",
            )
        with scale_col:
            scale_label = st.selectbox(
                "Scale",
                ["Ratio", "Percent"],
                key=f"calc_ratio_scale_{dataset}",
            )
        scale = 100.0 if scale_label == "Percent" else 1.0
        numerator = normalize_observation_frame(
            source_df,
            numerator_key,
            indicator_col,
            display_map[numerator_key],
        )
        denominator = normalize_observation_frame(
            source_df,
            denominator_key,
            indicator_col,
            display_map[denominator_key],
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
                key=f"calc_ratio_frequency_{dataset}_{numerator_key}_{denominator_key}",
            )
        elif common_freqs:
            selected_freq = common_freqs[0]
        ratio = compute_ratio(
            restrict_frequency(numerator, selected_freq),
            restrict_frequency(denominator, selected_freq),
            scale=scale,
        )
        ratio = filter_time_range(ratio, time_range)
        title = f"{display_map[numerator_key]} / {display_map[denominator_key]}"
        _render_calculated_chart(
            ratio,
            title=title,
            country_formatter=country_formatter,
            y_title=scale_label,
            chart_key=f"calc_ratio_chart_{dataset}_{numerator_key}_{denominator_key}",
        )
        st.caption(
            f"Formula: {display_map[numerator_key]} ÷ {display_map[denominator_key]}"
            f"{' × 100' if scale_label == 'Percent' else ''}. "
            "Only exact country/date/frequency matches are used; zero denominators are excluded."
        )
        return

    if calc_mode == "Cross-sectional share":
        indicator_key = st.selectbox(
            "Indicator",
            indicator_options,
            format_func=lambda x: display_map[x],
            key=f"calc_share_indicator_{dataset}",
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
                key=f"calc_share_frequency_{dataset}_{indicator_key}",
            )
        elif freq_options:
            selected_freq = freq_options[0]
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


def render_model_card_summary(
    scores: pd.DataFrame,
    features: pd.DataFrame | None,
    manifest: dict,
    pca: dict | None,
):
    """Render a UI-native model card instead of dumping repository Markdown."""
    snapshot_id = manifest.get("snapshot_id", "unversioned")
    status = str(manifest.get("snapshot_status", "not recorded")).replace("_", " ").title()
    training_date = (pca or {}).get("training_date") or manifest.get("model", {}).get("training_date")

    st.markdown("### Model Card")
    st.caption("Plain-English summary of the active scoring artifact shown in this app.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score Scale", "1–10", help="1 is lower relative risk; 10 is higher relative risk.")
    c2.metric("Countries Scored", f"{len(scores):,}")
    c3.metric("Snapshot", _display_value(snapshot_id))
    c4.metric("Status", status)
    st.caption(f"Training timestamp: {_display_value(training_date)}")

    st.markdown("#### Intended Use")
    use_col, no_use_col = st.columns(2)
    with use_col:
        st.markdown(
            """
Good uses:

- Cross-country banking-system risk screening.
- Peer comparison and watchlist prioritization.
- Finding data gaps and countries needing analyst review.
"""
        )
    with no_use_col:
        st.markdown(
            """
Do not use for:

- Automatic investment, lending, or supervisory decisions.
- Institution-level solvency calls.
- Precise crisis timing or causal claims.
"""
        )

    st.markdown("#### Current Score Construction")
    structure_rows = [
        {
            "Component": "Economic pillar",
            "Current role": "50% of pillar score",
            "What it captures": "Macro conditions, fiscal position, monetary/reserve proxies and governance inputs.",
        },
        {
            "Component": "Banking / industry pillar",
            "Current role": "50% of pillar score",
            "What it captures": "Capital, asset quality, liquidity, profitability, concentration and sovereign-bank nexus inputs.",
        },
        {
            "Component": "Crisis classifier",
            "Current role": "10% final-score blend",
            "What it captures": "Forward-looking three-year systemic-banking-crisis signal from historical crisis labels.",
        },
        {
            "Component": "Coverage policy",
            "Current role": "Confidence weighting and floors",
            "What it captures": "Limits overconfidence where inputs are sparse or heavily imputed.",
        },
    ]
    st.dataframe(pd.DataFrame(structure_rows), use_container_width=True, hide_index=True)

    with st.expander("Model limitations and review flags", expanded=True):
        limitations = [
            {
                "Issue": "Relative ranking",
                "Why it matters": "Pillar scores are percentiles within the current country universe, so rankings can move when the universe or source coverage changes.",
                "Status": "Disclose in app",
            },
            {
                "Issue": "PCA orientation",
                "Why it matters": "Unsupervised PCA can learn statistically strong but economically weak signs for some variables.",
                "Status": "Needs constrained/directional scoring review",
            },
            {
                "Issue": "Crisis overlay",
                "Why it matters": "The current 90/10 blend is not a literal probability and should not be described as a direct additive penalty.",
                "Status": "Needs formula review before formal approval",
            },
            {
                "Issue": "External liquidity gap",
                "Why it matters": "Debt service burden, gross external financing needs, current-account receipts, reserves adequacy and portfolio-flow stress are not yet fully modeled.",
                "Status": "Priority enhancement candidate",
            },
            {
                "Issue": "Imputation sensitivity",
                "Why it matters": "Countries with missing banking or external data can be materially affected by KNN imputation.",
                "Status": "Show coverage and missing fields",
            },
        ]
        st.dataframe(pd.DataFrame(limitations), use_container_width=True, hide_index=True)

    st.markdown("#### Release Governance")
    st.info(
        "The artifact is manifest-verified for serving. Formal production approval still requires "
        "grouped/out-of-time validation, material score-movement review, challenger comparison, "
        "and named approval with rollback artifacts."
    )


def render_data_card_summary(features: pd.DataFrame | None, manifest: dict):
    """Render a UI-native data card focused on source quality and model gaps."""
    st.markdown("### Data Card")
    st.caption("Current source inventory, freshness, coverage, and priority data gaps.")

    source_mode = str(manifest.get("source_mode", "not recorded")).replace("_", " ")
    sources = manifest.get("sources", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Source Mode", source_mode)
    c2.metric("Active Sources", f"{len(sources):,}")
    c3.metric("Snapshot Cutoff", _display_value(manifest.get("as_of_date") or manifest.get("snapshot_id")))

    source_rows = []
    for source_name, details in sorted(sources.items()):
        source_rows.append(
            {
                "Source": source_name,
                "Role": _source_role(source_name),
                "Rows": _display_value(details.get("rows"), integer=True),
                "Countries": _display_value(details.get("countries"), integer=True),
                "Indicators / Measures": _display_value(details.get("indicators"), integer=True),
                "Latest Observation": _display_value(details.get("latest_observation")),
                "Count Basis": details.get("indicator_basis", "unique indicator codes"),
            }
        )
    if source_rows:
        st.markdown("#### Active Sources")
        st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Snapshot Rules")
    st.markdown(
        """
- The model scores one country-level cross-section per snapshot.
- For each country and indicator, feature engineering uses the latest observation allowed by the cutoff.
- Historical monthly, quarterly and annual histories remain available in the Data Explorer.
- Estimates, projections, carried-forward values and imputations must not be silently presented as actuals.
"""
    )

    st.markdown("#### Coverage Watchlist")
    if features is not None and len(features) > 0:
        feature_cols = [
            c for c in features.columns
            if c != "country_code"
            and not c.endswith("_year")
            and not c.endswith("_period")
            and c != "crisis_prob"
        ]
        coverage = []
        for column in feature_cols:
            coverage.append(
                {
                    "Feature": column,
                    "Direct Coverage": f"{features[column].notna().mean():.0%}",
                    "Countries": int(features[column].notna().sum()),
                }
            )
        coverage_df = (
            pd.DataFrame(coverage)
            .assign(_coverage=lambda df: df["Direct Coverage"].str.rstrip("%").astype(int))
            .sort_values(["_coverage", "Feature"])
            .drop(columns="_coverage")
            .head(12)
        )
        st.dataframe(coverage_df, use_container_width=True, hide_index=True)
        st.caption("Lowest direct-coverage model/input features. Imputed values may still exist downstream.")
    else:
        st.info("Feature artifact is unavailable, so coverage cannot be summarized.")

    st.markdown("#### Priority Missing Data Families")
    gap_rows = [
        {
            "Gap": "Debt service burden",
            "Why it matters": "Debt affordability and refinancing stress.",
            "Likely source": "IMF BOP / World Bank IDS / QEDS / GFS",
            "Current status": "Not fully modeled",
        },
        {
            "Gap": "Gross external financing needs",
            "Why it matters": "Core external-liquidity pressure measure.",
            "Likely source": "BOP + IIP/external debt + reserves",
            "Current status": "Needs computed feature",
        },
        {
            "Gap": "Current account receipts and payments",
            "Why it matters": "Denominator for external debt-service and financing-need ratios.",
            "Likely source": "IMF BOP",
            "Current status": "Not in current cache stack",
        },
        {
            "Gap": "International reserves adequacy",
            "Why it matters": "Shock buffer against FX liquidity stress.",
            "Likely source": "IMF IRFCL / MFS / BOP",
            "Current status": "Partially proxied only",
        },
        {
            "Gap": "Portfolio flows and external liabilities",
            "Why it matters": "Market-access and fickle-capital risk.",
            "Likely source": "IMF BOP, IIP, PIP/CPIS",
            "Current status": "Not fully modeled",
        },
    ]
    st.dataframe(pd.DataFrame(gap_rows), use_container_width=True, hide_index=True)


def render_current_methodology(
    scores: pd.DataFrame,
    features: pd.DataFrame | None,
    manifest: dict,
    pca: dict | None,
):
    """Render current, manifest-backed methodology instead of stale README text."""
    st.markdown("## Methodology")

    snapshot_id = manifest.get('snapshot_id', 'unversioned')
    snapshot_status = manifest.get('snapshot_status', 'manifest unavailable')
    source_mode = manifest.get('source_mode', 'not recorded')
    training_date = (pca or {}).get('training_date', 'not recorded')

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Snapshot", snapshot_id)
    with c2:
        st.metric("Status", str(snapshot_status).replace('_', ' ').title())
    with c3:
        st.metric("Countries", f"{len(scores):,}")
    with c4:
        st.metric("Source Mode", str(source_mode).replace('_', ' '))

    st.caption(f"Model training timestamp: {training_date}")

    st.markdown("### Current Model Logic")
    st.markdown(
        """
The app serves a dated country-level snapshot. Source observations are filtered
to the active cutoff, then each feature uses the latest allowed observation
available for each country and indicator. Monthly, quarterly, and annual source
data are preserved in the Data Explorer, but the risk model scores one
cross-section per snapshot.

The final score is the two-pillar PCA score plus an upward-only crisis
overlay. Each pillar is a constrained principal component: every feature has a
declared credit-risk direction (for example, higher NPL ratios can only raise
risk), so the score cannot learn economically counterintuitive signs from
covariance alone. Countries missing critical banking soundness fields receive
a bounded risk penalty instead of relying on imputed values.

The supervised systemic-crisis classifier adds
`max(0, 0.1 x ((1 + 9 x P(crisis)) - pillar score))`: it is monotone in the
crisis probability and can never lower a high pillar-based risk score. The
classifier is trained on annual historical epochs and produces a
forward-looking three-year risk signal, not a monthly or quarterly crisis
probability.
"""
    )

    st.markdown("### Active Sources")
    source_rows = []
    for source_name, details in sorted(manifest.get('sources', {}).items()):
        indicators = details.get("indicators")
        source_rows.append({
            "Source": source_name,
            "Rows": details.get("rows"),
            "Countries": details.get("countries"),
            "Latest Observation": details.get("latest_observation"),
            "Indicators / Measures": indicators if indicators is not None else "—",
            "Count Basis": details.get("indicator_basis", "unique indicator codes"),
        })
    if source_rows:
        st.dataframe(
            pd.DataFrame(source_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No source manifest is available.")

    st.markdown("### Feature Set")
    if features is not None and len(features) > 0:
        feature_cols = [
            c for c in features.columns
            if c != 'country_code' and not c.endswith('_year') and not c.endswith('_period')
        ]
        coverage_rows = []
        for column in feature_cols:
            coverage = features[column].notna().mean()
            coverage_rows.append({
                "Feature": column,
                "Coverage": f"{coverage:.0%}",
                "Countries": int(features[column].notna().sum()),
            })
        coverage_df = pd.DataFrame(coverage_rows).sort_values(
            ["Coverage", "Feature"],
            ascending=[True, True],
        )
        st.caption(
            f"{len(feature_cols)} model/input features tracked across "
            f"{features['country_code'].nunique():,} countries in the feature artifact."
        )
        st.dataframe(
            coverage_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Feature artifact is unavailable.")

    st.markdown("### Validation And Governance")
    st.markdown(
        """
Historical README performance claims are not rendered here. Approved releases
must use country-grouped and out-of-time validation, include material
score-movement review, and record source/model checksums. The current active
artifact is verified by manifest, but formal policy approval and release
promotion remain separate governance steps.
"""
    )

    card_tab_model, card_tab_data = st.tabs(["Model Card", "Data Card"])
    with card_tab_model:
        render_model_card_summary(scores, features, manifest, pca)
    with card_tab_data:
        render_data_card_summary(features, manifest)


(
    scores_df, loader, wgi_data, model_features, pca_info,
    served_manifest, serving_status,
) = load_all_data()
# In fallback mode the manifest describing what is actually being served is
# the archived bundle's manifest, not the active one on disk.
data_manifest = served_manifest or load_data_manifest()
health_report = build_health_report(data_manifest, serving_status)

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
if "profile_country_code" not in st.session_state:
    st.session_state["profile_country_code"] = default_country_code
if "explorer_focus_country" not in st.session_state:
    st.session_state["explorer_focus_country"] = st.session_state.get(
        "profile_country_code",
        default_country_code,
    )

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

# ==============================================================================
# MAIN NAVIGATION: Tabs
# ==============================================================================
tab_global, tab_profile, tab_explorer, tab_methodology = st.tabs([
    "Global Summary", "Country Profile", "Data Explorer", "Methodology"
])

# ==============================================================================
# TAB: Global Summary
# ==============================================================================
with tab_global:
    render_global_summary(scores_df, model_features, loader)

# ==============================================================================
# TAB: Country Profile
# ==============================================================================
with tab_profile:
    profile_col1, profile_col2 = st.columns([2, 3])
    with profile_col1:
        selected_country_code = st.selectbox(
            "Country",
            options=available_country_codes,
            format_func=format_country_option,
            key="profile_country_code",
            help="This selector controls the Country Profile tab only.",
        )

    country_score_row = scores_df[scores_df['country_code'] == selected_country_code].iloc[0]
    selected_country_name = country_score_row['country_name']

    # 1. HEADER: Country Name + Risk Score Summary
    st.markdown(f"## {selected_country_name}")
    
    risk_score = country_score_row['risk_score']
    tier = score_to_tier(risk_score)
    percentile = (scores_df['risk_score'] < risk_score).mean()
    
    # Risk Summary Row (inline metrics instead of card)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Risk Score", f"{risk_score:.1f}/10")
    with m2:
        tier_labels = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}
        st.metric("Risk Tier", tier_labels.get(tier, "N/A"))
    with m3:
        st.metric("Global Rank", f"Top {percentile:.0%}")
    with m4:
        coverage = country_score_row.get('data_coverage', 0)
        st.metric("Data Coverage", f"{coverage:.0%}")
    
    # Confidence warning if needed
    if country_score_row.get('risk_floor_applied', False):
        st.warning("Risk score may be capped due to incomplete data. Interpret with caution.")
    
    st.markdown("---")
    
    # 2. MODEL BREAKDOWN (replaces spider chart in header)
    st.markdown("### Model Breakdown")
    bd1, bd2, bd3 = st.columns(3)
    with bd1:
        econ_score = country_score_row['economic_pillar']
        st.metric("Economic Pillar", f"{econ_score:.1f}/10", 
                  delta=f"{econ_score - scores_df['economic_pillar'].mean():.1f} vs avg")
    with bd2:
        ind_score = country_score_row['industry_pillar']
        st.metric("Industry Pillar", f"{ind_score:.1f}/10",
                  delta=f"{ind_score - scores_df['industry_pillar'].mean():.1f} vs avg")
    with bd3:
        if 'combined_pillar' in country_score_row:
            comb_score = country_score_row['combined_pillar']
            st.metric("Combined Pillar", f"{comb_score:.1f}/10")

    with st.expander("Score Drivers — feature-level attribution"):
        try:
            driver_model = {
                'country_scores': scores_df,
                'feature_values': model_features,
                'training_date': pca_info.get('training_date'),
                'pca_info': pca_info,
                'trained': True,
                'countries_trained': len(scores_df),
            }
            driver_pipeline = load_inference_pipeline(selected_snapshot)
            payload = compute_country_drivers(
                selected_snapshot, selected_country_code,
                driver_model, driver_pipeline,
            )
            if 'error' in payload:
                st.info(payload['error'])
            else:
                summary = payload.get('summary', {})
                sc1, sc2, sc3 = st.columns(3)
                crisis_uplift = summary.get('crisis_uplift')
                sc1.metric(
                    "Crisis Uplift",
                    "—" if crisis_uplift is None else f"+{crisis_uplift:.2f}",
                    help="Upward-only classifier overlay added to the pillar score.",
                )
                critical_missing = summary.get('critical_missing_share')
                sc2.metric(
                    "Critical Fields Missing",
                    "—" if critical_missing is None else f"{critical_missing:.0%}",
                    help="Share of core banking soundness fields that had to be imputed.",
                )
                critical_penalty = summary.get('critical_penalty')
                sc3.metric(
                    "Missingness Penalty",
                    "—" if critical_penalty is None else f"+{critical_penalty:.2f}",
                    help="Disclosed risk-score penalty for imputed critical fields.",
                )
                driver_rows = [
                    {
                        "Feature": driver["feature"],
                        "Pillar": driver["pillar"],
                        "Raw Value": (
                            np.nan if driver["raw_value"] is None
                            else float(driver["raw_value"])
                        ),
                        "Value Used": round(driver["used_value"], 3),
                        "Imputed": "yes" if driver["is_imputed"] else "",
                        "Critical": "yes" if driver.get("is_critical") else "",
                        "Risk Contribution": round(driver["risk_contribution"], 4),
                        "Peer Percentile": (
                            np.nan if driver["peer_percentile_raw"] is None
                            else float(driver["peer_percentile_raw"])
                        ),
                    }
                    for driver in payload.get("drivers", [])
                ]
                st.dataframe(
                    pd.DataFrame(driver_rows),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "Positive risk contributions push the country toward higher "
                    "risk relative to the training mean; contributions sum to the "
                    "raw pillar components before percentile mapping, confidence "
                    "weighting, floors, penalties, and the crisis uplift. "
                    "'Imputed' rows use model-filled values, not reported data."
                )
        except Exception as driver_error:
            st.info(
                "Feature-level attribution is unavailable for this snapshot: "
                f"{driver_error}"
            )

    st.markdown("---")
    
    # 3. KEY DATA: Left = Model Inputs, Right = WGI Governance
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.markdown("### Key Model Inputs")
        render_data_snapshot({}, loader=loader, country_code=selected_country_code, 
                           wgi_data=wgi_data, model_features=model_features, pca_info=pca_info)
    
    with right_col:
        st.markdown("### Governance Indicators (WGI)")
        if wgi_data is not None and len(wgi_data) > 0:
            country_wgi = wgi_data[wgi_data['country_code'] == selected_country_code]
            if len(country_wgi) > 0:
                latest_wgi = country_wgi.sort_values('year').iloc[-1]
                
                wgi_columns = {
                    'voice_accountability': 'Voice & Accountability',
                    'political_stability': 'Political Stability',
                    'govt_effectiveness': 'Govt Effectiveness',
                    'regulatory_quality': 'Regulatory Quality',
                    'rule_of_law': 'Rule of Law',
                    'control_corruption': 'Corruption Control'
                }
                
                # Display in 2-column grid
                wgi_col1, wgi_col2 = st.columns(2)
                items = list(wgi_columns.items())
                for i, (col, name) in enumerate(items):
                    target_col = wgi_col1 if i % 2 == 0 else wgi_col2
                    with target_col:
                        if col in latest_wgi.index and pd.notna(latest_wgi[col]):
                            val = latest_wgi[col]
                            st.metric(name, f"{val:.0f}/100")
                        else:
                            st.metric(name, "--")
            else:
                st.caption("No WGI data for this country.")
        else:
            st.caption("WGI data not loaded.")

    st.markdown("---")
    
    # 4. PEER COMPARISON (moved from separate page)
    st.markdown("### Peer Countries")
    
    # Note: find_peers expects (target_country, scores_df, n_peers)
    peers_df = find_peers(selected_country_code, scores_df, n_peers=6)
    nearest_peer_codes = (
        peers_df['country_code'].tolist()
        if peers_df is not None and len(peers_df) > 0
        else []
    )
    peer_options = [
        code for code in available_country_codes
        if code != selected_country_code
    ]
    custom_peer_codes = st.multiselect(
        "Peer set",
        options=peer_options,
        default=nearest_peer_codes[:4],
        format_func=format_country_option,
        key="custom_peer_codes",
        help=(
            "Defaults to nearest-neighbor peers from the model feature space. "
            "Edit this list to compare with a custom peer group."
        ),
    )
    peer_codes = custom_peer_codes or nearest_peer_codes[:4]
    
    if peer_codes:
        # Comparison table with key proximity indicators
        comparison_cols = ['country_name', 'risk_score', 'economic_pillar', 'industry_pillar', 'data_coverage']
        display_names = {
            'country_name': 'Country',
            'risk_score': 'Risk Score',
            'economic_pillar': 'Econ Pillar',
            'industry_pillar': 'Industry Pillar',
            'data_coverage': 'Coverage'
        }
        
        # Add selected country for comparison
        selected_row = country_score_row[comparison_cols].to_frame().T
        peer_rows = scores_df[scores_df['country_code'].isin(peer_codes)].copy()
        peer_rows['_peer_order'] = pd.Categorical(
            peer_rows['country_code'],
            categories=peer_codes,
            ordered=True,
        )
        peer_rows = peer_rows.sort_values('_peer_order')
        peers_comparison = pd.concat([selected_row, peer_rows[comparison_cols]], ignore_index=True)
        peers_comparison = peers_comparison.rename(columns=display_names)
        peers_comparison.insert(
            0,
            'Role',
            ['Selected'] + [
                'Nearest' if code in nearest_peer_codes else 'Custom'
                for code in peer_rows['country_code'].tolist()
            ],
        )
        
        # Format
        peers_comparison['Risk Score'] = peers_comparison['Risk Score'].apply(lambda x: f"{x:.1f}")
        peers_comparison['Econ Pillar'] = peers_comparison['Econ Pillar'].apply(lambda x: f"{x:.1f}")
        peers_comparison['Industry Pillar'] = peers_comparison['Industry Pillar'].apply(lambda x: f"{x:.1f}")
        peers_comparison['Coverage'] = peers_comparison['Coverage'].apply(lambda x: f"{x:.0%}")
        
        st.dataframe(peers_comparison, use_container_width=True, hide_index=True)
        
        st.caption("Nearest peers are selected from similar economic and industry risk profiles; the peer set can be edited above.")
    else:
        st.caption("Unable to find peer countries.")

# ==============================================================================
# TAB: Data Explorer
# ==============================================================================
with tab_explorer:
    explorer_col1, explorer_col2 = st.columns([2, 3])
    with explorer_col1:
        explorer_focus_country = st.selectbox(
            "Explorer focus country",
            options=available_country_codes,
            format_func=format_country_option,
            key="explorer_focus_country",
            help=(
                "Used for single-country source tabs and to seed comparison "
                "country defaults."
            ),
        )

    explorer_peers_df = find_peers(explorer_focus_country, scores_df, n_peers=4)
    explorer_nearest_peer_codes = (
        explorer_peers_df['country_code'].tolist()
        if explorer_peers_df is not None and len(explorer_peers_df) > 0
        else []
    )
    explorer_default_peers = explorer_nearest_peer_codes[:4]

    with explorer_col2:
        if explorer_default_peers:
            st.caption(
                "Default peers: "
                + ", ".join(format_country_option(code) for code in explorer_default_peers)
            )
        else:
            st.caption("No nearest-neighbor peers are available for this country.")

    tool_tab_compare, tool_tab_calc = st.tabs([
        "Compare indicators",
        "Calculated series",
    ])
    with tool_tab_compare:
        render_indicator_comparison(
            scores=scores_df,
            selected_country=explorer_focus_country,
            default_peer_codes=explorer_default_peers,
            country_formatter=format_country_option,
            wgi_panel=wgi_data,
        )
    with tool_tab_calc:
        render_calculated_series_builder(
            scores=scores_df,
            selected_country=explorer_focus_country,
            default_peer_codes=explorer_default_peers,
            country_formatter=format_country_option,
            wgi_panel=wgi_data,
        )

    st.markdown("#### Single-country source tabs")
    load_history = st.checkbox(
        f"Load single-country historical data for {explorer_focus_country}",
        value=False,
        help=(
            "Loads WEO, FSI, MFS, and WGI history for the explorer focus country only. "
            "This keeps hosted startup within Streamlit resource limits."
        ),
    )
    if not load_history:
        st.info(
            "Historical source data is loaded on demand. Enable the option "
            "above to inspect WEO, FSI, MFS, and WGI histories for the "
            "explorer focus country."
        )

    # Tabs for each dataset
    de_tab_weo, de_tab_fsi, de_tab_mfs, de_tab_wgi = st.tabs(["Economic (WEO)", "Banking (FSI)", "Monetary (MFS)", "Governance (WGI)"])
    
    with de_tab_weo:
        weo_data = load_country_history(explorer_focus_country, 'WEO') if load_history else pd.DataFrame()
        if weo_data is not None and len(weo_data) > 0:
            n_indicators = weo_data['indicator_code'].nunique() if 'indicator_code' in weo_data.columns else 0
            st.caption(f"{n_indicators} economic indicators available for {explorer_focus_country}")
            try:
                render_time_series_deep_dive(weo_data, "WEO", explorer_focus_country)
            except Exception as e:
                st.error(f"Chart error: {e}")
        else:
            if load_history:
                st.info("No WEO data available for this country.")
            else:
                st.caption("Enable selected-country historical data above to load WEO history.")

    
    with de_tab_fsi:
        st.markdown("#### Financial Soundness Indicators")
        
        # Sub-tabs for FSIC and FSIBSIS
        fsi_tab1, fsi_tab2 = st.tabs(["Core FSI (FSIC)", "Balance Sheet (FSIBSIS)"])
        
        with fsi_tab1:
            # Load FSIC Data (Core FSI) - show ALL indicators with exact names
            fsic_data = load_country_history(explorer_focus_country, 'FSIC') if load_history else pd.DataFrame()
            
            if fsic_data is not None and len(fsic_data) > 0:
                n_indicators = fsic_data['indicator_name'].nunique()
                st.caption(f"{n_indicators} indicators available for {explorer_focus_country}")
                render_time_series_deep_dive(fsic_data, "FSIC", explorer_focus_country)
            else:
                if load_history:
                    st.info("No FSIC data available for this country.")
                else:
                    st.caption("Enable selected-country historical data above to load FSIC history.")
        
        with fsi_tab2:
            load_fsibsis = st.checkbox(
                "Load balance-sheet history",
                value=False,
                help="Loads the larger FSIBSIS dataset only when needed.",
            )
            # Load FSIBSIS Data
            try:
                if load_fsibsis:
                    fsibsis_long = load_fsibsis_country_history(explorer_focus_country)
                else:
                    fsibsis_long = pd.DataFrame()
                
                if fsibsis_long is not None and len(fsibsis_long) > 0:
                    n_indicators = fsibsis_long['indicator_name'].nunique()
                    st.caption(f"{n_indicators} balance-sheet indicators available")
                    render_time_series_deep_dive(fsibsis_long, "FSIBSIS", explorer_focus_country)
                else:
                    if load_fsibsis:
                        st.info("No FSIBSIS data available for this country.")
                    else:
                        st.caption(
                            "Balance-sheet history is loaded on demand to "
                            "reduce startup time."
                        )
            except Exception as e:
                st.error(f"Error loading FSIBSIS: {e}")
    
    with de_tab_mfs:
        st.markdown("#### Monetary & Financial Statistics")
        mfs_data = load_country_history(explorer_focus_country, 'MFS') if load_history else pd.DataFrame()
        if mfs_data is not None and len(mfs_data) > 0:
            n_indicators = mfs_data['indicator_code'].nunique() if 'indicator_code' in mfs_data.columns else 0
            st.caption(f"{n_indicators} monetary indicators available for {explorer_focus_country}")
            render_time_series_deep_dive(mfs_data, "MFS", explorer_focus_country)
        else:
            if load_history:
                st.info("No MFS data available for this country.")
            else:
                st.caption("Enable selected-country historical data above to load MFS history.")
    
    with de_tab_wgi:
        if wgi_data is not None and len(wgi_data) > 0:
            country_wgi = wgi_data[wgi_data['country_code'] == explorer_focus_country]
            if len(country_wgi) > 0:
                # WGI data has columns: country_code, year, voice_accountability, political_stability, etc.
                # Melt to long format for plotting
                governance_cols = ['voice_accountability', 'political_stability', 'govt_effectiveness', 
                                   'regulatory_quality', 'rule_of_law', 'control_corruption']
                available_cols = [c for c in governance_cols if c in country_wgi.columns]
                
                if available_cols:
                    import plotly.express as px
                    melted = country_wgi.melt(
                        id_vars=['country_code', 'year'], 
                        value_vars=available_cols,
                        var_name='Indicator', 
                        value_name='Score'
                    )
                    fig = px.line(
                        melted, 
                        x='year', 
                        y='Score', 
                        color='Indicator',
                        title='Governance Indicators Over Time (0-100 scale)'
                    )
                    fig.update_layout(
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                else:
                    st.info("No governance score columns found.")
            else:
                st.info("No WGI data for this country.")
        else:
            st.info("WGI data not loaded.")


# ==============================================================================
# TAB: Methodology
# ==============================================================================
with tab_methodology:
    render_current_methodology(
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

