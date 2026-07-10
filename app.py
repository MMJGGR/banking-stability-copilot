import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os

from src.data_loader import IMFDataLoader, FSIBSISLoader, WGILoader
from src.country_names import fill_missing_country_names
from src.model_store import load_data_manifest, load_model_artifact
from src.dashboard.styles import STYLES, score_to_tier
from src.dashboard.components import (
    render_summary_card, 
    render_data_snapshot,
    render_time_series_deep_dive,
    WEO_INDICATORS,
    FSIC_NAME_PATTERNS
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
                st.info(f"📷 *{alt_text}* (Image will appear after model training)")
                
            i += 3 # skip (text, alt, path)
        else:
            i += 1


# Page Config
st.set_page_config(
    page_title="Banking System Stability Copilot",
    page_icon="B",
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
    """Load model artifacts and lightweight reference data."""
    # timestamp: force_reload_2026_01_12_v2
    try:
        model = load_model_artifact()
        scores_df = model['country_scores'].copy()
        # Artifacts built from official SDMX feeds may lack display names.
        fill_missing_country_names(scores_df, fallback_to_code=True)
        model_features = model.get('feature_values')
        pca_info = dict(model.get('pca_info', {}))
        pca_info.setdefault('training_date', model['training_date'])
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None

    loader = IMFDataLoader()

    try:
        wgi_loader = WGILoader()
        wgi_data = wgi_loader.load()
    except Exception as e:
        wgi_data = None
        
    return scores_df, loader, wgi_data, model_features, pca_info


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
            ["Economic (WEO)", "Banking (FSIC)", "Monetary (MFS)", "Governance (WGI)"],
            key="compare_source",
        )
    with control_col2:
        compare_countries = st.multiselect(
            "Countries",
            options=available_codes,
            default=default_countries,
            format_func=country_formatter,
            key="compare_countries",
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
        "Banking (FSIC)": "FSIC",
        "Monetary (MFS)": "MFS",
        "Governance (WGI)": "WGI",
    }
    dataset = source_to_dataset[source_choice]

    if dataset == "WGI":
        if wgi_panel is None or len(wgi_panel) == 0:
            st.info("WGI data is not available.")
            return
        governance_cols = [
            'voice_accountability', 'political_stability', 'govt_effectiveness',
            'regulatory_quality', 'rule_of_law', 'control_corruption'
        ]
        available_cols = [c for c in governance_cols if c in wgi_panel.columns]
        source_df = wgi_panel[wgi_panel['country_code'].isin(compare_countries)].copy()
        if len(source_df) == 0 or not available_cols:
            st.info("No WGI data is available for the selected countries.")
            return
        source_df = source_df.melt(
            id_vars=['country_code', 'year'],
            value_vars=available_cols,
            var_name='indicator_code',
            value_name='value',
        )
        source_df['indicator_name'] = source_df['indicator_code'].str.replace('_', ' ').str.title()
        source_df['period'] = pd.to_datetime(source_df['year'].astype(str) + '-12-31')
        source_df['frequency'] = 'A'
    else:
        with st.spinner(f"Loading {dataset} history for selected countries..."):
            source_df = load_multi_country_history(tuple(compare_countries), dataset)
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
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

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

The final score combines two PCA pillars with a supervised systemic-crisis
classifier. The current policy weighting is 90% pillar score and 10%
crisis-probability adjustment. The crisis classifier is trained on annual
historical epochs and produces a forward-looking three-year risk signal, not a
monthly or quarterly crisis probability.
"""
    )

    st.markdown("### Active Sources")
    source_rows = []
    for source_name, details in sorted(manifest.get('sources', {}).items()):
        source_rows.append({
            "Source": source_name,
            "Rows": details.get("rows"),
            "Countries": details.get("countries"),
            "Latest Observation": details.get("latest_observation"),
            "Indicators": details.get("indicators"),
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

    with st.expander("Model Card"):
        model_card = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "MODEL_CARD.md")
        if os.path.exists(model_card):
            with open(model_card, "r", encoding="utf-8") as source:
                st.markdown(source.read())
        else:
            st.info("Model card not found.")

    with st.expander("Data Card"):
        data_card = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "DATA_CARD.md")
        if os.path.exists(data_card):
            with open(data_card, "r", encoding="utf-8") as source:
                st.markdown(source.read())
        else:
            st.info("Data card not found.")


scores_df, loader, wgi_data, model_features, pca_info = load_all_data()
data_manifest = load_data_manifest()

if scores_df is None:
    st.error("Application cannot start without model data.")
    st.stop()

# Prepare data for Global View (Merge GDP for weighting)
if scores_df is not None and model_features is not None:
    if 'nominal_gdp' not in scores_df.columns and 'nominal_gdp' in model_features.columns:
        scores_df = scores_df.merge(model_features[['country_code', 'nominal_gdp']], on='country_code', how='left')

# ==============================================================================
# HEADER: Country Selector + Model Info
# ==============================================================================
available_countries = scores_df.sort_values('country_name')[['country_code', 'country_name']].drop_duplicates()
available_country_codes = available_countries['country_code'].tolist()
country_name_lookup = dict(
    zip(available_countries['country_code'], available_countries['country_name'])
)


def format_country_option(country_code: str) -> str:
    name = country_name_lookup.get(country_code, country_code)
    return f"{name} ({country_code})"


header_col1, header_col2, header_col3 = st.columns([2, 3, 1])

with header_col1:
    st.markdown("### Banking System Stability Copilot")
    training_date = pca_info.get('training_date', 'Unknown') if pca_info else 'Unknown'
    snapshot_id = data_manifest.get('snapshot_id', 'unversioned')
    snapshot_status = data_manifest.get('snapshot_status', 'manifest unavailable')
    st.caption(
        f"v2.0 | Snapshot {snapshot_id} | {snapshot_status.replace('_', ' ')}"
    )

with header_col2:
    default_idx = 0
    if 'USA' in available_countries['country_code'].values:
        default_idx = list(available_countries['country_code'].values).index('USA')
        
    selected_country_code = st.selectbox(
        "Select Country",
        options=available_country_codes,
        format_func=format_country_option,
        index=default_idx,
        label_visibility="collapsed"
    )
    
with header_col3:
    pass  # Reserved for future actions

# Get selected country data
country_score_row = scores_df[scores_df['country_code'] == selected_country_code].iloc[0]
selected_country_name = country_score_row['country_name']

st.markdown("---")

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
    st.markdown("### Historical Data Explorer")
    load_history = st.checkbox(
        f"Load selected-country historical data for {selected_country_code}",
        value=False,
        help=(
            "Loads WEO, FSI, and MFS history for the selected country only. "
            "This keeps hosted startup within Streamlit resource limits."
        ),
    )
    if not load_history:
        st.info(
            "Historical source data is loaded on demand. Enable the option "
            "above to inspect WEO, FSI, MFS, and WGI histories for the "
            "selected country."
        )

    explorer_peers_df = find_peers(selected_country_code, scores_df, n_peers=4)
    explorer_nearest_peer_codes = (
        explorer_peers_df['country_code'].tolist()
        if explorer_peers_df is not None and len(explorer_peers_df) > 0
        else []
    )
    explorer_default_peers = st.session_state.get(
        "custom_peer_codes",
        explorer_nearest_peer_codes[:4],
    )
    with st.expander("Compare one indicator across countries", expanded=False):
        render_indicator_comparison(
            scores=scores_df,
            selected_country=selected_country_code,
            default_peer_codes=explorer_default_peers,
            country_formatter=format_country_option,
            wgi_panel=wgi_data,
        )
    
    # Tabs for each dataset
    de_tab_weo, de_tab_fsi, de_tab_mfs, de_tab_wgi = st.tabs(["Economic (WEO)", "Banking (FSI)", "Monetary (MFS)", "Governance (WGI)"])
    
    with de_tab_weo:
        weo_data = load_country_history(selected_country_code, 'WEO') if load_history else pd.DataFrame()
        if weo_data is not None and len(weo_data) > 0:
            n_indicators = weo_data['indicator_code'].nunique() if 'indicator_code' in weo_data.columns else 0
            st.caption(f"📊 {n_indicators} economic indicators available for {selected_country_code}")
            try:
                render_time_series_deep_dive(weo_data, "WEO", selected_country_code)
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
            fsic_data = load_country_history(selected_country_code, 'FSIC') if load_history else pd.DataFrame()
            
            if fsic_data is not None and len(fsic_data) > 0:
                n_indicators = fsic_data['indicator_name'].nunique()
                st.caption(f"📊 {n_indicators} indicators available for {selected_country_code}")
                render_time_series_deep_dive(fsic_data, "FSIC", selected_country_code)
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
                    from src.data_loader import FSIBSISLoader
                    fsibsis_loader = FSIBSISLoader()
                    fsibsis_loader.load()
                    fsibsis_wide = fsibsis_loader.get_country_data(
                        selected_country_code
                    )
                else:
                    fsibsis_wide = pd.DataFrame()
                
                if fsibsis_wide is not None and len(fsibsis_wide) > 0:
                    # Convert to long format
                    time_cols = [
                        c for c in fsibsis_wide.columns
                        if c != 'INDICATOR'
                    ]
                    fsibsis_long = fsibsis_wide.melt(
                        id_vars=['INDICATOR'],
                        value_vars=time_cols,
                        var_name='period_label',
                        value_name='value'
                    )
                    fsibsis_long = fsibsis_long.dropna(subset=['value'])
                    fsibsis_long['indicator_name'] = fsibsis_long['INDICATOR']
                    fsibsis_long['indicator_code'] = 'FSIBSIS'
                    from src.data_loader import parse_period_label
                    fsibsis_long['period'] = fsibsis_long['period_label'].map(parse_period_label)
                    fsibsis_long['country_code'] = selected_country_code

                    def _fsibsis_frequency(label) -> str:
                        text = str(label).upper()
                        if 'Q' in text:
                            return 'Q'
                        if 'M' in text:
                            return 'M'
                        return 'A'

                    fsibsis_long['frequency'] = fsibsis_long['period_label'].map(_fsibsis_frequency)
                    
                    n_indicators = fsibsis_long['indicator_name'].nunique()
                    st.caption(f"📊 {n_indicators} balance sheet indicators available")
                    render_time_series_deep_dive(fsibsis_long, "FSIBSIS", selected_country_code)
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
        mfs_data = load_country_history(selected_country_code, 'MFS') if load_history else pd.DataFrame()
        if mfs_data is not None and len(mfs_data) > 0:
            n_indicators = mfs_data['indicator_code'].nunique() if 'indicator_code' in mfs_data.columns else 0
            st.caption(f"📊 {n_indicators} monetary indicators available for {selected_country_code}")
            render_time_series_deep_dive(mfs_data, "MFS", selected_country_code)
        else:
            if load_history:
                st.info("No MFS data available for this country.")
            else:
                st.caption("Enable selected-country historical data above to load MFS history.")
    
    with de_tab_wgi:
        if wgi_data is not None and len(wgi_data) > 0:
            country_wgi = wgi_data[wgi_data['country_code'] == selected_country_code]
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

