import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

from src.data_loader import IMFDataLoader, FSIBSISLoader, WGILoader
from train_model import BankingRiskModel
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
try:
    from streamlit_mermaid import st_mermaid
    HAS_STREAMLIT_MERMAID = True
except ImportError:
    HAS_STREAMLIT_MERMAID = False
    import streamlit.components.v1 as components

def render_mermaid(code: str, height: int = 500) -> None:
    """Render a mermaid diagram using streamlit-mermaid package.
    
    Falls back to components.html if package not available (local dev).
    """
    if HAS_STREAMLIT_MERMAID:
        # Use the proper Streamlit component (works on Cloud)
        st_mermaid(code, height=height)
    else:
        # Fallback for local development if package not installed
        import streamlit.components.v1 as components
        components.html(
            f'''
            <div class="mermaid" style="width: 100%; height: 100%;">
            {code}
            </div>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>
                mermaid.initialize({{
                    startOnLoad: true,
                    securityLevel: 'loose',
                    theme: 'default',
                }});
            </script>
            ''',
            height=height,
            scrolling=True,
        )

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
    """Load model and all datasets."""
    # timestamp: force_reload_2026_01_12_v2
    try:
        model = BankingRiskModel.load()
        scores_df = model.get_all_scores()
        model_features = model.feature_values  # Get stored feature values for comparison
        pca_info = model.pca_info if hasattr(model, 'pca_info') else {}
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None, None, None

    loader = IMFDataLoader()
    try:
        loader.load_from_cache() 
    except:
        st.warning("Cache not found, please run data pipeline.")

    # Load FSIBSIS data
    try:
        fsibsis_loader = FSIBSISLoader()
        fsibsis_data = fsibsis_loader.load()
    except Exception as e:
        fsibsis_data = None

    try:
        wgi_loader = WGILoader()
        wgi_data = wgi_loader.load()
    except Exception as e:
        wgi_data = None
        
    return scores_df, loader, wgi_data, model_features, pca_info, fsibsis_data

scores_df, loader, wgi_data, model_features, pca_info, fsibsis_data = load_all_data()

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
header_col1, header_col2, header_col3 = st.columns([2, 3, 1])

with header_col1:
    st.markdown("### Banking System Stability Copilot")
    training_date = pca_info.get('training_date', 'Unknown') if pca_info else 'Unknown'
    st.caption(f"v2.0 | Risk Model with PCA-based pillars")

with header_col2:
    available_countries = scores_df.sort_values('country_name')[['country_code', 'country_name']].drop_duplicates()
    default_idx = 0
    if 'USA' in available_countries['country_code'].values:
        default_idx = list(available_countries['country_code'].values).index('USA')
        
    selected_country_code = st.selectbox(
        "Select Country",
        options=available_countries['country_code'].tolist(),
        format_func=lambda x: available_countries[available_countries['country_code'] == x]['country_name'].values[0],
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
    peers_df = find_peers(selected_country_code, scores_df, n_peers=4)
    
    if peers_df is not None and len(peers_df) > 0:
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
        peers_comparison = pd.concat([selected_row, peers_df[comparison_cols]], ignore_index=True)
        peers_comparison = peers_comparison.rename(columns=display_names)
        
        # Format
        peers_comparison['Risk Score'] = peers_comparison['Risk Score'].apply(lambda x: f"{x:.1f}")
        peers_comparison['Econ Pillar'] = peers_comparison['Econ Pillar'].apply(lambda x: f"{x:.1f}")
        peers_comparison['Industry Pillar'] = peers_comparison['Industry Pillar'].apply(lambda x: f"{x:.1f}")
        peers_comparison['Coverage'] = peers_comparison['Coverage'].apply(lambda x: f"{x:.0%}")
        
        st.dataframe(peers_comparison, use_container_width=True, hide_index=True)
        
        st.caption("Peers selected based on similar economic and industry risk profiles (Euclidean distance).")
    else:
        st.caption("Unable to find peer countries.")

# ==============================================================================
# TAB: Data Explorer
# ==============================================================================
with tab_explorer:
    st.markdown("### Historical Data Explorer")
    
    # Tabs for each dataset
    de_tab_weo, de_tab_fsi, de_tab_mfs, de_tab_wgi = st.tabs(["Economic (WEO)", "Banking (FSI)", "Monetary (MFS)", "Governance (WGI)"])
    
    with de_tab_weo:
        weo_data = loader.get_country_data(selected_country_code, 'WEO')
        if weo_data is not None and len(weo_data) > 0:
            try:
                render_time_series_deep_dive(weo_data, "WEO", selected_country_code)
            except Exception as e:
                st.error(f"Chart error: {e}")
        else:
            st.info("No WEO data available for this country.")

    
    with de_tab_fsi:
        st.markdown("#### Financial Soundness Indicators")
        
        # Load FSIC Data (Core FSI)
        fsic_data = loader.get_country_data(selected_country_code, 'FSIC')
        
        # Load FSIBSIS Data (BIS format) - need to get loader instance
        try:
            from src.data_loader_fsibsis import FSIBSISLoader
            fsibsis_loader = FSIBSISLoader()
            fsibsis_country_data = fsibsis_loader.get_country_data(selected_country_code)
        except Exception as e:
            fsibsis_country_data = pd.DataFrame()
        
        # Combine both datasets
        combined_data = pd.DataFrame()
        if fsic_data is not None and len(fsic_data) > 0:
            combined_data = fsic_data.copy()
        
        if len(fsibsis_country_data) > 0:
            # Ensure column alignment before concatenation
            if len(combined_data) > 0:
                # Get common year columns
                fsic_cols = set(combined_data.columns)
                fsibsis_cols = set(fsibsis_country_data.columns)
                common_cols = ['INDICATOR'] + sorted([c for c in (fsic_cols & fsibsis_cols) if c.startswith('20')])
                
                # Align and concatenate
                if len(common_cols) > 1:  # More than just INDICATOR
                    fsic_aligned = combined_data[[c for c in common_cols if c in combined_data.columns]]
                    fsibsis_aligned = fsibsis_country_data[[c for c in common_cols if c in fsibsis_country_data.columns]]
                    combined_data = pd.concat([fsic_aligned, fsibsis_aligned], ignore_index=True)
            else:
                combined_data = fsibsis_country_data
        
        # Render combined data
        if len(combined_data) > 0:
            render_time_series_deep_dive(combined_data, "FSI", selected_country_code)
        else:
            st.info("No Financial Soundness Indicators available for this country.")
    
    with de_tab_mfs:
        st.markdown("#### Monetary & Financial Statistics")
        mfs_data = loader.get_country_data(selected_country_code, 'MFS')
        if mfs_data is not None and len(mfs_data) > 0:
            render_time_series_deep_dive(mfs_data, "MFS", selected_country_code)
        else:
            st.info("No MFS data available for this country.")
    
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
                    fig.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)
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
    st.markdown("## Methodology")
    
    with open('README.md', 'r', encoding='utf-8') as f:
        readme_content = f.read()
    
    # Extract and render mermaid diagram separately
    mermaid_code = extract_mermaid_code(readme_content)
    
    # Split content around the diagram
    parts = readme_content.split('```mermaid')
    
    # Render first part (text before diagram)
    render_markdown_with_images(parts[0])
    
    # Render diagram
    if mermaid_code:
        st.markdown("### Process Architecture")
        render_mermaid(mermaid_code, height=600)
    
    # Render rest (if any, skipping the code block itself)
    if len(parts) > 1:
        # Find end of code block
        after_diagram = parts[1].split('```', 1)[-1]
        render_markdown_with_images(after_diagram)

