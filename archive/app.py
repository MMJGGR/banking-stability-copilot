"""
Banking System Stability Copilot
Main Streamlit Application

A credit analyst tool for analyzing banking sector health using IMF datasets.
"""

import streamlit as st
import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import IMFDataLoader
from src.risk_scorer import compute_composite_bicra_score
from src.styles import MAIN_STYLES
from src.ui_components import render_header, render_metric_card, render_risk_gauge

# Page config
st.set_page_config(
    page_title="Banking System Stability Copilot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Global Styles
st.markdown(MAIN_STYLES, unsafe_allow_html=True)


@st.cache_resource
def load_data():
    """Load and cache IMF datasets."""
    loader = IMFDataLoader()
    # Try loading from cache first
    if loader.load_from_cache():
        return loader
    # Load from CSV files
    try:
        datasets = loader.load_all_datasets()
        if datasets:
            loader.save_cache()
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
    return loader


@st.cache_data
def get_country_list(_loader):
    """Get list of available countries."""
    countries = _loader.get_countries()
    if len(countries) > 0:
        unique = countries.groupby('country_code').agg({
            'country_name': 'first',
            'n_indicators': 'sum',
            'n_observations': 'sum'
        }).reset_index()
        return unique.sort_values('country_name')
    return pd.DataFrame()


def render_sidebar(loader):
    """Render sidebar with country selection and filters."""
    st.sidebar.title("Banking Copilot")
    
    # Data status
    datasets_loaded = sum([
        1 for d in ['FSIC', 'WEO', 'MFS'] 
        if d in loader._data_cache and len(loader._data_cache[d]) > 0
    ])
    
    status_color = "green" if datasets_loaded == 3 else "orange"
    st.sidebar.markdown(f"**Data Status**: :{status_color}[{datasets_loaded}/3 Datasets Loaded]")
    
    st.sidebar.markdown("---")
    
    # Country selection
    countries = get_country_list(loader)
    
    if len(countries) > 0:
        country_options = {
            f"{row['country_name']} ({row['country_code']})": row['country_code']
            for _, row in countries.iterrows()
            if row['country_name'] and row['country_code']
        }
        
        selected = st.sidebar.selectbox(
            "Select Country",
            options=list(country_options.keys()),
            index=0 if country_options else None
        )
        
        selected_country = country_options.get(selected, None)
    else:
        selected_country = None
        st.sidebar.warning("No countries available. Please load datasets.")
    
    st.sidebar.markdown("---")
    
    # Navigation Help
    st.sidebar.info(
        "**Navigation**\n\n"
        "Use the sidebar pages to navigate between:\n"
        "- **Country Scorecard**: Detailed country analysis\n"
        "- **Comparisons**: Peer benchmarking\n"
        "- **ML Risk Analysis**: Advanced risk clustering"
    )
    
    if st.sidebar.button("Refresh Data", type="secondary"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    
    return selected_country


def render_landing_kpis(loader, country_code):
    """Render high-level KPIs for the landing page."""
    fsic_data = loader._data_cache.get('FSIC', pd.DataFrame())
    weo_data = loader._data_cache.get('WEO', pd.DataFrame())
    
    if len(fsic_data) == 0:
        return
    
    country_fsic = fsic_data[fsic_data['country_code'] == country_code]
    country_weo = weo_data[weo_data['country_code'] == country_code] if len(weo_data) > 0 else pd.DataFrame()
    
    def get_latest(df, pattern):
        if df.empty: return None
        matching = df[df['indicator_code'].str.contains(pattern, case=False, na=False)]
        if matching.empty: return None
        return matching.sort_values('period').iloc[-1]['value']

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        val = get_latest(country_fsic, 'RCAR|T1RWA|CAR')
        # Simple rule: >10.5 is good (delta normal)
        delta_color = "normal" if val and val >= 10.5 else "inverse"
        render_metric_card("Capital Adequacy", val if val else "N/A", 
                           delta=f"{val-10.5:+.1f}% vs Min" if val else None,
                           delta_color=delta_color)

    with col2:
        val = get_latest(country_fsic, 'NPLGL')
        # Inverse rule: <5 is good (delta normal for inverse metric?)
        # For NPL, "normal" usually means green. If NPL is low, we want green.
        # If NPL is < 5, it's good. 
        delta_color = "normal" if val and val <= 5.0 else "inverse" 
        # Note: UI component logic handles color mapping.
        render_metric_card("NPL Ratio", val if val else "N/A", 
                           delta=f"{val:.1f}%" if val else None,
                           delta_color=delta_color)
                           
    with col3:
        val = get_latest(country_fsic, 'ROE')
        delta_color = "normal" if val and val >= 10.0 else "inverse"
        render_metric_card("Return on Equity", val if val else "N/A", 
                           delta=f"{val:.1f}%" if val else None,
                           delta_color=delta_color)

    with col4:
        val = get_latest(country_weo, 'NGDP_RPCH')
        delta_color = "normal" if val and val >= 0 else "inverse"
        render_metric_card("GDP Growth", val if val else "N/A", 
                           delta=f"{val:.1f}%" if val else None,
                           delta_color=delta_color)
    
    return fsic_data, weo_data


def main():
    loader = load_data()
    selected_country = render_sidebar(loader)
    
    if selected_country:
        # Resolve country name
        countries = get_country_list(loader)
        c_row = countries[countries['country_code'] == selected_country]
        country_name = c_row['country_name'].iloc[0] if len(c_row) > 0 else selected_country
        
        render_header(f"{country_name}", "Executive Summary & Key Metrics")
        
        # 1. Top Level KPIs
        fsic_data, weo_data = render_landing_kpis(loader, selected_country)
        
        st.markdown("---")
        
        # 2. Risk Score Preview (Centerpiece)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            try:
                composite = compute_composite_bicra_score(fsic_data, weo_data)
                if selected_country in composite['country_code'].values:
                    score = composite[composite['country_code'] == selected_country].iloc[0]['composite_score']
                    render_risk_gauge(score)
                else:
                    st.info("Insufficient data for risk score")
            except Exception as e:
                st.error(f"Risk score error: {e}")
                
        with col2:
            st.markdown("### 💡 Analyst Notes")
            st.info(
                f"**{country_name}** represents a component of the analyzed portfolio. "
                "Use the **Country Scorecard** page for a detailed breakdown of the risk factors "
                "driving this score, including Explainability analysis."
            )
            
            with st.expander("Available Data Points"):
                n_ind = c_row['n_indicators'].iloc[0] if len(c_row) > 0 else 0
                n_obs = c_row['n_observations'].iloc[0] if len(c_row) > 0 else 0
                st.write(f"Indicators Tracked: {n_ind}")
                st.write(f"Total Observations: {n_obs}")

    else:
        render_header("Banking System Stability Copilot", "No country selected")
        st.info("👈 Please select a country from the sidebar to begin.")


if __name__ == "__main__":
    main()
