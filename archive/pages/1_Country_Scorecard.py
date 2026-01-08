"""
Country Scorecard Page
BICRA-inspired comprehensive country assessment with Premium UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import IMFDataLoader
from src.risk_scorer import compute_composite_bicra_score
from src.insight_generator import InsightGenerator
from src.trend_analyzer import TrendAnalyzer
from src.explainability import RiskExplainer
from src.styles import MAIN_STYLES, COLORS, get_risk_color
from src.ui_components import render_header, render_metric_card, render_risk_gauge, render_score_breakdown
from src.report_generator import ReportGenerator

st.set_page_config(
    page_title="Country Scorecard",
    page_icon="📋",
    layout="wide"
)

# Inject Global Styles
st.markdown(MAIN_STYLES, unsafe_allow_html=True)


@st.cache_resource
def load_data():
    loader = IMFDataLoader()
    if not loader.load_from_cache():
        loader.load_all_datasets()
    return loader


def create_radar_chart(scores: dict):
    """Create premium radar chart."""
    categories = list(scores.keys())
    values = list(scores.values())
    
    # Close the polygon
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor=f"{COLORS['accent_blue']}40", # Transparent blue
        line=dict(color=COLORS['accent_blue'], width=2),
        name='Current Score'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color=COLORS['muted'])),
            angularaxis=dict(tickfont=dict(color=COLORS['text'], size=12)),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=20, b=20),
        height=300
    )
    return fig


def render_explainability(country_code, country_name, fsic_data, weo_data):
    """Render the Explainability section."""
    st.markdown("### 🔍 Why this Score?")
    
    explainer = RiskExplainer()
    # Combine data for explanation
    combined_df = pd.concat([fsic_data, weo_data])
    explanation = explainer.explain_score(combined_df, country_code)
    
    if not explanation:
        st.info("Insufficient data for detailed explanation.")
        return

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Positive Contributors")
        if explanation['positive_factors']:
            for ref in explanation['positive_factors']:
                st.markdown(
                    f"""
                    <div style="background-color: {COLORS['risk_very_low']}20; padding: 10px; border-radius: 4px; margin-bottom: 8px; border-left: 3px solid {COLORS['risk_very_low']};">
                        <div style="font-weight: bold; color: {COLORS['text']}">{ref['name']}</div>
                        <div style="font-size: 0.9rem; color: {COLORS['muted']}">Value: {ref['value']:.1f}% (+{ref['contribution']:.1f} pts)</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.markdown(f"<span style='color:{COLORS['muted']}'>No major positive factors identified.</span>", unsafe_allow_html=True)
            
    with col2:
        st.markdown("#### ⚠️ Negative Contributors")
        if explanation['negative_factors']:
            for ref in explanation['negative_factors']:
                st.markdown(
                    f"""
                    <div style="background-color: {COLORS['risk_high']}20; padding: 10px; border-radius: 4px; margin-bottom: 8px; border-left: 3px solid {COLORS['risk_high']};">
                        <div style="font-weight: bold; color: {COLORS['text']}">{ref['name']}</div>
                        <div style="font-size: 0.9rem; color: {COLORS['muted']}">Value: {ref['value']:.1f}% ({ref['contribution']:.1f} pts)</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
             st.markdown(f"<span style='color:{COLORS['muted']}'>No major negative factors identified.</span>", unsafe_allow_html=True)


def main():
    loader = load_data()
    
    # Sidebar
    countries = loader.get_countries()
    if len(countries) == 0:
        st.warning("No data loaded.")
        return
        
    unique = countries.groupby('country_code').first().reset_index().sort_values('country_name')
    country_map = dict(zip(unique['country_name'] + ' (' + unique['country_code'] + ')', unique['country_code']))
    
    st.sidebar.title("Scorecard")
    selected_label = st.sidebar.selectbox("Select Country", list(country_map.keys()))
    country_code = country_map[selected_label]
    country_name = selected_label.split(' (')[0]
    
    # Data Loading
    fsic_data = loader._data_cache.get('FSIC', pd.DataFrame())
    weo_data = loader._data_cache.get('WEO', pd.DataFrame())
    
    # Main Content
    render_header(f"{country_name}", f"Deep Dive Analysis ({country_code})")
    
    # Composite Score
    try:
        composite = compute_composite_bicra_score(fsic_data, weo_data)
        if country_code in composite['country_code'].values:
            row = composite[composite['country_code'] == country_code].iloc[0]
            score = row['composite_score']
            econ_score = row['economic_resilience_score']
            ind_score = row['industry_risk_score']
            tier = row['bicra_tier']
            scores_dict = {
                'composite': score,
                'economic': econ_score,
                'industry': ind_score,
                'tier': tier
            }
        else:
            score, econ_score, ind_score = 50, 50, 50
            scores_dict = {'composite': 50, 'economic': 50, 'industry': 50, 'tier': 'N/A'}
    except:
        score, econ_score, ind_score = 50, 50, 50
        scores_dict = {'composite': 50, 'economic': 50, 'industry': 50, 'tier': 'N/A'}
        
    # Top Section: Gauge + Radar
    col_gauge, col_radar = st.columns([1, 1.5])
    
    with col_gauge:
        render_risk_gauge(score, "Composite Risk Score")
        st.markdown("<br>", unsafe_allow_html=True)
        render_score_breakdown(econ_score, ind_score)
        
    with col_radar:
        st.markdown("### Risk Profile")
        radar_data = {
            'Economic': econ_score,
            'Industry': ind_score,
            'Capital': min(100, ind_score * 1.1), # Placeholder logic
            'Asset Quality': min(100, ind_score * 0.9),
            'Liquidity': min(100, ind_score * 1.05),
            'Fiscal': min(100, econ_score * 0.9)
        }
        fig = create_radar_chart(radar_data)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # Explainability Section
    render_explainability(country_code, country_name, fsic_data, weo_data)
    
    st.markdown("---")
    
    # Key Indicators Grid
    st.markdown("### 📊 Key Indicators")
    
    def get_val(df, code):
        if df.empty: return None
        matches = df[df['indicator_code'].str.contains(code, case=False, na=False)]
        if matches.empty: return None
        return matches.sort_values('period').iloc[-1]['value']
        
    curr_fsic = fsic_data[fsic_data['country_code'] == country_code]
    curr_weo = weo_data[weo_data['country_code'] == country_code] if len(weo_data) > 0 else pd.DataFrame()
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_metric_card("Capital (RCAR)", get_val(curr_fsic, 'RCAR') or "N/A", delta="Target >10.5%")
    with kpi2:
        render_metric_card("NPL Ratio", get_val(curr_fsic, 'NPLGL') or "N/A", delta="Target <5%", delta_color="inverse")
    with kpi3:
        render_metric_card("Liquidity (LASTL)", get_val(curr_fsic, 'LASTL') or "N/A", delta="Target >20%")
    with kpi4:
        render_metric_card("GDP Growth", get_val(curr_weo, 'NGDP_RPCH') or "N/A", delta="Target >2%")
        
    st.markdown("---")
    
    # Insights Generator (Text)
    st.markdown("### 📝 Analyst Insights")
    gen = InsightGenerator()
    insights = gen.generate_country_summary(country_code, country_name, fsic_data, weo_data)
    
    for s in insights['key_strengths']:
         st.markdown(f"✅ {s}")
    for r in insights['key_risks']:
         st.markdown(f"⚠️ {r}")
         
    # PDF Export
    st.markdown("---")
    st.markdown("### 📥 Reports")
    
    col_pdf, col_spacer = st.columns([1, 4])
    
    with col_pdf:
        if st.button("Generate PDF Report", type="primary"):
            try:
                reporter = ReportGenerator()
                pdf_bytes = reporter.generate_report(country_name, country_code, scores_dict, insights)
                
                st.download_button(
                    label="📄 Download PDF Briefing",
                    data=pdf_bytes,
                    file_name=f"{country_code}_Briefing_Pack.pdf",
                    mime="application/pdf"
                )
                st.success("Ready for download!")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
