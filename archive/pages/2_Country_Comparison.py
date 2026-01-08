"""
Country Comparison Page
Multi-country comparison with visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import IMFDataLoader
from src.risk_scorer import compute_composite_bicra_score

st.set_page_config(
    page_title="Country Comparison",
    page_icon="📊",
    layout="wide"
)

st.markdown("# 📊 Country Comparison")
st.markdown("*Compare banking sector metrics across countries*")


@st.cache_resource
def load_data():
    loader = IMFDataLoader()
    if not loader.load_from_cache():
        loader.load_all_datasets()
    return loader


def create_bar_comparison(data: pd.DataFrame, countries: list, 
                          indicator: str, title: str):
    """Create bar chart comparing countries."""
    
    filtered = data[
        (data['country_code'].isin(countries)) & 
        (data['indicator_code'].str.contains(indicator, case=False, na=False))
    ]
    
    if filtered.empty:
        return None
    
    # Get latest values
    latest = filtered.sort_values('period').groupby('country_code').last().reset_index()
    
    fig = px.bar(
        latest,
        x='country_code',
        y='value',
        color='value',
        color_continuous_scale=[
            [0, '#D32F2F'],
            [0.25, '#FF7043'],
            [0.5, '#FFC107'],
            [0.75, '#66BB6A'],
            [1, '#2E7D32']
        ],
        title=title,
        labels={'value': indicator, 'country_code': 'Country'}
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=350
    )
    
    return fig


def create_scatter_comparison(data: pd.DataFrame, countries: list,
                              x_indicator: str, y_indicator: str):
    """Create scatter plot comparing two indicators."""
    
    # Get latest values for both indicators
    x_data = data[
        (data['country_code'].isin(countries)) & 
        (data['indicator_code'].str.contains(x_indicator, case=False, na=False))
    ].sort_values('period').groupby('country_code').last()[['value']].rename(columns={'value': 'x'})
    
    y_data = data[
        (data['country_code'].isin(countries)) & 
        (data['indicator_code'].str.contains(y_indicator, case=False, na=False))
    ].sort_values('period').groupby('country_code').last()[['value']].rename(columns={'value': 'y'})
    
    merged = x_data.join(y_data, how='inner').reset_index()
    
    if merged.empty:
        return None
    
    fig = px.scatter(
        merged,
        x='x',
        y='y',
        text='country_code',
        size=[30] * len(merged),
        color='y',
        color_continuous_scale=[
            [0, '#D32F2F'],
            [0.5, '#FFC107'],
            [1, '#2E7D32']
        ]
    )
    
    fig.update_traces(textposition='top center', textfont_size=12)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title=x_indicator,
        yaxis_title=y_indicator,
        showlegend=False,
        height=400
    )
    
    return fig


def create_time_series_comparison(data: pd.DataFrame, countries: list,
                                  indicator: str, title: str):
    """Create time series chart comparing countries over time."""
    
    filtered = data[
        (data['country_code'].isin(countries)) & 
        (data['indicator_code'].str.contains(indicator, case=False, na=False))
    ]
    
    if filtered.empty:
        return None
    
    fig = px.line(
        filtered,
        x='period',
        y='value',
        color='country_code',
        markers=True,
        title=title,
        labels={'value': indicator, 'period': 'Period', 'country_code': 'Country'}
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def main():
    loader = load_data()
    
    # Get country list
    countries = loader.get_countries()
    if len(countries) == 0:
        st.warning("No data loaded.")
        return
    
    unique_countries = countries.groupby('country_code').agg({
        'country_name': 'first'
    }).reset_index().sort_values('country_name')
    
    country_options = {
        f"{row['country_name']} ({row['country_code']})": row['country_code']
        for _, row in unique_countries.iterrows()
        if row['country_name']
    }
    
    # Country selection
    st.sidebar.markdown("### 🌍 Select Countries")
    
    selected = st.sidebar.multiselect(
        "Countries to compare",
        options=list(country_options.keys()),
        max_selections=10,
        help="Select up to 10 countries to compare"
    )
    
    selected_codes = [country_options[s] for s in selected]
    
    if len(selected_codes) < 2:
        st.info("👈 Select at least 2 countries from the sidebar to compare.")
        return
    
    # Load data
    fsic_data = loader._data_cache.get('FSIC', pd.DataFrame())
    weo_data = loader._data_cache.get('WEO', pd.DataFrame())
    
    # Composite scores comparison
    st.markdown("### 🎯 Risk Score Comparison")
    
    try:
        composite = compute_composite_bicra_score(fsic_data, weo_data)
        comparison_data = composite[composite['country_code'].isin(selected_codes)]
        
        if len(comparison_data) > 0:
            fig = px.bar(
                comparison_data.sort_values('composite_score', ascending=True),
                x='composite_score',
                y='country_code',
                orientation='h',
                color='composite_score',
                color_continuous_scale=[
                    [0, '#D32F2F'],
                    [0.35, '#FF7043'],
                    [0.5, '#FFC107'],
                    [0.65, '#66BB6A'],
                    [1, '#2E7D32']
                ],
                text='bicra_tier'
            )
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Composite Score (0-100)",
                yaxis_title="",
                showlegend=False,
                height=max(250, len(selected_codes) * 40)
            )
            fig.update_traces(textposition='outside')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Score breakdown table
            st.markdown("#### Score Details")
            st.dataframe(
                comparison_data[['country_code', 'economic_resilience_score', 
                                'industry_risk_score', 'composite_score', 'bicra_tier']]
                .rename(columns={
                    'country_code': 'Country',
                    'economic_resilience_score': 'Economic',
                    'industry_risk_score': 'Industry',
                    'composite_score': 'Composite',
                    'bicra_tier': 'Tier'
                })
                .sort_values('Composite', ascending=False),
                use_container_width=True,
                hide_index=True
            )
    except Exception as e:
        st.error(f"Error computing scores: {e}")
    
    st.markdown("---")
    
    # Indicator comparisons
    st.markdown("### 📊 Indicator Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_bar_comparison(
            fsic_data, selected_codes, 
            'RCAR|T1RWA', 'Capital Adequacy Ratio (%)'
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_bar_comparison(
            fsic_data, selected_codes,
            'NPLGL', 'NPL Ratio (%)'
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_bar_comparison(
            fsic_data, selected_codes,
            'ROE', 'Return on Equity (%)'
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_bar_comparison(
            fsic_data, selected_codes,
            'LASTL', 'Liquid Assets / ST Liabilities (%)'
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Scatter plot: NPL vs Capital
    st.markdown("### 🔄 Risk-Capital Trade-off")
    
    fig = create_scatter_comparison(
        fsic_data, selected_codes,
        'NPLGL', 'RCAR'
    )
    if fig:
        fig.update_layout(
            title="NPL Ratio vs Capital Adequacy",
            xaxis_title="NPL Ratio (%)",
            yaxis_title="Capital Adequacy Ratio (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("""
        *Interpretation: Countries in the upper-left quadrant (low NPL, high capital) 
        demonstrate stronger banking sector fundamentals.*
        """)
    
    st.markdown("---")
    
    # Time series comparison
    st.markdown("### 📈 Historical Trends")
    
    indicator_choice = st.selectbox(
        "Select indicator for time series comparison",
        options=['Capital Ratio (RCAR)', 'NPL Ratio (NPLGL)', 'ROE', 'Liquidity (LASTL)']
    )
    
    indicator_map = {
        'Capital Ratio (RCAR)': 'RCAR',
        'NPL Ratio (NPLGL)': 'NPLGL',
        'ROE': 'ROE',
        'Liquidity (LASTL)': 'LASTL'
    }
    
    indicator_code = indicator_map.get(indicator_choice, 'RCAR')
    
    fig = create_time_series_comparison(
        fsic_data, selected_codes,
        indicator_code, f"{indicator_choice} Over Time"
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Raw data table
    with st.expander("📋 View Raw Data"):
        st.markdown("#### Latest Values by Country")
        
        # Pivot table of latest values
        latest = fsic_data[fsic_data['country_code'].isin(selected_codes)].copy()
        latest = latest.sort_values('period').groupby(
            ['country_code', 'indicator_code']
        ).last().reset_index()
        
        pivot = latest.pivot_table(
            index='country_code',
            columns='indicator_code',
            values='value'
        )
        
        st.dataframe(pivot, use_container_width=True)
        
        # Download
        csv = pivot.to_csv()
        st.download_button(
            "Download Comparison Data",
            csv,
            "country_comparison.csv",
            "text/csv"
        )


if __name__ == "__main__":
    main()
