"""
ML Risk Analysis Page
Deep dive into ML-based risk scoring with feature importance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import IMFDataLoader
from src.risk_scorer import RiskScorer, compute_composite_bicra_score
from src.imputation import GapImputer

st.set_page_config(
    page_title="ML Risk Analysis",
    page_icon="🤖",
    layout="wide"
)

st.markdown("# 🤖 ML Risk Analysis")
st.markdown("*Feature importance and clustering insights*")


@st.cache_resource
def load_data():
    loader = IMFDataLoader()
    if not loader.load_from_cache():
        loader.load_all_datasets()
    return loader


@st.cache_data
def train_risk_model(_fsic_data):
    """Train risk model and return scorer."""
    if len(_fsic_data) == 0:
        return None
    
    scorer = RiskScorer(n_clusters=5)
    try:
        scorer.fit(_fsic_data)
        return scorer
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None


def main():
    loader = load_data()
    fsic_data = loader._data_cache.get('FSIC', pd.DataFrame())
    weo_data = loader._data_cache.get('WEO', pd.DataFrame())
    
    if len(fsic_data) == 0:
        st.warning("No FSIC data available. Please load datasets first.")
        return
    
    # Train model
    with st.spinner("Training ML model..."):
        scorer = train_risk_model(fsic_data)
    
    if scorer is None:
        st.error("Could not train risk model.")
        return
    
    st.success("Model trained successfully!")
    
    # Feature importance
    st.markdown("### 🎯 Feature Importance")
    st.markdown("*Indicators most predictive of risk tier differentiation*")
    
    if scorer.feature_importance_:
        importance_df = pd.DataFrame([
            {'indicator': k, 'importance': v}
            for k, v in scorer.feature_importance_.items()
        ]).head(20).sort_values('importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='importance',
            y='indicator',
            orientation='h',
            color='importance',
            color_continuous_scale='Blues',
            title="Top 20 Most Important Indicators"
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            height=600,
            yaxis_title="",
            xaxis_title="Relative Importance"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        > **Interpretation**: These indicators have the highest mutual information 
        with risk cluster assignments. They are the most predictive of which 
        risk tier a country falls into.
        """)
    
    st.markdown("---")
    
    # Country scores
    st.markdown("### 🌍 Country Risk Scores")
    
    try:
        scores = scorer.score_countries(fsic_data)
        
        if len(scores) > 0:
            # Distribution by tier
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.histogram(
                    scores,
                    x='risk_score',
                    nbins=20,
                    color='risk_tier',
                    color_discrete_map={
                        'Very Strong': '#2E7D32',
                        'Strong': '#66BB6A',
                        'Adequate': '#FFC107',
                        'Weak': '#FF7043',
                        'Very Weak': '#D32F2F'
                    },
                    title="Distribution of Risk Scores"
                )
                
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Risk Score",
                    yaxis_title="Number of Countries",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                tier_counts = scores['risk_tier'].value_counts()
                
                fig = px.pie(
                    values=tier_counts.values,
                    names=tier_counts.index,
                    color=tier_counts.index,
                    color_discrete_map={
                        'Very Strong': '#2E7D32',
                        'Strong': '#66BB6A',
                        'Adequate': '#FFC107',
                        'Weak': '#FF7043',
                        'Very Weak': '#D32F2F'
                    },
                    title="Countries by Risk Tier"
                )
                
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Full country table
            st.markdown("#### All Countries")
            
            # Add rank
            scores_display = scores.sort_values('risk_score', ascending=False).copy()
            scores_display['rank'] = range(1, len(scores_display) + 1)
            
            st.dataframe(
                scores_display[['rank', 'country_code', 'risk_score', 'risk_tier', 'confidence']]
                .rename(columns={
                    'rank': 'Rank',
                    'country_code': 'Country',
                    'risk_score': 'Score',
                    'risk_tier': 'Tier',
                    'confidence': 'Data Confidence'
                }),
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            # Download
            csv = scores_display.to_csv(index=False)
            st.download_button(
                "Download Risk Scores",
                csv,
                "ml_risk_scores.csv",
                "text/csv"
            )
            
    except Exception as e:
        st.error(f"Error scoring countries: {e}")
    
    st.markdown("---")
    
    # Deterioration detection
    st.markdown("### ⚠️ Deterioration Signals")
    st.markdown("*Countries showing signs of weakening fundamentals*")
    
    try:
        deterioration = scorer.detect_deterioration(fsic_data, lookback_periods=8)
        
        if len(deterioration) > 0:
            # Aggregate by country
            country_deterioration = deterioration.groupby('country_code').agg({
                'indicator_code': 'count',
                'deterioration_severity': lambda x: (x == 'High').sum()
            }).reset_index()
            country_deterioration.columns = ['country_code', 'n_deteriorating', 'n_high_severity']
            country_deterioration = country_deterioration.sort_values('n_high_severity', ascending=False)
            
            # Top countries with deterioration
            st.markdown("#### Countries with Most Deteriorating Indicators")
            
            fig = px.bar(
                country_deterioration.head(15),
                x='country_code',
                y='n_deteriorating',
                color='n_high_severity',
                color_continuous_scale='Reds',
                title="Number of Deteriorating Indicators by Country"
            )
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Country",
                yaxis_title="# Deteriorating Indicators",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            with st.expander("View All Deterioration Signals"):
                st.dataframe(deterioration, use_container_width=True, height=400)
        else:
            st.info("No significant deterioration signals detected in recent data.")
            
    except Exception as e:
        st.warning(f"Could not compute deterioration signals: {e}")
    
    st.markdown("---")
    
    # Data quality
    st.markdown("### 📊 Data Quality Analysis")
    
    # Coverage analysis
    countries = loader.get_countries('FSIC')
    
    if len(countries) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Countries with FSIC Data", len(countries['country_code'].unique()))
        
        with col2:
            avg_indicators = countries['n_indicators'].mean()
            st.metric("Avg Indicators per Country", f"{avg_indicators:.0f}")
        
        # Missing data heatmap would go here
        st.markdown("""
        **Note**: Countries with less data coverage will have lower confidence scores 
        in their risk assessments. The ML model uses imputation to fill gaps, but 
        confidence decreases as the proportion of imputed data increases.
        """)


if __name__ == "__main__":
    main()
