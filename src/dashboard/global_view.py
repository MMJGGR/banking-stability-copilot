
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycountry_convert as pc
from typing import Dict, Any, Optional

def get_continent_name(alpha3_code: str) -> str:
    """Convert ISO-3 country code to Continent Name."""
    try:
        alpha2 = pc.country_alpha3_to_country_alpha2(alpha3_code)
        continent_code = pc.country_alpha2_to_continent_code(alpha2)
        continent_names = {
            'NA': 'North America',
            'SA': 'South America', 
            'AS': 'Asia',
            'EU': 'Europe',
            'AF': 'Africa',
            'OC': 'Oceania',
            'AN': 'Antarctica'
        }
        return continent_names.get(continent_code, 'Other')
    except:
        return 'Other'

def calculate_weighted_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate GDP-weighted global metrics.
    Weights = nominal_gdp (USD).
    """
    metrics = {}
    
    # Ensure weighting column exists and has data
    if 'nominal_gdp' not in df.columns:
        return {}
        
    valid_df = df.dropna(subset=['nominal_gdp'])
    if len(valid_df) == 0:
        return {}
        
    total_gdp = valid_df['nominal_gdp'].sum()
    metrics['total_gdp_trillions'] = total_gdp / 1000  # Assuming B -> T (if data is Billions)
    # Note: WEO NGDPD is usually Billions.
    
    # helper for weighted avg
    def w_avg(col):
        if col not in valid_df.columns: return 0.0 # Return 0.0 instead of None
        mask = valid_df[col].notna()
        sub = valid_df[mask]
        if sub['nominal_gdp'].sum() == 0: return 0.0
        return (sub[col] * sub['nominal_gdp']).sum() / sub['nominal_gdp'].sum()

    metrics['global_risk_score'] = w_avg('risk_score')
    metrics['global_economic_pillar'] = w_avg('economic_pillar')
    metrics['global_industry_pillar'] = w_avg('industry_pillar')
    
    # Key indicators
    metrics['global_npl'] = w_avg('npl_ratio')
    metrics['global_capital_adequacy'] = w_avg('capital_adequacy')
    
    return metrics

def render_global_summary(scores_df: pd.DataFrame, model_features: pd.DataFrame, loader):
    """
    Render the Global Summary tab with weighted metrics and maps.
    Accepts full model_features to access raw indicators (NPL, GDP) for weighting.
    """
    st.markdown("## Global Risk Landscape")
    st.caption("Risk scores weighted by Nominal GDP (USD). Larger economies contribute more to the global aggregate.")
    
    if scores_df is None or len(scores_df) == 0:
        st.warning("No data available for summary.")
        return

    # 1. Processing Data - Merge Scores with Features (GDP, NPL, etc)
    df = scores_df.copy()
    if model_features is not None:
         # Avoid duplicate columns in merge
         cols_to_use = [c for c in model_features.columns if c not in df.columns or c == 'country_code']
         df = df.merge(model_features[cols_to_use], on='country_code', how='left')

    df['Region'] = df['country_code'].apply(get_continent_name)
    
    # 2. Weighted Metrics
    metrics = calculate_weighted_metrics(df)
    
    # 3. KPI Cards - More actionable metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        score = metrics.get('global_risk_score', 0) or 0
        st.metric("Global Weighted Risk", f"{score:.1f}/10", help="GDP-weighted average of all country risk scores")
        
    with kpi2:
        countries_covered = len(df)
        high_risk_count = len(df[df['risk_score'] > 6.0])
        st.metric("Countries Analyzed", f"{countries_covered}", delta=f"{high_risk_count} High Risk", delta_color="inverse")
        
    with kpi3:
        # Find highest risk region
        if 'Region' in df.columns:
            region_risk = df.groupby('Region')['risk_score'].mean().sort_values(ascending=False)
            highest_region = region_risk.index[0] if len(region_risk) > 0 else "N/A"
            highest_score = region_risk.iloc[0] if len(region_risk) > 0 else 0
            st.metric("Highest Risk Region", highest_region, delta=f"Avg: {highest_score:.1f}")
        else:
            st.metric("Highest Risk Region", "N/A")
        
    with kpi4:
        low_risk_count = len(df[df['risk_score'] < 4.0])
        countries_total = len(df)
        pct = (low_risk_count / countries_total * 100) if countries_total > 0 else 0
        st.metric("Low Risk Economies", f"{low_risk_count}", delta=f"{pct:.0f}% of total", delta_color="off")

    st.markdown("---")
    
    # 4. Global Map
    st.markdown("### Risk Distribution Map")
    
    # Format GDP for cleaner hover display
    df['gdp_billions'] = df['nominal_gdp'].apply(lambda x: f"${x/1:.0f}B" if pd.notna(x) else "N/A")
    
    fig_map = px.choropleth(
        df,
        locations="country_code",
        color="risk_score",
        hover_name="country_name",
        hover_data={
            "risk_score": ":.1f",
            "economic_pillar": ":.1f",
            "industry_pillar": ":.1f",
            "gdp_billions": True,
            "country_code": False  # Hide code in hover
        },
        color_continuous_scale="RdYlGn_r", # Low(1)=Green, High(10)=Red
        range_color=(1, 10),
        projection="natural earth",
        title=""
    )
    fig_map.update_layout(
        margin={"r":0,"t":10,"l":0,"b":0},
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        geo_bgcolor='rgba(0,0,0,0)',
        coloraxis_colorbar=dict(title="Risk Score", tickvals=[1,3,5,7,9])
    )
    st.plotly_chart(fig_map, use_container_width=True)
    
    # 5. Regional Analysis & Scatter
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### Regional Risk Profile")
        # Calc weighted risk by region
        region_stats = []
        for reg in df['Region'].unique():
            reg_df = df[df['Region'] == reg]
            if len(reg_df) == 0: continue
            
            # Weighted avg for region
            tot_gdp = reg_df['nominal_gdp'].sum()
            if tot_gdp > 0:
                w_score = (reg_df['risk_score'] * reg_df['nominal_gdp']).sum() / tot_gdp
            else:
                w_score = reg_df['risk_score'].mean()
                
            region_stats.append({'Region': reg, 'Weighted Risk': w_score, 'Countries': len(reg_df)})
            
        reg_summary = pd.DataFrame(region_stats).sort_values('Weighted Risk', ascending=False)
        
        fig_bar = px.bar(
            reg_summary,
            x='Region',
            y='Weighted Risk',
            color='Weighted Risk',
            color_continuous_scale='RdYlGn_r',
            range_color=[1, 10],
            text_auto='.1f'
        )
        fig_bar.update_layout(xaxis_title="", yaxis_title="Weighted Risk Score")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_right:
        st.markdown("### Stability vs Growth")
        # Scatter: X=GDP Growth, Y=Risk Score, Size=GDP
        # Filter to only actual countries (exclude regional aggregates like ASEAN, Euro Area)
        plot_df = df[df['Region'] != 'Other'].copy()
        if 'gdp_growth' in plot_df.columns:
            plot_df = plot_df[plot_df['gdp_growth'].between(-10, 15)] # Filter outliers
            
            fig_scat = px.scatter(
                plot_df,
                x="gdp_growth",
                y="risk_score",
                size="nominal_gdp",
                color="Region",
                hover_name="country_name",
                size_max=60,
                title="Risk vs GDP Growth (Size = GDP)"
            )
            fig_scat.update_layout(
                xaxis_title="Real GDP Growth (%)",
                yaxis_title="Risk Score (1-10)"
            )
            st.plotly_chart(fig_scat, use_container_width=True)
        else:
            st.info("GDP Growth data needed for scatter plot.")

    # 6. Top/Bottom Lists
    st.markdown("### Systemic Risk Watchlist")
    st.caption("Significant economies with elevated risk scores (> 6.0)")
    
    # Filter large economies (e.g. top 50 percentile of GDP) and High Risk
    if 'nominal_gdp' in df.columns:
        gdp_median = df['nominal_gdp'].median()
        watchlist = df[ (df['nominal_gdp'] > gdp_median) & (df['risk_score'] > 6.0) ].copy()
        watchlist = watchlist.sort_values('risk_score', ascending=False).head(10)
        
        if len(watchlist) > 0:
            # Derive which pillar is the main risk contributor
            def get_risk_driver(row):
                econ = row.get('economic_pillar', 5)
                ind = row.get('industry_pillar', 5)
                if pd.isna(econ) and pd.isna(ind):
                    return "—"
                elif pd.isna(econ):
                    return "Industry"
                elif pd.isna(ind):
                    return "Economic"
                elif econ > ind:
                    return "Economic"
                else:
                    return "Industry"
            
            watchlist['risk_driver'] = watchlist.apply(get_risk_driver, axis=1)
            
            # Build display dataframe with friendly names
            display_df = pd.DataFrame({
                'Country': watchlist['country_name'],
                'Region': watchlist['Region'],
                'Risk Score': watchlist['risk_score'].apply(lambda x: f"{x:.1f}"),
                'Economic': watchlist['economic_pillar'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—"),
                'Industry': watchlist['industry_pillar'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—"),
                'Main Driver': watchlist['risk_driver'],
            })
            display_df = display_df.reset_index(drop=True)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.success("✓ No significant economies currently flagged as high risk (>6.0).")

