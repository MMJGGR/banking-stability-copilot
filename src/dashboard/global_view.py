
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pycountry_convert as pc
from typing import Dict, Any, Optional

# Codes pycountry_convert cannot map (IMF aggregates and small territories).
CONTINENT_OVERRIDES = {
    'TLS': 'Asia',
    'WBG': 'Asia',
    'KOS': 'Europe',
    'UVK': 'Europe',
    'SXM': 'North America',
    'CUW': 'North America',
}


def get_continent_name(alpha3_code: str) -> str:
    """Convert ISO-3 country code to Continent Name."""
    if alpha3_code in CONTINENT_OVERRIDES:
        return CONTINENT_OVERRIDES[alpha3_code]
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

def render_global_summary(
    scores_df: pd.DataFrame,
    model_features: pd.DataFrame,
    loader,
    risk_architecture: pd.DataFrame | None = None,
    risk_validation: dict | None = None,
):
    """
    Render the Global Summary tab with weighted metrics and maps.
    Accepts full model_features to access raw indicators (NPL, GDP) for weighting.
    """
    st.markdown("## Global Risk Landscape")
    st.caption(
        "The active composite score and research early-warning signals are shown "
        "separately. The global active score is weighted by nominal GDP."
    )
    
    if scores_df is None or len(scores_df) == 0:
        st.warning("No data available for summary.")
        return

    # 1. Processing Data - Merge Scores with Features (GDP, NPL, etc)
    df = scores_df.copy()
    if model_features is not None:
         # Avoid duplicate columns in merge
         cols_to_use = [c for c in model_features.columns if c not in df.columns or c == 'country_code']
         df = df.merge(model_features[cols_to_use], on='country_code', how='left')

    if (
        risk_architecture is not None
        and not risk_architecture.empty
        and 'country_code' in risk_architecture.columns
    ):
        architecture = risk_architecture.copy()
        architecture['country_code'] = architecture['country_code'].astype(str).str.upper()
        architecture = architecture.drop_duplicates('country_code')
        architecture_columns = [
            column for column in architecture.columns
            if column == 'country_code' or column not in df.columns
        ]
        df = df.merge(
            architecture[architecture_columns],
            on='country_code',
            how='left',
        )

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
        
    alert_series = (
        df['alert_status'].astype(str).str.lower()
        if 'alert_status' in df.columns
        else pd.Series('', index=df.index)
    )
    architecture_is_production = (
        'effective_production' in df.columns
        and not df['effective_production'].empty
        and df['effective_production'].fillna(False).astype(bool).all()
    )
    architecture_is_research = not architecture_is_production
    architecture_reportable = (
        'outputs_reportable' in df.columns
        and not df['outputs_reportable'].empty
        and df['outputs_reportable'].fillna(False).astype(bool).all()
    )
    promotion_gate = (
        (risk_validation or {}).get('promotion_gates', {}).get('passed')
        if isinstance(risk_validation, dict)
        else None
    )

    with kpi3:
        if 'alert_status' in df.columns:
            if architecture_is_research:
                st.metric("Early-warning Model", "Research")
            else:
                st.metric(
                    "Red Alerts",
                    f"{int(alert_series.eq('red').sum())}",
                    help="High-conviction alerts requiring corroborating evidence.",
                )
        else:
            high_risk_count = len(df[df['risk_score'] > 6.0])
            st.metric("High Active Risk", f"{high_risk_count}")

    with kpi4:
        if 'alert_status' in df.columns:
            if architecture_is_research:
                gate_text = (
                    "Passed" if promotion_gate is True
                    else "Failed" if promotion_gate is False
                    else "Not recorded"
                )
                st.metric("Forward Validation Gate", gate_text)
            else:
                st.metric(
                    "Amber Watches",
                    f"{int(alert_series.eq('amber').sum())}",
                    help="Recall-oriented surveillance signals for analyst review.",
                )
        else:
            low_risk_count = len(df[df['risk_score'] < 4.0])
            countries_total = len(df)
            pct = (low_risk_count / countries_total * 100) if countries_total > 0 else 0
            st.metric(
                "Low Structural Risk",
                f"{low_risk_count}",
                delta=f"{pct:.0f}% of total",
                delta_color="off",
            )

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
    st.plotly_chart(fig_map, use_container_width=True, theme="streamlit")
    
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

        def risk_color(value: float) -> str:
            if value >= 7:
                return "#B91C1C"
            if value >= 5:
                return "#F59E0B"
            if value >= 3:
                return "#A3E635"
            return "#16A34A"

        fig_bar = go.Figure(
            go.Bar(
                x=reg_summary['Region'],
                y=reg_summary['Weighted Risk'],
                marker_color=[risk_color(v) for v in reg_summary['Weighted Risk']],
                text=reg_summary['Weighted Risk'].map(lambda v: f"{v:.1f}"),
                textposition="inside",
                cliponaxis=False,
            )
        )
        fig_bar.update_layout(
            xaxis_title="",
            yaxis_title="Weighted Risk Score",
            showlegend=False,
            margin={"r": 10, "t": 10, "l": 10, "b": 10},
            yaxis=dict(range=[0, 10]),
        )
        st.plotly_chart(fig_bar, use_container_width=True, theme="streamlit")
        
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
            st.plotly_chart(fig_scat, use_container_width=True, theme="streamlit")
        else:
            st.info("GDP Growth data needed for scatter plot.")

    # 6. Keep structural vulnerability and early-warning decisions distinct.
    st.markdown("### Watchlists")
    watch_early, watch_structural = st.tabs(
        ["Research diagnostics" if architecture_is_research else "Early warning", "Active risk"]
    )

    with watch_early:
        if 'alert_status' not in df.columns:
            st.caption(
                "The hierarchical early-warning artifact is not available for "
                "this snapshot."
            )
        elif not architecture_reportable:
            if promotion_gate is False:
                st.warning(
                    "The early-warning challenger failed its untouched forward "
                    "validation gate. Country probabilities, rankings, and review "
                    "tiers are suppressed."
                )
            else:
                st.warning(
                    "The early-warning validation gate is unavailable or the "
                    "artifact does not match the active snapshot. Country outputs "
                    "are not reportable."
                )
            coverage_columns = [
                column
                for column in (
                    'hazard_evidence_coverage',
                    'mechanism_evidence_coverage',
                )
                if column in df.columns
            ]
            if coverage_columns:
                gaps = df.copy()
                gaps['_minimum_coverage'] = gaps[coverage_columns].apply(
                    pd.to_numeric, errors='coerce'
                ).min(axis=1)
                gaps = gaps.sort_values('_minimum_coverage', na_position='first').head(20)
                gap_display = pd.DataFrame({
                    'Country': gaps['country_name'],
                    'Hazard inputs': pd.to_numeric(
                        gaps.get(
                            'hazard_evidence_coverage',
                            pd.Series(np.nan, index=gaps.index),
                        ),
                        errors='coerce',
                    ),
                    'Mechanism taxonomy': pd.to_numeric(
                        gaps.get(
                            'mechanism_evidence_coverage',
                            pd.Series(np.nan, index=gaps.index),
                        ),
                        errors='coerce',
                    ),
                    'Selected expert': gaps.get(
                        'hazard_expert', pd.Series('Not available', index=gaps.index)
                    ).astype(str).str.replace('_', ' ').str.title(),
                })
                st.markdown("#### Largest evidence gaps")
                st.dataframe(
                    gap_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Hazard inputs': st.column_config.ProgressColumn(
                            min_value=0.0, max_value=1.0
                        ),
                        'Mechanism taxonomy': st.column_config.ProgressColumn(
                            min_value=0.0, max_value=1.0
                        ),
                    },
                )
            else:
                st.caption("Evidence coverage is unavailable for this research snapshot.")
        else:
            alerts = df[alert_series.isin(['red', 'amber'])].copy()
            alerts['_alert_rank'] = (
                alerts['alert_status'].astype(str).str.lower().map(
                    {'red': 0, 'amber': 1}
                )
            )
            sort_columns = ['_alert_rank']
            ascending = [True]
            if 'systemic_hazard_1y' in alerts.columns:
                sort_columns.append('systemic_hazard_1y')
                ascending.append(False)
            alerts = alerts.sort_values(
                sort_columns, ascending=ascending
            ).head(20)
            if alerts.empty:
                st.caption(
                    "No countries meet the current Red or Amber research "
                    "thresholds."
                )
            else:
                display = pd.DataFrame({
                    'Country': alerts['country_name'],
                    'Status': alerts['alert_status'].astype(str).str.title(),
                    '1-year onset probability': pd.to_numeric(
                        alerts.get('systemic_hazard_1y'), errors='coerce'
                    ),
                    'Years 2-3 onset probability': pd.to_numeric(
                        alerts.get('systemic_hazard_2_3y'), errors='coerce'
                    ),
                    'Dominant mechanism': alerts.get(
                        'dominant_mechanism', 'Not available'
                    ),
                    'Evidence': pd.to_numeric(
                        alerts.get('evidence_confidence'), errors='coerce'
                    ),
                })
                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        '1-year onset probability': st.column_config.NumberColumn(
                            format='percent'
                        ),
                        'Years 2-3 onset probability': st.column_config.NumberColumn(
                            format='percent'
                        ),
                        'Evidence': st.column_config.ProgressColumn(
                            min_value=0.0, max_value=1.0
                        ),
                    },
                )
            if not architecture_is_production:
                st.caption(
                    "Research outputs are not used in the active risk score. "
                    f"Current candidate count: {int(alert_series.eq('amber').sum())} "
                    "Amber and no operational Red tier."
                )

    with watch_structural:
        if 'nominal_gdp' not in df.columns:
            st.caption(
                "Nominal GDP is unavailable, so the significance filter "
                "cannot be applied."
            )
        else:
            gdp_median = df['nominal_gdp'].median()
            watchlist = df[
                (df['nominal_gdp'] > gdp_median)
                & (df['risk_score'] > 6.0)
            ].copy().sort_values('risk_score', ascending=False).head(15)
            if watchlist.empty:
                st.caption(
                    "No significant economies currently exceed the "
                    "structural-risk threshold."
                )
            else:
                display_df = pd.DataFrame({
                    'Country': watchlist['country_name'],
                    'Region': watchlist['Region'],
                    'Risk Score': pd.to_numeric(
                        watchlist['risk_score'], errors='coerce'
                    ),
                    'Operating Environment': pd.to_numeric(
                        watchlist['economic_pillar'], errors='coerce'
                    ),
                    'Banking System': pd.to_numeric(
                        watchlist['industry_pillar'], errors='coerce'
                    ),
                })
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Risk Score': st.column_config.NumberColumn(
                            format='%.1f'
                        ),
                        'Operating Environment': st.column_config.NumberColumn(
                            format='%.1f'
                        ),
                        'Banking System': st.column_config.NumberColumn(
                            format='%.1f'
                        ),
                    },
                )

