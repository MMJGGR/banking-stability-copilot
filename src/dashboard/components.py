"""
Dashboard components for Banking System Stability Copilot.
Provides UI components for data visualization, trends, and prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
from datetime import datetime
from src.dashboard.styles import get_risk_color_hex, get_risk_label, COLORS

RESERVE_CURRENCY_COUNTRIES = ['USA', 'GBR', 'JPN', 'CHE', 'EMU', 'DEU', 'FRA', 'ITA', 'ESP', 'NLD', 'BEL', 'AUT']

# ==============================================================================
# SUMMARY COMPONENTS
# ==============================================================================

def render_summary_card(country_name: str, score: float, tier: int, confidence: float,
                        risk_floor_applied: bool = False, econ_coverage: float = None, 
                        ind_coverage: float = None):
    """Renders the high-level summary card for a country with confidence indicators."""
    color = get_risk_color_hex(tier)
    label = get_risk_label(tier)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="summary-header">Risk Score</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-value" style="color: {color}">{score:.1f}/10</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown(f'<div class="summary-header">Risk Tier</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-value" style="color: {color}">{label}</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown(f'<div class="summary-header">Category</div>', unsafe_allow_html=True)
        # Map tier to category description
        categories = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}
        st.markdown(f'<div class="summary-value">{categories.get(tier, "N/A")} Risk</div>', unsafe_allow_html=True)

    with col4:
        st.markdown(f'<div class="summary-header">Data Coverage</div>', unsafe_allow_html=True)
        # Color code coverage
        cov_color = "#22c55e" if confidence >= 0.70 else "#f59e0b" if confidence >= 0.50 else "#ef4444"
        st.markdown(f'<div class="summary-value" style="color: {cov_color}">{confidence:.0%}</div>', unsafe_allow_html=True)

    # Confidence warning banner
    if confidence < 0.70 or risk_floor_applied:
        warning_parts = []
        if confidence < 0.50:
            warning_parts.append("⚠️ Very low data coverage (<50%)")
        elif confidence < 0.70:
            warning_parts.append("⚠️ Limited data coverage (<70%)")
        
        if risk_floor_applied:
            warning_parts.append("Risk score capped due to low confidence")
        
        # Show pillar-specific coverage issues
        if econ_coverage is not None and econ_coverage < 0.30:
            warning_parts.append(f"Economic data: {econ_coverage:.0%}")
        if ind_coverage is not None and ind_coverage < 0.30:
            warning_parts.append(f"Banking data: {ind_coverage:.0%}")
        
        if warning_parts:
            st.warning(" | ".join(warning_parts))

    st.markdown("---")


# ==============================================================================
# DATA SNAPSHOT COMPONENTS
# ==============================================================================

# Mapping from our feature names to IMF indicator codes (for WEO)
# Harmonized with feature_engineering.py extract_weo_features
WEO_INDICATORS = {
    'gdp_per_capita': 'NGDPDPC',
    'gdp_growth': 'NGDP_RPCH',
    'inflation': 'PCPIPCH',
    'unemployment': 'LUR',
    'govt_debt_gdp': 'GGXWDG_NGDP',
    'fiscal_balance_gdp': 'GGXCNL_NGDP',
    'current_account_gdp': 'BCA_NGDPD',
    'nominal_gdp': 'NGDPD',  # Nominal GDP in USD billions - matches feature_engineering.py
}

# FSIC uses same code (S12CFSI) for all - differentiate by name pattern
# Patterns from src/feature_engineering.py - FULLY HARMONIZED
# IMPORTANT: Add 'Percent' to patterns to avoid matching absolute currency values
FSIC_NAME_PATTERNS = {
    'capital_adequacy': 'Regulatory capital to risk-weighted assets.*Core FSI',
    'tier1_capital': 'Tier 1 capital to risk-weighted assets.*Core FSI',
    'npl_ratio': 'Nonperforming loans to total gross loans.*Core FSI',
    'roe': 'Return on equity.*Core FSI',
    'roa': 'Return on assets.*Core FSI',
    'liquid_assets_st_liab': 'Liquid assets to short term liabilities.*Core FSI',
    'liquid_assets_total': 'Liquid assets to total assets.*Percent',
    'customer_deposits_loans': 'Customer deposits to total.*loans.*Percent',
    'fx_loan_exposure': 'Foreign currency.*loans.*Percent',
    'loan_concentration': 'Loan concentration.*Percent',
    'real_estate_loans': 'Residential real estate loans to total gross loans.*Core FSI',
    'npl_provisions': 'Provisions to nonperforming loans.*Percent',
}

# Combined mapping for backwards compatibility
FEATURE_TO_INDICATOR = {**WEO_INDICATORS}

def render_data_snapshot(country_data: Dict[str, Any], features_df: pd.DataFrame = None, 
                         loader=None, country_code: str = None, wgi_data: pd.DataFrame = None,
                         model_features: pd.DataFrame = None, pca_info: Dict = None):
    """
    Renders key indicators used in the model for a country.
    
    IMPORTANT: All feature values are read directly from crisis_features.parquet
    (the model's computed features) to ensure consistency between dashboard and model.
    No redundant calculations - single source of truth.
    """
    
    # SINGLE SOURCE OF TRUTH: Read all features from crisis_features.parquet
    # This is exact same data the model uses - no recalculation needed
    actual_values = {}
    
    from src.config import CACHE_DIR
    import os
    parquet_path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
    
    if os.path.exists(parquet_path) and country_code:
        try:
            crisis_df = pd.read_parquet(parquet_path)
            crisis_row = crisis_df[crisis_df['country_code'] == country_code]
            if len(crisis_row) > 0:
                row = crisis_row.iloc[0]
                # Extract all numeric values from model features
                for col in crisis_df.columns:
                    if col != 'country_code' and pd.notna(row.get(col)):
                        actual_values[col] = row[col]
        except Exception as e:
            pass
    
    # Fallback: If parquet not available, try model_features (for testing/dev)
    if not actual_values and model_features is not None and country_code:
        try:
            model_row = model_features[model_features['country_code'] == country_code]
            if len(model_row) > 0:
                row = model_row.iloc[0]
                for col in model_features.columns:
                    if col != 'country_code' and pd.notna(row.get(col)):
                        actual_values[col] = row[col]
        except Exception:
            pass

    
    # Define indicator groups with display names and format
    # NOTE: Shows the KEY indicators used in the model - not all features
    # Dropped features (tier1, private/total credit_to_gdp) removed
    indicator_groups = {
        "Economic Fundamentals": [
            ("gdp_per_capita", "GDP per Capita", "${:,.0f}"),
            ("gdp_growth", "GDP Growth", "{:.1f}%"),
            ("inflation", "Inflation Rate", "{:.1f}%"),
            ("govt_debt_gdp", "Govt Debt/GDP", "{:.1f}%"),
            ("fiscal_balance_gdp", "Fiscal Balance/GDP", "{:.1f}%"),
            ("credit_to_gdp_gap", "Credit-to-GDP Gap", "{:+.1f}pp"), # Moved from Liquidity
            ("sovereign_liability_to_reserves", "Sov Liab/Reserves", "{:.1f}x"), # NEW
        ],
        "Banking Sector Health": [
            ("capital_adequacy", "Capital Adequacy Ratio", "{:.1f}%"),
            ("npl_ratio", "NPL Ratio", "{:.1f}%"),
            ("roe", "Return on Equity", "{:.1f}%"),
            ("sovereign_exposure_ratio", "Sovereign Exposure", "{:.1f}%"),
            ("net_interest_margin", "Net Interest Margin", "{:.2f}%"),
        ],
        "Liquidity & Funding": [
            ("liquid_assets_st_liab", "Liquid Assets/ST Liab", "{:.1f}%"),
            ("bank_liability_to_nfa", "Bank Liab/NFA", "{:.1f}x"), # NEW
            ("customer_deposits_loans", "Deposits/Loans", "{:.1f}%"),
            ("deposit_funding_ratio", "Deposit Funding Ratio", "{:.1f}%"),
        ],
    }
    
    # Create columns for the groups
    cols = st.columns(len(indicator_groups))
    
    # Load crisis_features.parquet to get imputed values
    from src.config import CACHE_DIR
    import os
    import os
    # Read IMPUTED features (from train_model.py) for the "imputed_values" dict
    # This allows us to detect when Raw (crisis_features) differs from Imputed (imputed_features)
    parquet_path = os.path.join(CACHE_DIR, 'imputed_features.parquet')
    imputed_values = {}
    if os.path.exists(parquet_path) and country_code:
        try:
            imputed_df = pd.read_parquet(parquet_path)
            imputed_row = imputed_df[imputed_df['country_code'] == country_code]
            if len(imputed_row) > 0:
                imputed_values = imputed_row.iloc[0].to_dict()
        except Exception:
            pass
    
    for idx, (group_name, indicators) in enumerate(indicator_groups.items()):
        with cols[idx]:
            st.markdown(f"**{group_name}**")
            
            for key, display_name, fmt in indicators:
                value = actual_values.get(key)
                imputed_val = imputed_values.get(key)
                
                if value is not None and not pd.isna(value):
                    # Actual data available
                    try:
                        formatted = fmt.format(value)
                    except:
                        formatted = str(value)
                    st.markdown(f"<div class='snapshot-row'><span class='snapshot-label'>{display_name}</span><span class='snapshot-value'>{formatted}</span></div>", unsafe_allow_html=True)
                elif imputed_val is not None and not pd.isna(imputed_val):
                    # Check for Smart Proxy (Verified Data)
                    is_proxy = False
                    if key == 'sovereign_exposure_ratio':
                        sec_val = imputed_values.get('securities_to_assets')
                        if sec_val is not None and pd.notna(sec_val) and abs(imputed_val - sec_val) < 0.1:
                            is_proxy = True
                    
                    try:
                        formatted = fmt.format(imputed_val)
                    except:
                        formatted = str(imputed_val)
                    
                    if is_proxy:
                        # Show as Actual (Verified Proxy)
                        st.markdown(f"<div class='snapshot-row'><span class='snapshot-label'>{display_name}</span><span class='snapshot-value'>{formatted}</span></div>", unsafe_allow_html=True)
                    else:
                        # Show imputed value with flag
                        st.markdown(f"<div class='snapshot-row'><span class='snapshot-label'>{display_name}</span><span class='snapshot-value imputed'>{formatted}*</span></div>", unsafe_allow_html=True)
                else:
                    # No data at all
                    st.markdown(f"<div class='snapshot-row'><span class='snapshot-label'>{display_name}</span><span class='snapshot-value missing'>—</span></div>", unsafe_allow_html=True)
    
    st.caption("*Values marked with asterisk are from model training data (may include imputations)")

    
    # Add expandable Model Features & Weights section
    with st.expander("Model Features & Weights"):
        st.markdown("**Risk Score = 50% Economic Pillar + 50% Industry Pillar**")
        st.markdown("Each pillar uses PCA to combine features. Missing values are imputed via KNN (or median fallback).")
        st.caption("✓ = actual data, ≈ = imputed value used in model")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Economic Pillar (50%)**")
            economic_features = [
                ("gdp_growth", "GDP Growth", "{:.1f}%"),
                ("gdp_per_capita", "GDP per Capita", "${:,.0f}"),
                ("inflation", "Inflation Rate", "{:.1f}%"),
                ("current_account_gdp", "Current Account/GDP", "{:.1f}%"),
                ("govt_debt_gdp", "Govt Debt/GDP", "{:.1f}%"),
                ("fiscal_balance_gdp", "Fiscal Balance/GDP", "{:.1f}%"),
                ("unemployment", "Unemployment", "{:.1f}%"),
                ("credit_to_gdp_gap", "Credit-to-GDP Gap", "{:+.1f}pp"),
                ("sovereign_liability_to_reserves", "Sov Liab/Reserves", "{:.1f}x"), # NEW (Ratio)
                # Governance indicators from WGI (only 2 kept after correlation filtering)
                ("voice_accountability", "Voice & Accountability", "{:.0f}/100"),
                ("political_stability", "Political Stability", "{:.0f}/100"),
            ]
            for key, name, fmt in economic_features:
                val = actual_values.get(key)
                year = actual_values.get(f'{key}_year', '')
                year_str = ""
                if year:
                    try:
                        # Convert float year (2026.0) to int (2026)
                        year_int = int(float(year))
                        year_str = f" ({year_int})"
                    except:
                        year_str = f" ({year})"
                
                if val is not None and not pd.isna(val):
                    try:
                        formatted = fmt.format(val)
                    except:
                        formatted = f"{val:.2f}"
                    st.caption(f"✓ {name}: **{formatted}**{year_str}")
                else:
                    imp_val = imputed_values.get(key)
                    if imp_val is not None and not pd.isna(imp_val):
                        try:
                            formatted = fmt.format(imp_val)
                        except:
                            formatted = f"{imp_val:.2f}"
                        st.caption(f"≈ {name}: {formatted} (imputed)")
                    else:
                        st.caption(f"— {name}: (no data)")
        
        with col2:
            st.markdown("**Industry Pillar (50%)**")
            industry_features = [
                ("capital_adequacy", "Capital Adequacy", "{:.1f}%"),
                ("capital_quality", "Capital Quality (T1/CAR)", "{:.1f}%"),  # NEW
                ("npl_ratio", "NPL Ratio", "{:.1f}%"),
                ("roe", "Return on Equity", "{:.1f}%"),
                ("roa", "Return on Assets", "{:.2f}%"),
                ("liquid_assets_st_liab", "Liquid Assets/ST Liab", "{:.1f}%"),
                ("liquid_assets_total", "Liquid Assets/Total", "{:.1f}%"),
                ("sovereign_exposure_ratio", "Sovereign Exposure (Govt/Assets)", "{:.1f}%"),
                ("customer_deposits_loans", "Deposits/Loans", "{:.1f}%"),
                ("fx_loan_exposure", "FX Loan Exposure", "{:.1f}%"),
                ("loan_concentration", "Loan Concentration", "{:.1f}%"),
                ("real_estate_loans", "Real Estate Loans", "{:.1f}%"),
                ("bank_liability_to_nfa", "Bank Liab/NFA", "{:.1f}x"), # NEW (Ratio)
                # NEW FSIBSIS Features (2026-01)
                ("net_interest_margin", "Net Interest Margin", "{:.2f}%"),
                ("interbank_funding_ratio", "Interbank Funding Ratio", "{:.1f}%"),
                ("income_diversification", "Income Diversification", "{:.1f}%"),
                ("securities_to_assets", "Securities/Assets", "{:.1f}%"),
                ("npl_provisions", "NPL Coverage Ratio", "{:.1f}%"),
                ("large_exposure_ratio", "Large Exposure Ratio", "{:.1f}%"),
                ("deposit_funding_ratio", "Deposit Funding Ratio", "{:.1f}%"),
                # Note: regulatory_quality, rule_of_law, control_corruption dropped due to high correlation
            ]
            for key, name, fmt in industry_features:
                val = actual_values.get(key)
                year = actual_values.get(f'{key}_year', '')
                year_str = ""
                if year:
                    try:
                        year_int = int(float(year))
                        year_str = f" ({year_int})"
                    except:
                        year_str = f" ({year})"
                
                if val is not None and not pd.isna(val):
                    try:
                        formatted = fmt.format(val)
                    except:
                        formatted = f"{val:.2f}"
                    st.caption(f"✓ {name}: **{formatted}**{year_str}")
                else:
                    imp_val = imputed_values.get(key)
                    if imp_val is not None and not pd.isna(imp_val):
                        # Special check for Sovereign Exposure Proxy (matches updated data loader logic)
                        is_proxy = False
                        if key == 'sovereign_exposure_ratio':
                             sec_val = imputed_values.get('securities_to_assets')
                             if sec_val is not None and pd.notna(sec_val) and abs(imp_val - sec_val) < 0.1:
                                 is_proxy = True
                        
                        try:
                            formatted = fmt.format(imp_val)
                        except:
                            formatted = f"{imp_val:.2f}"
                            
                        if is_proxy:
                            st.caption(f"✓ {name}: **{formatted}** (Securities)")
                        else:
                            st.caption(f"≈ {name}: {formatted} (imputed)")
                    else:
                        st.caption(f"≈ {name}: *imputed*")
        
        # Sanity check - flag suspicious values
        sanity_bounds = {
            'capital_adequacy': (5, 50),   # Typical: 10-25%
            'tier1_capital': (5, 40),
            'npl_ratio': (0, 50),          # High NPL >30% is crisis level
            'roe': (-50, 50),
            'roa': (-5, 10),
            'liquid_assets_st_liab': (10, 200),
            'liquid_assets_total': (5, 80),
            'customer_deposits_loans': (20, 300),
            'fx_loan_exposure': (0, 80),
            'loan_concentration': (0, 200),  # % of capital, can exceed 100%
            'real_estate_loans': (0, 80),
            'sovereign_exposure_ratio': (0, 100), # Bank claims on govt / GDP
            'gdp_growth': (-20, 20),
            'inflation': (-10, 100),
            'unemployment': (0, 50),
            'govt_debt_gdp': (0, 300),
            'private_credit_to_gdp': (10, 250),
            'total_credit_to_gdp': (20, 400),
        }
        
        warnings = []
        for key, (low, high) in sanity_bounds.items():
            val = actual_values.get(key)
            if val is not None and not pd.isna(val):
                if val < low or val > high:
                    warnings.append(f"⚠️ {key}: {val:.1f}% outside expected range ({low}-{high}%)")
        
        # Compare dashboard values with model's stored values
        mismatches = []
        if model_features is not None and country_code:
            try:
                model_row = model_features[model_features['country_code'] == country_code]
                if len(model_row) > 0:
                    model_row = model_row.iloc[0]
                    for key in list(actual_values.keys()):
                        if key.endswith('_year'):
                            continue
                        dashboard_val = actual_values.get(key)
                        model_val = model_row.get(key) if key in model_row.index else None
                        
                        if dashboard_val is not None and model_val is not None:
                            if not pd.isna(dashboard_val) and not pd.isna(model_val):
                                diff = abs(dashboard_val - model_val)
                                if diff > 0.01 and diff / max(abs(model_val), 0.1) > 0.05:  # >5% difference
                                    mismatches.append(f"🔄 {key}: Dashboard={dashboard_val:.2f} vs Model={model_val:.2f}")
            except Exception:
                pass
        
        if warnings or mismatches:
            st.markdown("---")
            st.markdown("**Sanity Checks:**")
            for w in warnings[:3]:
                st.caption(w)
            if mismatches:
                st.caption("**Dashboard vs Model:**")
                for m in mismatches[:3]:
                    st.caption(m)
        
        # Show pillar weights
        if pca_info:
            st.markdown("---")
            econ_w = pca_info.get('economic_weight', 0.5) * 100
            ind_w = pca_info.get('industry_weight', 0.5) * 100
            st.caption(f"**Weights:** Economic Pillar: {econ_w:.0f}% | Industry Pillar: {ind_w:.0f}%")



def render_additional_data(country_code: str, loader, expanded: bool = False):
    """
    Renders an expandable section with additional dataset indicators.
    Allows toggling between FSIC, WEO, and MFS data.
    """
    with st.expander("📋 Additional Data (All Indicators)", expanded=expanded):
        # Dataset selector
        dataset_choice = st.radio(
            "Select Dataset",
            ["FSIC (Banking)", "WEO (Economic)", "MFS (Monetary)"],
            horizontal=True,
            key="additional_data_selector"
        )
        
        dataset_map = {
            "FSIC (Banking)": "FSIC",
            "WEO (Economic)": "WEO",
            "MFS (Monetary)": "MFS"
        }
        dataset = dataset_map[dataset_choice]
        
        render_time_series_deep_dive(country_code, loader, dataset)


# ==============================================================================
# DEEP DIVE TIME SERIES COMPONENTS
# ==============================================================================

def render_time_series_deep_dive(df: pd.DataFrame, dataset_name: str, country_code: str):
    """
    Renders a deep dive view for a specific dataset (WEO, FSIC, MFS).
    Features:
    - Indicator Selector (Dropdown)
    - Time-series Chart (Line)
    - Detailed Data Table (Bottom)
    
    Args:
        df: Pre-loaded DataFrame with columns: indicator_code, indicator_name, period, value
        dataset_name: Name of dataset (for display)
        country_code: Country code (for display)
    """
    if df is None or len(df) == 0:
        st.info(f"No {dataset_name} data available for {country_code}")
        return


    # 2. Process Data & Get Names
    # FSIC/FSIBSIS has many indicator_names per indicator_code (10 codes, 87+ names)
    # So for FSIC/FSI/FSIBSIS, we need to use indicator_name as the unique key
    # WEO and MFS have 1:1 or many:1 code:name ratios, so indicator_code works fine
    
    use_name_as_key = dataset_name in ("FSIC", "FSI", "FSIBSIS") and 'indicator_name' in df.columns
    
    if use_name_as_key:
        # For FSIC: Use indicator_name as the unique identifier
        options = df['indicator_name'].dropna().unique().tolist()
        # Small truncation in dropdown only for readability
        options_display = {name: name[:80] + "..." if len(name) > 80 else name for name in options}
        sorted_options = sorted(options, key=lambda x: x.lower())
    else:
        # For WEO/MFS: Use indicator_code as the unique identifier
        if 'indicator_name' in df.columns:
            mapping = df[['indicator_code', 'indicator_name']].dropna().drop_duplicates('indicator_code')
            name_map = dict(zip(mapping['indicator_code'], mapping['indicator_name']))
        else:
            name_map = {}

        def get_display_name(code):
            if code in name_map and str(name_map[code]) != code and str(name_map[code]).strip():
                 return str(name_map[code])
            return code.replace('_', ' ').title()

        options = df['indicator_code'].unique().tolist()
        options_display = {code: f"{get_display_name(code)} ({code})" for code in options}
        sorted_options = sorted(options, key=lambda x: options_display[x])
    
    # 3. UI Controls
    col_sel, col_range = st.columns([3, 1])
    
    with col_sel:
        selected_key = st.selectbox(
            "Select Indicator to Visualize",
            options=sorted_options,
            format_func=lambda x: options_display[x],
            key=f"dd_select_{dataset_name}"
        )
        
    with col_range:
         time_range = st.selectbox(
            "Time Range",
            ["5 Years", "10 Years", "20 Years", "All Data"],
            index=1, # Default 10 years
            key=f"dd_range_{dataset_name}"
        )

    # 4. Filter Data - use the appropriate column based on dataset type
    if use_name_as_key:
        chart_data = df[df['indicator_name'] == selected_key].copy()
    else:
        chart_data = df[df['indicator_code'] == selected_key].copy()
    
    # Parse period/date
    chart_data['date'] = pd.to_datetime(chart_data['period'].astype(str), errors='coerce')
    chart_data = chart_data.dropna(subset=['date', 'value']).sort_values('date')
    
    # Aggregate to ANNUAL data to avoid quarterly zigzag pattern
    # Extract year and take the last (most recent) value per year
    if len(chart_data) > 0:
        chart_data['year'] = chart_data['date'].dt.year
        chart_data = chart_data.groupby('year').agg({'date': 'last', 'value': 'last'}).reset_index()
    
    # Apply Time Filter
    if len(chart_data) > 0:
        max_date = chart_data['date'].max()
        if time_range == "5 Years":
            cutoff = max_date - pd.DateOffset(years=5)
            chart_data = chart_data[chart_data['date'] >= cutoff]
        elif time_range == "10 Years":
            cutoff = max_date - pd.DateOffset(years=10)
            chart_data = chart_data[chart_data['date'] >= cutoff]
        elif time_range == "20 Years":
            cutoff = max_date - pd.DateOffset(years=20)
            chart_data = chart_data[chart_data['date'] >= cutoff]


    # 5. Render Chart
    if len(chart_data) > 0:
        # For FSIC, selected_key IS the indicator name; for WEO/MFS, use get_display_name
        if use_name_as_key:
            # Use full name for chart title
            indicator_name_clean = selected_key
        else:
            indicator_name_clean = get_display_name(selected_key)
        
        fig = px.line(
            chart_data,
            x='date',
            y='value',
            markers=True,
            title=f"{indicator_name_clean} - Historical Trend"
        )
        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title=None,
            yaxis_title=None,
            # Trendline color
        )
        fig.update_traces(line_color='#60A5FA', line_width=2)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data found for this period.")

    # 6. Render Data Table (Pivot/Detailed)
    st.caption("Detailed Data")
    
    if len(chart_data) > 0:
        # Use 'year' column from aggregation (not 'period' which was lost during groupby)
        display_table = chart_data[['year', 'value']].copy()
        display_table.columns = ['Year', 'Value']
        display_table['Value'] = display_table['Value'].apply(lambda x: f"{x:,.2f}")
        display_table = display_table.sort_values('Year', ascending=False)
        
        # Always display as a simple vertical table
        st.dataframe(display_table, use_container_width=True, hide_index=True)




# ==============================================================================
# PREDICTION COMPONENTS
# ==============================================================================

def render_prediction_form(current_values: Dict[str, float] = None) -> Dict[str, float]:
    """
    Renders an input form for prediction with sliders and number inputs.
    Returns the user-entered values.
    """
    st.subheader("🔮 Predict Risk Score")
    st.caption("Enter hypothetical values to see predicted risk classification.")
    
    inputs = {}
    
    # Define input fields with ranges based on typical data
    input_fields = {
        "Economic Inputs": [
            ("gdp_per_capita", "GDP per Capita (USD)", 500, 200000, 20000, 100),
            ("gdp_growth", "GDP Growth (%)", -15.0, 15.0, 2.5, 0.1),
            ("inflation", "Inflation Rate (%)", -5.0, 50.0, 3.0, 0.1),
            ("unemployment", "Unemployment (%)", 0.0, 40.0, 5.0, 0.1),
            ("govt_debt_gdp", "Govt Debt/GDP (%)", 0.0, 300.0, 60.0, 1.0),
        ],
        "Banking Inputs": [
            ("capital_adequacy", "Capital Adequacy Ratio (%)", 5.0, 30.0, 15.0, 0.1),
            ("npl_ratio", "NPL Ratio (%)", 0.0, 30.0, 3.0, 0.1),
            ("roe", "Return on Equity (%)", -20.0, 30.0, 10.0, 0.1),
            ("liquid_assets_st_liab", "Liquid Assets/ST Liab (%)", 10.0, 300.0, 40.0, 1.0),
            ("sovereign_exposure_ratio", "Sovereign Exposure (Govt/GDP, %)", 0.0, 100.0, 10.0, 0.5), # NEW
        ],
        "Credit Inputs": [
            ("private_credit_to_gdp", "Private Credit to GDP (%)", 10.0, 250.0, 50.0, 1.0), # Renamed
            ("total_credit_to_gdp", "Total Credit to GDP (%)", 10.0, 350.0, 80.0, 1.0), # NEW
            ("credit_to_gdp_gap", "Credit-to-GDP Gap (pp)", -30.0, 30.0, 0.0, 0.5),
        ],
    }
    
    # Create columns for input groups
    for group_name, fields in input_fields.items():
        st.markdown(f"**{group_name}**")
        
        cols = st.columns(len(fields))
        for idx, (key, label, min_val, max_val, default, step) in enumerate(fields):
            with cols[idx]:
                # Use current value as default if available, clamped to valid range
                if current_values and key in current_values and not pd.isna(current_values.get(key)):
                    # Clamp the value to be within the min/max range to avoid StreamlitAPIException
                    default = max(min_val, min(max_val, float(current_values[key])))
                
                inputs[key] = st.number_input(
                    label,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default),
                    step=float(step),
                    key=f"predict_{key}"
                )
        
        st.markdown("")  # Spacing
    
    return inputs


def render_prediction_result(score: float, category: str, comparison_score: float = None):
    """
    Renders the prediction result with visual styling.
    """
    # Determine tier from score
    if score <= 2:
        tier = 1
    elif score <= 4:
        tier = 2
    elif score <= 6:
        tier = 3
    elif score <= 8:
        tier = 4
    else:
        tier = 5
    
    color = get_risk_color_hex(tier)
    label = get_risk_label(tier)
    
    st.markdown("---")
    st.markdown("### Prediction Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">Predicted Score</div>
            <div class="prediction-score" style="color: {color}">{score:.1f}/10</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">Risk Tier</div>
            <div class="prediction-tier" style="color: {color}">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">Category</div>
            <div class="prediction-category">{category}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Comparison if provided
    if comparison_score is not None:
        diff = score - comparison_score
        direction = "higher" if diff > 0 else "lower"
        st.info(f"This is **{abs(diff):.1f} points {direction}** than the current country score ({comparison_score:.1f})")


# ==============================================================================
# CHART COMPONENTS (Legacy + New)
# ==============================================================================

def render_drivers_chart(drivers: list):
    """Renders a bar chart of key risk drivers."""
    if not drivers:
        st.info("No drivers data available.")
        return

    df = pd.DataFrame(drivers)
    df = df.sort_values('z_score', ascending=True)
    colors = df['impact'].map({'risk': COLORS['danger'], 'strength': COLORS['success']})
    
    fig = go.Figure(go.Bar(
        x=df['z_score'],
        y=df['indicator'],
        orientation='h',
        marker=dict(color=colors)
    ))
    
    fig.update_layout(
        title="Key Risk Drivers (Deviation from Peer Median)",
        xaxis_title="Z-Score (Standard Deviations)",
        yaxis_title=None,
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_comparison_table(main_country: str, peers_df: pd.DataFrame, use_full_names: bool = False):
    """Renders a dense comparison table with optional full country names."""
    st.markdown("### Regional Peer Comparison")
    
    peers_df = peers_df.copy()
    
    # Use full names if available and requested
    if use_full_names and 'country_name_full' in peers_df.columns:
        cols = ['country_name_full', 'risk_score', 'risk_category', 'data_coverage']
        rename_map = {
            'country_name_full': 'Country',
            'risk_score': 'Risk Score',
            'risk_category': 'Category',
            'data_coverage': 'Data Coverage'
        }
    else:
        cols = ['country_code', 'risk_score', 'risk_tier', 'confidence']
        rename_map = {
            'country_code': 'Country',
            'risk_score': 'Risk Score', 
            'risk_tier': 'Tier',
            'confidence': 'Data Coverage'
        }
    
    # Handle data_coverage/confidence swap
    if 'data_coverage' in peers_df.columns and 'confidence' not in peers_df.columns:
        peers_df['confidence'] = peers_df['data_coverage']
    
    available_cols = [c for c in cols if c in peers_df.columns]
    display_df = peers_df[available_cols].copy()
    
    # Format data coverage as percentage
    if 'data_coverage' in display_df.columns:
        display_df['data_coverage'] = display_df['data_coverage'].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
    if 'confidence' in display_df.columns:
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
    
    # Rename columns for display
    display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})
    
    # Sort by risk score
    if 'Risk Score' in display_df.columns:
        display_df = display_df.sort_values('Risk Score').reset_index(drop=True)
    
    # Highlight main country
    try:
        if use_full_names and 'country_name_full' in peers_df.columns:
            main_name = peers_df[peers_df['country_code'] == main_country]['country_name_full'].iloc[0]
            main_idx = display_df[display_df['Country'] == main_name].index[0]
        else:
            main_idx = display_df[display_df['Country'] == main_country].index[0]
        
        st.dataframe(
            display_df.style.apply(
                lambda x: ['background-color: #30363D' if x.name == main_idx else '' for i in x],
                axis=1
            ),
            use_container_width=True,
            hide_index=True
        )
    except:
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_spider_chart(country_data: str, labels: List[str], values: List[float]):
    """Renders a radar/spider chart for component scores."""
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='Score'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        template="plotly_dark",
        height=300,
        margin=dict(t=20, b=20, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
