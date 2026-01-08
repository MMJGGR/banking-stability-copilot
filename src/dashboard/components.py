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
    Extracts actual values from WEO, FSIC, and WGI data via the loader.
    Compares with model_features to show any differences from training data.
    Shows PCA pillar weights from pca_info.
    """
    
    # Build a dict of actual values from the loader
    actual_values = {}
    
    if loader and country_code:
        # Dynamic max year filter - exclude IMF forecasts
        from datetime import datetime
        max_data_year = datetime.now().year
        
        # Load WEO data and extract latest values
        try:
            weo_data = loader.get_country_data(country_code, 'WEO')
            if weo_data is not None and len(weo_data) > 0:
                # Filter to historical data only (exclude forecasts)
                weo_data = weo_data.copy()
                weo_data['year'] = pd.to_datetime(weo_data['period']).dt.year
                weo_data = weo_data[weo_data['year'] <= max_data_year]
                
                # Get latest value AND year per indicator
                latest_weo = weo_data.sort_values('period').groupby('indicator_code').agg({
                    'value': 'last',
                    'year': 'last'
                })
                for feature_name, indicator_code in WEO_INDICATORS.items():
                    if indicator_code in latest_weo.index:
                        actual_values[feature_name] = latest_weo.loc[indicator_code, 'value']
                        actual_values[f'{feature_name}_year'] = int(latest_weo.loc[indicator_code, 'year'])
                
                # Store nominal GDP for credit_to_gdp computation
                if 'NGDP' in latest_weo.index:
                    nominal_gdp = latest_weo.loc['NGDP', 'value']  # GDP in billions (local currency)
                else:
                    nominal_gdp = None
        except Exception as e:
            nominal_gdp = None
            
        # Load FSIC data and extract latest values (filter by name pattern)
        try:
            fsic_data = loader.get_country_data(country_code, 'FSIC')
            if fsic_data is not None and len(fsic_data) > 0:
                fsic_data = fsic_data.copy()
                fsic_data['year'] = pd.to_datetime(fsic_data['period']).dt.year
                for feature_name, name_pattern in FSIC_NAME_PATTERNS.items():
                    import re
                    matches = fsic_data[fsic_data['indicator_name'].str.contains(name_pattern, case=False, na=False, regex=True)]
                    if len(matches) > 0:
                        # Get the latest value and year
                        latest_row = matches.sort_values('period').iloc[-1]
                        actual_values[feature_name] = latest_row['value']
                        actual_values[f'{feature_name}_year'] = int(latest_row['year'])
        except Exception as e:
            pass
            
        # IMPUTATION: FX Loan Exposure for Reserve Currency Countries
        # If missing, we impute 0.0% as per feature_engineering.py logic
        if 'fx_loan_exposure' not in actual_values and country_code in RESERVE_CURRENCY_COUNTRIES:
            actual_values['fx_loan_exposure'] = 0.0
            actual_values['fx_loan_exposure_year'] = datetime.now().year
        
        # Compute credit and sovereign ratios from MFS + WEO (per feature_engineering.py logic)
        try:
            mfs_data = loader.get_country_data(country_code, 'MFS')
            if mfs_data is not None and len(mfs_data) > 0 and nominal_gdp is not None and nominal_gdp > 0:
                # 1. Private Credit: DCORP_A_ACO_PS (Claims on Private Sector)
                priv_mask = mfs_data['indicator_code'].str.contains('DCORP_A_ACO_PS', case=False, na=False)
                priv_data = mfs_data[priv_mask]
                if len(priv_data) > 0:
                    latest_priv = priv_data.sort_values('period')['value'].iloc[-1]  # in millions
                    # Convert credit from millions to billions, compute ratio
                    priv_ratio = (latest_priv / 1000 / nominal_gdp) * 100
                    if 0 < priv_ratio < 500:  # Sanity check
                        actual_values['private_credit_to_gdp'] = priv_ratio
                        actual_values['credit_to_gdp'] = priv_ratio # Backward compatibility

                # 2. Total Credit: DCORP_A_ACO_S1_Z (Claims on Real Sector)
                tot_mask = mfs_data['indicator_code'].str.contains('DCORP_A_ACO_S1_Z', case=False, na=False)
                tot_data = mfs_data[tot_mask]
                if len(tot_data) > 0:
                    latest_tot = tot_data.sort_values('period')['value'].iloc[-1]
                    tot_ratio = (latest_tot / 1000 / nominal_gdp) * 100
                    if 0 < tot_ratio < 600:
                        actual_values['total_credit_to_gdp'] = tot_ratio

                # 3. Sovereign Nexus: DCORP_A_ACO_S13M1 (Claims on Central Govt)
                sov_mask = mfs_data['indicator_code'].str.contains('DCORP_A_ACO_S13M1', case=False, na=False)
                sov_data = mfs_data[sov_mask]
                if len(sov_data) > 0:
                    latest_sov = sov_data.sort_values('period')['value'].iloc[-1]
                    sov_ratio = (latest_sov / 1000 / nominal_gdp) * 100
                    if 0 <= sov_ratio < 200:
                        actual_values['sovereign_exposure_ratio'] = sov_ratio

        except Exception as e:
            pass
        
        # Extract WGI governance indicators (harmonized with feature_engineering.py)
        if wgi_data is not None and len(wgi_data) > 0:
            try:
                country_wgi = wgi_data[wgi_data['country_code'] == country_code]
                if len(country_wgi) > 0:
                    # Get latest year for this country
                    latest_wgi = country_wgi.sort_values('year').iloc[-1]
                    wgi_year = int(latest_wgi['year'])
                    
                    # WGI columns used in model (from feature_engineering.py)
                    wgi_cols = {
                        'voice_accountability': 'voice_accountability',
                        'political_stability': 'political_stability',
                        'govt_effectiveness': 'govt_effectiveness',
                        'regulatory_quality': 'regulatory_quality',
                        'rule_of_law': 'rule_of_law',
                        'control_corruption': 'control_corruption'
                    }
                    for feature_name, col_name in wgi_cols.items():
                        if col_name in latest_wgi.index and pd.notna(latest_wgi[col_name]):
                            actual_values[feature_name] = latest_wgi[col_name]
                            actual_values[f'{feature_name}_year'] = wgi_year
            except Exception as e:
                pass

    # FSIBSIS-derived features: Pull from crisis_features.parquet directly since these are computed values
    # not available in the raw IMF API data, and model_features may be from older training.
    # Includes: NIM, interbank funding, income diversification, securities/assets, provisions, 
    # large exposures, deposit funding, capital_quality
    fsibsis_features = [
        'net_interest_margin', 'interbank_funding_ratio', 'income_diversification',
        'securities_to_assets', 'specific_provisions_ratio', 'large_exposure_ratio',
        'deposit_funding_ratio', 'sovereign_exposure_fsibsis', 'capital_quality',
        'credit_to_gdp_gap'  # Also computed in feature engineering
    ]
    
    # First try model_features (for features that exist there)
    if model_features is not None and country_code:
        try:
            model_row = model_features[model_features['country_code'] == country_code]
            if len(model_row) > 0:
                model_row = model_row.iloc[0]
                for feat in fsibsis_features:
                    if feat in model_row.index and pd.notna(model_row[feat]):
                        if feat not in actual_values:
                            actual_values[feat] = model_row[feat]
                            # Copy year metadata if available
                            year_col = f"{feat}_year"
                            if year_col in model_row.index and pd.notna(model_row[year_col]):
                                try:
                                    actual_values[year_col] = int(model_row[year_col])
                                except: pass
        except Exception:
            pass
    
    # Also load from crisis_features.parquet for FSIBSIS features not in model_features
    # This ensures new features show even before model retraining
    from src.config import CACHE_DIR
    import os
    parquet_path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
    if os.path.exists(parquet_path) and country_code:
        try:
            crisis_df = pd.read_parquet(parquet_path)
            crisis_row = crisis_df[crisis_df['country_code'] == country_code]
            if len(crisis_row) > 0:
                crisis_row = crisis_row.iloc[0]
                for feat in fsibsis_features:
                    # Logic 1: Always try to get Year metadata (because model_features might be stale and lack it)
                    year_col = f"{feat}_year"
                    if f"{feat}_year" not in actual_values:
                        if year_col in crisis_row.index and pd.notna(crisis_row[year_col]):
                            try:
                                actual_values[year_col] = int(crisis_row[year_col])
                            except: pass

                    # Logic 2: Get Value if missing
                    if feat not in actual_values:  # Don't override existing values
                        if feat in crisis_row.index and pd.notna(crisis_row[feat]):
                            actual_values[feat] = crisis_row[feat]
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
            ("unemployment", "Unemployment", "{:.1f}%"),
            ("govt_debt_gdp", "Govt Debt/GDP", "{:.1f}%"),
            ("fiscal_balance_gdp", "Fiscal Balance/GDP", "{:.1f}%"),
        ],
        "Banking Sector Health": [
            ("capital_adequacy", "Capital Adequacy Ratio", "{:.1f}%"),
            ("capital_quality", "Capital Quality (T1/CAR)", "{:.1f}%"),
            ("npl_ratio", "NPL Ratio", "{:.1f}%"),
            ("roe", "Return on Equity", "{:.1f}%"),
            ("sovereign_exposure_ratio", "Sovereign Exposure", "{:.1f}%"),
            ("net_interest_margin", "Net Interest Margin", "{:.2f}%"),
        ],
        "Liquidity & Funding": [
            ("liquid_assets_st_liab", "Liquid Assets/ST Liab", "{:.1f}%"),
            ("credit_to_gdp_gap", "Credit-to-GDP Gap", "{:+.1f}pp"),
            ("customer_deposits_loans", "Deposits/Loans", "{:.1f}%"),
            ("deposit_funding_ratio", "Deposit Funding Ratio", "{:.1f}%"),
        ],
    }
    
    # Create columns for the groups
    cols = st.columns(len(indicator_groups))
    
    # Load crisis_features.parquet to get imputed values
    from src.config import CACHE_DIR
    import os
    parquet_path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
    imputed_values = {}
    if os.path.exists(parquet_path) and country_code:
        try:
            crisis_df = pd.read_parquet(parquet_path)
            crisis_row = crisis_df[crisis_df['country_code'] == country_code]
            if len(crisis_row) > 0:
                imputed_values = crisis_row.iloc[0].to_dict()
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
                # Governance indicators from WGI
                ("voice_accountability", "Voice & Accountability", "{:.0f}/100"),
                ("political_stability", "Political Stability", "{:.0f}/100"),
                ("govt_effectiveness", "Govt Effectiveness", "{:.0f}/100"),
            ]
            for key, name, fmt in economic_features:
                val = actual_values.get(key)
                year = actual_values.get(f'{key}_year', '')
                year_str = f" ({year})" if year else ""
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
                # NEW FSIBSIS Features (2026-01)
                ("net_interest_margin", "Net Interest Margin", "{:.2f}%"),
                ("interbank_funding_ratio", "Interbank Funding Ratio", "{:.1f}%"),
                ("income_diversification", "Income Diversification", "{:.1f}%"),
                ("securities_to_assets", "Securities/Assets", "{:.1f}%"),
                ("specific_provisions_ratio", "Specific Provisions", "{:.2f}%"),
                ("large_exposure_ratio", "Large Exposure Ratio", "{:.1f}%"),
                ("deposit_funding_ratio", "Deposit Funding Ratio", "{:.1f}%"),
                # Governance
                ("regulatory_quality", "Regulatory Quality", "{:.0f}/100"),
                ("rule_of_law", "Rule of Law", "{:.0f}/100"),
                ("control_corruption", "Corruption Control", "{:.0f}/100"),
            ]
            for key, name, fmt in industry_features:
                val = actual_values.get(key)
                year = actual_values.get(f'{key}_year', '')
                year_str = f" ({year})" if year else ""
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
    # FSIC has many indicator_names per indicator_code (10 codes, 161 names)
    # So for FSIC, we need to use indicator_name as the unique key
    # WEO and MFS have 1:1 or many:1 code:name ratios, so indicator_code works fine
    
    use_name_as_key = dataset_name == "FSIC" and 'indicator_name' in df.columns
    
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
