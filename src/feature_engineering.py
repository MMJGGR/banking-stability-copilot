"""
==============================================================================
Banking Crisis Early Warning System - Feature Engineering & EDA Module
==============================================================================
CRISP-DM Phase: Data Preparation

This module:
1. Computes credit-to-GDP gap (BIS methodology) 
2. Extracts debt service ratios from WEO
3. Engineers liquidity metrics from FSIC
4. Provides EDA visualizations for iterative improvement

Academic Foundation:
- Laeven & Valencia (2018) - Crisis definitions
- BIS (2019) - Credit-to-GDP gap as early warning indicator
- S&P BICRA - Two-pillar risk framework

Author: Banking Copilot
Date: 2025-01-02
==============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CACHE_DIR


class CrisisFeatureEngineer:
    """
    Feature engineering for banking crisis prediction.
    
    Implements academically-validated features:
    - Credit-to-GDP gap (BIS methodology)
    - Debt service ratio (external vulnerability)
    - Liquidity ratios (banking sector resilience)
    
    Follows CRISP-DM iterative process with built-in EDA.
    """
    
    # Reserve currency countries - FX exposure is minimal for these
    RESERVE_CURRENCY_COUNTRIES = ['USA', 'GBR', 'JPN', 'CHE', 'EMU', 'DEU', 'FRA', 'ITA', 'ESP', 'NLD', 'BEL', 'AUT']
    
    # Feature categories aligned with S&P BICRA two-pillar structure
    ECONOMIC_PILLAR_FEATURES = {
        # WEO-derived macro features
        'credit_to_gdp_gap': 'Credit-to-GDP gap (HP-filtered)',
        'private_credit_to_gdp': 'Private sector credit (% GDP)',  # NEW: explicit private credit
        'total_credit_to_gdp': 'Total domestic credit (% GDP)',    # NEW: total credit incl. govt
        'debt_service_gdp': 'External debt service (% GDP)',
        'external_debt_gdp': 'External debt (% GDP)',
        'current_account_gdp': 'Current account (% GDP)',
        'govt_debt_gdp': 'Government debt (% GDP)',
        'gdp_growth': 'Real GDP growth (%)',
        'inflation': 'Inflation rate (%)',
        # WGI governance indicators (BICRA: Economic Risk)
        'voice_accountability': 'Voice & Accountability (0-100)',
        'political_stability': 'Political Stability (0-100)',
        'govt_effectiveness': 'Government Effectiveness (0-100)',
    }
    
    INDUSTRY_PILLAR_FEATURES = {
        # FSIC-derived banking sector features
        'capital_adequacy': 'Regulatory capital ratio (%)',
        'npl_ratio': 'Non-performing loans (% gross loans)',
        'liquid_assets_st_liab': 'Liquid assets / ST liabilities (%)',
        'liquid_assets_total': 'Liquid assets / Total assets (%)',
        'roe': 'Return on equity (%)',
        'customer_deposits_loans': 'Customer deposits / Loans (%)',
        'fx_loan_exposure': 'FX-denominated loans (% total)',
        'loan_concentration': 'Loan concentration by activity (%)',
        'real_estate_loans': 'Residential real estate loans (% total loans)',  # BICRA: Real Estate Risk
        'sovereign_exposure_ratio': 'Bank claims on govt (% GDP)',  # NEW: Sovereign-bank nexus
        # WGI governance indicators (BICRA: Institutional Framework)
        'regulatory_quality': 'Regulatory Quality (0-100)',
        'rule_of_law': 'Rule of Law (0-100)',
        'control_corruption': 'Control of Corruption (0-100)',
    }
    
    def __init__(self, output_dir: str = None):
        """
        Initialize feature engineer.
        
        Args:
            output_dir: Directory for saving EDA visualizations
        """
        self.output_dir = output_dir or os.path.join(CACHE_DIR, 'eda')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.features_df = None
        self.eda_stats = {}
        
    # ==========================================================================
    # CRISP-DM: DATA PREPARATION - Feature Extraction
    # ==========================================================================
    
    def extract_weo_features(self, weo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract Economic Pillar features from WEO dataset.
        
        CRISP-DM: Data Preparation
        Mapping WEO indicators to crisis-predictive features per academic literature.
        """
        print("\n" + "="*70)
        print("EXTRACTING ECONOMIC PILLAR FEATURES (WEO)")
        print("="*70)
        
        if weo_df is None or len(weo_df) == 0:
            print("  WARNING: WEO data is empty")
            return pd.DataFrame()
        
        # WEO uses indicator_code column for data storage (not indicator_name)
        # Key: target feature name, Value: (indicator_code, higher_is_better)
        weo_mappings = {
            'gdp_growth': ('NGDP_RPCH', True),           # Real GDP growth %
            'gdp_per_capita': ('NGDPDPC', True),         # GDP per capita USD
            'inflation': ('PCPIPCH', False),              # CPI inflation %
            'current_account_gdp': ('BCA_NGDPD', True),  # Current account % GDP
            'govt_debt_gdp': ('GGXWDG_NGDP', False),     # Govt gross debt % GDP
            'fiscal_balance_gdp': ('GGXCNL_NGDP', True), # Fiscal balance % GDP
            'unemployment': ('LUR', False),               # Unemployment rate
            'nominal_gdp': ('NGDPD', True),              # Nominal GDP (for credit/GDP)
        }
        
        # Vectorized extraction: get latest value per country-indicator
        # IMPORTANT: Filter to historical data only (no forecasts)
        # Use current year as max to dynamically update without code changes
        from datetime import datetime
        max_data_year = datetime.now().year
        weo_df = weo_df.copy()
        weo_df['year'] = pd.to_datetime(weo_df['period']).dt.year
        weo_df = weo_df[weo_df['year'] <= max_data_year]
        print(f"  Filtered to data <= {max_data_year} (excluding forecasts)")
        
        weo_df = weo_df.sort_values('period')
        latest = weo_df.groupby(['country_code', 'indicator_code']).agg({
            'value': 'last',
            'period': 'last'
        }).reset_index()
        
        # Pivot to wide format for each mapped indicator
        features_list = []
        for feature_name, (code, _) in weo_mappings.items():
            code_data = latest[latest['indicator_code'] == code][['country_code', 'value', 'period']]
            code_data = code_data.rename(columns={'value': feature_name, 'period': f'{feature_name}_period'})
            features_list.append(code_data.set_index('country_code'))
        
        # Merge all features
        if features_list:
            weo_features = features_list[0]
            for df in features_list[1:]:
                weo_features = weo_features.join(df, how='outer')
            weo_features = weo_features.reset_index()
        else:
            weo_features = pd.DataFrame()
        
        print(f"  Extracted {len(weo_features.columns)-1} features for {len(weo_features)} countries")
        
        # EDA: Basic statistics
        self._print_feature_stats(weo_features, "WEO Features")
        
        return weo_features
    
    def extract_fsic_features(self, fsic_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract Industry Pillar features from FSIC dataset.
        
        CRISP-DM: Data Preparation
        Includes liquidity metrics as part of banking sector risk assessment.
        """
        print("\n" + "="*70)
        print("EXTRACTING INDUSTRY PILLAR FEATURES (FSIC)")
        print("="*70)
        
        if fsic_df is None or len(fsic_df) == 0:
            print("  WARNING: FSIC data is empty")
            return pd.DataFrame()
        
        # FSIC indicator mappings - Core FSI metrics
        # IMPORTANT: Patterns must include 'Percent' where relevant to avoid matching currency values
        fsic_mappings = {
            'capital_adequacy': ('Regulatory capital to risk-weighted assets.*Core FSI', True),
            'npl_ratio': ('Nonperforming loans to total gross loans.*Core FSI', False),
            'roe': ('Return on equity.*Core FSI', True),
            'roa': ('Return on assets.*Core FSI', True),
            'liquid_assets_st_liab': ('Liquid assets to short term liabilities.*Core FSI', True),
            'liquid_assets_total': ('Liquid assets to total assets.*Percent', True),
            'deposit_to_total_assets': ('Deposits to total.*assets.*Percent', True),  # NEW: Supplementary liquidity
            'customer_deposits_loans': ('Customer deposits to total.*loans.*Percent', True),
            'fx_loan_exposure': ('Foreign currency.*loans.*Percent', False),
            'tier1_capital': ('Tier 1 capital to risk-weighted assets.*Core FSI', True), # Fixed ambiguity
            'npl_provisions': ('Provisions to nonperforming loans.*Percent', True),
            'loan_concentration': ('Loan concentration.*Percent', False),
            'real_estate_loans': ('Residential real estate loans to total gross loans.*Core FSI', False),  # BICRA: Real Estate Risk
        }
        
        features_list = []
        
        for country in fsic_df['country_code'].unique():
            country_data = fsic_df[fsic_df['country_code'] == country]
            country_features = {'country_code': country}
            
            for feature_name, (pattern, _) in fsic_mappings.items():
                mask = country_data['indicator_name'].str.contains(
                    pattern, case=False, na=False, regex=True
                )
                if mask.any():
                    matched = country_data[mask].sort_values('period')
                    if len(matched) > 0:
                        country_features[feature_name] = matched['value'].iloc[-1]
                        country_features[f'{feature_name}_period'] = matched['period'].iloc[-1]
            
            if len(country_features) > 1:
                features_list.append(country_features)
        
        fsic_features = pd.DataFrame(features_list)
        
        # NOTE: features are kept in their natural units (e.g. NPL ratio is positive percent)
        # Directionality (higher is better vs lower is better) is handled during:
        # 1. Model Scaling/PCA phase
        # 2. Risk Scoring phase
        # This ensures the dashboard displays intuitive positive values.

        
        # CAPITAL QUALITY: Tier1/CAR ratio - measures equity-like capital composition
        # Higher is better: more Tier 1 (equity, reserves) vs Tier 2 (subordinated debt)
        # This ratio has near-zero correlation with CAR level itself (r=-0.01)
        if 'tier1_capital' in fsic_features.columns and 'capital_adequacy' in fsic_features.columns:
            # Calculate ratio where both are available
            mask = (fsic_features['tier1_capital'].notna() & 
                    fsic_features['capital_adequacy'].notna() &
                    (fsic_features['capital_adequacy'] != 0))
            fsic_features.loc[mask, 'capital_quality'] = (
                fsic_features.loc[mask, 'tier1_capital'] / 
                fsic_features.loc[mask, 'capital_adequacy'] * 100
            )
            # Clip to reasonable range (50-110% - can exceed 100% if T1 > CAR due to timing)
            fsic_features['capital_quality'] = fsic_features['capital_quality'].clip(50, 110)
            quality_count = fsic_features['capital_quality'].notna().sum()
            print(f"    Computed capital_quality (Tier1/CAR) for {quality_count} countries")
        
        print(f"  Extracted {len(fsic_features.columns)-1} features for {len(fsic_features)} countries")
        
        # EDA: Basic statistics
        self._print_feature_stats(fsic_features, "FSIC Features")
        
        return fsic_features
    
    def compute_credit_to_gdp_gap(self, mfs_df: pd.DataFrame, 
                                   weo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute credit-to-GDP metrics using BIS methodology.
        
        CRISP-DM: Data Preparation
        
        The credit-to-GDP gap is the single most robust early warning indicator
        for banking crises (BIS, 2019). It measures the deviation of the 
        credit-to-GDP ratio from its long-term trend.
        
        Methodology:
        1. Private Credit = MFS "DCORP_A_ACO_PS" (Claims on Private Sector)
        2. Total Credit = MFS "DCORP_A_ACO_S1_Z" (Total Domestic Credit)
        3. GDP = WEO nominal GDP (local currency)
        4. Gap = private_credit_to_gdp - median(private_credit_to_gdp)
        
        Returns:
            DataFrame with private_credit_to_gdp, total_credit_to_gdp, and credit_to_gdp_gap
        """
        print("\n" + "="*70)
        print("COMPUTING CREDIT-TO-GDP METRICS (BIS Methodology)")
        print("="*70)
        
        if mfs_df is None or weo_df is None:
            print("  WARNING: MFS or WEO data missing")
            return pd.DataFrame()
        
        # Get PRIVATE credit from MFS (DCORP_A_ACO_PS)
        private_credit_mask = mfs_df['indicator_code'].str.contains(
            'DCORP_A_ACO_PS', case=False, na=False, regex=False
        )
        private_credit_data = mfs_df[private_credit_mask].copy()
        
        # Get TOTAL domestic credit from MFS (DCORP_A_ACO_S1_Z or similar)
        # S1 = Domestic sectors, Z = Total
        total_credit_mask = mfs_df['indicator_code'].str.match(
            r'DCORP_A_ACO_S1(_Z)?$', case=False, na=False
        )
        total_credit_data = mfs_df[total_credit_mask].copy()
        
        # Get nominal GDP from WEO in LOCAL CURRENCY (NGDP)
        # Units: NGDP is in BILLIONS, MFS credit is in MILLIONS
        gdp_mask = weo_df['indicator_code'] == 'NGDP'
        gdp_data = weo_df[gdp_mask].copy()
        
        print(f"  Private credit data rows: {len(private_credit_data)}")
        print(f"  Total credit data rows: {len(total_credit_data)}")
        print(f"  GDP data rows (local currency): {len(gdp_data)}")
        
        if len(private_credit_data) == 0 or len(gdp_data) == 0:
            print("  WARNING: Could not find credit or GDP data")
            return pd.DataFrame()
        
        # Process each country
        results = []
        countries_processed = 0
        
        for country in private_credit_data['country_code'].unique():
            country_private = private_credit_data[private_credit_data['country_code'] == country]
            country_total = total_credit_data[total_credit_data['country_code'] == country]
            country_gdp = gdp_data[gdp_data['country_code'] == country]
            
            if len(country_private) > 0 and len(country_gdp) > 0:
                # Get latest values
                latest_private = country_private.sort_values('period')['value'].iloc[-1]
                latest_gdp = country_gdp.sort_values('period')['value'].iloc[-1]
                
                # Get total credit if available
                latest_total = None
                if len(country_total) > 0:
                    latest_total = country_total.sort_values('period')['value'].iloc[-1]
                
                if latest_gdp > 0 and not pd.isna(latest_private):
                    # UNIT CONVERSION: MFS in millions -> billions
                    private_in_billions = latest_private / 1000
                    private_ratio = (private_in_billions / latest_gdp) * 100
                    
                    # Sanity check: credit-to-GDP typically 20-300%
                    if 5 < private_ratio < 500:
                        result = {
                            'country_code': country,
                            'credit_to_gdp': private_ratio,  # Keep for backwards compat
                            'private_credit_to_gdp': private_ratio,
                        }
                        
                        # Add total credit if available
                        if latest_total is not None and not pd.isna(latest_total):
                            total_in_billions = latest_total / 1000
                            total_ratio = (total_in_billions / latest_gdp) * 100
                            if 5 < total_ratio < 600:  # Slightly higher range for total
                                result['total_credit_to_gdp'] = total_ratio
                        
                        results.append(result)
                        countries_processed += 1
        
        if len(results) > 0:
            gap_df = pd.DataFrame(results)
            
            # Compute gap as deviation from median (simplified BIS approach)
            median_ratio = gap_df['private_credit_to_gdp'].median()
            gap_df['credit_to_gdp_gap'] = gap_df['private_credit_to_gdp'] - median_ratio
            
            print(f"  Computed credit-to-GDP for {countries_processed} countries")
            print(f"  Median private credit/GDP: {median_ratio:.1f}%")
            print(f"  Gap range: {gap_df['credit_to_gdp_gap'].min():.1f} to {gap_df['credit_to_gdp_gap'].max():.1f}")
            if 'total_credit_to_gdp' in gap_df.columns:
                total_coverage = gap_df['total_credit_to_gdp'].notna().sum()
                print(f"  Total credit/GDP coverage: {total_coverage}/{len(gap_df)} countries")
            
            return gap_df
        
        return pd.DataFrame()
    
    def compute_sovereign_bank_nexus(self, mfs_df: pd.DataFrame, 
                                      weo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute sovereign-bank nexus feature from MFS.
        
        CRISP-DM: Data Preparation
        
        The sovereign-bank nexus measures how exposed banks are to government debt.
        High exposure creates a feedback loop: bank stress -> sovereign stress -> bank stress.
        
        Methodology (Updated 2026-01):
        1. Claims on Govt = Sum of DCORP and S121 claims on Central Government (S1311MIXED)
        2. Total Banking Assets = Sum of all claims (PS + S1311MIXED + S12R + S11001 + S13M1 + NRES)
        3. sovereign_exposure_ratio = Claims_on_Govt / Total_Banking_Assets * 100
        
        Note: This uses Total Banking Assets as denominator (not GDP) per BICRA methodology.
        This correctly reflects concentration risk within the banking sector.
        
        Returns:
            DataFrame with sovereign_exposure_ratio per country
        """
        print("\n" + "="*70)
        print("COMPUTING SOVEREIGN-BANK NEXUS (Claims on Govt / Total Banking Assets)")
        print("="*70)
        
        if mfs_df is None:
            print("  WARNING: MFS data missing")
            return pd.DataFrame()
        
        # Define sector codes for claims - EXCLUDE S121 (Central Bank)
        # S121 = Central Bank - doesn't create doom-loop risk (can't fail)
        # DCORP = All Depository Corporations, ODCORP = Other Depository Corps
        # We only want commercial bank exposure to sovereign debt
        govt_claim_codes = ['DCORP_A_ACO_S1311MIXED', 'ODCORP_A_ACO_S1311MIXED']
        
        # Build government claims per country
        govt_claims_dict = {}
        for code in govt_claim_codes:
            mask = mfs_df['indicator_code'].str.contains(code, case=False, na=False, regex=False)
            sector_data = mfs_df[mask].copy()
            latest = sector_data.sort_values('period').groupby('country_code')['value'].last()
            for country, val in latest.items():
                if pd.notna(val) and val > 0:
                    if country not in govt_claims_dict:
                        govt_claims_dict[country] = 0
                    govt_claims_dict[country] += val
        
        print(f"  Govt claims data: {len(govt_claims_dict)} countries")
        
        # Total domestic credit - use DCORP_A_ACO_S1 or S1_Z (already includes all sectors)
        # This is cleaner than summing individual sectors
        total_credit_codes = ['DCORP_A_ACO_S1_Z', 'DCORP_A_ACO_S1']
        
        total_credit_dict = {}
        for code in total_credit_codes:
            mask = mfs_df['indicator_code'].str.match(f'^{code}$', case=False, na=False)
            sector_data = mfs_df[mask].copy()
            latest = sector_data.sort_values('period').groupby('country_code')['value'].last()
            for country, val in latest.items():
                if pd.notna(val) and val > 0:
                    if country not in total_credit_dict:
                        total_credit_dict[country] = val
        
        print(f"  Total credit data: {len(total_credit_dict)} countries")
        
        # Compute ratio for countries with both metrics
        results = []
        countries_processed = 0
        
        for country in govt_claims_dict.keys():
            if country in total_credit_dict and total_credit_dict[country] > 0:
                govt_claims = govt_claims_dict[country]
                total_credit = total_credit_dict[country]
                
                ratio = (govt_claims / total_credit) * 100
                
                # Sanity check: sovereign exposure typically 0-80% of credit
                if 0 <= ratio <= 100:
                    results.append({
                        'country_code': country,
                        'sovereign_exposure_ratio': ratio,
                    })
                    countries_processed += 1
        
        if len(results) > 0:
            nexus_df = pd.DataFrame(results)
            
            print(f"  Computed sovereign exposure for {countries_processed} countries")
            print(f"  Median sovereign exposure: {nexus_df['sovereign_exposure_ratio'].median():.1f}%")
            print(f"  Range: {nexus_df['sovereign_exposure_ratio'].min():.1f}% to {nexus_df['sovereign_exposure_ratio'].max():.1f}%")
            
            return nexus_df
        
        return pd.DataFrame()
    
    def merge_features(self, weo_features: pd.DataFrame,
                       fsic_features: pd.DataFrame,
                       credit_gap: pd.DataFrame,
                       sovereign_nexus: pd.DataFrame = None,
                       wgi_features: pd.DataFrame = None,
                       fsibsis_features: pd.DataFrame = None) -> pd.DataFrame:
        """
        Merge all feature sets into unified country-level dataset.
        
        CRISP-DM: Data Preparation - Feature Integration
        
        Includes:
        - FX exposure imputation: Set to 0 for reserve currency countries
        - FSIBSIS new features integration (2026-01)
        - High correlation detection and handling
        """
        print("\n" + "="*70)
        print("MERGING FEATURE SETS")
        print("="*70)
        
        # Start with WEO features (broadest coverage)
        if len(weo_features) > 0:
            merged = weo_features.copy()
        else:
            merged = pd.DataFrame()
        
        # Merge FSIC features
        if len(fsic_features) > 0:
            if len(merged) > 0:
                merged = merged.merge(fsic_features, on='country_code', how='outer')
            else:
                merged = fsic_features.copy()
        
        # Merge credit-to-GDP gap and metrics
        if len(credit_gap) > 0:
            merged = merged.merge(credit_gap, on='country_code', how='outer')
        
        # Merge sovereign-bank nexus
        if sovereign_nexus is not None and len(sovereign_nexus) > 0:
            merged = merged.merge(sovereign_nexus, on='country_code', how='outer')
            print(f"  Merged sovereign-bank nexus feature")
        
        # Merge WGI governance indicators
        if wgi_features is not None and len(wgi_features) > 0:
            merged = merged.merge(wgi_features, on='country_code', how='outer')
            print(f"  Merged WGI governance features: {len(wgi_features.columns)-1} indicators")
        
        # Merge FSIBSIS features (NEW 2026-01: NIM, interbank funding, etc.)
        if fsibsis_features is not None and len(fsibsis_features) > 0:
            # FSIBSIS provides enhanced versions of some features - use as supplement
            # sovereign_exposure_fsibsis supplements sovereign_exposure_ratio
            # real_estate_loans_fsibsis supplements real_estate_loans
            merged = merged.merge(fsibsis_features, on='country_code', how='outer')
            print(f"  Merged FSIBSIS features: {len(fsibsis_features.columns)-1} new indicators")
            
            # Use FSIBSIS data to fill gaps in existing features where available
            if 'sovereign_exposure_ratio' in merged.columns:
                # 1. Fill gaps with FSIBSIS specific sovereign exposure metric
                if 'sovereign_exposure_fsibsis' in merged.columns:
                    gap_filled = merged['sovereign_exposure_ratio'].isna() & merged['sovereign_exposure_fsibsis'].notna()
                    merged.loc[gap_filled, 'sovereign_exposure_ratio'] = merged.loc[gap_filled, 'sovereign_exposure_fsibsis']
                    if gap_filled.sum() > 0:
                        print(f"    Filled {gap_filled.sum()} sovereign_exposure gaps with FSIBSIS data")
                
                # 2. Deep fallback: Use 'securities_to_assets' if exposure is still missing OR suspiciously low (< 2%)
                # Many emerging markets hold govt debt mainly as securities, not loans.
                if 'securities_to_assets' in merged.columns:
                    # Missing values
                    still_missing = merged['sovereign_exposure_ratio'].isna() & merged['securities_to_assets'].notna()
                    merged.loc[still_missing, 'sovereign_exposure_ratio'] = merged.loc[still_missing, 'securities_to_assets']
                    
                    # Low values (< 2%) but high securities (> 5%) -> Use securities (Likely data definition mismatch)
                    low_val_mask = (merged['sovereign_exposure_ratio'] < 2.0) & (merged['securities_to_assets'] > 5.0)
                    merged.loc[low_val_mask, 'sovereign_exposure_ratio'] = merged.loc[low_val_mask, 'securities_to_assets']
                    
                    if still_missing.sum() + low_val_mask.sum() > 0:
                        print(f"    Fixed {still_missing.sum() + low_val_mask.sum()} sovereign exposures using securities_to_assets (e.g. Kenya/Uganda)")

            
            if 'real_estate_loans' in merged.columns and 'real_estate_loans_fsibsis' in merged.columns:
                gap_filled = merged['real_estate_loans'].isna() & merged['real_estate_loans_fsibsis'].notna()
                merged.loc[gap_filled, 'real_estate_loans'] = merged.loc[gap_filled, 'real_estate_loans_fsibsis']
                if gap_filled.sum() > 0:
                    print(f"    Filled {gap_filled.sum()} real_estate_loans gaps with FSIBSIS data")
        
        # FX EXPOSURE IMPUTATION for reserve currency countries
        # These countries issue global reserve currencies, so FX mismatch risk is minimal
        if 'fx_loan_exposure' in merged.columns:
            fx_imputed = 0
            for country in self.RESERVE_CURRENCY_COUNTRIES:
                if country in merged['country_code'].values:
                    idx = merged[merged['country_code'] == country].index
                    if pd.isna(merged.loc[idx, 'fx_loan_exposure']).any():
                        merged.loc[idx, 'fx_loan_exposure'] = 0.0
                        fx_imputed += 1
            if fx_imputed > 0:
                print(f"  Imputed FX exposure = 0 for {fx_imputed} reserve currency countries")
        
        # LIQUIDITY CROSS-IMPUTATION
        liq_st = 'liquid_assets_st_liab'
        liq_total = 'liquid_assets_total'
        
        if liq_st in merged.columns and liq_total in merged.columns:
            both_avail = merged[[liq_st, liq_total]].dropna()
            
            if len(both_avail) > 10:
                median_ratio = (both_avail[liq_st] / both_avail[liq_total]).median()
                liq_imputed = 0
                
                missing_st = merged[liq_st].isna() & merged[liq_total].notna()
                if missing_st.any():
                    merged.loc[missing_st, liq_st] = merged.loc[missing_st, liq_total] * median_ratio
                    liq_imputed += missing_st.sum()
                
                missing_total = merged[liq_total].isna() & merged[liq_st].notna()
                if missing_total.any():
                    merged.loc[missing_total, liq_total] = merged.loc[missing_total, liq_st] / median_ratio
                    liq_imputed += missing_total.sum()
                
                if liq_imputed > 0:
                    print(f"  Liquidity cross-imputation: {liq_imputed} values filled (ratio={median_ratio:.2f})")
        
        # Filter to numeric columns only (drop period columns for modeling)
        numeric_cols = ['country_code'] + [
            c for c in merged.columns 
            if c != 'country_code' and not c.endswith('_period')
        ]
        merged = merged[numeric_cols]
        
        # HIGH CORRELATION DETECTION AND HANDLING
        merged = self._handle_high_correlations(merged)
        
        print(f"  Final dataset: {len(merged)} countries, {len(merged.columns)-1} features")
        print(f"  Feature completeness: {merged.notna().mean().mean()*100:.1f}%")
        
        self.features_df = merged
        return merged
    
    def _handle_high_correlations(self, df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
        """
        Detect and handle highly correlated features.
        
        Strategy:
        1. Identify pairs with |correlation| > threshold
        2. Keep the feature with better coverage (fewer missing values)
        3. Log dropped features for transparency
        
        Args:
            df: Feature DataFrame
            threshold: Correlation threshold (default 0.85)
            
        Returns:
            DataFrame with redundant features removed
        """
        print(f"\n  --- Correlation Analysis (threshold={threshold}) ---")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return df
        
        # Compute correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) > threshold:
                    high_corr_pairs.append((col1, col2, corr_val))
        
        if len(high_corr_pairs) == 0:
            print(f"    No feature pairs with |correlation| > {threshold}")
            return df
        
        print(f"    Found {len(high_corr_pairs)} highly correlated pairs:")
        
        # Track which features to drop
        features_to_drop = set()
        
        # Define feature priority (lower = more important, keep these)
        # Core features that should be preserved
        priority_features = {
            'credit_to_gdp_gap': 1,      # BIS validated
            'capital_adequacy': 1,        # Core FSI
            'npl_ratio': 1,               # Core FSI
            'sovereign_exposure_ratio': 2,
            'gdp_per_capita': 2,
            'gdp_growth': 2,
        }
        
        for col1, col2, corr in high_corr_pairs:
            print(f"      {col1} <-> {col2}: r={corr:.3f}")
            
            # Determine which to drop
            # Priority 1: Drop _fsibsis suffix versions if original exists
            if col1.endswith('_fsibsis') and not col2.endswith('_fsibsis'):
                to_drop = col1
            elif col2.endswith('_fsibsis') and not col1.endswith('_fsibsis'):
                to_drop = col2
            # Priority 2: Keep higher priority features
            elif priority_features.get(col1, 10) < priority_features.get(col2, 10):
                to_drop = col2
            elif priority_features.get(col2, 10) < priority_features.get(col1, 10):
                to_drop = col1
            # Priority 3: Keep feature with better coverage
            else:
                missing1 = df[col1].isna().sum()
                missing2 = df[col2].isna().sum()
                to_drop = col1 if missing1 > missing2 else col2
            
            features_to_drop.add(to_drop)
        
        if features_to_drop:
            print(f"    Dropping {len(features_to_drop)} redundant features: {sorted(features_to_drop)}")
            df = df.drop(columns=list(features_to_drop), errors='ignore')
        
        return df
    
    # ==========================================================================
    # CRISP-DM: EDA - Exploratory Data Analysis
    # ==========================================================================
    
    def run_eda(self, save_plots: bool = True) -> Dict:
        """
        Run exploratory data analysis on engineered features.
        
        CRISP-DM: Data Understanding (iterative)
        Generates visualizations to inform feature selection and transformation.
        """
        if self.features_df is None or len(self.features_df) == 0:
            print("No features to analyze. Run feature extraction first.")
            return {}
        
        print("\n" + "="*70)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        df = self.features_df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        eda_results = {
            'n_countries': len(df),
            'n_features': len(numeric_cols),
            'missing_rates': {},
            'distributions': {},
            'correlations': None,
        }
        
        # 1. Missing data analysis
        print("\n--- Missing Data Analysis ---")
        for col in numeric_cols:
            missing_rate = df[col].isna().mean() * 100
            eda_results['missing_rates'][col] = missing_rate
            if missing_rate > 30:
                print(f"  WARNING: {col} has {missing_rate:.1f}% missing")
        
        # 2. Distribution statistics
        print("\n--- Distribution Statistics ---")
        for col in numeric_cols[:10]:  # Top 10 features
            if df[col].notna().sum() > 5:
                eda_results['distributions'][col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'median': df[col].median(),
                    'skew': df[col].skew(),
                }
                skew = df[col].skew()
                if abs(skew) > 2:
                    print(f"  {col}: High skewness ({skew:.2f}) - consider log transform")
        
        # 3. Generate visualizations
        if save_plots and len(numeric_cols) > 0:
            self._plot_feature_distributions(df, numeric_cols[:8])
            self._plot_correlation_heatmap(df, numeric_cols)
            self._plot_missing_data(df, numeric_cols)
        
        self.eda_stats = eda_results
        return eda_results
    
    def _print_feature_stats(self, df: pd.DataFrame, title: str):
        """Print summary statistics for extracted features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n  {title} Summary:")
            print(f"    Countries: {len(df)}")
            print(f"    Features: {len(numeric_cols)}")
            for col in list(numeric_cols)[:5]:
                non_null = df[col].notna().sum()
                print(f"    - {col}: {non_null} values, "
                      f"mean={df[col].mean():.2f}, std={df[col].std():.2f}")
    
    def _plot_feature_distributions(self, df: pd.DataFrame, cols: List[str]):
        """Plot histograms for key features."""
        n_cols = min(len(cols), 8)
        if n_cols == 0:
            return
            
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, col in enumerate(cols[:8]):
            if df[col].notna().sum() > 3:
                axes[i].hist(df[col].dropna(), bins=20, edgecolor='black', alpha=0.7)
                axes[i].set_title(col, fontsize=10)
                axes[i].axvline(df[col].median(), color='red', linestyle='--', 
                               label=f'Median: {df[col].median():.1f}')
                axes[i].legend(fontsize=8)
        
        # Hide unused subplots
        for j in range(i+1, 8):
            axes[j].set_visible(False)
        
        plt.suptitle('Feature Distributions (CRISP-DM: Data Understanding)', fontsize=12)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'feature_distributions.png')
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"\n  Saved: {filepath}")
    
    def _plot_correlation_heatmap(self, df: pd.DataFrame, cols: List[str]):
        """Plot correlation matrix for feature selection."""
        numeric_df = df[cols].dropna(axis=1, how='all')
        if len(numeric_df.columns) < 3:
            return
            
        corr = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                    cmap='RdYlBu_r', center=0, ax=ax,
                    annot_kws={'size': 8})
        
        plt.title('Feature Correlations (CRISP-DM: Feature Selection)', fontsize=12)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"  Saved: {filepath}")
    
    def _plot_missing_data(self, df: pd.DataFrame, cols: List[str]):
        """Plot missing data patterns."""
        missing = df[cols].isna().mean() * 100
        missing = missing.sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if x < 20 else 'orange' if x < 50 else 'red' for x in missing]
        missing.plot(kind='barh', ax=ax, color=colors)
        
        ax.set_xlabel('Missing Rate (%)')
        ax.set_title('Missing Data by Feature (CRISP-DM: Data Quality)', fontsize=12)
        ax.axvline(20, color='orange', linestyle='--', label='20% threshold')
        ax.axvline(50, color='red', linestyle='--', label='50% threshold')
        ax.legend()
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'missing_data.png')
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"  Saved: {filepath}")


# =============================================================================
# MAIN: Run feature engineering pipeline
# =============================================================================

def run_feature_engineering_pipeline():
    """
    Main entry point for feature engineering.
    
    CRISP-DM: Data Preparation + EDA (iterative)
    """
    from src.data_loader import IMFDataLoader
    
    print("="*70)
    print("BANKING CRISIS EWS - FEATURE ENGINEERING PIPELINE")
    print("CRISP-DM Phase: Data Preparation")
    print("="*70)
    
    # Load data
    loader = IMFDataLoader()
    if not loader.load_from_cache():
        print("\nLoading from CSV files...")
        loader.load_all_datasets()
        loader.save_cache()
    
    fsic_df = loader._data_cache.get('FSIC', pd.DataFrame())
    weo_df = loader._data_cache.get('WEO', pd.DataFrame())
    mfs_df = loader._data_cache.get('MFS', pd.DataFrame())
    
    # Initialize feature engineer
    engineer = CrisisFeatureEngineer()
    
    # Extract features from each dataset
    weo_features = engineer.extract_weo_features(weo_df)
    fsic_features = engineer.extract_fsic_features(fsic_df)
    
    # Compute credit-to-GDP metrics (private + total + gap)
    credit_gap = engineer.compute_credit_to_gdp_gap(mfs_df, weo_df)
    
    # Compute sovereign-bank nexus
    sovereign_nexus = engineer.compute_sovereign_bank_nexus(mfs_df, weo_df)
    
    # Load FSIBSIS features (NEW 2026-01: NIM, interbank funding, etc.)
    fsibsis_features = None
    try:
        from src.data_loader_fsibsis import load_fsibsis_features
        import glob
        fsibsis_files = glob.glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), '*FSIBSIS*.csv'))
        if fsibsis_files:
            print("\n" + "="*70)
            print("LOADING FSIBSIS FEATURES (New Funding Stability Indicators)")
            print("="*70)
            fsibsis_features = load_fsibsis_features(fsibsis_files[0])
            print(f"  Loaded {len(fsibsis_features)} countries with {len(fsibsis_features.columns)-1} new indicators")
    except Exception as e:
        print(f"  WARNING: Could not load FSIBSIS features: {e}")
    
    # Merge all features (including FX imputation and correlation handling)
    features = engineer.merge_features(
        weo_features, fsic_features, credit_gap, 
        sovereign_nexus=sovereign_nexus,
        fsibsis_features=fsibsis_features
    )
    
    # Run EDA
    eda_results = engineer.run_eda(save_plots=True)
    
    # Save features for modeling phase
    features_path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
    features.to_parquet(features_path, index=False)
    print(f"\n  Saved features to: {features_path}")
    
    return features, eda_results


if __name__ == "__main__":
    features, eda = run_feature_engineering_pipeline()
