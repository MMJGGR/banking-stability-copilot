
"""
Evidence Generator: Credit-to-GDP Gap Analysis
==============================================
Purpose: Mathematically prove the divergence between the implemented "Median Gap" 
and the academic "HP Filter Gap" (Basel III standard).

Outputs:
- analysis_scripts/output/gap_analysis_chart.png
- analysis_scripts/output/gap_divergence_stats.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import IMFDataLoader
from src.feature_engineering import CrisisFeatureEngineer
from src.config import CACHE_DIR

OUTPUT_DIR = os.path.join("analysis_scripts", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def hp_filter(y, lamb=1600):
    """
    Hodrick-Prescott filter implementation using Scipy Sparse.
    Solves the minimization problem:
    min sum((y - trend)^2) + lambda * sum((trend_t+1 - 2*trend_t + trend_t-1)^2)
    """
    n = len(y)
    diag = np.ones(n)
    D = sparse.spdiags([diag, -2*diag, diag], [0, 1, 2], n-2, n)
    I = sparse.eye(n)
    # The optimization problem solution is (I + lambda * D'D) * trend = y
    # But D is (n-2) x n, so D.T @ D gives n x n matrix
    
    # Standard implementation:
    # 1. Construct second-difference matrix D
    # 2. Solve (I + lambda * D.T @ D) * x = y
    
    D_mat = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n-2, n))
    F = sparse.eye(n) + lamb * D_mat.T @ D_mat
    trend = spsolve(F, y)
    return trend

def run_gap_verification():
    print("Loading data...")
    loader = IMFDataLoader()
    if not loader.load_from_cache():
        loader.load_all_datasets()
        loader.save_cache()
        
    mfs_df = loader._data_cache.get('MFS')
    weo_df = loader._data_cache.get('WEO')
    
    engineer = CrisisFeatureEngineer()
    
    # 1. Get raw Credit-to-GDP ratios using current logic
    # We want the TIME SERIES of ratios, not just the latest value
    # Re-implementing the ratio calculation from feature_engineering but keeping time series
    
    print("Extracting time-series Credit-to-GDP ratios...")
    # Private Credit: DCORP_A_ACO_PS
    priv_mask = mfs_df['indicator_code'].str.contains('DCORP_A_ACO_PS', case=False, na=False)
    priv_data = mfs_df[priv_mask].copy()
    
    # GDP: NGDP
    gdp_mask = weo_df['indicator_code'] == 'NGDP'
    gdp_data = weo_df[gdp_mask].copy()
    
    # Merge on country, period (using year for simplicity if needed, but period is better)
    # Ensure periods match (they are usually strings like '2020' or '2020Q1'). WEO is annual '2020', MFS is monthly/quarterly/annual?
    # Let's standardize to Annual 'YYYY' for this analysis to match WEO frequency
    
    priv_data['year'] = pd.to_datetime(priv_data['period'], errors='coerce').dt.year
    gdp_data['year'] = pd.to_datetime(gdp_data['period'], errors='coerce').dt.year
    
    # Aggregate to Annual max value
    priv_annual = priv_data.groupby(['country_code', 'year'])['value'].max().reset_index()
    gdp_annual = gdp_data.groupby(['country_code', 'year'])['value'].max().reset_index()
    
    merged = pd.merge(priv_annual, gdp_annual, on=['country_code', 'year'], suffixes=('_credit', '_gdp'))
    
    # Calculate Ratio: (Credit / 1000) / GDP * 100
    merged['credit_to_gdp'] = (merged['value_credit'] / 1000 / merged['value_gdp']) * 100
    
    # Filter for valid range
    merged = merged[(merged['credit_to_gdp'] > 10) & (merged['credit_to_gdp'] < 300)]
    
    print(f"Computed ratios for {len(merged)} country-years.")
    
    # 2. Compute "Median Gap" (Current Implementation)
    # Gap = Value - Median(All Countries for that Year)
    yearly_medians = merged.groupby('year')['credit_to_gdp'].median()
    merged['median_global'] = merged['year'].map(yearly_medians)
    merged['gap_current_impl'] = merged['credit_to_gdp'] - merged['median_global']
    
    # 3. Compute "HP Filter Gap" (Academic Standard)
    # Gap = Value - HP_Trend(Value for that Country)
    merged['gap_hp_filter'] = np.nan
    merged['trend_hp'] = np.nan
    
    print("Applying HP Filter (lambda=100 for annual)...")
    for country in merged['country_code'].unique():
        country_df = merged[merged['country_code'] == country].sort_values('year')
        if len(country_df) > 10: # Need enough data for trend
            y = country_df['credit_to_gdp'].values
            try:
                trend = hp_filter(y, lamb=100) # lambda=100 is standard for annual data (Backus and Kehoe)
                gap = y - trend
                
                # Assign back
                merged.loc[country_df.index, 'trend_hp'] = trend
                merged.loc[country_df.index, 'gap_hp_filter'] = gap
            except Exception as e:
                print(f"HP filter failed for {country}: {e}")

    # 4. Visualization for Key Countries
    # focus_countries = ['USA', 'DEU', 'NGA', 'JPN']
    focus_countries = ['USA', 'DEU', 'NGA', 'TUR'] # TUR is interesting for crises
    
    plt.figure(figsize=(15, 10))
    
    for i, country in enumerate(focus_countries):
        data = merged[merged['country_code'] == country].sort_values('year')
        if len(data) == 0: continue
        
        ax = plt.subplot(2, 2, i+1)
        ax.plot(data['year'], data['gap_current_impl'], label='Current Impl (Median-based)', color='red', linestyle='--')
        ax.plot(data['year'], data['gap_hp_filter'], label='Basel Standard (HP Filter)', color='green')
        ax.axhline(0, color='black', alpha=0.3)
        ax.set_title(f"{country} Credit-to-GDP Gap Analysis")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gap_analysis_chart.png"))
    print(f"Saved chart to {os.path.join(OUTPUT_DIR, 'gap_analysis_chart.png')}")
    
    # 5. Save Stats
    stats = merged[['country_code', 'year', 'credit_to_gdp', 'gap_current_impl', 'gap_hp_filter']].dropna()
    stats.to_csv(os.path.join(OUTPUT_DIR, "gap_divergence_stats.csv"), index=False)
    
    # Correlation analysis
    corr = stats[['gap_current_impl', 'gap_hp_filter']].corr().iloc[0,1]
    print(f"\nCorrection between Implementation and Standard: {corr:.3f}")
    if corr < 0.5:
        print("CRITICAL: High divergence confirmed. The current implementation does NOT proxy the standard metric.")
    
if __name__ == "__main__":
    run_gap_verification()
