
import pandas as pd
import sys
import os
from src.data_loader import IMFDataLoader
from src.feature_engineering import CrisisFeatureEngineer
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def identify_gaps():
    print("=" * 70)
    print("IDENTIFYING HIGH-VALUE DATA GAPS")
    print("=" * 70)
    
    # Load data
    loader = IMFDataLoader()
    # Ensure we use the latest cache (already rebuilt)
    loader.load_from_cache()
    
    fsic_df = loader._data_cache.get('FSIC', pd.DataFrame())
    weo_df = loader._data_cache.get('WEO', pd.DataFrame())
    mfs_df = loader._data_cache.get('MFS', pd.DataFrame())
    
    engineer = CrisisFeatureEngineer()
    weo_features = engineer.extract_weo_features(weo_df)
    fsic_features = engineer.extract_fsic_features(fsic_df)
    credit_gap = engineer.compute_credit_to_gdp_gap(mfs_df, weo_df)
    
    features = engineer.merge_features(weo_features, fsic_features, credit_gap)
    
    # Identify pillars
    gdp_col = 'nominal_gdp' # Raw GDP size in USD
    
    # If nominal_gdp not in features (it's in Weo features 'nominal_gdp'), check columns
    if 'nominal_gdp' not in features.columns:
        print("Error: Nominal GDP not found in features.")
        return

    # Identify FSIC columns
    industry_cols = [
        'capital_adequacy', 
        'npl_ratio', 
        'roe', 
        'roa', 
        'liquid_assets_st_liab', 
        'liquid_assets_total', 
        'customer_deposits_loans', 
        'fx_loan_exposure', 
        'tier1_capital', 
        'npl_provisions'
    ]
    fsic_cols = [c for c in industry_cols if c in features.columns]
    
    print(f"Tracking {len(fsic_cols)} Banking Indicators.")
    
    # Calculate Coverage
    features['fsic_coverage'] = features[fsic_cols].notna().mean(axis=1)
    
    # Sort by GDP Magnitude to prioritize relevant economies
    # (High GDP + Low Coverage = Priority)
    
    # Get country names
    names = weo_df[['country_code', 'country_name']].drop_duplicates().set_index('country_code')
    features = features.join(names, on='country_code')
    
    # Filter: Coverage < 50%
    missing_data = features[features['fsic_coverage'] < 0.5].copy()
    
    # Sort by GDP
    missing_data = missing_data.sort_values('nominal_gdp', ascending=False)
    
    print(f"\nFound {len(missing_data)} countries with < 50% Banking Data.")
    print("\nTOP PRIORITY CANDIDATES (Sorted by GDP Size):")
    print("-" * 80)
    print(f"{'Code':<6} {'Country Name':<25} {'GDP (Billions)':<15} {'Coverage':<10} {'Data Status':<15}")
    print("-" * 80)
    
    for _, row in missing_data.head(20).iterrows():
        gdp_val = row['nominal_gdp']
        gdp_str = f"${gdp_val:,.1f} B" if pd.notna(gdp_val) else "N/A"
        cov = row['fsic_coverage']
        
        status = "CRITICAL (0%)" if cov == 0 else f"PARTIAL ({cov:.0%})"
        
        print(f"{row['country_code']:<6} {str(row['country_name'])[:25]:<25} {gdp_str:<15} {cov:.1%}      {status:<15}")

if __name__ == "__main__":
    identify_gaps()
