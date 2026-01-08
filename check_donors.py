
import pandas as pd
import numpy as np
import sys
import os
from src.data_loader import IMFDataLoader

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_donors():
    print("=" * 70)
    print("CHECKING FOR VALID KNN DONORS")
    print("=" * 70)
    
    # Load data
    loader = IMFDataLoader()
    loader.load_from_cache()
    
    fsic_df = loader._data_cache.get('FSIC', pd.DataFrame())
    weo_df = loader._data_cache.get('WEO', pd.DataFrame())
    mfs_df = loader._data_cache.get('MFS', pd.DataFrame())
    
    from train_model import build_country_feature_matrix, identify_anchor_indicator, BankingRiskModel
    
    # We need to replicate the feature set used in the model to measure RELEVANT coverage
    from src.feature_engineering import CrisisFeatureEngineer
    engineer = CrisisFeatureEngineer()
    
    weo_features = engineer.extract_weo_features(weo_df)
    fsic_features = engineer.extract_fsic_features(fsic_df)
    credit_gap = engineer.compute_credit_to_gdp_gap(mfs_df, weo_df)
    
    features = engineer.merge_features(weo_features, fsic_features, credit_gap)
    
    if 'country_code' not in features.columns:
        features = features.reset_index()
        
    # Get GDP per capita (Anchor)
    gdp_col = 'gdp_per_capita'
    
    # Identify Industry (FSIC) columns
    industry_cols = [
        'capital_adequacy', 'npl_ratio', 'roe', 'roa', 'liquid_assets_st_liab', 
        'liquid_assets_total', 'customer_deposits_loans', 'fx_loan_exposure', 
        'tier1_capital', 'npl_provisions'
    ]
    fsic_cols = [c for c in industry_cols if c in features.columns]
    
    print(f"Tracking {len(fsic_cols)} Industry Features")
    
    # Calculate Industry Coverage
    features['fsic_coverage'] = features[fsic_cols].notna().mean(axis=1)
    
    # Filter to Developed Economies (GDP per cap > $20,000 approx)
    # Note: Using raw GDP/cap might need scaling, but let's just sort
    developed = features[features[gdp_col] > 20000].copy()
    
    print(f"\nPotential Donors (GDP/cap > $20k): {len(developed)} countries")
    
    # Sort by coverage
    donors = developed.sort_values('fsic_coverage', ascending=False)
    
    print(f"{'Country':<10} {'GDP/Cap':<15} {'FSIC Coverage':<15} {'Name'}")
    print("-" * 60)
    for _, row in donors.head(20).iterrows():
        print(f"{row['country_code']:<10} ${row[gdp_col]:<14,.0f} {row['fsic_coverage']:.1%}            {row.get('country_name', '')}")
        
    # Check specific countries
    print("\nTarget Countries:")
    for code in ['DEU', 'GBR', 'USA', 'FRA', 'JPN', 'CAN', 'AUS']:
        row = features[features['country_code'] == code]
        if len(row) > 0:
            cov = row.iloc[0]['fsic_coverage']
            gdp = row.iloc[0][gdp_col]
            print(f"{code:<10} ${gdp:<14,.0f} {cov:.1%}")

if __name__ == "__main__":
    check_donors()
