
import pandas as pd
import sys
import os
from src.data_loader import IMFDataLoader
from src.feature_engineering import CrisisFeatureEngineer

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_germany_missing():
    print("=" * 70)
    print("MISSING DATA DIAGNOSTIC: GERMANY (DEU)")
    print("=" * 70)
    
    # Load data
    loader = IMFDataLoader()
    loader.load_from_cache()
    
    fsic_df = loader._data_cache.get('FSIC', pd.DataFrame())
    weo_df = loader._data_cache.get('WEO', pd.DataFrame())
    mfs_df = loader._data_cache.get('MFS', pd.DataFrame())
    
    # Run engineering to get the exact columns used in model
    engineer = CrisisFeatureEngineer()
    
    # OPTIMIZATION: Filter for DEU only to debug quickly
    weo_deu = weo_df[weo_df['country_code'] == 'DEU']
    fsic_deu = fsic_df[fsic_df['country_code'] == 'DEU']
    
    weo_features = engineer.extract_weo_features(weo_deu)
    fsic_features = engineer.extract_fsic_features(fsic_deu)
    credit_gap = engineer.compute_credit_to_gdp_gap(mfs_df, weo_df)
    
    features = engineer.merge_features(weo_features, fsic_features, credit_gap)
    
    # Filter for Germany
    deu_data = features[features['country_code'] == 'DEU']
    
    if len(deu_data) == 0:
        print("Germany (DEU) not found in engineered features!")
        return
        
    deu_row = deu_data.iloc[0]
    
    # Define Industry (Banking) Columns
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
    
    print("\n--- Industry Pillar (Banking Sector) Status ---")
    missing_count = 0
    for col in industry_cols:
        val = deu_row.get(col, float('nan'))
        if pd.isna(val):
            print(f"  [MISSING] {col}")
            missing_count += 1
        else:
            print(f"  [PRESENT] {col}: {val}")
            
    print(f"\nSummary: Germany is missing {missing_count} out of {len(industry_cols)} Banking Indicators.")
    
    if missing_count == len(industry_cols):
        print("\nCONFIRMED: Germany has ZERO data for the Industry Pillar.")

if __name__ == "__main__":
    check_germany_missing()
