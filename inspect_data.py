
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import IMFDataLoader
from src.feature_engineering import CrisisFeatureEngineer
from src.wgi_loader import WGILoader

def inspect_industry_data():
    print("Loading data...")
    loader = IMFDataLoader()
    loader.load_from_cache()
    
    fsic_df = loader._data_cache.get('FSIC')
    weo_df = loader._data_cache.get('WEO')
    mfs_df = loader._data_cache.get('MFS')
    
    print("Engineering features...")
    engineer = CrisisFeatureEngineer()
    weo_features = engineer.extract_weo_features(weo_df)
    fsic_features = engineer.extract_fsic_features(fsic_df)
    credit_gap = engineer.compute_credit_to_gdp_gap(mfs_df, weo_df)
    
    wgi_loader = WGILoader()
    wgi_features = wgi_loader.get_latest_scores()
    
    features = engineer.merge_features(weo_features, fsic_features, credit_gap, wgi_features)
    
    # Industry Columns used in train_model.py
    industry_cols = [
        'capital_adequacy', 'npl_ratio', 'roe', 'roa', 'liquid_assets_st_liab', 
        'liquid_assets_total', 'customer_deposits_loans', 'fx_loan_exposure', 
        'tier1_capital', 'npl_provisions', 'loan_concentration',
        'real_estate_loans',
        'regulatory_quality', 'rule_of_law', 'control_corruption'
    ]
    
    print("\n" + "="*80)
    print("USA vs NGA Industry Indicators")
    print("="*80)
    
    countries = ['USA', 'NGA']
    comparison = features[features['country_code'].isin(countries)].set_index('country_code')
    
    # Transpose for easier reading
    comp_T = comparison[industry_cols].T
    print(comp_T)
    
    # Check for missing values (imputation impact)
    print("\nMissing Values:")
    print(comp_T.isna())

if __name__ == "__main__":
    inspect_industry_data()
