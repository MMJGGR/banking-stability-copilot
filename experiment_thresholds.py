
import pandas as pd
import numpy as np
import sys
import os
from src.data_loader import IMFDataLoader
from src.imputation import GapImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_experiment():
    print("=" * 70)
    print("DATA COMPLETENESS THRESHOLD EXPERIMENT")
    print("=" * 70)
    
    # Load data
    loader = IMFDataLoader()
    if not loader.load_from_cache():
        loader.load_all_datasets()
        loader.save_cache()
    
    fsic_df = loader._data_cache.get('FSIC', pd.DataFrame())
    weo_df = loader._data_cache.get('WEO', pd.DataFrame())
    mfs_df = loader._data_cache.get('MFS', pd.DataFrame())
    
    # helper to build matrix (simplified from train_model.py)
    from train_model import build_country_feature_matrix, identify_anchor_indicator
    
    features_df, country_names = build_country_feature_matrix(fsic_df, weo_df, mfs_df)
    
    if 'country_code' in features_df.columns:
        features_indexed = features_df.set_index('country_code')
    else:
        features_indexed = features_df

    # Calculate coverage
    numeric_cols = features_indexed.select_dtypes(include=[np.number]).columns
    data_coverage = features_indexed[numeric_cols].notna().mean(axis=1)
    
    # Key countries to track
    key_countries = ['USA', 'JPN', 'DEU', 'GBR', 'CHE', 'FRA', 'CAN', 'AUS',
                     'CHN', 'IND', 'BRA', 'MEX', 'IDN', 'TUR', 'ZAF',
                     'NGA', 'PAK', 'ARG', 'EGY', 'VEN', 'KEN']
    
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    print(f"\nTotal Indicators: {len(numeric_cols)}")
    
    for thresh in thresholds:
        print(f"\n--- Threshold: {thresh:.0%} Data Required ---")
        
        passed_countries = data_coverage[data_coverage >= thresh].index
        params_dropped = [c for c in key_countries if c not in passed_countries]
        params_kept = [c for c in key_countries if c in passed_countries]
        
        print(f"  Total Countries Kept: {len(passed_countries)} / {len(data_coverage)}")
        
        if params_dropped:
            print(f"  DROPPED Key Countries ({len(params_dropped)}): {', '.join(sorted(params_dropped))}")
        else:
            print(f"  DROPPED Key Countries: None")
            
        # Analyze coverage for dropped key countries
        if params_dropped:
            print("  Why dropped?")
            for c in params_dropped:
                cov = data_coverage.loc[c] if c in data_coverage.index else 0
                print(f"    - {c}: {cov:.1%} coverage")

if __name__ == "__main__":
    run_experiment()
