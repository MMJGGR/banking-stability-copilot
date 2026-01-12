
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_engineering import CrisisFeatureEngineer
from src.data_loader import IMFDataLoader

def verify_and_fix():
    print("Loading data...")
    loader = IMFDataLoader()
    loader.load_from_cache()
    
    fe = CrisisFeatureEngineer()
    
    print("Extracting WEO features...")
    weo = loader._data_cache.get('WEO')
    weo_feats = fe.extract_weo_features(weo)
    
    year_cols = [c for c in weo_feats.columns if 'year' in c]
    print(f"WEO Year columns ({len(year_cols)}):")
    print(year_cols)
    
    if 'gdp_per_capita_year' in weo_feats.columns:
        print("SUCCESS: gdp_per_capita_year found in extraction.")
    else:
        print("FAILURE: gdp_per_capita_year NOT found in extraction.")
        
    print("  [1a] Extracting core IMF features (WEO, FSIC, MFS)...")
    weo_features = fe.extract_weo_features(weo)
    fsic_features = fe.extract_fsic_features(loader._data_cache.get('FSIC'))
    credit_gap = fe.compute_credit_to_gdp_gap(loader._data_cache.get('MFS'), weo)
    sovereign_nexus = fe.compute_sovereign_bank_nexus(loader._data_cache.get('MFS'), weo)
    
    # Load WGI
    from src.data_loader import WGILoader
    try:
        wgi_loader = WGILoader()
        wgi_features = wgi_loader.get_latest_scores()
    except Exception as e:
        print(f"WGI Error: {e}")
        wgi_features = None

    # Load FSIBSIS
    try:
        from src.data_loader import load_fsibsis_features
        fsibsis_features = load_fsibsis_features()
    except Exception as e:
        print(f"FSIBSIS Error: {e}")
        fsibsis_features = None

    print("Merging all features...")
    full_feats = fe.merge_features(
        weo_features=weo_features,
        fsic_features=fsic_features,
        credit_gap=credit_gap,
        sovereign_nexus=sovereign_nexus,
        wgi_features=wgi_features,
        fsibsis_features=fsibsis_features
    )
    
    full_year_cols = [c for c in full_feats.columns if 'year' in c]
    print(f"Full Feature Matrix Year columns ({len(full_year_cols)}):")
    # print(full_year_cols)
    
    if 'gdp_per_capita_year' in full_feats.columns:
        print("SUCCESS: gdp_per_capita_year found in full matrix.")
        
        # Determine strict save path
        from src.config import CACHE_DIR
        save_path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
        
        print(f"Saving corrected feature matrix to {save_path}...")
        full_feats.to_parquet(save_path)
        print("Saved.")
        
    else:
        print("FAILURE: gdp_per_capita_year lost during merge.")

if __name__ == "__main__":
    verify_and_fix()
