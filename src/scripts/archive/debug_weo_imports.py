import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_loader import IMFDataLoader

def scan_weo_imports():
    print("Scanning WEO for Import indicators...")
    loader = IMFDataLoader()
    loader.load_from_cache()
    weo = loader._data_cache.get('WEO')
    
    # Search for "Imports" in indicator names if available, or just list common codes
    # WEO usually has:
    # TM_RP_CIF: Imports of Goods, Reported/CIF
    # BM: Imports of goods and services, debit
    
    print("\nChecking coverage for potential IMPORT codes:")
    possible_codes = ['BM', 'TM_RP_CIF', 'TMG_RP_CIF', 'TM', 'M']
    
    for code in possible_codes:
        subset = weo[weo['indicator_code'] == code]
        countries = subset['country_code'].nunique()
        print(f"  Code '{code}': {countries} countries")
        
    print("\nChecking Feature Engineering Logic (NFA):")
    # Verify NFA calculation return
    from src.feature_engineering import CrisisFeatureEngineer
    eng = CrisisFeatureEngineer()
    
    # Mock data check
    # We can't easily mock the full pipeline here, but we can check the merge_features signature
    # by inspecting the file content in the next step.

if __name__ == "__main__":
    scan_weo_imports()
