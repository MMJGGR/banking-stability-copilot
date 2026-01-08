
import sys
import os
import pandas as pd
sys.path.append(os.getcwd())
from src.data_loader import IMFDataLoader
from src.config import FSIC_CORE_INDICATORS, WEO_CORE_INDICATORS, MFS_CORE_INDICATORS

def check_country_names():
    loader = IMFDataLoader()
    print("Loading datasets to check country names...")
    datasets = loader.load_all_datasets()
    
    for name, df in datasets.items():
        if df is None or df.empty:
            continue
            
        print(f"\n--- Checking {name} ---")
        if 'country_name' in df.columns:
            unique_names = df['country_name'].unique()
            print(f"Sample country names ({len(unique_names)} unique):")
            print(unique_names[:20])
            
            # Check for suspicious names (digits, dates)
            suspicious = [n for n in unique_names if str(n).replace('-','').replace('/','').isdigit() or 'Q' in str(n) or len(str(n)) < 3]
            if suspicious:
                print(f"⚠️ SUSPICIOUS NAMES FOUND in {name}:")
                print(suspicious[:20])
        else:
            print(f"No 'country_name' column in {name}")

def check_indicators():
    loader = IMFDataLoader()
    datasets = loader.load_all_datasets()
    
    all_codes = set()
    for name, df in datasets.items():
        if df is not None and not df.empty:
            all_codes.update(df['indicator_code'].unique())
            
    print(f"\nTotal unique indicator codes found: {len(all_codes)}")
    
    # Check against known mappings
    from app import ALL_INDICATOR_NAMES
    
    missing_names = [code for code in all_codes if code not in ALL_INDICATOR_NAMES]
    print(f"Indicators missing from mapping: {len(missing_names)}")
    if missing_names:
        print("Sample missing indicators:")
        print(missing_names[:20])

if __name__ == "__main__":
    check_country_names()
    check_indicators()
