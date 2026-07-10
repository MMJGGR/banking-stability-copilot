import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_loader import IMFDataLoader

def scan_mfs_indicators():
    print("Scanning MFS for specific FX indicators...")
    loader = IMFDataLoader()
    loader.load_from_cache()
    mfs = loader._data_cache.get('MFS')
    
    # 1. Banking Sector Net Foreign Assets (ODCORP vs DCORP)
    # We used DCORP_NETAL_NFRA before. Let's look for ODCORP equivalent.
    print("\nSearching for ODCORP Net Foreign Assets:")
    nfa_codes = mfs[mfs['indicator_code'].str.contains('NETAL_NFRA', na=False)]['indicator_code'].unique()
    for c in nfa_codes:
        count = mfs[mfs['indicator_code'] == c]['country_code'].nunique()
        print(f"  {c}: {count} countries")

    # 2. Central Bank External Liabilities
    # Central Bank is S121 or FCS (Central Bank). 
    # Liabilities to Non-residents: L_LT_NRES or similar.
    print("\nSearching for Central Bank External Liabilities:")
    # Filter for S121 (Central Bank) + Liabilities + Non-Residents
    # Just search codes approx
    cb_liab = [c for c in mfs['indicator_code'].unique() if 'S121' in c and 'L_' in c and 'NRES' in c]
    for c in cb_liab:
        count = mfs[mfs['indicator_code'] == c]['country_code'].nunique()
        print(f"  {c}: {count} countries")

if __name__ == "__main__":
    scan_mfs_indicators()
