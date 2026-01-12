import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_loader import IMFDataLoader

def search_weo_imports():
    print("Searching WEO for Import indicators by DESCRIPTION...")
    loader = IMFDataLoader()
    loader.load_from_cache()
    weo = loader._data_cache.get('WEO')
    
    # Filter for unique indicators with 'Import' in description or name
    # We don't have descriptions loaded in the cache usually, just codes.
    # But wait, look at data_loader.py. 
    # Usually WEO cache has: country_code, indicator_code, period, value.
    # It might NOT have descriptions.
    
    # Plan B: Look at unique indicator codes that look like imports
    codes = weo['indicator_code'].unique()
    import_like = [c for c in codes if 'IMP' in c.upper() or 'TM' in c.upper() or 'BM' in c.upper()]
    
    print(f"Found {len(import_like)} potential codes:")
    for c in import_like:
        count = weo[weo['indicator_code'] == c]['country_code'].nunique()
        if count > 50:
            print(f"  {c}: {count} countries")

if __name__ == "__main__":
    search_weo_imports()
