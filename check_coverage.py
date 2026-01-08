
import pandas as pd
from src.data_loader import IMFDataLoader

def check_coverage():
    loader = IMFDataLoader()
    if not loader.load_from_cache():
        loader.load_all_datasets()
    
    fsic = loader._data_cache.get('FSIC')
    
    for code in ['S14', 'REM']:
        df = fsic[fsic['indicator_code'] == code]
        n_countries = df['country_code'].nunique()
        latest_year = df['period'].max().year if not df.empty else 'N/A'
        print(f"Indicator {code}: {n_countries} countries, detailed coverage check...")
        
        # Check recent coverage (last 5 years)
        recent = df[df['period'] >= '2020-01-01']
        recent_countries = recent['country_code'].nunique()
        print(f"  Countries with data since 2020: {recent_countries}")

if __name__ == "__main__":
    check_coverage()
