
import pandas as pd
import numpy as np
import os
from datetime import datetime
from src.data_loader import IMFDataLoader
from src.dashboard.components import WEO_INDICATORS, FSIC_NAME_PATTERNS, RESERVE_CURRENCY_COUNTRIES

# CONFIG
CACHE_DIR = 'cache'
OUTPUT_FILE = 'feature_comparison.csv'

# 1. LOAD MODEL FEATURES
try:
    model_features = pd.read_parquet(os.path.join(CACHE_DIR, 'crisis_features.parquet'))
    print(f"Loaded {len(model_features)} model feature rows.")
except Exception as e:
    print(f"ERROR: Could not load model features: {e}")
    exit(1)

# 2. SETUP DASHBOARD LOADER
loader = IMFDataLoader(cache_dir=CACHE_DIR)
loader.load_from_cache()

# Get list of countries from model
countries = sorted(model_features['country_code'].unique())

results = []

print(f"Processing {len(countries)} countries...")

for country in countries:
    # --- GET DASHBOARD VALUES ---
    dash_values = {}
    dash_periods = {}
    
    # WEO - Filtered
    weo_data = loader.get_country_data(country, 'WEO')
    nominal_gdp = None 
    
    if weo_data is not None and len(weo_data) > 0:
        weo_data['year'] = pd.to_datetime(weo_data['period']).dt.year
        current_year = datetime.now().year
        weo_data = weo_data[weo_data['year'] <= current_year]
        
        for feat, code in WEO_INDICATORS.items():
            matches = weo_data[weo_data['indicator_code'] == code]
            if len(matches) > 0:
                latest = matches.sort_values('year').iloc[-1]
                dash_values[feat] = latest['value']
                dash_periods[feat] = int(latest['year'])
                if feat == 'nominal_gdp':
                    nominal_gdp = latest['value']

    # FSIC
    fsic_data = loader.get_country_data(country, 'FSIC')
    if fsic_data is not None and len(fsic_data) > 0:
        fsic_data['year'] = pd.to_datetime(fsic_data['period']).dt.year
        for feat, pattern in FSIC_NAME_PATTERNS.items():
            matches = fsic_data[fsic_data['indicator_name'].str.contains(pattern, case=False, na=False, regex=True)]
            if len(matches) > 0:
                latest = matches.sort_values('period').iloc[-1]
                dash_values[feat] = latest['value']
                dash_periods[feat] = int(latest['year'])
    
    # FX Imputation
    if 'fx_loan_exposure' not in dash_values and country in RESERVE_CURRENCY_COUNTRIES:
        dash_values['fx_loan_exposure'] = 0.0
        dash_periods['fx_loan_exposure'] = datetime.now().year

    # MFS (Ratios) - Simplified emulation
    mfs_data = loader.get_country_data(country, 'MFS')
    if mfs_data is not None and len(mfs_data) > 0 and nominal_gdp:
        mfs_data['year'] = pd.to_datetime(mfs_data['period']).dt.year
        # Private Credit
        priv = mfs_data[mfs_data['indicator_code'].str.contains('DCORP_A_ACO_PS', case=False)]
        if len(priv) > 0:
            latest = priv.sort_values('period').iloc[-1]
            val = latest['value']
            ratio = (val / 1000 / nominal_gdp) * 100
            if 0 < ratio < 500:
                dash_values['private_credit_to_gdp'] = ratio
                dash_periods['private_credit_to_gdp'] = int(latest['year'])
        # Sovereign
        sov = mfs_data[mfs_data['indicator_code'].str.contains('DCORP_A_ACO_S13M1', case=False)]
        if len(sov) > 0:
            latest = sov.sort_values('period').iloc[-1]
            val = latest['value']
            ratio = (val / 1000 / nominal_gdp) * 100
            if 0 <= ratio < 200:
                dash_values['sovereign_exposure_ratio'] = ratio
                dash_periods['sovereign_exposure_ratio'] = int(latest['year'])

    # --- COMPARE ---
    model_row = model_features[model_features['country_code'] == country]
    if len(model_row) == 0: continue
    
    # Identify all features
    all_keys = set(list(dash_values.keys()) + [c for c in model_row.columns if c not in ['country_code', 'country_name', 'year', 'period'] and not c.endswith('_period')])
    all_keys.discard('nominal_gdp')
    all_keys.discard('credit_to_gdp') # Legacy alias
    
    # Inverted Features (FSIC)
    INVERTED_FEATURES = ['npl_ratio', 'fx_loan_exposure', 'loan_concentration', 'real_estate_loans']
    
    for key in all_keys:
        if key.endswith('_period'): continue
        
        # Model Data
        m_val = model_row[key].values[0] if key in model_row.columns else None
        m_period = model_row[f'{key}_period'].values[0] if f'{key}_period' in model_row.columns else None
        
        # Dash Data
        d_val = dash_values.get(key)
        d_period = dash_periods.get(key)
        
        # Inversion Handling for Display
        m_val_raw = m_val # Keep raw from parquet
        m_val_abs = abs(m_val) if (m_val is not None and key in INVERTED_FEATURES) else m_val
        
        results.append({
            'Country': country,
            'Feature': key,
            'Model_Value_Raw': m_val_raw,
            'Model_Value_Abs': m_val_abs,
            'Model_Period': m_period,
            'Dashboard_Value': d_val,
            'Dashboard_Period': d_period,
            'Diff': abs(m_val_abs - d_val) if (m_val_abs is not None and d_val is not None) else None
        })

df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Exported {len(df)} rows to {OUTPUT_FILE}")
