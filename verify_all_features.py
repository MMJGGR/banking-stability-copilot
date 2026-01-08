
import pandas as pd
import numpy as np
import os
from src.data_loader import IMFDataLoader
from src.dashboard.components import WEO_INDICATORS, FSIC_NAME_PATTERNS

# CONFIG
COUNTRIES = ['KEN', 'NGA', 'USA', 'DEU', 'BRA']
CACHE_DIR = 'cache' # Relative to root
RESERVE_CURRENCY_COUNTRIES = ['USA', 'GBR', 'JPN', 'CHE', 'EMU', 'DEU', 'FRA', 'ITA', 'ESP', 'NLD', 'BEL', 'AUT']

# 1. LOAD MODEL FEATURES (The Truth Used for Training)
try:
    model_features = pd.read_parquet(os.path.join(CACHE_DIR, 'crisis_features.parquet'))
    print(f"Loaded {len(model_features)} model feature rows from {CACHE_DIR}/crisis_features.parquet")
except Exception as e:
    print(f"ERROR: Could not load model features: {e}")
    print(f"Try running src/feature_engineering.py first to generate them.")
    exit(1)

# 2. SETUP DASHBOARD LOADER
loader = IMFDataLoader(cache_dir=CACHE_DIR)
loader.load_from_cache()

# 3. VERIFICATION LOOP
print(f"\n{'COUNTRY':<8} | {'FEATURE':<30} | {'MODEL':<10} | {'DASHBOARD':<10} | {'DIFF':<10} | {'STATUS':<5} | {'EXPLANATION'}")
print("-" * 110)

discrepancies = []
from datetime import datetime

for country in COUNTRIES:
    # --- GET DASHBOARD VALUES (Emulate components.py logic) ---
    dash_values = {}
    
    # WEO - Filtered by Year
    weo_data = loader.get_country_data(country, 'WEO')
    nominal_gdp = None 
    
    if weo_data is not None and len(weo_data) > 0:
        weo_data['year'] = pd.to_datetime(weo_data['period']).dt.year
        # FILTER: Prevailing period only
        current_year = datetime.now().year
        weo_data = weo_data[weo_data['year'] <= current_year]
        
        for feat, code in WEO_INDICATORS.items():
            matches = weo_data[weo_data['indicator_code'] == code]
            if len(matches) > 0:
                latest = matches.sort_values('year').iloc[-1]
                dash_values[feat] = latest['value']
                if feat == 'nominal_gdp':
                    nominal_gdp = latest['value']

    # FSIC
    fsic_data = loader.get_country_data(country, 'FSIC')
    if fsic_data is not None and len(fsic_data) > 0:
        for feat, pattern in FSIC_NAME_PATTERNS.items():
            matches = fsic_data[fsic_data['indicator_name'].str.contains(pattern, case=False, na=False, regex=True)]
            if len(matches) > 0:
                latest = matches.sort_values('period').iloc[-1]
                dash_values[feat] = latest['value']
    
    # IMPUTATION: FX Loan Exposure
    if 'fx_loan_exposure' not in dash_values and country in RESERVE_CURRENCY_COUNTRIES:
        dash_values['fx_loan_exposure'] = 0.0

    # MFS (Ratios)

    # MFS (Ratios)
    mfs_data = loader.get_country_data(country, 'MFS')
    if mfs_data is not None and len(mfs_data) > 0 and nominal_gdp:
        # Private Credit
        priv = mfs_data[mfs_data['indicator_code'].str.contains('DCORP_A_ACO_PS', case=False)]
        if len(priv) > 0:
            val = priv.sort_values('period')['value'].iloc[-1]
            ratio = (val / 1000 / nominal_gdp) * 100
            if 0 < ratio < 500:
                dash_values['private_credit_to_gdp'] = ratio
        
        # Total Credit
        tot = mfs_data[mfs_data['indicator_code'].str.contains('DCORP_A_ACO_S1_Z', case=False)]
        if len(tot) > 0:
            val = tot.sort_values('period')['value'].iloc[-1]
            ratio = (val / 1000 / nominal_gdp) * 100
            if 0 < ratio < 600:
                dash_values['total_credit_to_gdp'] = ratio
                
        # Sovereign
        sov = mfs_data[mfs_data['indicator_code'].str.contains('DCORP_A_ACO_S13M1', case=False)]
        if len(sov) > 0:
            val = sov.sort_values('period')['value'].iloc[-1]
            ratio = (val / 1000 / nominal_gdp) * 100
            if 0 <= ratio < 200:
                dash_values['sovereign_exposure_ratio'] = ratio

    # --- COMPARE WITH MODEL ---
    model_row = model_features[model_features['country_code'] == country]
    if len(model_row) == 0:
        print(f"{country:<8} | ALL {'MISSING':<30} | Model has no data for this country")
        continue
    
    # Check all keys present in Dashboard logic OR Model logic
    all_keys = set(list(WEO_INDICATORS.keys()) + list(FSIC_NAME_PATTERNS.keys()) + 
                   ['private_credit_to_gdp', 'total_credit_to_gdp', 'sovereign_exposure_ratio'])
    
    # Remove keys intended to be ignored or internal
    all_keys.discard('nominal_gdp')
    
    # Features inverted in Model (FSIC only)
    INVERTED_FEATURES = ['npl_ratio', 'fx_loan_exposure', 'loan_concentration', 'real_estate_loans']
    
    for key in sorted(all_keys):
        # Model value
        m_val = model_row[key].values[0] if key in model_row.columns else None
        
        # Dash value
        d_val = dash_values.get(key)
        
        # Formating
        m_str = f"{m_val:.2f}" if m_val is not None and not pd.isna(m_val) else "N/A"
        d_str = f"{d_val:.2f}" if d_val is not None and not pd.isna(d_val) else "N/A"
        
        status = "OK"
        diff_str = "-"
        explanation = ""
        
        if m_val is None or pd.isna(m_val):
            if d_val is not None:
                status = "WARN"
                explanation = "Missing in Model (Imputed?)"
            else:
                status = "INFO"
                explanation = "Missing in Both"
        elif d_val is None or pd.isna(d_val):
             status = "INFO"
             explanation = "Dash Missing (Model Imputed)"
        else:
            # Handle Inversion
            if key in INVERTED_FEATURES:
                model_cmp = abs(m_val)
                # dash value is usually positive, but check sign
            else:
                model_cmp = m_val
                
            diff = abs(model_cmp - d_val)
            pct = diff / abs(d_val) if d_val != 0 else 0
            diff_str = f"{diff:.2f}"
            
            # Thresholds
            # WEO features might differ due to 2026 truncation vs Latest (2029)
            if key in WEO_INDICATORS:
                if diff < 2.0 or pct < 0.10: # 10% tolerance for Projections mismatch
                     status = "OK"
                     if diff > 0.1: explanation = "Time Mismatch (Benign)"
                else:
                    status = "FAIL"
                    explanation = "Large WEO Mismatch"
            elif key in INVERTED_FEATURES:
                if diff < 0.1: 
                    status = "OK"
                    explanation = "Inverted (Benign)"
                else:
                     status = "FAIL"
                     explanation = "Value Mismatch"
            else:
                if diff < 0.1 or pct < 0.05:
                    status = "OK"
                else:
                    status = "FAIL"
                    explanation = "Mismatch"
            
            if status == "FAIL":
                 discrepancies.append((country, key, m_val, d_val))
            
        print(f"{country:<8} | {key:<30} | {m_str:<10} | {d_str:<10} | {diff_str:<10} | {status:<5} | {explanation}")


print("-" * 90)
if discrepancies:
    print(f"\nFOUND {len(discrepancies)} DISCREPANCIES!")
else:
    print("\nSUCCESS: All features match within tolerance (or explained by imputation).")
