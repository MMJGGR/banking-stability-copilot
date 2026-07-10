"""Check if KNN imputation is using country-specific neighbors."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np

# Load original and imputed features
original = pd.read_parquet('cache/crisis_features.parquet')
imputed = pd.read_parquet('cache/imputed_features.parquet')

# Set index
original = original.set_index('country_code')
imputed = imputed.set_index('country_code')

# Check specific features that should vary by country
test_features = ['npl_ratio', 'capital_adequacy', 'roe', 'liquid_assets_st_liab', 'deposit_funding_ratio']

print("="*80)
print("KNN IMPUTATION VARIATION CHECK")
print("="*80)

# Compare imputed values across diverse countries
diverse_countries = ['USA', 'VEN', 'JPN', 'NGA', 'KEN', 'DEU', 'BRA', 'CIV', 'BEN', 'SEN']

for feature in test_features:
    if feature not in imputed.columns:
        continue
    
    print(f"\n--- {feature} ---")
    
    # Count unique imputed values
    imputed_vals = []
    for country in diverse_countries:
        if country in imputed.index and country in original.index:
            orig_val = original.loc[country, feature] if feature in original.columns else np.nan
            imp_val = imputed.loc[country, feature]
            was_imputed = pd.isna(orig_val)
            
            if was_imputed:
                imputed_vals.append(imp_val)
                print(f"  {country}: {imp_val:.2f} (IMPUTED)")
            else:
                print(f"  {country}: {imp_val:.2f} (real data)")
    
    # Check variance in imputed values
    if len(imputed_vals) > 1:
        variance = np.var(imputed_vals)
        if variance < 0.01:
            print(f"  WARNING: Very low variance in imputed values ({variance:.4f})")
            print(f"  -> This suggests KNN is NOT working correctly!")
        else:
            print(f"  OK: Good variance in imputed values ({variance:.2f})")

# Also check: do all West African countries get similar values?
print("\n" + "="*80)
print("REGIONAL COHERENCE CHECK (West Africa)")
print("="*80)

west_africa = ['SEN', 'CIV', 'BEN', 'TGO', 'GIN', 'MLI', 'NER', 'BFA', 'GHA', 'NGA']
for feature in ['npl_ratio', 'capital_adequacy']:
    if feature not in imputed.columns:
        continue
    print(f"\n{feature}:")
    for country in west_africa:
        if country in imputed.index:
            val = imputed.loc[country, feature]
            was_real = country in original.index and pd.notna(original.loc[country, feature]) if feature in original.columns else False
            marker = "(real)" if was_real else "(imputed)"
            print(f"  {country}: {val:.2f} {marker}")
