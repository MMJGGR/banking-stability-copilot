"""Investigate risk score changes and imputation issues."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import pandas as pd
import numpy as np

# 1. Load model and check overall stats
print("="*70)
print("MODEL RISK SCORE ANALYSIS")
print("="*70)

model = pickle.load(open('cache/risk_model.pkl', 'rb'))
if hasattr(model, 'country_scores'):
    scores = model.country_scores
elif isinstance(model, dict) and 'country_scores' in model:
    scores = model['country_scores']
else:
    # Try loading directly
    scores = pd.read_pickle('cache/risk_model.pkl')
    if isinstance(scores, dict):
        scores = scores.get('country_scores', pd.DataFrame(scores))

print(f"Total countries: {len(scores)}")
print(f"Mean risk score: {scores['risk_score'].mean():.2f}")
print(f"Median risk score: {scores['risk_score'].median():.2f}")

# 2. Check specific countries mentioned
print("\n--- Specific Countries ---")
for country in ['VEN', 'CIV', 'BEN', 'SEN', 'NGA', 'KEN', 'USA', 'DEU']:
    c = scores[scores['country_code'] == country]
    if len(c) > 0:
        r = c.iloc[0]
        print(f"{country}: risk={r['risk_score']:.1f}, coverage={r['data_coverage']*100:.0f}%")

# 3. Check imputed features for Venezuela
print("\n--- Venezuela Imputation Check ---")
imputed = pd.read_parquet('cache/imputed_features.parquet')
original = pd.read_parquet('cache/crisis_features.parquet')

if 'VEN' in imputed['country_code'].values:
    ven_imp = imputed[imputed['country_code'] == 'VEN'].iloc[0]
    ven_orig = original[original['country_code'] == 'VEN'].iloc[0] if 'VEN' in original['country_code'].values else None
    
    print("Feature comparison (original vs imputed):")
    key_features = ['npl_ratio', 'capital_adequacy', 'roe', 'inflation', 'gdp_growth']
    for f in key_features:
        if f in imputed.columns:
            orig_val = ven_orig[f] if ven_orig is not None and f in original.columns else np.nan
            imp_val = ven_imp[f]
            is_imputed = pd.isna(orig_val) if ven_orig is not None else True
            print(f"  {f}: orig={orig_val}, imputed={imp_val}, was_imputed={is_imputed}")

# 4. Check what changed - aggregate codes removed?
print("\n--- Aggregate Codes Check ---")
aggregates_removed = ['G00', 'G11', 'G16', 'G20', 'G40', 'G50', 'G51', 'G60', 'G90', 'G99', 'GX1']
removed_count = 0
for code in aggregates_removed:
    if code not in scores['country_code'].values:
        removed_count += 1
print(f"Aggregate codes removed: {removed_count}/{len(aggregates_removed)}")

# 5. Distribution comparison
print("\n--- Risk Score Distribution ---")
print(f"1-2 (Very Low): {(scores['risk_score'] <= 2).sum()}")
print(f"3-4 (Low): {((scores['risk_score'] > 2) & (scores['risk_score'] <= 4)).sum()}")
print(f"5-6 (Moderate): {((scores['risk_score'] > 4) & (scores['risk_score'] <= 6)).sum()}")
print(f"7-8 (High): {((scores['risk_score'] > 6) & (scores['risk_score'] <= 8)).sum()}")
print(f"9-10 (Very High): {(scores['risk_score'] > 8).sum()}")
