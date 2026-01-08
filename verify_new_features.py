"""Verify new MFS features implementation."""
import sys
sys.path.insert(0, '.')

from src.feature_engineering import run_feature_engineering_pipeline

print("Running feature engineering pipeline...")
features, _ = run_feature_engineering_pipeline()

print("\n" + "="*70)
print("NEW FEATURES VERIFICATION")
print("="*70)

print(f"\nTotal countries: {len(features)}")
print(f"Total features: {len(features.columns) - 1}")

# Check new features
new_features = ['sovereign_exposure_ratio', 'private_credit_to_gdp', 'total_credit_to_gdp', 'deposit_to_total_assets']
print("\n--- NEW FEATURE COVERAGE ---")
for feat in new_features:
    if feat in features.columns:
        coverage = features[feat].notna().sum()
        pct = coverage / len(features) * 100
        print(f"  {feat}: {coverage}/{len(features)} ({pct:.0f}%)")
    else:
        print(f"  {feat}: NOT FOUND")

# Sample values for key countries
print("\n--- SAMPLE VALUES (USA, DEU, JPN) ---")
sample_countries = ['USA', 'DEU', 'JPN']
for code in sample_countries:
    row = features[features['country_code'] == code]
    if len(row) > 0:
        sov = row['sovereign_exposure_ratio'].values[0] if 'sovereign_exposure_ratio' in features.columns else None
        priv = row['private_credit_to_gdp'].values[0] if 'private_credit_to_gdp' in features.columns else None
        total = row['total_credit_to_gdp'].values[0] if 'total_credit_to_gdp' in features.columns else None
        fx = row['fx_loan_exposure'].values[0] if 'fx_loan_exposure' in features.columns else None
        
        sov_str = f"{sov:.1f}%" if sov and not pd.isna(sov) else "N/A"
        priv_str = f"{priv:.1f}%" if priv and not pd.isna(priv) else "N/A"
        total_str = f"{total:.1f}%" if total and not pd.isna(total) else "N/A"
        fx_str = f"{fx:.1f}%" if fx is not None and not pd.isna(fx) else "N/A"
        
        print(f"  {code}: sov_exp={sov_str}, priv_cred={priv_str}, total_cred={total_str}, fx_exp={fx_str}")
    else:
        print(f"  {code}: NOT FOUND")

import pandas as pd
# Check FX imputation for reserve currencies
print("\n--- FX IMPUTATION CHECK ---")
reserve_currencies = ['USA', 'GBR', 'JPN', 'CHE', 'DEU', 'FRA']
for code in reserve_currencies:
    row = features[features['country_code'] == code]
    if len(row) > 0 and 'fx_loan_exposure' in features.columns:
        fx = row['fx_loan_exposure'].values[0]
        status = "imputed=0" if fx == 0.0 else f"value={fx:.1f}" if not pd.isna(fx) else "missing"
        print(f"  {code}: {status}")

print("\n--- SUCCESS ---")
