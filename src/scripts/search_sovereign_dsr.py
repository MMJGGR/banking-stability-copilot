"""Comprehensive search for sovereign debt service indicators across all datasets."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

print("="*80)
print("COMPREHENSIVE SOVEREIGN DSR INDICATOR SEARCH")
print("="*80)

# 1. WEO - Primary source for sovereign indicators
print("\n--- WEO: ALL SOVEREIGN/DEBT INDICATORS ---")
weo = pd.read_parquet('cache/WEO_cache.parquet')
weo_codes = weo[['indicator_code', 'indicator_name']].drop_duplicates()

for _, row in weo_codes.iterrows():
    name_lower = str(row['indicator_name']).lower()
    code = row['indicator_code']
    
    # Fiscal and debt keywords
    if any(kw in name_lower for kw in ['interest', 'debt', 'fiscal', 'deficit', 'revenue', 'expenditure', 'primary balance']):
        n = weo[weo['indicator_code'] == code]['country_code'].nunique()
        if n >= 50:  # Only show well-covered indicators
            print(f"  {code}: {n} countries")
            print(f"    {row['indicator_name'][:70]}")

# 2. Check what FSIC has for sovereign exposure
print("\n--- FSIC: SOVEREIGN-RELATED ---")
fsic = pd.read_parquet('cache/FSIC_cache.parquet')
fsic_names = fsic[['indicator_code', 'indicator_name']].drop_duplicates()

for _, row in fsic_names.iterrows():
    name_lower = str(row['indicator_name']).lower()
    if any(kw in name_lower for kw in ['government', 'sovereign', 'public sector']):
        n = fsic[fsic['indicator_code'] == code]['country_code'].nunique()
        print(f"  {row['indicator_name'][:70]}")

# 3. Calculate potential DSR proxy from WEO
print("\n" + "="*80)
print("SOVEREIGN DSR PROXY OPTIONS")
print("="*80)

# Check availability of key ingredients
key_indicators = {
    'GGR_NGDP': 'General government revenue (% GDP)',
    'GGXWDG_NGDP': 'General government gross debt (% GDP)',
    'GGXCNL_NGDP': 'General government net lending/borrowing (% GDP)',
    'GGX_NGDP': 'General government total expenditure (% GDP)',
    'NGDPDPC': 'GDP per capita',
    'GGXWDN_NGDP': 'General government net debt (% GDP)',
}

print("\nKey WEO indicators for DSR calculation:")
for code, name in key_indicators.items():
    matches = weo[weo['indicator_code'] == code]
    n = matches['country_code'].nunique() if len(matches) > 0 else 0
    print(f"  {code}: {n} countries - {name}")

# 4. Proposed DSR formula
print("\n" + "="*80)
print("PROPOSED SOVEREIGN DSR PROXY FORMULA")
print("="*80)
print("""
Option 1: Interest Burden Ratio (best available)
  Formula: (GGX_NGDP - GGXCNL_NGDP) / GGR_NGDP * 100
  Meaning: (Expenditure - Primary Balance) / Revenue
  Interpretation: % of revenue going to interest payments
  
Option 2: Debt Sustainability Indicator
  Formula: GGXWDG_NGDP / GGR_NGDP
  Meaning: Years of revenue to pay off debt
  Interpretation: Lower = more sustainable
  
Option 3: Use existing govt_debt_gdp + interest rate assumption
  Formula: govt_debt_gdp * assumed_interest_rate / GDP
  Problem: Requires interest rate data
""")

# 5. Check current model features for overlap
print("\n" + "="*80)
print("CURRENT SOVEREIGN FEATURES IN MODEL")
print("="*80)
features = pd.read_parquet('cache/crisis_features.parquet')
sovereign_cols = ['govt_debt_gdp', 'fiscal_balance_gdp', 'external_debt_gdp', 
                  'sovereign_exposure_ratio', 'sovereign_liability_to_reserves']
for col in sovereign_cols:
    if col in features.columns:
        n = features[col].notna().sum()
        print(f"  {col}: {n} countries")
