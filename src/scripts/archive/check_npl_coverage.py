"""Check NPL Coverage Ratio calculation for validation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from src.data_loader import FSIBSISLoader

loader = FSIBSISLoader()
loader.load()

# Check a few countries for NPL and Provisions data
test_countries = ['USA', 'KEN', 'GBR', 'DEU', 'NGA', 'ZAF', 'BRA']

print("="*80)
print("NPL COVERAGE RATIO VALIDATION")
print("NPL Coverage = Specific Provisions / NPL * 100")
print("="*80)

for country in test_countries:
    country_data = loader.bank_data[loader.bank_data['country_code'] == country]
    
    if len(country_data) == 0:
        print(f"\n{country}: No FSIBSIS data")
        continue
    
    # Find NPL
    npl_rows = country_data[country_data['INDICATOR'].str.contains('Nonperforming loans, Domestic currency', na=False)]
    
    # Find Specific Provisions
    prov_rows = country_data[country_data['INDICATOR'].str.contains('Specific provisions, Assets, Domestic currency', na=False)]
    
    if len(npl_rows) == 0 or len(prov_rows) == 0:
        print(f"\n{country}: Missing NPL or Provisions in FSIBSIS")
        # Try to find what we have
        npl_any = country_data[country_data['INDICATOR'].str.contains('Nonperform', case=False, na=False)]
        prov_any = country_data[country_data['INDICATOR'].str.contains('provision', case=False, na=False)]
        if len(npl_any) > 0:
            print(f"  Available NPL indicators: {npl_any['INDICATOR'].unique()[:3]}")
        if len(prov_any) > 0:
            print(f"  Available Provisions indicators: {prov_any['INDICATOR'].unique()[:3]}")
        continue
    
    # Get latest year with both values
    npl_row = npl_rows.iloc[0]
    prov_row = prov_rows.iloc[0]
    
    # Find years with both
    year_cols = [c for c in country_data.columns if c.isdigit() and len(c) == 4]
    
    for year in sorted(year_cols, reverse=True):
        npl_val = npl_row.get(year)
        prov_val = prov_row.get(year)
        
        if pd.notna(npl_val) and pd.notna(prov_val) and npl_val > 0:
            coverage = (prov_val / npl_val) * 100
            print(f"\n{country} ({year}):")
            print(f"  NPL: {npl_val:,.0f}")
            print(f"  Provisions: {prov_val:,.0f}")
            print(f"  NPL Coverage Ratio: {coverage:.1f}%")
            
            # Sanity check
            if coverage > 200:
                print(f"  WARNING: Coverage > 200% is unusual")
            elif coverage < 30:
                print(f"  WARNING: Coverage < 30% indicates under-provisioning")
            else:
                print(f"  OK - Looks reasonable")
            break

# Also check what FSIC has for "Provisions to nonperforming loans" (the direct ratio)
print("\n" + "="*80)
print("CROSS-CHECK: FSIC Direct Ratio")
print("="*80)

fsic = pd.read_parquet('cache/FSIC_cache.parquet')
fsic_prov = fsic[fsic['indicator_name'].str.contains('Provisions to nonperforming loans.*Percent', case=False, na=False, regex=True)]

for country in test_countries:
    country_fsic = fsic_prov[fsic_prov['country_code'] == country]
    if len(country_fsic) > 0:
        latest = country_fsic.sort_values('period')['value'].iloc[-1]
        latest_period = country_fsic.sort_values('period')['period'].iloc[-1]
        print(f"{country}: FSIC NPL Coverage = {latest:.1f}% ({latest_period})")
    else:
        print(f"{country}: No FSIC NPL coverage data")
