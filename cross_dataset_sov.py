"""
Cross-dataset check for sovereign exposure across FSIC, WEO, MFS
Goal: Find plausible sovereign exposure figure for Kenya (~25-30% expected)
"""
import pandas as pd
import numpy as np

# Load all datasets
fsic_df = pd.read_parquet(r"c:\Users\Richard\Banking\cache\FSIC_cache.parquet")
weo_df = pd.read_parquet(r"c:\Users\Richard\Banking\cache\WEO_cache.parquet")
mfs_df = pd.read_parquet(r"c:\Users\Richard\Banking\cache\MFS_cache.parquet")

test_countries = ['KEN', 'NGA', 'DEU', 'USA', 'ZAF', 'TUR', 'JPN', 'BRA']

print("=" * 80)
print("CROSS-DATASET SOVEREIGN EXPOSURE CHECK")
print("=" * 80)

# === OPTION 1: FSIC - Direct sovereign exposure indicators ===
print("\n--- OPTION 1: FSIC Sovereign Exposure Indicators ---")

# Search for any govt/sovereign related FSIC indicators
fsic_sov_patterns = [
    'sovereign', 'government', 'public sector', 'claims on central', 
    'domestic debt', 'treasury', 'securities.*government'
]

for pattern in fsic_sov_patterns:
    mask = fsic_df['indicator_name'].str.contains(pattern, case=False, na=False, regex=True)
    if mask.any():
        print(f"\nPattern '{pattern}':")
        sample = fsic_df[mask].groupby('indicator_name').agg({
            'country_code': 'nunique', 
            'value': ['mean', 'count']
        }).head(5)
        print(sample.to_string())

# === OPTION 2: WEO - Govt Debt to GDP (approximate) ===
print("\n\n--- OPTION 2: WEO Government Debt Indicators ---")

weo_debt_codes = ['GGXWDG_NGDP', 'GGXCNL_NGDP', 'D']  # Govt debt % GDP
weo_debt = weo_df[weo_df['indicator_code'].isin(weo_debt_codes)]

for country in test_countries:
    latest = weo_debt[weo_debt['country_code'] == country].sort_values('period')
    if len(latest) > 0:
        val = latest['value'].iloc[-1]
        print(f"  {country} Govt Debt/GDP: {val:.1f}%")

# === OPTION 3: MFS - Alternative calculation methods ===
print("\n\n--- OPTION 3: MFS Alternative Calculations ---")

# For countries missing DCORP govt claims, try:
# 1. S122 (Other Depository Corps) instead of DCORP
# 2. Combine ODCORP only
# 3. Use total domestic credit and govt debt

for country in test_countries:
    country_mfs = mfs_df[mfs_df['country_code'] == country]
    
    # Check ODCORP govt claims
    odcorp_govt = country_mfs[country_mfs['indicator_code'].str.contains('ODCORP_A_ACO_S1311', case=False, na=False, regex=False)]
    if len(odcorp_govt) > 0:
        latest_odcorp = odcorp_govt.sort_values('period')['value'].iloc[-1]
        print(f"  {country} ODCORP Govt Claims: {latest_odcorp:,.0f}")
    
    # Check DCORP govt claims
    dcorp_govt = country_mfs[country_mfs['indicator_code'].str.contains('DCORP_A_ACO_S1311', case=False, na=False, regex=False)]
    if len(dcorp_govt) > 0:
        latest_dcorp = dcorp_govt.sort_values('period')['value'].iloc[-1]
        print(f"  {country} DCORP Govt Claims: {latest_dcorp:,.0f}")     
    
    # Check total domestic credit
    dcorp_total = country_mfs[country_mfs['indicator_code'].str.match('DCORP_A_ACO_S1(_Z)?$', case=False, na=False)]
    if len(dcorp_total) > 0:
        latest_total = dcorp_total.sort_values('period')['value'].iloc[-1]
        print(f"  {country} DCORP Total Credit: {latest_total:,.0f}")
    
    # Compute ratio if both available
    if len(dcorp_govt) > 0 and len(dcorp_total) > 0:
        ratio = (latest_dcorp / latest_total) * 100
        print(f"  {country} => Sovereign Exposure Ratio: {ratio:.1f}%")
    print()

# === OPTION 4: Kenya Specific - Use ODCORP only ===
print("\n\n--- OPTION 4: Kenya ODCORP-Only Calculation ---")
kenya_mfs = mfs_df[mfs_df['country_code'] == 'KEN']

# Get ODCORP govt claims
kenya_odcorp_govt = kenya_mfs[kenya_mfs['indicator_code'].str.contains('ODCORP_A_ACO_S1311', case=False, na=False)]
# Get ODCORP total assets
kenya_odcorp_total = kenya_mfs[kenya_mfs['indicator_code'].str.match('ODCORP_A_ACO', case=False, na=False)]

print(f"  Kenya ODCORP Govt Claims records: {len(kenya_odcorp_govt)}")
print(f"  Kenya ODCORP Total records: {len(kenya_odcorp_total)}")

if len(kenya_odcorp_govt) > 0:
    latest_govt = kenya_odcorp_govt.sort_values('period')['value'].iloc[-1]
    print(f"  Latest ODCORP Govt Claims: {latest_govt:,.0f}")
    
if len(kenya_odcorp_total) > 0:
    # Sum all asset codes
    latest_period = kenya_odcorp_total['period'].max()
    total_assets = kenya_odcorp_total[kenya_odcorp_total['period'] == latest_period]['value'].sum()
    print(f"  Total ODCORP Assets (sum): {total_assets:,.0f}")
    
    if len(kenya_odcorp_govt) > 0:
        ratio = (latest_govt / total_assets) * 100
        print(f"  Kenya ODCORP Sovereign Exposure: {ratio:.1f}%")
