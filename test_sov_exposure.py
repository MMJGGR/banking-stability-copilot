import pandas as pd
import numpy as np

# Load cached data
mfs_df = pd.read_parquet(r"c:\Users\Richard\Banking\cache\MFS_cache.parquet")

# Get claims on government (Central Government)
govt_claims_mask = mfs_df['indicator_code'].str.contains('DCORP_A_ACO_S1311MIXED', case=False, na=False, regex=False)
govt_claims_data = mfs_df[govt_claims_mask].copy()

# Get Total Domestic Credit (Claims on all domestic sectors)
# DCORP_A_ACO_S1_Z = Claims on Other sectors (approximation for total domestic credit assets)
# Better: Sum of all claims = PS + S1311MIXED + S12R + S11001 + S13M1
# Or: Use DCORP total assets if available

# Option 1: Sum claims on different sectors (more accurate)
# Sectors: PS (Private Sector), S1311MIXED (Central Govt), S12R (Other Financial Corps), S11001 (Public NFCs), S13M1 (State/Local Govt)
sector_codes = ['DCORP_A_ACO_PS', 'DCORP_A_ACO_S1311MIXED', 'DCORP_A_ACO_S12R', 'DCORP_A_ACO_S11001', 'DCORP_A_ACO_S13M1', 'DCORP_A_ACO_NRES']

# Build total claims per country
total_claims_dict = {}
for sector in sector_codes:
    mask = mfs_df['indicator_code'].str.contains(sector, case=False, na=False, regex=False)
    sector_data = mfs_df[mask].copy()
    # Get latest value per country
    latest = sector_data.sort_values('period').groupby('country_code')['value'].last()
    for country, val in latest.items():
        if country not in total_claims_dict:
            total_claims_dict[country] = 0
        if pd.notna(val):
            total_claims_dict[country] += val

# Create DataFrame
total_claims_df = pd.DataFrame(list(total_claims_dict.items()), columns=['country_code', 'total_claims'])

# Get latest govt claims per country
latest_govt = govt_claims_data.sort_values('period').groupby('country_code')['value'].last().reset_index()
latest_govt.columns = ['country_code', 'govt_claims']

# Merge
merged = latest_govt.merge(total_claims_df, on='country_code', how='inner')

# Compute ratio: Govt Claims / Total Claims * 100
merged['sov_exposure_pct_assets'] = (merged['govt_claims'] / merged['total_claims']) * 100

# Print results for key countries
key_countries = ['DEU', 'KEN', 'USA', 'NGA', 'GBR', 'JPN', 'BRA', 'ZAF', 'IND', 'TUR']

print("Sovereign Exposure Ratio (Claims on Govt / Total Banking Assets)")
print("=" * 70)
for country in key_countries:
    row = merged[merged['country_code'] == country]
    if len(row) > 0:
        ratio = row.iloc[0]['sov_exposure_pct_assets']
        govt = row.iloc[0]['govt_claims']
        total = row.iloc[0]['total_claims']
        print(f"{country}: {ratio:6.1f}% (Govt: {govt/1e6:.1f}M, Total: {total/1e6:.1f}M)")
    else:
        print(f"{country}: No data")

print("\n--- Full distribution ---")
print(f"Median: {merged['sov_exposure_pct_assets'].median():.1f}%")
print(f"Mean: {merged['sov_exposure_pct_assets'].mean():.1f}%")
print(f"Range: {merged['sov_exposure_pct_assets'].min():.1f}% - {merged['sov_exposure_pct_assets'].max():.1f}%")

# Sanity check: Kenya should be ~30% as per user
kenya = merged[merged['country_code'] == 'KEN']
if len(kenya) > 0:
    print(f"\n*** Kenya Validation: {kenya.iloc[0]['sov_exposure_pct_assets']:.1f}% (Expected ~30%) ***")
