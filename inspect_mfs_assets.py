import pandas as pd
import os
import sys

# Load cached MFS data
mfs_path = r"c:\Users\Richard\Banking\cache\MFS_cache.parquet"

df = pd.read_parquet(mfs_path)

# Much broader search - get unique codes and names
indicators = df.groupby(['indicator_code', 'indicator_name']).size().reset_index(name='count')
indicators = indicators.sort_values('count', ascending=False)

print(f"Total unique MFS indicators: {len(indicators)}")

# Option 1: Find Total Claims on all sectors (this is "Total Assets" equivalent)
# MFS code: DCORP_A_ACO_S1 = Depository Corps, Claims on ALL Domestic Sectors
# OR DCS_A_NFA_ZZ or similar

print("\n--- Searching for 'Claims on' indicators ---")
claims_mask = indicators['indicator_name'].str.contains('Claims on', case=False, na=False)
claims_data = indicators[claims_mask].head(30)
for _, row in claims_data.iterrows():
    print(f"{row['indicator_code']} | {row['indicator_name'][:80]} | {row['count']}")

# Option 2: Check the DCORP_A_ACO code variants
print("\n--- DCORP_A_ACO variants ---")
dcorp_mask = indicators['indicator_code'].str.contains('DCORP_A_ACO', case=False, na=False)
dcorp_data = indicators[dcorp_mask].head(20)
for _, row in dcorp_data.iterrows():
    print(f"{row['indicator_code']} | {row['indicator_name'][:70]} | {row['count']}")

# Option 3: Check for Kenya specifically (30% sovereign exposure claim)
print("\n--- Kenya's Claims on Government ---")
kenya = df[df['country_code'] == 'KEN']
kenya_govt = kenya[kenya['indicator_code'].str.contains('S13', case=False, na=False)]
print(kenya_govt[['indicator_code', 'indicator_name', 'value', 'period']].drop_duplicates().tail(10).to_string())

# Check Kenya's Total Domestic Credit
print("\n--- Kenya's Total Domestic Credit ---")
kenya_total = kenya[kenya['indicator_code'].str.contains('DCORP_A_ACO_S1', case=False, na=False)]
print(kenya_total[['indicator_code', 'indicator_name', 'value', 'period']].drop_duplicates().tail(10).to_string())
