import pandas as pd
import numpy as np

mfs_df = pd.read_parquet(r"c:\Users\Richard\Banking\cache\MFS_cache.parquet")

# Check all Kenya MFS data containing "Govt" or "Government"
kenya = mfs_df[mfs_df['country_code'] == 'KEN']

print(f"Kenya MFS records: {len(kenya)}")
print("\n--- Kenya claims on government (any code) ---")

# Filter for Central Govt / Government in name
govt_mask = kenya['indicator_name'].str.contains('Claim.*Government|Government.*Claim', case=False, na=False, regex=True)
govt_data = kenya[govt_mask][['indicator_code', 'indicator_name', 'value', 'period']].drop_duplicates('indicator_code')
print(govt_data.to_string())

# Check for S13 codes (Government sector in SNA classification)
print("\n--- Kenya S13 (Government sector) codes ---")
s13_mask = kenya['indicator_code'].str.contains('S13', case=False, na=False)
s13_data = kenya[s13_mask].groupby(['indicator_code', 'indicator_name']).agg({'value': 'last', 'period': 'last'}).reset_index()
print(s13_data.to_string())

# Check for S121 (Central Bank sector) - often holds government claims
print("\n--- Kenya S121 claims on govt ---")
s121_govt = kenya[(kenya['indicator_code'].str.contains('S121', case=False, na=False)) & 
                   (kenya['indicator_name'].str.contains('Claim', case=False, na=False))]
s121_latest = s121_govt.sort_values('period').drop_duplicates('indicator_code', keep='last')
print(s121_latest[['indicator_code', 'indicator_name', 'value', 'period']].to_string())
