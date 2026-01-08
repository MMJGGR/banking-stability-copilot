"""
Search for 'Claims on Government sector by deposit money banks' indicator
"""
import pandas as pd

# Load all datasets
fsic_df = pd.read_parquet(r"c:\Users\Richard\Banking\cache\FSIC_cache.parquet")
weo_df = pd.read_parquet(r"c:\Users\Richard\Banking\cache\WEO_cache.parquet")
mfs_df = pd.read_parquet(r"c:\Users\Richard\Banking\cache\MFS_cache.parquet")

search_term = "Claims on Government.*deposit money|deposit money.*claim.*government"

print("=" * 70)
print("Searching for 'Claims on Government sector by deposit money banks'")
print("=" * 70)

# Search in FSIC
print("\n--- FSIC Search ---")
mask = fsic_df['indicator_name'].str.contains(search_term, case=False, na=False, regex=True)
if mask.any():
    matches = fsic_df[mask].groupby(['indicator_code', 'indicator_name']).agg({
        'country_code': 'nunique'
    }).head(10)
    print(matches.to_string())
else:
    print("No matches in FSIC")

# Search in MFS
print("\n--- MFS Search ---")
mask = mfs_df['indicator_name'].str.contains(search_term, case=False, na=False, regex=True)
if mask.any():
    matches = mfs_df[mask].groupby(['indicator_code', 'indicator_name']).agg({
        'country_code': 'nunique'
    }).head(10)
    print(matches.to_string())
else:
    print("No exact matches in MFS, trying broader search...")
    # Try broader search
    mask2 = mfs_df['indicator_name'].str.contains('deposit.*money.*bank|DMB', case=False, na=False, regex=True)
    if mask2.any():
        matches = mfs_df[mask2].groupby(['indicator_code', 'indicator_name']).agg({
            'country_code': 'nunique'
        }).head(15)
        print(matches.to_string())

# Search in WEO
print("\n--- WEO Search ---")
mask = weo_df['indicator_name'].str.contains(search_term, case=False, na=False, regex=True)
if mask.any():
    matches = weo_df[mask].groupby(['indicator_code', 'indicator_name']).agg({
        'country_code': 'nunique'
    }).head(10)
    print(matches.to_string())
else:
    print("No matches in WEO")

# Check Kenya specifically for any "Claims on Central Government" indicators
print("\n--- Kenya: All 'Claims on Central Government' indicators ---")
kenya_all = pd.concat([
    fsic_df[fsic_df['country_code'] == 'KEN'],
    mfs_df[mfs_df['country_code'] == 'KEN']
])
mask = kenya_all['indicator_name'].str.contains('Claims on Central Government|Claims on.*Government', case=False, na=False, regex=True)
kenya_govt = kenya_all[mask].sort_values('period')
latest = kenya_govt.groupby(['indicator_code', 'indicator_name']).agg({
    'value': 'last',
    'period': 'last'
}).reset_index()
print(latest.to_string())
