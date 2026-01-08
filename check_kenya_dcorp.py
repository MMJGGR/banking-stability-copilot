import pandas as pd

mfs_df = pd.read_parquet(r"c:\Users\Richard\Banking\cache\MFS_cache.parquet")

# Check Kenya's DCORP vs S121 data
kenya = mfs_df[mfs_df['country_code'] == 'KEN']

print("=== KENYA MFS DATA STRUCTURE ===")
print(f"Total Kenya records: {len(kenya)}")

# Check for DCORP codes
dcorp_mask = kenya['indicator_code'].str.contains('DCORP', case=False, na=False)
print(f"\nDCORP records: {dcorp_mask.sum()}")

# Check for ODCORP codes
odcorp_mask = kenya['indicator_code'].str.contains('ODCORP', case=False, na=False)
print(f"ODCORP records: {odcorp_mask.sum()}")

# Check for S121 codes (Central Bank)
s121_mask = kenya['indicator_code'].str.contains('S121', case=False, na=False)
print(f"S121 (Central Bank) records: {s121_mask.sum()}")

# Check for S122 codes (Other Depository Corps) - alternative sector code
s122_mask = kenya['indicator_code'].str.contains('S122', case=False, na=False)
print(f"S122 (Other Depository Corps) records: {s122_mask.sum()}")

# Show sample of Kenya's indicator codes
print("\n--- Sample Kenya indicator codes ---")
unique_codes = kenya['indicator_code'].value_counts().head(20)
for code, count in unique_codes.items():
    print(f"  {code}: {count}")

# Check what's available for govt claims
print("\n--- Kenya Govt-related indicator codes ---")
govt_related = kenya[kenya['indicator_name'].str.contains('Government|Govt', case=False, na=False)]
unique_govt = govt_related.groupby('indicator_code').agg({'indicator_name': 'first', 'country_code': 'count'}).head(15)
print(unique_govt.to_string())
