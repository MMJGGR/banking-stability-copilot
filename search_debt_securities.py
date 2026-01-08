"""
Check Kenya for S12CFSI (Credit to private sector)
and search MFS for Debt securities
"""
import pandas as pd

fsic_df = pd.read_parquet(r"c:\Users\Richard\Banking\cache\FSIC_cache.parquet")
mfs_df = pd.read_parquet(r"c:\Users\Richard\Banking\cache\MFS_cache.parquet")

print("=" * 70)

# 1. Check Kenya S12CFSI (Credit to private sector, Domestic currency)
print("\n--- Kenya: S12CFSI (Credit to private sector, Domestic currency) ---")
kenya_credit = fsic_df[(fsic_df['country_code'] == 'KEN') & 
                        (fsic_df['indicator_code'] == 'S12CFSI')]
if len(kenya_credit) > 0:
    latest = kenya_credit.sort_values('period').iloc[-1]
    print(f"Value: {latest['value']:,.0f} KES")
    print(f"Period: {latest['period']}")
else:
    print("No S12CFSI data for Kenya")

# 2. Search MFS for Debt securities
print("\n--- MFS: 'Debt securities' indicators ---")
mask = mfs_df['indicator_name'].str.contains('Debt securities', case=False, na=False)
if mask.any():
    results = mfs_df[mask].groupby(['indicator_code', 'indicator_name']).agg({
        'country_code': 'nunique'
    }).head(20)
    print(results.to_string())
    
    # Check Kenya specifically
    print("\n--- Kenya: MFS Debt securities ---")
    kenya_mfs = mfs_df[(mfs_df['country_code'] == 'KEN') & mask]
    if len(kenya_mfs) > 0:
        latest = kenya_mfs.sort_values('period').groupby('indicator_name').agg({
            'value': 'last',
            'period': 'last'
        })
        print(latest.to_string())
    else:
        print("No debt securities data for Kenya in MFS")
else:
    print("No 'Debt securities' indicators in MFS")

# 3. Alternative: Search for "securities" in MFS
print("\n--- MFS: 'securities' indicators (broader) ---")
mask2 = mfs_df['indicator_name'].str.contains('securities.*domestic|domestic.*securities', case=False, na=False, regex=True)
if mask2.any():
    results = mfs_df[mask2].groupby(['indicator_code', 'indicator_name']).agg({
        'country_code': 'nunique'
    }).head(10)
    print(results.to_string())
