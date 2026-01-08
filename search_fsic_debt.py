"""
Search FSIC for specific indicators:
1. Debt securities, Assets, Domestic currency
2. Credit to the private sector, Domestic currency
"""
import pandas as pd

fsic_df = pd.read_parquet(r"c:\Users\Richard\Banking\cache\FSIC_cache.parquet")

print("=" * 70)
print("FSIC Indicator Search")
print("=" * 70)

# Search 1: Debt securities
print("\n--- 1. 'Debt securities' + 'Assets' + 'Domestic' ---")
mask1 = fsic_df['indicator_name'].str.contains('Debt securities.*Assets.*Domestic|Domestic.*Debt securities', case=False, na=False, regex=True)
if mask1.any():
    results = fsic_df[mask1].groupby(['indicator_code', 'indicator_name']).agg({
        'country_code': 'nunique',
        'value': 'count'
    }).reset_index()
    print(results.to_string())
else:
    print("No exact match. Trying broader search...")
    mask1b = fsic_df['indicator_name'].str.contains('Debt securities', case=False, na=False)
    if mask1b.any():
        results = fsic_df[mask1b].groupby(['indicator_code', 'indicator_name']).agg({
            'country_code': 'nunique'
        }).head(15)
        print(results.to_string())

# Search 2: Credit to private sector
print("\n--- 2. 'Credit to the private sector' + 'Domestic' ---")
mask2 = fsic_df['indicator_name'].str.contains('Credit.*private.*Domestic|Domestic.*Credit.*private', case=False, na=False, regex=True)
if mask2.any():
    results = fsic_df[mask2].groupby(['indicator_code', 'indicator_name']).agg({
        'country_code': 'nunique',
        'value': 'count'
    }).reset_index()
    print(results.to_string())
else:
    print("No exact match. Trying broader search...")
    mask2b = fsic_df['indicator_name'].str.contains('Credit.*private sector', case=False, na=False, regex=True)
    if mask2b.any():
        results = fsic_df[mask2b].groupby(['indicator_code', 'indicator_name']).agg({
            'country_code': 'nunique'
        }).head(15)
        print(results.to_string())

# Check Kenya specifically
print("\n--- Kenya: All available indicators ---")
kenya = fsic_df[fsic_df['country_code'] == 'KEN']
print(f"Total Kenya FSIC records: {len(kenya)}")
print(f"Unique indicators: {kenya['indicator_name'].nunique()}")

# Check for debt/securities in Kenya
debt_mask = kenya['indicator_name'].str.contains('debt|securities|government|sovereign', case=False, na=False)
if debt_mask.any():
    print("\nKenya - Debt/Securities related indicators:")
    latest = kenya[debt_mask].sort_values('period').groupby('indicator_name').agg({
        'value': 'last',
        'period': 'last'
    })
    print(latest.to_string())
