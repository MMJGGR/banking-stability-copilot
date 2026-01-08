"""
Check FSIC 'Sectoral distribution of investments' indicator for sovereign exposure
"""
import pandas as pd

fsic_df = pd.read_parquet(r"c:\Users\Richard\Banking\cache\FSIC_cache.parquet")

# Find the sectoral distribution indicator
mask = fsic_df['indicator_name'].str.contains('Sectoral distribution.*Central Government', case=False, na=False, regex=True)
govt_invest = fsic_df[mask].copy()

print("=== FSIC Sectoral Distribution: Central Government ===")
print(f"Total records: {len(govt_invest)}")
print(f"Countries: {govt_invest['country_code'].nunique()}")

# Get latest per country
latest = govt_invest.sort_values('period').groupby('country_code').agg({
    'value': 'last',
    'period': 'last',
    'indicator_name': 'first'
}).reset_index()

latest = latest.sort_values('value', ascending=False)

print("\n--- Top 20 Countries by Govt Securities Exposure ---")
print(latest[['country_code', 'value', 'period']].head(20).to_string(index=False))

# Check specific countries
print("\n--- Key Countries ---")
key = ['KEN', 'NGA', 'DEU', 'USA', 'ZAF', 'TUR', 'JPN', 'BRA', 'GBR', 'IND']
for c in key:
    row = latest[latest['country_code'] == c]
    if len(row) > 0:
        val = row.iloc[0]['value']
        period = row.iloc[0]['period']
        print(f"  {c}: {val:.1f}% ({period})")
    else:
        print(f"  {c}: No data")
