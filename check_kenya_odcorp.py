"""
Check Kenya ODCORP (Other Depository Corps = Deposit Money Banks) data
"""
import pandas as pd

mfs_df = pd.read_parquet(r"c:\Users\Richard\Banking\cache\MFS_cache.parquet")

kenya = mfs_df[mfs_df['country_code'] == 'KEN']

print("=== Kenya ODCORP (Deposit Money Banks) Data ===")
print(f"Total Kenya records: {len(kenya)}")

# Check ODCORP records
odcorp = kenya[kenya['indicator_code'].str.contains('ODCORP', case=False, na=False)]
print(f"\nODCORP records: {len(odcorp)}")

if len(odcorp) > 0:
    unique = odcorp.groupby(['indicator_code', 'indicator_name']).agg({
        'value': 'last',
        'period': 'last'
    }).reset_index()
    print("\n--- Latest Values per Indicator ---")
    print(unique.to_string())
    
    # Check for govt claims specifically
    govt_odcorp = odcorp[odcorp['indicator_code'].str.contains('S1311|S13', case=False, na=False)]
    if len(govt_odcorp) > 0:
        print("\n--- ODCORP Govt Claims ---")
        latest = govt_odcorp.sort_values('period').groupby('indicator_code').last()
        print(latest[['indicator_name', 'value', 'period']].to_string())
else:
    print("No ODCORP data for Kenya")
    
    # Alternative: Check for S122 (alternative code for Other Depository Corps)
    s122 = kenya[kenya['indicator_code'].str.contains('S122', case=False, na=False)]
    print(f"\nS122 records: {len(s122)}")
    
    if len(s122) > 0:
        unique = s122.groupby(['indicator_code', 'indicator_name']).agg({
            'value': 'last',
            'period': 'last'
        }).reset_index().head(10)
        print(unique.to_string())

# Cross-check: What do we have for Kenya in total?
print("\n=== Kenya Summary ===")
print(f"S121 (Central Bank): {len(kenya[kenya['indicator_code'].str.contains('S121', na=False)])}")
print(f"DCORP (All Depository): {len(kenya[kenya['indicator_code'].str.contains('DCORP', na=False)])}")
print(f"ODCORP (Other Depository): {len(kenya[kenya['indicator_code'].str.contains('ODCORP', na=False)])}")
