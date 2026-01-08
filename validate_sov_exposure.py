"""
Test script to validate updated sovereign exposure calculation.
Expected: Kenya should be ~30% (per user knowledge).
"""
import sys
sys.path.insert(0, r'c:\Users\Richard\Banking')

from src.data_loader import IMFDataLoader
from src.feature_engineering import CrisisFeatureEngineer

# Load data
loader = IMFDataLoader()
loader.load_from_cache()

mfs_df = loader._data_cache.get('MFS')
weo_df = loader._data_cache.get('WEO')

# Run updated calculation
engineer = CrisisFeatureEngineer()
nexus_df = engineer.compute_sovereign_bank_nexus(mfs_df, weo_df)

# Print key countries
print("\n" + "="*60)
print("VALIDATION: Sovereign Exposure (Govt/Total Assets)")
print("="*60)

key_countries = ['DEU', 'KEN', 'USA', 'NGA', 'GBR', 'JPN', 'BRA', 'ZAF', 'IND', 'TUR']
for country in key_countries:
    row = nexus_df[nexus_df['country_code'] == country]
    if len(row) > 0:
        val = row.iloc[0]['sovereign_exposure_ratio']
        print(f"  {country}: {val:.1f}%")
    else:
        print(f"  {country}: No data")

# Kenya validation
kenya = nexus_df[nexus_df['country_code'] == 'KEN']
if len(kenya) > 0:
    kenya_val = kenya.iloc[0]['sovereign_exposure_ratio']
    expected = 30.0
    diff = abs(kenya_val - expected)
    status = "PASS" if diff < 10 else "FAIL"
    print(f"\n*** KENYA VALIDATION: {kenya_val:.1f}% (expected ~{expected:.0f}%) [{status}] ***")
else:
    print("\n*** KENYA: No data available ***")
