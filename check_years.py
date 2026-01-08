
import pandas as pd
import os
from src.config import CACHE_DIR

path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
df = pd.read_parquet(path)
ken = df[df['country_code'] == 'KEN'].iloc[0]

print("Columns in Parquet related to Interest Margin:")
print([c for c in df.columns if 'interest_margin' in c])

print("\nKenya Values:")
if 'net_interest_margin' in ken:
    print(f"NIM: {ken['net_interest_margin']}")
if 'net_interest_margin_year' in ken:
    print(f"NIM Year: {ken['net_interest_margin_year']}")
else:
    print("NIM Year column MISSING")

print("\nColumns related to Large Exposure:")
print([c for c in df.columns if 'large_exposure' in c])
if 'large_exposure_ratio_year' in ken:
    print(f"Large Exp Year: {ken['large_exposure_ratio_year']}")
