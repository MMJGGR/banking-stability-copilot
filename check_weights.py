
import pandas as pd
import os
from src.config import CACHE_DIR

path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
df = pd.read_parquet(path)

print("Columns available:")
print(df.columns.tolist())

# Check for weighting candidates
weight_candidates = ['total_assets', 'gdp', 'nominal_gdp', 'population', 'quota', 'gdp_per_capita']
print("\nWeighting Candidates present:")
for c in weight_candidates:
    if c in df.columns:
        print(f" - {c}: {df[c].count()} non-nulls")
    else:
        print(f" - {c}: MISSING")

# Check coverage for 'total_assets' specifically (best for banking weights)
if 'total_assets' in df.columns:
    print(f"\nTotal Assets Coverage: {df['total_assets'].count()}/{len(df)}")
