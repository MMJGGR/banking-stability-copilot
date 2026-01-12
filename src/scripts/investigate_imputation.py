"""Investigate imputation issues and identify non-country aggregates."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np

# 1. Find aggregate/non-country codes
print("="*70)
print("IDENTIFYING NON-COUNTRY AGGREGATES")
print("="*70)

weo = pd.read_parquet('cache/WEO_cache.parquet')
names = weo.groupby('country_code')['country_name'].first()

# Patterns that indicate aggregates
aggregate_patterns = [
    'Area', 'ASEAN', 'Euro area', 'World', 'Advanced', 'Emerging', 
    'Sub-Saharan', 'Middle East', 'Commonwealth', 'European Union', 
    'OECD', 'G7', 'G20', 'Latin America', ' and ', 'Western Hemisphere',
    'Developing', 'Low-income', 'Heavily indebted'
]

aggregate_codes = []
for code, name in names.items():
    for pattern in aggregate_patterns:
        if pattern.lower() in str(name).lower():
            aggregate_codes.append(code)
            print(f"  {code}: {name}")
            break

print(f"\nTotal aggregates found: {len(aggregate_codes)}")
print(f"Codes to exclude: {sorted(aggregate_codes)}")

# 2. Check imputed values for Senegal, Kenya, etc.
print("\n" + "="*70)
print("CHECKING IMPUTED VALUES")
print("="*70)

try:
    imputed = pd.read_parquet('cache/imputed_features.parquet')
    imputed = imputed.set_index('country_code')
    
    # Compare deposit_funding_ratio across African countries
    african_codes = ['SEN', 'NGA', 'KEN', 'GHA', 'UGA', 'TZA', 'ZAF', 'EGY', 'ETH', 'BEN']
    
    if 'deposit_funding_ratio' in imputed.columns:
        print("\nDeposit Funding Ratio comparison:")
        for code in african_codes:
            if code in imputed.index:
                val = imputed.loc[code, 'deposit_funding_ratio']
                print(f"  {code}: {val:.1f}%")
    
    # Check if many countries have identical values (sign of bad imputation)
    print("\nChecking for identical imputed values:")
    for col in ['deposit_funding_ratio', 'net_interest_margin', 'interbank_funding_ratio']:
        if col in imputed.columns:
            val_counts = imputed[col].value_counts()
            if len(val_counts) > 0:
                most_common_val = val_counts.index[0]
                most_common_count = val_counts.iloc[0]
                print(f"  {col}: Most common value = {most_common_val:.2f}, count = {most_common_count}")
                if most_common_count > 20:
                    print(f"    WARNING: {most_common_count} countries have the same value!")
except Exception as e:
    print(f"Could not load imputed features: {e}")

# 3. Check if KNN is using feature similarity
print("\n" + "="*70)
print("VERIFYING KNN FEATURE SIMILARITY")
print("="*70)

# Load crisis features and compute pairwise distances for a few test countries
features = pd.read_parquet('cache/crisis_features.parquet')
features = features.set_index('country_code')

# Numeric only
numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
features_numeric = features[numeric_cols].dropna(how='all')

# Fill NaN with median for distance calculation
features_filled = features_numeric.fillna(features_numeric.median())

# Compute distances from Senegal
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

if 'SEN' in features_filled.index:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_filled)
    scaled_df = pd.DataFrame(scaled, index=features_filled.index, columns=features_filled.columns)
    
    sen_vec = scaled_df.loc[['SEN']].values
    distances = cdist(sen_vec, scaled_df.values, metric='euclidean')[0]
    
    closest = pd.Series(distances, index=scaled_df.index).sort_values()
    
    print("\nCountries closest to Senegal (in feature space):")
    for i, (code, dist) in enumerate(closest[:15].items()):
        name = names.get(code, 'Unknown')
        print(f"  {i+1}. {code} ({name[:25]}): distance = {dist:.2f}")
    
    print("\nFarthest countries from Senegal:")
    for i, (code, dist) in enumerate(closest[-5:].items()):
        name = names.get(code, 'Unknown')
        print(f"  {code} ({name[:25]}): distance = {dist:.2f}")
