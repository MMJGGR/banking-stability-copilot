
import pandas as pd
import os
from src.config import CACHE_DIR
from sklearn.impute import SimpleImputer

# 1. Load Features
path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
print(f"Loading features from {path}...")
df = pd.read_parquet(path)

# 2. Identify Numeric Columns for Imputation
# Exclude metadata like country_code, period, country_name if present
numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int']).columns.tolist()
# Ensure we don't impute the target variable if it exists (it shouldn't be here but good practice)
cols_to_impute = [c for c in numeric_cols if c not in ['crisis_prob', 'year', 'period']]

print(f"Imputing {len(cols_to_impute)} features using Median strategy...")

# 3. Apply Imputation
imputer = SimpleImputer(strategy='median')
df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

# 4. Verify Kenya Large Exposure
# Before verification, check if it was NaN
# (We already overwrote it, but we can check the result)
ken = df[df['country_code'] == 'KEN'].iloc[0]
print(f"Kenya Large Exposure (Imputed): {ken.get('large_exposure_ratio')}")

# 5. Save
df.to_parquet(path, index=False)
print(f"SUCCESS: Imputed dataset saved to {path}")
