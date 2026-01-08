
import pandas as pd
from src.data_loader import IMFDataLoader
from src.feature_engineering import CrisisFeatureEngineer
import re

# Initialize
loader = IMFDataLoader()
loader.load_from_cache()

# 1. EMULATE DASHBOARD LOGIC
print("--- DASHBOARD LOGIC ---")
fsic_data = loader.get_country_data('DEU', 'FSIC')
# Pattern from components.py
pattern = 'Tier 1 capital to risk-weighted assets.*Core FSI'

matches = fsic_data[fsic_data['indicator_name'].str.contains(pattern, case=False, na=False, regex=True)]
print(f"Matches found: {len(matches)}")
if len(matches) > 0:
    for idx, row in matches.sort_values('period').iterrows():
        print(f"  Period: {row['period']}, Value: {row['value']}")
    
    latest_row = matches.sort_values('period').iloc[-1]
    dashboard_val = latest_row['value']
    print(f"DASHBOARD VALUE: {dashboard_val}")
else:
    print("No dashboard matches")

# 2. EMULATE FEATURE ENGINEERING LOGIC
print("\n--- FEATURE ENGINEERING LOGIC (Class Test) ---")
full_fsic = loader.load_fsic()
engineer = CrisisFeatureEngineer()

# This uses the current code in src/feature_engineering.py
print("Running extract_fsic_features...")
fsic_features = engineer.extract_fsic_features(full_fsic)
deu_feats = fsic_features[fsic_features['country_code'] == 'DEU']

if 'tier1_capital' in deu_feats.columns:
    model_val = deu_feats['tier1_capital'].values[0]
    print(f"MODEL VALUE (Processed): {model_val}")
else:
    print("tier1_capital not in features")
    
# Print the mapping used by the engineer
# Access private attribute context or just check source
print("\nRegex used in feature_engineering.py (Verification):")
import inspect
src = inspect.getsource(engineer.extract_fsic_features)
for line in src.split('\n'):
    if 'tier1_capital' in line:
        print(f"  {line.strip()}")
