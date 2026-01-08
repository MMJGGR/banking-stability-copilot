"""
Debug script to trace FSIBSIS data through the pipeline.
Compares raw CSV -> extracted features -> crisis_features.parquet
"""
import pandas as pd
import os

# Configuration
CACHE_DIR = r'C:\Users\Richard\Banking\cache'
FSIBSIS_CSV = r'C:\Users\Richard\Banking\dataset_2026-01-07T14_49_15.772533145Z_DEFAULT_INTEGRATION_IMF.STA_FSIBSIS_18.0.0.csv'

# Test countries
TEST_COUNTRIES = ['KEN', 'DEU', 'NGA', 'USA', 'GBR', 'BRA']

# FSIBSIS features to check
FSIBSIS_FEATURES = [
    'net_interest_margin', 'interbank_funding_ratio', 'income_diversification',
    'securities_to_assets', 'specific_provisions_ratio', 'large_exposure_ratio',
    'deposit_funding_ratio', 'capital_quality'
]

# Country name mapping
COUNTRY_NAMES = {
    'KEN': 'Kenya',
    'DEU': 'Germany', 
    'NGA': 'Nigeria',
    'USA': 'United States',
    'GBR': 'United Kingdom',
    'BRA': 'Brazil'
}

print("=" * 70)
print("FSIBSIS DATA PIPELINE DEBUG")
print("=" * 70)

# Step 1: Check raw CSV
print("\n=== STEP 1: RAW FSIBSIS CSV ===")
raw_df = pd.read_csv(FSIBSIS_CSV, low_memory=False)
bank_df = raw_df[raw_df['SECTOR'] == 'Deposit takers']

# Key indicators in raw form
raw_indicators = {
    'interest_income': 'Interest income, Domestic currency',
    'interest_expenses': 'Interest expenses, Domestic currency',
    'total_assets': 'Total assets, Assets, Domestic currency',
    'interbank_deposits': 'Interbank deposits, Liabilities, Domestic currency',
    'total_liabilities': 'Total liabilities, Domestic currency',
    'deposits': 'Currency and deposits, Liabilities, Domestic currency',
    'debt_securities': 'Debt securities, Assets, Domestic currency',
    'specific_provisions': 'Specific provisions, Assets, Domestic currency',
    'gross_loans': 'Gross loans, Assets, Domestic currency',
}

for country_code in TEST_COUNTRIES:
    country_name = COUNTRY_NAMES.get(country_code, country_code)
    country_df = bank_df[bank_df['COUNTRY'].str.contains(country_name, case=False, na=False)]
    
    print(f"\n{country_code} ({country_name}):")
    print(f"  Rows in FSIBSIS: {len(country_df)}")
    
    for key, indicator in raw_indicators.items():
        ind_df = country_df[country_df['INDICATOR'] == indicator]
        if len(ind_df) > 0:
            val_2024 = ind_df.iloc[0].get('2024')
            has_data = 'YES' if pd.notna(val_2024) else 'no'
            print(f"  {key}: {has_data}")
        else:
            print(f"  {key}: NOT FOUND")

# Step 2: Check what FSIBSIS loader extracted
print("\n\n=== STEP 2: FSIBSIS LOADER OUTPUT ===")
from src.data_loader_fsibsis import load_fsibsis_features

fsibsis_features_df = load_fsibsis_features(FSIBSIS_CSV)
print(f"Total countries extracted: {len(fsibsis_features_df)}")
print(f"Features: {list(fsibsis_features_df.columns)}")

for country_code in TEST_COUNTRIES:
    row = fsibsis_features_df[fsibsis_features_df['country_code'] == country_code]
    if len(row) > 0:
        row = row.iloc[0]
        print(f"\n{country_code}:")
        for feat in FSIBSIS_FEATURES:
            if feat in row.index:
                val = row[feat]
                status = f"{val:.2f}" if pd.notna(val) else "MISSING"
            else:
                status = "NOT IN DF"
            print(f"  {feat}: {status}")
    else:
        print(f"\n{country_code}: NOT IN EXTRACTED FEATURES")

# Step 3: Check crisis_features.parquet
print("\n\n=== STEP 3: CRISIS_FEATURES.PARQUET ===")
parquet_path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
if os.path.exists(parquet_path):
    crisis_df = pd.read_parquet(parquet_path)
    print(f"Total countries: {len(crisis_df)}")
    print(f"Total features: {len(crisis_df.columns)}")
    
    for country_code in TEST_COUNTRIES:
        row = crisis_df[crisis_df['country_code'] == country_code]
        if len(row) > 0:
            row = row.iloc[0]
            print(f"\n{country_code}:")
            for feat in FSIBSIS_FEATURES:
                if feat in row.index:
                    val = row[feat]
                    status = f"{val:.2f}" if pd.notna(val) else "MISSING - will be imputed"
                else:
                    status = "NOT IN PARQUET"
                print(f"  {feat}: {status}")
        else:
            print(f"\n{country_code}: NOT IN PARQUET")
else:
    print(f"Parquet not found: {parquet_path}")

print("\n" + "=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)
