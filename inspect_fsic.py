
import sys
import os
sys.path.insert(0, os.getcwd())
import pandas as pd
from src.data_loader import IMFDataLoader

loader = IMFDataLoader()
loader.load_from_cache()

fsic_df = loader._data_cache['FSIC']
print(f"FSIC Columns: {fsic_df.columns.tolist()}")
print(f"Number of countries: {fsic_df['country_code'].nunique()}")

# Check for DEU
deu_data = fsic_df[fsic_df['country_code'] == 'DEU']
print(f"Rows for DEU: {len(deu_data)}")

if len(deu_data) > 0:
    print("Indicators for DEU:")
    print(deu_data['indicator_name'].unique()[:10])

# Check for Germany by name if DEU is missing
if len(deu_data) == 0:
    print("Searching for 'Germany' in country_name...")
    germany_rows = fsic_df[fsic_df['country_name'].str.contains('Germany', case=False, na=False)]
    if len(germany_rows) > 0:
        print(f"Found Germany with codes: {germany_rows['country_code'].unique()}")
    else:
        print("Germany not found in country_name either.")
