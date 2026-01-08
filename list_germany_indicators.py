
import sys
import os
sys.path.insert(0, os.getcwd())
import pandas as pd
from src.data_loader import IMFDataLoader

loader = IMFDataLoader()
loader.load_from_cache()
fsic_df = loader._data_cache['FSIC']
deu_data = fsic_df[fsic_df['country_code'] == 'DEU']
indicators = deu_data['indicator_name'].unique()
print("All Germany FSIC Indicators:")
for i in sorted(indicators):
    print(f"- {i}")
