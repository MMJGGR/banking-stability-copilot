
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CACHE_DIR

path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
df = pd.read_parquet(path)
print("Index:", df.index)
print("Columns:", df.columns)
print("Head:\n", df.head())
