import pickle
import sys
import os
from datetime import datetime

model_path = r"c:\Users\Richard\Banking\cache\risk_model.pkl"

try:
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Model Training Date: {data.get('training_date', 'Unknown')}")
    print(f"Countries Trained: {data.get('countries_trained', 'Unknown')}")
    print(f"Scores Available: {len(data.get('country_scores', []))}")
    print("Verification: SUCCESS")
    
except Exception as e:
    print(f"Verification: FAILED - {e}")
