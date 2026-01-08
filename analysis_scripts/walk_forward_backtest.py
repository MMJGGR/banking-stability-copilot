
"""
Evidence Generator: Walk-Forward Validation
===========================================
Purpose: Test the model in a realistic "Forward-Looking" scenario.
Train on past (e.g., <2015), Predict future (>2015).
This exposes "Look-Ahead Bias".

Outputs:
- analysis_scripts/output/rolling_accuracy.png
- analysis_scripts/output/walk_forward_metrics.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CACHE_DIR
from src.crisis_classifier import CrisisClassifier
from src.crisis_labels import CrisisLabels

OUTPUT_DIR = os.path.join("analysis_scripts", "output")

def run_walk_forward():
    print("Loading data for Backtest...")
    features_path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
    if not os.path.exists(features_path):
        return

    df = pd.read_parquet(features_path)
    labels = CrisisLabels()
    # Target: Crisis in next 2 years (usually 1-3 years horizon)
    df['crisis_target'] = df['country_code'].apply(
        lambda c: labels.get_crisis_target(c, 2005, horizon=3) 
    )
    
    # Check if 'year' is in index
    if 'year' not in df.columns:
        print("Resetting index to find 'year'...")
        df = df.reset_index()
        
    # We need a proper time column. 'year' should be there.
    # Ensure year is int
    if 'year' not in df.columns:
        print("Error: 'year' column missing even after reset_index. Columns:", df.columns)
        return
        
    df['year'] = df['year'].astype(int)
    
    # Clean data
    feature_cols = [c for c in df.columns 
                   if c not in ['country_code', 'country_name', 'year', 'period', 'crisis_target'] 
                   and not c.endswith('_period')]
    
    # Walk-Forward Loop
    # Start training on < 2012, Test 2013
    # Step forward 1 year
    
    start_year = 2012
    end_year = 2022 # Assuming data goes up to here?
    
    results = []
    
    print(f"Starting Walk-Forward Validation ({start_year}-{end_year})...")
    
    for test_year in range(start_year, end_year + 1):
        train_mask = df['year'] < test_year
        test_mask = df['year'] == test_year
        
        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, 'crisis_target'].fillna(0)
        
        X_test = df.loc[test_mask, feature_cols]
        y_test = df.loc[test_mask, 'crisis_target'].fillna(0)
        
        if len(X_test) == 0:
            continue
            
        # Train
        model = CrisisClassifier(n_estimators=50) # Faster training for loop
        model.fit(X_train, y_train, cv=2)
        
        # Predict
        probs = model.predict_proba(X_test)
        preds = (probs > 0.5).astype(int)
        
        # Metrics
        # Handle cases with no crises in test set (AUC undefined)
        try:
            auc_val = roc_auc_score(y_test, probs) if y_test.sum() > 0 else np.nan
        except:
            auc_val = np.nan
            
        res = {
            'Test_Year': test_year,
            'Train_Size': len(X_train),
            'Test_Size': len(X_test),
            'Crisis_Count': y_test.sum(),
            'AUC': auc_val,
            'Precision': precision_score(y_test, preds, zero_division=0),
            'Recall': recall_score(y_test, preds, zero_division=0)
        }
        results.append(res)
        print(f"  {test_year}: AUC={auc_val:.3f}, Crises={int(y_test.sum())}")
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "walk_forward_metrics.csv"), index=False)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Test_Year'], results_df['AUC'], marker='o', label='AUC-ROC')
    plt.axhline(0.7, color='r', linestyle='--', label='Academic Threshold (0.7)')
    plt.title("Walk-Forward Validation Performance (2012-2022)")
    plt.xlabel("Test Year")
    plt.ylabel("AUC Score")
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "rolling_accuracy.png"))
    print(f"Saved plot to {os.path.join(OUTPUT_DIR, 'rolling_accuracy.png')}")

if __name__ == "__main__":
    run_walk_forward()
