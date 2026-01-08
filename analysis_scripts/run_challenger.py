
"""
Evidence Generator: Challenger Model Benchmarking
=================================================
Purpose: Verify if the complex 2-Pillar XGBoost architecture 
actually adds value over a simple Logistic Regression baseline.

Outputs:
- analysis_scripts/output/roc_comparison.png
- analysis_scripts/output/model_metrics.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CACHE_DIR
# We import the CrisisClassifier to run the "Current Model" logic
from src.crisis_classifier import CrisisClassifier

OUTPUT_DIR = os.path.join("analysis_scripts", "output")

def run_challenger_benchmark():
    print("Loading features...")
    features_path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
    if not os.path.exists(features_path):
        print("Features missing.")
        return

    df = pd.read_parquet(features_path)
    
    # Needs crisis target. We can reuse the logic from train_model or crisis_classifier
    # For simplicity, let's load crisis_labels or assume target exists or recreate it
    # Ideally, use the same dataset preparation as crisis_classifier
    
    from src.crisis_labels import CrisisLabels
    labels = CrisisLabels()
    df['crisis_target'] = df['country_code'].apply(
        lambda c: labels.get_crisis_target(c, 2005, horizon=3) # consistent with train_model
    )
    
    # Drop rows where we can't determine target (usually future data)
    # The 'crisis_target' logic handles periods.
    
    # Features
    feature_cols = [c for c in df.columns 
                   if c not in ['country_code', 'country_name', 'year', 'period', 'crisis_target'] 
                   and not c.endswith('_period')]
    
    X = df[feature_cols]
    y = df['crisis_target']
    
    # Handle NaN in target (maybe some countries not in label DB?)
    # Usually 0 or drop. Let's fill 0 (No Crisis confirmed)
    y = y.fillna(0)
    
    # 1. Baseline: Logistic Regression (The "Dumb" Model)
    # Pipeline: Impute Median -> Scale -> LogReg
    baseline_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42))
    ])
    
    # 2. Comparison: Naive Bayes (Another simple baseline)
    nb_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', GaussianNB())
    ])
    
    # 3. The "State of the Art" (Current Copilot XGBoost)
    # We'll re-instantiate it.
    copilot = CrisisClassifier()
    
    print(f"Benchmarking on {len(df)} samples...")
    print(f"Crisis prevalence: {y.mean():.1%}")
    
    # Cross-Validation Predictions (5-Fold Stratified)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Get probs
    # Note: Copilot handles internal imputation/scaling, Baselines assume pipeline does it
    
    preds_baseline = cross_val_predict(baseline_pipe, X, y, cv=cv, method='predict_proba')[:, 1]
    preds_nb = cross_val_predict(nb_pipe, X, y, cv=cv, method='predict_proba')[:, 1]
    
    # Copilot requires manual CV loop because it's a custom class that might not fully comply with sklearn clone
    # We can use its fit/predict_proba, but cross_val_predict might fail if not fully sklearn compliant.
    # Let's do manual loop for Copilot to be safe.
    
    preds_copilot = np.zeros(len(y))
    # We need to fill NaNs for Copilot wrapper if it expects clean DF? 
    # Wrapper does fillna.
    
    # Actually, let's trust the Wrapper's internal CV logic or use sklearn
    # CrisisClassifier has fit/predict_proba.
    # But it assumes X is DataFrame. cross_val_predict passes numpy often.
    # So we do manual CV.
    
    print("Running CV for Copilot...")
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = CrisisClassifier()
        model.fit(X_train, y_train, cv=2) # Internal CV inside fit? No, current code uses CV inside fit only for print.
        # We just want to train and predict.
        
        prob = model.predict_proba(X_test)
        preds_copilot[test_idx] = prob
        
    # Compute Metrics
    fpr_b, tpr_b, _ = roc_curve(y, preds_baseline)
    roc_auc_b = auc(fpr_b, tpr_b)
    
    fpr_c, tpr_c, _ = roc_curve(y, preds_copilot)
    roc_auc_c = auc(fpr_c, tpr_c)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_b, tpr_b, label=f'Baseline (LogReg) AUC = {roc_auc_b:.3f}', linestyle='--')
    plt.plot(fpr_c, tpr_c, label=f'Copilot (XGBoost) AUC = {roc_auc_c:.3f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Challenger Model Benchmark: Does Complexity Pay Off?')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_comparison.png"))
    print(f"Saved ROC to {os.path.join(OUTPUT_DIR, 'roc_comparison.png')}")
    
    # Save Metrics
    metrics = pd.DataFrame({
        'Model': ['Logistic Regression', 'Banking Copilot'],
        'AUC-ROC': [roc_auc_b, roc_auc_c]
    })
    metrics.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), index=False)
    print("\nMetrics:")
    print(metrics)

if __name__ == "__main__":
    run_challenger_benchmark()
