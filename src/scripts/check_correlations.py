
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path (three levels up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import CACHE_DIR

def check_correlations():
    print("="*70)
    print("ANALYZING NEW FEATURE CORRELATIONS")
    print("="*70)
    
    # Load features
    path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
    if not os.path.exists(path):
        print(f"Error: {path} not found. Please run train_model.py first.")
        return

    df = pd.read_parquet(path)
    print(f"Loaded features for {len(df)} countries")
    
    # Select specific features of interest
    new_features = [
        'reserves_to_imports', 
        'bank_ext_liabilities_to_assets', 
        'net_foreign_assets_gdp'
    ]
    
    existing_features = [
        'sovereign_exposure_ratio',
        'current_account_gdp',
        'external_debt_gdp',
        'fx_loan_exposure',
        'gdp_per_capita'
    ]
    
    # Filter to available columns
    available_new = [c for c in new_features if c in df.columns]
    available_existing = [c for c in existing_features if c in df.columns]
    
    if not available_new:
        print("WARNING: New features not found in dataset! Integration failed?")
        return

    print(f"\nFound new features: {available_new}")
    
    # 1. Missingness Analysis
    print("\n" + "="*60)
    print("MISSINGNESS ANALYSIS")
    print("="*60)
    print(f"{'Feature':<35} | {'Missing':<10} | {'% Missing':<10}")
    print("-" * 65)
    for feat in available_new:
        missing = df[feat].isna().sum()
        pct = (missing / len(df)) * 100
        print(f"{feat:<35} | {missing:<10} | {pct:<10.1f}%")

    # 2. Correlation Matrix
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    cols_to_check = available_new + available_existing
    corr_matrix = df[cols_to_check].corr()
    
    print("\nCorrelation with Existing Features:")
    print("-" * 60)
    for new_feat in available_new:
        print(f"\n{new_feat.upper()} correlations:")
        corrs = corr_matrix[new_feat].drop(available_new) # Drop self-corrs
        # Sort by absolute correlation
        sorted_corrs = corrs.abs().sort_values(ascending=False)
        for other_feat, corr_val in sorted_corrs.items():
             # Original corr value (with sign)
             real_val = corrs[other_feat]
             impact = "HIGH" if abs(real_val) > 0.7 else ("MED" if abs(real_val) > 0.4 else "LOW")
             print(f"  vs {other_feat:30s}: {real_val:+.2f} ({impact})")
             
    # Check complementarity (low correlation is good)
    print("\nComplementarity Assessment:")
    max_corr = corr_matrix.loc[available_new, available_existing].abs().max().max()
    if max_corr < 0.7:
        print(f"  [PASS] All new features provide distinct information (Max Corr < 0.7)")
    else:
        print(f"  [NOTE] Some overlap detected (Max Corr {max_corr:.2f}). Check high correlations above.")

if __name__ == "__main__":
    check_correlations()
