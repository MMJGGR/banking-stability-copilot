
"""
Evidence Generator: Data Sparsity Audit
=======================================
Purpose: Visualizing the "Confidence Floor" problem.
Generating a heatmap of data availability by region/country.

Outputs:
- analysis_scripts/output/data_sparsity_heatmap.png
- analysis_scripts/output/missingness_stats.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CACHE_DIR

OUTPUT_DIR = os.path.join("analysis_scripts", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_sparsity_audit():
    print("Loading engineered features...")
    features_path = os.path.join(CACHE_DIR, "crisis_features.parquet")
    if not os.path.exists(features_path):
        print("Features file not found. Please run src/feature_engineering.py first.")
        return

    features = pd.read_parquet(features_path)
    
    # Identify key columns (exclude periods and metadata)
    metadata_cols = ['country_code', 'country_name', 'year', 'period']
    data_cols = [c for c in features.columns if c not in metadata_cols and not c.endswith('_period')]
    
    print(f"Auditing {len(data_cols)} features for {len(features)} countries...")
    
    # Calculate missingness matrix
    missing_matrix = features.set_index('country_code')[data_cols].isna()
    
    # Cluster countries by missingness pattern (simple sort by % missing)
    missing_pct = missing_matrix.mean(axis=1).sort_values()
    sorted_countries = missing_pct.index
    sorted_matrix = missing_matrix.loc[sorted_countries]
    
    # Plot Heatmap
    plt.figure(figsize=(12, 20)) # Tall plot
    sns.heatmap(sorted_matrix, cbar=False, cmap="binary", xticklabels=True, yticklabels=True)
    plt.title(f"Data Sparsity Map (Black = Missing)\nSorted by Data Completeness")
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "data_sparsity_heatmap.png"))
    print(f"Saved heatmap to {os.path.join(OUTPUT_DIR, 'data_sparsity_heatmap.png')}")
    
    # Identify "Ghost Countries" (Countries that are mostly imputed)
    # In the code (train_model.py), countries with < 50% coverage get floored to Risk 6.
    # Let's list who these are.
    coverage = 1 - missing_pct
    ghost_countries = coverage[coverage < 0.50]
    
    print(f"\nIdentified {len(ghost_countries)} 'Ghost Countries' (<50% actual data):")
    print(ghost_countries.head(10))
    
    ghost_countries.to_csv(os.path.join(OUTPUT_DIR, "ghost_countries_list.csv"))
    
    # Bias Check: Are they mostly African/Low-Income?
    # We don't have region tags easily loaded, but we can infer from ISO codes visually in the report.

if __name__ == "__main__":
    run_sparsity_audit()
