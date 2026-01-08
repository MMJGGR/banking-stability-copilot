"""
Diagnose model outputs for Kenya, USA, Germany, Nigeria.
Compares ALL features to explain risk scores.
"""
import pandas as pd
import numpy as np
import pickle

# Load model and data
with open('cache/risk_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

scores = model_data['country_scores']
# Use feature_values stored in model to ensure we see what the model saw (imputed/processed)
# But for meaningful "raw" comparison, reading parquet is better, just be aware of missingness
features = pd.read_parquet('cache/crisis_features.parquet')
pca_info = model_data.get('pca_info', {})
econ_loadings = pca_info.get('economic_loadings', {})
ind_loadings = pca_info.get('industry_loadings', {})

# Target countries
countries = ['KEN', 'USA', 'DEU', 'NGA']

print("="*100)
print("FULL FEATURE COMPARISON: Kenya vs USA vs Germany vs Nigeria")
print("="*100)

# 1. SCORES
print("\n--- MODEL SCORES ---")
score_cols = ['risk_score', 'economic_pillar', 'industry_pillar', 'combined_pillar', 'risk_category']
for code in countries:
    row = scores[scores['country_code'] == code]
    if len(row) > 0:
        res = ", ".join([f"{c}={row[c].values[0]}" for c in score_cols if c in row.columns])
        print(f"{code}: {res}")
    else:
        print(f"{code}: No score generated")

# 2. FEATURE COMPARISON
# Group by pillar based on loadings keys, plus extras

all_cols = sorted([c for c in features.columns if c not in ['country_code', 'country_name', 'year', 'period'] and features[c].dtype in [np.float64, np.float32]])

# Define categories
econ_feats = list(econ_loadings.keys())
ind_feats = list(ind_loadings.keys())
other_feats = [c for c in all_cols if c not in econ_feats and c not in ind_feats]

def print_feature_table(title, feat_list, loadings_dict=None):
    print(f"\n--- {title.upper()} ---")
    if loadings_dict:
        # Sort by absolute weight impact
        feat_list = sorted(feat_list, key=lambda x: abs(loadings_dict.get(x, 0)), reverse=True)
    
    # Header
    header = f"{'FEATURE':<30} | {'WEIGHT':<8} |"
    for code in countries:
        header += f" {code:<10} |"
    print(header)
    print("-" * len(header))
    
    for feat in feat_list:
        if feat not in features.columns:
            continue
            
        weight_str = ""
        if loadings_dict and feat in loadings_dict:
            w = loadings_dict[feat]
            weight_str = f"{w:+.3f}"
        
        row_str = f"{feat:<30} | {weight_str:<8} |"
        
        for code in countries:
            country_row = features[features['country_code'] == code]
            if len(country_row) > 0:
                val = country_row[feat].values[0]
                if pd.isna(val):
                    val_str = "   N/A"
                else:
                    # Format nicely depending on magnitude
                    if abs(val) > 1000:
                        val_str = f"{val:>9.0f}"
                    elif abs(val) < 0.1 and val != 0:
                        val_str = f"{val:>9.4f}"
                    else:
                        val_str = f"{val:>9.2f}"
            else:
                val_str = "   MISSing"
            row_str += f" {val_str} |"
        print(row_str)

print_feature_table("Economic Pillar Features (Sorted by Weight)", econ_feats, econ_loadings)
print_feature_table("Industry Pillar Features (Sorted by Weight)", ind_feats, ind_loadings)
print_feature_table("Other / Unused Features", other_feats)

print("\n" + "="*100)
print("INTERPRETATION NOTES:")
print("1. WEIGHTS: Showing PCA loadings from the trained model.")
print("   - High POSITIVE weight (+): Higher feature value -> Higher Pillar Score (Better/Lower Risk)")
print("   - High NEGATIVE weight (-): Higher feature value -> Lower Pillar Score (Worse/Higher Risk)")
print("2. VALUES: Raw values from 'crisis_features.parquet' (before log/scaling).")
print("   - Note that 'loan_concentration' and 'npl_ratio' might be inverted in the raw file or pipeline.")
print("   - Check if USA has N/A for key features like 'loan_concentration'.")
print("="*100)
