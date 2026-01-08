"""
Inspect training data for specific countries to diagnose validation failures.
"""
import pandas as pd
import pickle
import os
import sys

# Setup
cache_dir = r'c:\Users\Richard\Banking\cache'
sys.path.insert(0, r'c:\Users\Richard\Banking')

from src.crisis_labels import CrisisLabels

print("=" * 70)
print("TRAINING DATA INSPECTION FOR VALIDATION DIAGNOSTICS")
print("=" * 70)

# Countries to inspect
countries_of_interest = ['ARG', 'DEU', 'PAK', 'GBR', 'KEN']
validation_countries = ['USA', 'JPN', 'NGA', 'CHE', 'VEN', 'CAN', 'TUR']
all_countries = countries_of_interest + validation_countries

# 1. Load crisis features
print("\n=== 1. CRISIS FEATURES DATASET ===")
features = pd.read_parquet(os.path.join(cache_dir, 'crisis_features.parquet'))
print(f"Shape: {features.shape}")
print(f"Columns: {list(features.columns)}")

# 2. Load risk model scores
print("\n=== 2. RISK MODEL SCORES ===")
with open(os.path.join(cache_dir, 'risk_model.pkl'), 'rb') as f:
    model_data = pickle.load(f)
    
scores_df = model_data['country_scores']
print(f"Total countries: {len(scores_df)}")

# 3. Show validation country scores
print("\n=== 3. VALIDATION COUNTRIES SCORES ===")
validation_pairs = [
    ('USA', 'NGA', 'USA < Nigeria'),
    ('JPN', 'NGA', 'Japan < Nigeria'),
    ('DEU', 'ARG', 'Germany < Argentina'),
    ('CHE', 'VEN', 'Switzerland < Venezuela'),
    ('GBR', 'PAK', 'UK < Pakistan'),
    ('CAN', 'TUR', 'Canada < Turkey'),
]

for c1, c2, desc in validation_pairs:
    r1 = scores_df[scores_df['country_code'] == c1]
    r2 = scores_df[scores_df['country_code'] == c2]
    
    if len(r1) > 0 and len(r2) > 0:
        s1, s2 = r1.iloc[0]['risk_score'], r2.iloc[0]['risk_score']
        status = "PASS" if s1 < s2 else "FAIL"
        print(f"  [{status}] {desc}: {c1}={s1:.1f}, {c2}={s2:.1f}")
    else:
        missing = []
        if len(r1) == 0: missing.append(c1)
        if len(r2) == 0: missing.append(c2)
        print(f"  [SKIP] {desc} - missing: {missing}")

# 4. Show features for specific countries
print("\n=== 4. RAW FEATURES FOR COUNTRIES OF INTEREST ===")
feature_cols = ['gdp_per_capita', 'gdp_growth', 'inflation', 'current_account_gdp',
                'govt_debt_gdp', 'credit_to_gdp', 'credit_to_gdp_gap', 
                'npl_ratio', 'capital_adequacy', 'roe', 'roa', 
                'liquid_assets_st_liab', 'crisis_target']

for code in ['ARG', 'DEU', 'PAK', 'GBR', 'KEN'] + ['USA', 'NGA']:
    row = features[features['country_code'] == code]
    if len(row) > 0:
        print(f"\n{code}:")
        r = row.iloc[0]
        for col in feature_cols:
            if col in r.index:
                val = r[col]
                if pd.notna(val):
                    print(f"  {col}: {val:.4f}")
                else:
                    print(f"  {col}: NaN")
    else:
        print(f"\n{code}: NOT FOUND in features")

# 5. Check crisis labels (hardcoded historical crises)
print("\n=== 5. HARDCODED CRISIS LABELS (Laeven-Valencia) ===")
labels = CrisisLabels()

for code in all_countries:
    if code in labels.SYSTEMIC_CRISES:
        crises = labels.SYSTEMIC_CRISES[code]
        print(f"  {code}: {crises}")
    else:
        print(f"  {code}: No recorded crises")

# 6. Show crisis_target values (what model learns from)
print("\n=== 6. CRISIS TARGET BY COUNTRY (Training Labels) ===")
if 'crisis_target' in features.columns:
    for code in all_countries:
        row = features[features['country_code'] == code]
        if len(row) > 0:
            target = row.iloc[0]['crisis_target']
            print(f"  {code}: crisis_target={target}")
        else:
            print(f"  {code}: NOT IN TRAINING DATA")

# 7. Show model scores breakdown
print("\n=== 7. MODEL SCORE COMPONENTS ===")
print("(Development-weighted approach: 50% GDP/cap + 25% Economic + 25% Industry)")
score_cols = ['country_code', 'risk_score', 'economic_pillar', 'industry_pillar', 
              'development_level', 'data_coverage', 'risk_category']

for code in ['ARG', 'DEU', 'PAK', 'GBR', 'KEN'] + ['USA', 'NGA', 'CHE', 'VEN']:
    row = scores_df[scores_df['country_code'] == code]
    if len(row) > 0:
        r = row.iloc[0]
        print(f"\n{code}:")
        print(f"  risk_score: {r['risk_score']:.1f}")
        print(f"  economic_pillar: {r['economic_pillar']:.1f}")
        print(f"  industry_pillar: {r['industry_pillar']:.1f}")
        print(f"  development_level: {r['development_level']:.2f}")
        print(f"  data_coverage: {r['data_coverage']:.2%}")
        print(f"  category: {r['risk_category']}")
    else:
        print(f"\n{code}: NOT IN MODEL")

# 8. Potential bias analysis
print("\n=== 8. BIAS ANALYSIS ===")
print("\nDeveloped vs Developing correlation:")
if 'gdp_per_capita' in features.columns and 'crisis_target' in features.columns:
    merged = features.merge(scores_df[['country_code', 'risk_score']], on='country_code', how='inner')
    
    # Correlation between GDP per capita and risk score
    corr = merged[['gdp_per_capita', 'risk_score']].corr().iloc[0, 1]
    print(f"  Correlation(GDP per capita, risk_score): {corr:.3f}")
    
    # Check if poor countries always get high scores
    gdp_median = merged['gdp_per_capita'].median()
    rich = merged[merged['gdp_per_capita'] > gdp_median]['risk_score'].mean()
    poor = merged[merged['gdp_per_capita'] <= gdp_median]['risk_score'].mean()
    print(f"  Average risk score (rich countries): {rich:.2f}")
    print(f"  Average risk score (poor countries): {poor:.2f}")
    print(f"  Bias factor: {poor/rich:.2f}x")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
