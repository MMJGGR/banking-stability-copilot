import joblib
import pandas as pd

m = joblib.load('cache/risk_model.pkl')
scores = m['country_scores']
f = m['feature_values']

countries = ['KEN', 'USA', 'DEU', 'UGA']

# All FSIC columns (after sign flipping)
fsic_cols = [
    'capital_adequacy', 'tier1_capital', 
    'npl_ratio',  # Stored as negative (inverted)
    'roe', 'roa', 
    'liquid_assets_st_liab', 'liquid_assets_total',
    'customer_deposits_loans', 
    'fx_loan_exposure',  # Stored as negative (inverted)
    'loan_concentration',  # Stored as negative (inverted)
    'real_estate_loans',  # Stored as negative (inverted)
    'npl_provisions'
]

print('=' * 100)
print('EXHAUSTIVE FSIC FEATURE COMPARISON')
print('=' * 100)

# Build comparison table
data = []
for c in countries:
    row = scores[scores['country_code'] == c]
    feat_row = f[f['country_code'] == c]
    
    if len(row) == 0 or len(feat_row) == 0:
        continue
        
    row = row.iloc[0]
    feat = feat_row.iloc[0]
    
    entry = {
        'Country': row['country_name'],
        'Code': c,
        'Risk Score': f"{row['risk_score']:.1f}",
        'Econ Pillar': f"{row['economic_pillar']:.1f}",
        'Industry Pillar': f"{row['industry_pillar']:.1f}",
        'Coverage': f"{row['data_coverage']*100:.0f}%"
    }
    
    for col in fsic_cols:
        val = feat.get(col)
        if pd.isna(val):
            entry[col] = 'NaN'
        else:
            # Un-invert for display (negative stored = positive real for inverted cols)
            if col in ['npl_ratio', 'fx_loan_exposure', 'loan_concentration', 'real_estate_loans']:
                entry[col] = f"{-val:.1f}"  # Show real value
            else:
                entry[col] = f"{val:.1f}"
    
    data.append(entry)

df = pd.DataFrame(data)
print("\nSCORES:")
print(df[['Country', 'Risk Score', 'Econ Pillar', 'Industry Pillar', 'Coverage']].to_string(index=False))

print("\n\nFSIC FEATURES (after sign flip for model, SHOWN AS REAL VALUES):")
print("Note: Higher values for capital/ROE/liquidity = BETTER banking health")
print("      Higher values for NPL/FX exposure/concentration = WORSE banking health")
print()

for entry in data:
    print(f"\n{entry['Country']} ({entry['Code']}):")
    print(f"  Capital Adequacy:      {entry['capital_adequacy']:>8}")
    print(f"  Tier 1 Capital:        {entry['tier1_capital']:>8}")
    print(f"  NPL Ratio:             {entry['npl_ratio']:>8} (HIGHER = WORSE)")
    print(f"  ROE:                   {entry['roe']:>8}")
    print(f"  ROA:                   {entry['roa']:>8}")
    print(f"  Liquid/ST Liab:        {entry['liquid_assets_st_liab']:>8}")
    print(f"  Liquid/Total:          {entry['liquid_assets_total']:>8}")
    print(f"  Deposits/Loans:        {entry['customer_deposits_loans']:>8}")
    print(f"  FX Loan Exposure:      {entry['fx_loan_exposure']:>8} (HIGHER = WORSE)")
    print(f"  Loan Concentration:    {entry['loan_concentration']:>8} (HIGHER = WORSE)")
    print(f"  Real Estate Loans:     {entry['real_estate_loans']:>8} (HIGHER = WORSE)")
    print(f"  NPL Provisions:        {entry['npl_provisions']:>8}")

print("\n" + "=" * 100)
print("DIAGNOSIS: Compare features to understand why Industry Pillar scores differ")
print("=" * 100)
