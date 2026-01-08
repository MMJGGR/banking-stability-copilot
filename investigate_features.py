"""
COMPREHENSIVE EXPLORATION: Sovereign-Bank Nexus and Credit Features
Explore ALL datasets: WEO, FSIC, MFS for:
1. Sovereign-bank nexus (bank holdings of govt debt)
2. Private sector credit / GDP
3. Total credit / GDP
"""

from src.data_loader import IMFDataLoader
import pandas as pd
import re

l = IMFDataLoader()
l.load_from_cache()

print("=" * 100)
print("COMPREHENSIVE FEATURE EXPLORATION ACROSS ALL DATASETS")
print("=" * 100)

# Sample countries for coverage testing
sample_countries = ['USA', 'DEU', 'GBR', 'JPN', 'CHN', 'IND', 'BRA', 'ZAF', 
                   'NGA', 'KEN', 'EGY', 'MEX', 'ARG', 'TUR', 'FRA', 'AUS', 'CAN', 'ITA', 'ESP', 'KOR']

# ============================================================================
# 1. SOVEREIGN-BANK NEXUS OPTIONS
# ============================================================================
print("\n" + "=" * 100)
print("1. SOVEREIGN-BANK NEXUS - EXPLORING ALL DATASETS")
print("=" * 100)

# 1A. FSIC Options
print("\n--- 1A. FSIC Indicators Related to Government/Sovereign ---")
fsic_sample = l.get_country_data('USA', 'FSIC')
if fsic_sample is not None:
    govt_patterns = ['government', 'sovereign', 'public sector', 'central', 'treasury']
    for pattern in govt_patterns:
        matches = fsic_sample[fsic_sample['indicator_name'].str.contains(pattern, case=False, na=False)]
        if len(matches) > 0:
            print(f"\n  Pattern '{pattern}':")
            for name in matches['indicator_name'].unique()[:5]:
                print(f"    - {name}")

# 1B. MFS Options - Claims on Government
print("\n--- 1B. MFS Indicators Related to Government Claims ---")
mfs_sample = l.get_country_data('USA', 'MFS')
if mfs_sample is not None:
    # Look for government-related claims
    govt_codes = mfs_sample[mfs_sample['indicator_code'].str.contains('S13|GOV|S1311', case=False, na=False)]
    print(f"  Found {len(govt_codes['indicator_code'].unique())} government-related indicator codes:")
    for code in govt_codes['indicator_code'].unique()[:10]:
        sample_name = govt_codes[govt_codes['indicator_code'] == code]['indicator_name'].iloc[0] if 'indicator_name' in govt_codes.columns else 'N/A'
        print(f"    {code}")

# 1C. WEO Options - Government debt held by domestic banks
print("\n--- 1C. WEO Indicators Related to Government ---")
weo_sample = l.get_country_data('USA', 'WEO')
if weo_sample is not None:
    govt_indicators = weo_sample[weo_sample['indicator_code'].str.contains('GG|DEBT|GOV', case=False, na=False)]
    print(f"  Found {len(govt_indicators['indicator_code'].unique())} government-related indicators:")
    for code in govt_indicators['indicator_code'].unique()[:10]:
        print(f"    {code}")

# 1D. Coverage test for best sovereign-bank nexus candidates
print("\n--- 1D. Coverage Test for Sovereign-Bank Nexus Candidates ---")

# Candidate 1: MFS DCORP_A_ACO_S13M1 or S1311 (Claims on Government)
# Candidate 2: FSIC "Sectoral distribution of investments"
# Candidate 3: Compute from WEO govt debt metrics

candidates = {
    'MFS_ClaimsOnGovt': 'DCORP_A_ACO_S13',  # Claims on govt sector
    'MFS_ClaimsOnCentralGovt': 'DCORP_A_ACO_S1311',  # Claims on central govt
}

for candidate_name, code_pattern in candidates.items():
    coverage = 0
    for cc in sample_countries:
        mfs = l.get_country_data(cc, 'MFS')
        if mfs is not None:
            matches = mfs[mfs['indicator_code'].str.contains(code_pattern, case=False, na=False)]
            if len(matches) > 0:
                coverage += 1
    print(f"  {candidate_name}: {coverage}/{len(sample_countries)} countries ({coverage/len(sample_countries)*100:.0f}%)")

# ============================================================================
# 2. CREDIT TO GDP - BOTH PRIVATE AND TOTAL
# ============================================================================
print("\n" + "=" * 100)
print("2. CREDIT TO GDP - PRIVATE SECTOR AND TOTAL CREDIT")
print("=" * 100)

# 2A. MFS Credit Indicators
print("\n--- 2A. MFS Credit Indicators ---")
if mfs_sample is not None:
    credit_patterns = {
        'Private_Sector': 'DCORP_A_ACO_PS',       # Claims on Private Sector
        'Total_Domestic': 'DCORP_A_ACO_S1',       # Total domestic claims
        'Household': 'DCORP_A_ACO_S14',           # Claims on households
        'NonFinCorps': 'DCORP_A_ACO_S11',         # Claims on non-financial corps
    }
    
    for name, pattern in credit_patterns.items():
        matches = mfs_sample[mfs_sample['indicator_code'].str.contains(pattern, case=False, na=False)]
        print(f"  {name} ({pattern}): {len(matches['indicator_code'].unique())} codes, {len(matches)} records")

# 2B. Coverage test for credit indicators
print("\n--- 2B. Coverage Test for Credit Indicators ---")

credit_candidates = {
    'MFS_PrivateSector': 'DCORP_A_ACO_PS',
    'MFS_TotalDomestic': 'DCORP_A_ACO_S1_',  # Total economy
    'MFS_Households': 'DCORP_A_ACO_S14',
    'MFS_NonFinCorps': 'DCORP_A_ACO_S11001',
}

for candidate_name, code_pattern in credit_candidates.items():
    coverage = 0
    sample_values = []
    for cc in sample_countries:
        mfs = l.get_country_data(cc, 'MFS')
        if mfs is not None:
            matches = mfs[mfs['indicator_code'].str.contains(code_pattern, case=False, na=False)]
            if len(matches) > 0:
                coverage += 1
                latest = matches.sort_values('period')['value'].iloc[-1]
                sample_values.append((cc, latest))
    print(f"  {candidate_name}: {coverage}/{len(sample_countries)} countries ({coverage/len(sample_countries)*100:.0f}%)")
    if sample_values[:3]:
        print(f"    Sample values: {[(c, f'{v:.0f}') for c, v in sample_values[:3]]}")

# 2C. Check FSIC for credit ratios
print("\n--- 2C. FSIC Credit-Related Ratios ---")
if fsic_sample is not None:
    credit_fsic = fsic_sample[fsic_sample['indicator_name'].str.contains('loan|credit|asset', case=False, na=False)]
    unique_names = credit_fsic['indicator_name'].unique()
    print(f"  Found {len(unique_names)} credit-related FSIC indicators:")
    for name in unique_names[:10]:
        print(f"    - {name[:80]}...")

# 2D. WEO GDP indicators for denominator
print("\n--- 2D. WEO GDP Indicators for Denominator ---")
if weo_sample is not None:
    gdp_indicators = weo_sample[weo_sample['indicator_code'].str.contains('NGD', case=False, na=False)]
    for code in gdp_indicators['indicator_code'].unique()[:8]:
        print(f"    {code}")

print("\n" + "=" * 100)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 100)
