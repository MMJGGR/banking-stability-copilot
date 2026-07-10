"""
Diagnostic script to analyze FSIC and FSIBSIS indicators.
Identifies percent vs currency issues and categorizes all indicators.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from src.data_loader import IMFDataLoader, FSIBSISLoader

print("="*80)
print("FSI DATA DIAGNOSTIC ANALYSIS")
print("="*80)

# =============================================================================
# FSIC ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("SECTION 1: FSIC INDICATORS")
print("="*80)

loader = IMFDataLoader()
loader.load_from_cache()
fsic = loader._data_cache.get('FSIC')

print(f"\nTotal FSIC records: {len(fsic):,}")
print(f"Unique indicator_code: {fsic['indicator_code'].nunique()}")
print(f"Unique indicator_name: {fsic['indicator_name'].nunique()}")

# Get unique indicator names with sample values
indicators = fsic.groupby('indicator_name').agg({
    'value': ['mean', 'min', 'max', 'count'],
    'country_code': 'nunique'
}).reset_index()
indicators.columns = ['indicator_name', 'mean', 'min', 'max', 'count', 'n_countries']

# Categorize by indicator name patterns
def categorize(name):
    name_lower = name.lower()
    if 'percent' in name_lower:
        return 'PERCENT'
    elif 'domestic currency' in name_lower:
        return 'CURRENCY'
    elif 'basis points' in name_lower:
        return 'BASIS_POINTS'
    elif 'number' in name_lower or 'count' in name_lower:
        return 'COUNT'
    elif 'ratio' in name_lower:
        return 'RATIO'
    else:
        return 'UNKNOWN'

indicators['type'] = indicators['indicator_name'].apply(categorize)

print(f"\n--- Indicators by Type ---")
type_counts = indicators['type'].value_counts()
for t, c in type_counts.items():
    print(f"  {t}: {c}")

# Check for problematic indicators (currency values shown as if percent)
print("\n--- POTENTIAL ISSUES: Non-percent indicators ---")
non_pct = indicators[indicators['type'] != 'PERCENT'].sort_values('count', ascending=False)
print(f"Total non-percent indicators: {len(non_pct)}")
for _, r in non_pct.head(20).iterrows():
    print(f"  [{r['type']:12}] {r['indicator_name'][:65]}")
    print(f"               mean={r['mean']:,.0f}, range=[{r['min']:,.0f}, {r['max']:,.0f}]")

# Check PERCENT indicators look reasonable (0-200% range typically)
print("\n--- PERCENT indicators with suspicious values ---")
pct = indicators[indicators['type'] == 'PERCENT']
suspicious = pct[(pct['mean'] > 500) | (pct['mean'] < -100)]
if len(suspicious) > 0:
    for _, r in suspicious.iterrows():
        print(f"  {r['indicator_name'][:70]}")
        print(f"    mean={r['mean']:.1f}, range=[{r['min']:.1f}, {r['max']:.1f}]")
else:
    print("  None found - all percent values look reasonable")

# =============================================================================
# FSIBSIS ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("SECTION 2: FSIBSIS INDICATORS")
print("="*80)

fsibsis_loader = FSIBSISLoader()
fsibsis_loader.load()
fsibsis = fsibsis_loader.bank_data

print(f"\nTotal FSIBSIS records: {len(fsibsis):,}")
print(f"Countries: {fsibsis['country_code'].nunique()}")

# Get unique indicators
if 'INDICATOR' in fsibsis.columns:
    fsibsis_indicators = fsibsis['INDICATOR'].unique()
    print(f"Unique indicators: {len(fsibsis_indicators)}")
    
    # Categorize
    def categorize_fsibsis(name):
        name_lower = str(name).lower()
        if 'percent' in name_lower or 'pct' in name_lower or '%' in name_lower:
            return 'PERCENT'
        elif 'ratio' in name_lower:
            return 'RATIO'
        elif 'number' in name_lower or 'count' in name_lower:
            return 'COUNT'
        elif any(x in name_lower for x in ['usd', 'currency', 'million', 'billion', 'domestic']):
            return 'CURRENCY'
        else:
            return 'UNKNOWN'
    
    fsibsis_cats = {ind: categorize_fsibsis(ind) for ind in fsibsis_indicators}
    
    print("\n--- FSIBSIS Indicators by Type ---")
    from collections import Counter
    cat_counts = Counter(fsibsis_cats.values())
    for t, c in cat_counts.most_common():
        print(f"  {t}: {c}")
    
    print("\n--- Sample FSIBSIS Indicators ---")
    for i, ind in enumerate(sorted(fsibsis_indicators)[:30]):
        cat = fsibsis_cats[ind]
        print(f"  [{cat:8}] {ind[:70]}")

# =============================================================================
# RECOMMENDATIONS
# =============================================================================
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

# Count non-banking indicators
non_bank_keywords = ['gdp', 'money market', 'insurance', 'pension', 'household', 'other financial']
non_bank = indicators[indicators['indicator_name'].str.lower().str.contains('|'.join(non_bank_keywords), na=False)]
print(f"\n1. Potentially non-banking indicators in FSIC: {len(non_bank)}")
for _, r in non_bank.head(10).iterrows():
    print(f"   - {r['indicator_name'][:70]}")

print("\n2. Currency vs Percent disambiguation:")
print("   - Pattern 'Percent' in name -> PERCENT type")
print("   - Pattern 'Domestic currency' in name -> CURRENCY type (absolute values)")
print("   - These should be displayed differently in the Data Explorer")

print("\n3. Indicator deduplication needed:")
# Check for similar names
from difflib import SequenceMatcher
similar_pairs = []
names = list(indicators['indicator_name'])[:50]  # Check first 50 for speed
for i, n1 in enumerate(names):
    for n2 in names[i+1:]:
        ratio = SequenceMatcher(None, n1.lower(), n2.lower()).ratio()
        if ratio > 0.9:
            similar_pairs.append((n1[:50], n2[:50], ratio))

if similar_pairs:
    print("   Similar indicator names found:")
    for n1, n2, r in similar_pairs[:5]:
        print(f"   - {n1}...")
        print(f"     {n2}... (similarity: {r:.0%})")
else:
    print("   No highly similar indicator names found")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
