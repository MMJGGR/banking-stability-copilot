"""Deep search for External DSR components and verification of Fiscal codes."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np

print("="*80)
print("VERIFYING WEO FISCAL INDICATORS FOR INTEREST CALCULATION")
print("="*80)

weo = pd.read_parquet('cache/WEO_cache.parquet')

# Check Primary Balance vs Overall Balance
fiscal_codes = {
    'GGXCNL_NGDP': 'General government net lending/borrowing (% GDP)',
    'GGXONLB_NGDP': 'General government primary net lending/borrowing (% GDP)',
    'GGX_NGDP': 'General government total expenditure (% GDP)',
    'GGR_NGDP': 'General government revenue (% GDP)',
}

print("\nFiscal Indicator Coverage:")
for code, name in fiscal_codes.items():
    if code in weo['indicator_code'].values:
        n = weo[weo['indicator_code'] == code]['country_code'].nunique()
        print(f"  {code}: {n} countries - {name}")
    else:
        # Try finding the code if name matches
        print(f"  {code}: NOT FOUND directly. Searching by name parts...")
        
# Check if we can calculate Interest Payments
print("\nCalculating Derived Interest Payments (Interest = Primary Balance - Overall Balance):")
# If Primary (GGXONLB) > Overall (GGXCNL), the difference is Interest Payments (negative impact on overall)
# Actually: Overall = Primary - Interest  =>  Interest = Primary - Overall
# Let's check a sample country (e.g., BRA, USA)
sample_countries = ['BRA', 'USA', 'ZAF', 'KEN']
relevant_weo = weo[weo['indicator_code'].isin(fiscal_codes.keys())]

if 'GGXONLB_NGDP' in relevant_weo['indicator_code'].values and 'GGXCNL_NGDP' in relevant_weo['indicator_code'].values:
    for c in sample_countries:
        try:
            c_data = relevant_weo[relevant_weo['country_code'] == c]
            if c_data.empty: continue
            
            # Pivot to wide to align years
            wide = c_data.pivot(index='period', columns='indicator_code', values='value').sort_index()
            
            if 'GGXONLB_NGDP' in wide and 'GGXCNL_NGDP' in wide:
                wide['Implied_Interest_GDP'] = wide['GGXONLB_NGDP'] - wide['GGXCNL_NGDP']
                print(f"\n{c} Sample Calculation (Most Recent 3 Years):")
                print(wide[['GGXONLB_NGDP', 'GGXCNL_NGDP', 'Implied_Interest_GDP']].tail(3))
        except Exception as e:
            print(f"Error calculating for {c}: {e}")

print("\n" + "="*80)
print("DEEP SEARCH FOR EXTERNAL DEBT SERVICE (SOVEREIGN)")
print("="*80)

# Look for specific keywords in ALL datasets
keywords = ['external', 'foreign', 'non-resident', 'rest of the world']
service_keywords = ['interest', 'service', 'amortization', 'payment', 'servicing']

def search_dataset(name, df):
    print(f"\nScanning {name} for External + DSR terms...")
    
    unique_inds = df[['indicator_code', 'indicator_name']].drop_duplicates()
    
    found_any = False
    for _, row in unique_inds.iterrows():
        iname = str(row['indicator_name']).lower()
        if any(k in iname for k in keywords) and any(s in iname for s in service_keywords):
            # Check coverage
            cnt = df[df['indicator_code'] == row['indicator_code']]['country_code'].nunique()
            if cnt > 5: # Filter out the ones we already know have 5 countries
                print(f"  [POTENTIAL] {row['indicator_code']} ({cnt} countries): {row['indicator_name']}")
                found_any = True
            elif cnt > 0 and 'WEO' not in name: # Show even low counts for non-WEO
                print(f"  [LOW COVERAGE] {row['indicator_code']} ({cnt} countries): {row['indicator_name']}")
                found_any = True
                
    if not found_any:
        print("  No high-coverage indicators found matching 'External/Foreign' AND 'Interest/Service'.")

search_dataset("WEO", weo)

mfs = pd.read_parquet('cache/MFS_cache.parquet')
search_dataset("MFS", mfs)

# Check FSIBSIS for any Sovereign/External details not caught before
from src.data_loader import FSIBSISLoader
fsibsis = FSIBSISLoader()
fsibsis.load()
# FSIBSIS structure is slightly different
print("\nScanning FSIBSIS...")
fs_inds = fsibsis.bank_data['INDICATOR'].unique()
for ind in fs_inds:
    iname = ind.lower()
    if any(k in iname for k in keywords) and any(s in iname for s in service_keywords):
        print(f"  {ind}")

print("\n" + "="*80)
print("CRISIS DATABASE STRUCTURE CHECK")
print("="*80)
# Check current crisis_labels.py structure
try:
    with open('src/crisis_labels.py', 'r') as f:
        content = f.read()
        if "CRISIS_EVENTS" in content:
            print("CRISIS_EVENTS list found.")
            # Print first few lines of definition to check format
            for line in content.splitlines():
                if "CRISIS_EVENTS = [" in line:
                    print("Structure:", line[:100], "...")
                    break
except Exception as e:
    print(f"Could not read crisis_labels.py: {e}")
