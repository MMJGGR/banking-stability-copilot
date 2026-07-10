"""Quick check on Senegal's deposit data."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

fsic = pd.read_parquet('cache/FSIC_cache.parquet')
sen = fsic[fsic['country_code'] == 'SEN']

print("Senegal FSIC deposit-related indicators:")
dep = sen[sen['indicator_name'].str.contains('deposit|funding', case=False, na=False)]
for ind in dep['indicator_name'].unique():
    latest = dep[dep['indicator_name'] == ind].sort_values('period')['value'].iloc[-1]
    print(f"  {ind[:70]}: {latest:,.2f}")

# Also check what the Data Explorer sees for "deposit funding"
print("\n\nChecking deposits to total assets percent indicator:")
deposits_pct = sen[sen['indicator_name'].str.contains('Deposits to total.*assets.*Percent', case=False, na=False, regex=True)]
if len(deposits_pct) > 0:
    for ind in deposits_pct['indicator_name'].unique():
        latest = deposits_pct[deposits_pct['indicator_name'] == ind].sort_values('period')['value'].iloc[-1]
        print(f"  {ind}: {latest:.2f}%")
else:
    print("  No Deposits to total assets Percent indicator found")
