"""Quick check on NPL coverage features."""
import pandas as pd

f = pd.read_parquet('cache/crisis_features.parquet')
print('Columns with provision/npl:')
cols = [c for c in f.columns if 'provision' in c.lower() or c == 'npl_provisions']
print(cols)
for c in cols:
    print(f'  {c}: {f[c].notna().sum()} countries')

# Check imputed features
try:
    imp = pd.read_parquet('cache/imputed_features.parquet')
    print('\nIn imputed_features.parquet:')
    cols2 = [c for c in imp.columns if 'provision' in c.lower() or c == 'npl_provisions']
    for c in cols2:
        print(f'  {c}: {imp[c].notna().sum()} countries')
except:
    print('No imputed_features.parquet found')
