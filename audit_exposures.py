
import pandas as pd
import os
from src.data_loader_fsibsis import FSIBSISLoader
from src.config import CACHE_DIR

# 1. Load FSIBSIS Data (Raw)
loader = FSIBSISLoader()
loader.load()
# Removed invalid attribution access

# Helper to get raw
countries = ['KEN', 'NGA', 'ZAF', 'GHA', 'UGA', 'TZA', 'RWA', 'ETH', 'AGO', 'SEN']
mappings = loader.INDICATOR_MAPPINGS

print(f"{'Country':<8} {'Large(LCY)':<12} {'Assets(LCY)':<12} {'RegCap(LCY)':<12} {'L.Exp/Assets(%)':<18} {'L.Exp/RegCap(%)':<18} {'Sov(MFS)':<10} {'Sec/Assets':<10}")

# Load MFS for comparison (load from cache if possible, else simplified)
mfs_path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
mfs_data = pd.read_parquet(mfs_path)
mfs_map = mfs_data.set_index('country_code')['sovereign_exposure_ratio'].to_dict()

for country in countries:
    # Get raw values (most recent year found)
    l_exp = loader.get_indicator_value(country, 'large_exposures')
    assets = loader.get_indicator_value(country, 'total_assets')
    reg_cap = loader.get_indicator_value(country, 'regulatory_capital')
    
    # Ratios
    ratio_assets = (l_exp / assets * 100) if (l_exp and assets) else 0.0
    ratio_reg_cap = (l_exp / reg_cap * 100) if (l_exp and reg_cap) else 0.0
    
    # Sec to Assets (calculated internally in fsibsis)
    sec = loader.get_indicator_value(country, 'debt_securities')
    sec_ratio = (sec / assets * 100) if (sec and assets) else 0.0
    
    # MFS Sovereign
    mfs_val = mfs_map.get(country, float('nan'))
    
    def fmt_scientific(val):
        if not val: return "-"
        return f"{val:.2e}"

    print(f"{country:<8} {fmt_scientific(l_exp):<12} {fmt_scientific(assets):<12} {fmt_scientific(reg_cap):<12} {ratio_assets: <18.2f} {ratio_reg_cap: <18.2f} {mfs_val:<10.2f} {sec_ratio:<10.2f}")
