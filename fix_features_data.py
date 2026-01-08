
import pandas as pd
import sys
import os
from src.config import CACHE_DIR
from src.data_loader import IMFDataLoader
from src.feature_engineering import CrisisFeatureEngineer
try:
    from src.data_loader_fsibsis import load_fsibsis_features
except ImportError:
    print("WARNING: Could not import FSIBSIS loader")

print("FORCE UPDATING FEATURE DATABASE...")

# 1. Load Data
loader = IMFDataLoader()
loader.load_from_cache()
fsic = loader.load_fsic()
weo = loader.load_weo()
mfs = loader.load_mfs()

print(f"Loaded raw data: FSIC {len(fsic)}, WEO {len(weo)}, MFS {len(mfs)}")

# 2. Run Engineer
engineer = CrisisFeatureEngineer()
print("Extracting features...")
fsic_feats = engineer.extract_fsic_features(fsic)
weo_feats = engineer.extract_weo_features(weo)
credit_gap = engineer.compute_credit_to_gdp_gap(mfs, weo)
sov_nexus = engineer.compute_sovereign_bank_nexus(mfs, weo)

print("Loading FSIBSIS features...")
fsibsis = load_fsibsis_features()

# 3. Merge
print("Merging features...")
features = engineer.merge_features(
    weo_features=weo_feats,
    fsic_features=fsic_feats,
    credit_gap=credit_gap,
    sovereign_nexus=sov_nexus,
    fsibsis_features=fsibsis
)

# 4. Save
out_path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
features.to_parquet(out_path, index=False)
print(f"SUCCESS: Saved {len(features)} rows to {out_path}")

# Verify
ken = features[features['country_code'] == 'KEN'].iloc[0]
print(f"\nVERIFICATION:")
print(f"Kenya Sovereign Exposure: {ken.get('sovereign_exposure_ratio', 'MISSING')}")
print(f"Kenya NPL Ratio: {ken.get('npl_ratio', 'MISSING')}")
