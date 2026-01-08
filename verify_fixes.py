
import pandas as pd
import sys
import os
from src.data_loader import IMFDataLoader
from src.feature_engineering import CrisisFeatureEngineer
from src.data_loader_fsibsis import load_fsibsis_features

print("VERIFYING FIXES...")

# 1. Load Data
loader = IMFDataLoader()
loader.load_from_cache()
fsic = loader.load_fsic()
weo = loader.load_weo()
mfs = loader.load_mfs()


# Filter to just USA/KEN to speed up
countries = ['USA', 'KEN', 'DEU']
fsic = fsic[fsic['country_code'].isin(countries)]
weo = weo[weo['country_code'].isin(countries)]
mfs = mfs[mfs['country_code'].isin(countries)]

# 2. Run Engineer
engineer = CrisisFeatureEngineer()
fsic_feats = engineer.extract_fsic_features(fsic)
weo_feats = engineer.extract_weo_features(weo)
credit_gap = engineer.compute_credit_to_gdp_gap(mfs, weo)
sov_nexus = engineer.compute_sovereign_bank_nexus(mfs, weo)
fsibsis = load_fsibsis_features()
if fsibsis is not None:
    fsibsis = fsibsis[fsibsis['country_code'].isin(countries)]

# 3. Merge
features = engineer.merge_features(
    weo_features=weo_feats,
    fsic_features=fsic_feats,
    credit_gap=credit_gap,
    sovereign_nexus=sov_nexus,
    fsibsis_features=fsibsis
)

print("\nRESULTS:")
cols = ['country_code', 'npl_ratio', 'sovereign_exposure_ratio', 'securities_to_assets']
print(features[cols])

# Checks
ken = features[features['country_code'] == 'KEN'].iloc[0]
usa = features[features['country_code'] == 'USA'].iloc[0]

print("\nVERIFICATION CHECKS:")

# CHECK 1: NPL Positive?
if ken['npl_ratio'] > 0:
    print(f"[PASS] Kenya NPL is positive: {ken['npl_ratio']:.2f}%")
else:
    print(f"[FAIL] Kenya NPL is negative: {ken['npl_ratio']:.2f}%")

if usa['npl_ratio'] > 0:
    print(f"[PASS] USA NPL is positive: {usa['npl_ratio']:.2f}%")
else:
    print(f"[FAIL] USA NPL is negative: {usa['npl_ratio']:.2f}%")

# CHECK 2: Kenya Sovereign Exposure > 2%?
if ken['sovereign_exposure_ratio'] > 2.0:
    print(f"[PASS] Kenya Sovereign Exposure fixed: {ken['sovereign_exposure_ratio']:.2f}% (Target ~27%)")
else:
    print(f"[FAIL] Kenya Sovereign Exposure still low: {ken['sovereign_exposure_ratio']:.2f}%")
