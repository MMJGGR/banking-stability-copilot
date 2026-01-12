"""
Feature Comparison: Raw Extraction vs Parquet

Extracts features from raw datasets using the same logic as train_model.py,
then compares with crisis_features.parquet to identify divergences.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CACHE_DIR
from src.data_loader import IMFDataLoader, FSIBSISLoader, WGILoader, load_fsibsis_features
from src.feature_engineering import CrisisFeatureEngineer

OUTPUT_PATH = os.path.join('replication', 'outputs', 'feature_comparison.csv')


def extract_raw_features():
    """Extract features from raw datasets using same logic as train_model.py"""
    print("="*70)
    print("EXTRACTING FEATURES FROM RAW DATASETS")
    print("="*70)
    
    # Load raw datasets
    loader = IMFDataLoader()
    loader.load_from_cache()
    
    fsic_df = loader._data_cache.get('FSIC', pd.DataFrame())
    weo_df = loader._data_cache.get('WEO', pd.DataFrame())
    mfs_df = loader._data_cache.get('MFS', pd.DataFrame())
    
    print(f"  FSIC: {len(fsic_df):,} records")
    print(f"  WEO: {len(weo_df):,} records")
    print(f"  MFS: {len(mfs_df):,} records")
    
    # Feature Engineering
    engineer = CrisisFeatureEngineer()
    
    print("\n  Extracting WEO features...")
    weo_features = engineer.extract_weo_features(weo_df)
    
    print("  Extracting FSIC features...")
    fsic_features = engineer.extract_fsic_features(fsic_df)
    
    print("  Computing credit-to-GDP gap...")
    credit_gap = engineer.compute_credit_to_gdp_gap(mfs_df, weo_df)
    
    print("  Computing sovereign-bank nexus...")
    sovereign_nexus = engineer.compute_sovereign_bank_nexus(mfs_df, weo_df)
    
    # WGI
    print("  Loading WGI...")
    wgi_loader = WGILoader()
    wgi_features = wgi_loader.get_latest_scores()
    
    # FSIBSIS
    print("  Loading FSIBSIS...")
    fsibsis_features = load_fsibsis_features()
    
    # Merge (without correlation dropping for raw comparison)
    print("\n  Merging features...")
    features = engineer.merge_features(
        weo_features=weo_features,
        fsic_features=fsic_features,
        credit_gap=credit_gap,
        sovereign_nexus=sovereign_nexus,
        wgi_features=wgi_features,
        fsibsis_features=fsibsis_features
    )
    
    print(f"  Raw features: {len(features)} countries, {len(features.columns)} columns")
    return features


def load_parquet_features():
    """Load features from parquet file"""
    parquet_path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    print(f"\nParquet features: {len(df)} countries, {len(df.columns)} columns")
    return df


def compare_features(raw_df, parquet_df):
    """Compare raw extraction vs parquet, identify divergences"""
    print("\n" + "="*70)
    print("COMPARING RAW VS PARQUET FEATURES")
    print("="*70)
    
    # Find common countries
    common_countries = set(raw_df['country_code']) & set(parquet_df['country_code'])
    print(f"\n  Common countries: {len(common_countries)}")
    
    # Find common feature columns (exclude metadata)
    exclude_cols = ['country_code', 'country_name']
    raw_cols = set(c for c in raw_df.columns if c not in exclude_cols and not c.endswith('_period'))
    parquet_cols = set(c for c in parquet_df.columns if c not in exclude_cols and not c.endswith('_period') and not c.endswith('_year'))
    
    common_cols = raw_cols & parquet_cols
    only_raw = raw_cols - parquet_cols
    only_parquet = parquet_cols - raw_cols
    
    print(f"  Common features: {len(common_cols)}")
    print(f"  Only in raw: {len(only_raw)} - {sorted(only_raw)[:5]}{'...' if len(only_raw) > 5 else ''}")
    print(f"  Only in parquet: {len(only_parquet)} - {sorted(only_parquet)[:5]}{'...' if len(only_parquet) > 5 else ''}")
    
    # Build comparison dataframe
    comparison_rows = []
    
    for country in sorted(common_countries):
        raw_row = raw_df[raw_df['country_code'] == country].iloc[0]
        parquet_row = parquet_df[parquet_df['country_code'] == country].iloc[0]
        
        for col in sorted(common_cols):
            raw_val = raw_row.get(col)
            parquet_val = parquet_row.get(col)
            
            # Calculate difference
            raw_num = pd.to_numeric(raw_val, errors='coerce')
            parquet_num = pd.to_numeric(parquet_val, errors='coerce')
            
            if pd.notna(raw_num) and pd.notna(parquet_num):
                diff = raw_num - parquet_num
                pct_diff = (diff / parquet_num * 100) if parquet_num != 0 else 0
                match = 'MATCH' if abs(pct_diff) < 1 else ('CLOSE' if abs(pct_diff) < 5 else 'DIVERGE')
            elif pd.isna(raw_num) and pd.isna(parquet_num):
                diff = 0
                pct_diff = 0
                match = 'BOTH_NULL'
            elif pd.isna(raw_num):
                diff = None
                pct_diff = None
                match = 'RAW_NULL'
            else:
                diff = None
                pct_diff = None
                match = 'PARQUET_NULL'
            
            comparison_rows.append({
                'country_code': country,
                'feature': col,
                'raw_value': raw_val,
                'parquet_value': parquet_val,
                'difference': diff,
                'pct_difference': pct_diff,
                'status': match
            })
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Summary stats
    print("\n  Comparison Summary:")
    status_counts = comparison_df['status'].value_counts()
    for status, count in status_counts.items():
        pct = count / len(comparison_df) * 100
        print(f"    {status}: {count:,} ({pct:.1f}%)")
    
    # Show top divergences
    diverge_df = comparison_df[comparison_df['status'] == 'DIVERGE'].copy()
    if len(diverge_df) > 0:
        diverge_df['abs_pct'] = diverge_df['pct_difference'].abs()
        top_diverge = diverge_df.nlargest(20, 'abs_pct')
        print(f"\n  Top 20 Divergences (>5% difference):")
        for _, row in top_diverge.iterrows():
            print(f"    {row['country_code']} {row['feature']}: raw={row['raw_value']:.2f}, parquet={row['parquet_value']:.2f} ({row['pct_difference']:+.1f}%)")
    
    return comparison_df


def main():
    # Extract from raw
    raw_features = extract_raw_features()
    
    # Load parquet
    parquet_features = load_parquet_features()
    
    # Compare
    comparison = compare_features(raw_features, parquet_features)
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    comparison.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved comparison to: {OUTPUT_PATH}")
    print(f"  Total rows: {len(comparison):,}")
    
    # Also save a summary by feature
    feature_summary = comparison.groupby('feature').agg({
        'status': lambda x: (x == 'MATCH').sum() / len(x) * 100,
        'pct_difference': lambda x: x.abs().mean()
    }).rename(columns={'status': 'match_rate_pct', 'pct_difference': 'avg_abs_diff_pct'})
    feature_summary = feature_summary.sort_values('match_rate_pct')
    
    summary_path = OUTPUT_PATH.replace('.csv', '_by_feature.csv')
    feature_summary.to_csv(summary_path)
    print(f"  Saved feature summary to: {summary_path}")
    
    return comparison


if __name__ == "__main__":
    main()
