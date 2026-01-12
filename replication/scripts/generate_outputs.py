"""
Generate Replication Outputs

This script generates all outputs needed to validate the model methodology:
1. Model scores for all countries (CSV)
2. Full feature matrix (CSV)
3. PCA loadings (JSON)
4. Sample data extracts

Run from project root: python replication/scripts/generate_outputs.py
"""

import os
import sys
import json
import pandas as pd

# Add project root to path (two levels up from this script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config import CACHE_DIR

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)


def generate_model_scores():
    """Export final risk scores for all countries."""
    print("\n[1/4] Generating model scores...")
    
    from train_model import BankingRiskModel
    
    try:
        model = BankingRiskModel.load()
        scores = model.get_all_scores()
        
        # Select key columns for output
        output_cols = [
            'country_code', 'country_name', 'risk_score', 'risk_category',
            'economic_pillar', 'industry_pillar', 'combined_pillar',
            'data_coverage', 'economic_coverage', 'industry_coverage',
            'risk_floor_applied'
        ]
        
        available_cols = [c for c in output_cols if c in scores.columns]
        output_df = scores[available_cols].copy()
        
        output_path = os.path.join(OUTPUT_DIR, 'model_scores.csv')
        output_df.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")
        print(f"  Countries: {len(output_df)}")
        
        return output_df
        
    except FileNotFoundError:
        print("  ERROR: Model not found. Run 'python train_model.py' first.")
        return None


def generate_feature_matrix():
    """Export full feature matrix used in model."""
    print("\n[2/4] Generating feature matrix...")
    
    parquet_path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
    
    if not os.path.exists(parquet_path):
        print(f"  ERROR: Features not found at {parquet_path}")
        return None
    
    features = pd.read_parquet(parquet_path)
    
    # Remove period columns for cleaner output
    feature_cols = [c for c in features.columns if not c.endswith('_period')]
    output_df = features[feature_cols].copy()
    
    output_path = os.path.join(OUTPUT_DIR, 'feature_matrix.csv')
    output_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    print(f"  Shape: {output_df.shape[0]} countries x {output_df.shape[1]} features")
    
    return output_df


def generate_pca_loadings():
    """Export PCA component loadings."""
    print("\n[3/4] Generating PCA loadings...")
    
    from train_model import BankingRiskModel
    
    try:
        model = BankingRiskModel.load()
        pca_info = model.pca_info
        
        output_path = os.path.join(OUTPUT_DIR, 'pca_loadings.json')
        with open(output_path, 'w') as f:
            json.dump(pca_info, f, indent=2)
        
        print(f"  Saved: {output_path}")
        print(f"  Economic features: {len(pca_info.get('economic_loadings', {}))}")
        print(f"  Industry features: {len(pca_info.get('industry_loadings', {}))}")
        
        return pca_info
        
    except FileNotFoundError:
        print("  ERROR: Model not found. Run 'python train_model.py' first.")
        return None


def generate_sample_data():
    """Generate sample data extracts (1000 rows each)."""
    print("\n[4/4] Generating sample data...")
    
    from src.data_loader import IMFDataLoader, FSIBSISLoader
    
    loader = IMFDataLoader()
    loader.load_from_cache()
    
    samples_created = 0
    
    # FSIC sample
    fsic_df = loader._data_cache.get('FSIC')
    if fsic_df is not None and len(fsic_df) > 0:
        sample = fsic_df.head(1000)
        sample.to_csv(os.path.join(SAMPLE_DIR, 'sample_fsic.csv'), index=False)
        samples_created += 1
        print(f"  sample_fsic.csv: {len(sample)} rows")
    
    # WEO sample
    weo_df = loader._data_cache.get('WEO')
    if weo_df is not None and len(weo_df) > 0:
        sample = weo_df.head(1000)
        sample.to_csv(os.path.join(SAMPLE_DIR, 'sample_weo.csv'), index=False)
        samples_created += 1
        print(f"  sample_weo.csv: {len(sample)} rows")
    
    # MFS sample
    mfs_df = loader._data_cache.get('MFS')
    if mfs_df is not None and len(mfs_df) > 0:
        sample = mfs_df.head(1000)
        sample.to_csv(os.path.join(SAMPLE_DIR, 'sample_mfs.csv'), index=False)
        samples_created += 1
        print(f"  sample_mfs.csv: {len(sample)} rows")
    
    # FSIBSIS sample
    try:
        fsibsis_loader = FSIBSISLoader()
        fsibsis_loader.load()
        if fsibsis_loader.bank_data is not None:
            sample = fsibsis_loader.bank_data.head(1000)
            sample.to_csv(os.path.join(SAMPLE_DIR, 'sample_fsibsis.csv'), index=False)
            samples_created += 1
            print(f"  sample_fsibsis.csv: {len(sample)} rows")
    except Exception as e:
        print(f"  FSIBSIS sample failed: {e}")
    
    print(f"  Total samples: {samples_created}")
    return samples_created


def main():
    print("=" * 60)
    print("GENERATING REPLICATION OUTPUTS")
    print("=" * 60)
    
    generate_model_scores()
    generate_feature_matrix()
    generate_pca_loadings()
    generate_sample_data()
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"Sample data saved to: {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
