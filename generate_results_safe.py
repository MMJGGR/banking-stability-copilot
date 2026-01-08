
import pandas as pd
from train_model import BankingRiskModel
from src.data_loader import IMFDataLoader

print("Running SAFE Result Generator...")

def generate_results():
    print("Loading data...")
    loader = IMFDataLoader()
    if not loader.load_from_cache():
        loader.load_all_datasets()
    
    fsic_df = loader._data_cache.get('FSIC', pd.DataFrame())
    weo_df = loader._data_cache.get('WEO', pd.DataFrame())
    mfs_df = loader._data_cache.get('MFS', pd.DataFrame())
    
    print("Training model...")
    model = BankingRiskModel()
    ret = model.train(fsic_df, weo_df, mfs_df)
    
    # Dynamic unpacking
    if isinstance(ret, tuple):
        print(f"Model returned tuple of length {len(ret)}")
        results = ret[0]
    else:
        print("Model returned single object")
        results = ret
    
    output_path = "model_results.csv"
    results.to_csv(output_path, index=False)
    print(f"\nSaved model results to: {output_path}")

if __name__ == "__main__":
    generate_results()
