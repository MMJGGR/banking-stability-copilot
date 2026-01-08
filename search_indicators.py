
import pandas as pd
import os
from src.data_loader import IMFDataLoader

def search_keywords():
    print("Loading data for search...")
    loader = IMFDataLoader()
    # Try loading from cache first
    if not loader.load_from_cache():
        print("Cache missing, loading from CSVs...")
        loader.load_all_datasets()
    
    keywords = ['real estate', 'house', 'housing', 'residential', 'household', 'corporate', 'private sector', 'm2', 'broad money', 'money supply']
    
    print(f"\nSearching for keywords: {keywords}")
    
    hits = []
    
    for dataset_name in ['FSIC', 'WEO', 'MFS']:
        df = loader._data_cache.get(dataset_name)
        if df is None:
            continue
            
        print(f"Scanning {dataset_name} ({len(df)} rows)...")
        
        # Get unique indicators
        indicators = df[['indicator_code', 'indicator_name']].drop_duplicates()
        
        for _, row in indicators.iterrows():
            code = str(row['indicator_code'])
            name = str(row['indicator_name']).lower()
            
            for k in keywords:
                if k in name:
                    hits.append({
                        'dataset': dataset_name,
                        'code': code,
                        'name': row['indicator_name'],
                        'keyword': k
                    })
    
    # Sort and print
    hits_df = pd.DataFrame(hits)
    if not hits_df.empty:
        hits_df = hits_df.drop_duplicates(subset=['dataset', 'code'])
        print(f"\nFound {len(hits_df)} matching indicators:")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', 100)
        print(hits_df[['dataset', 'code', 'name']].to_string(index=False))
    else:
        print("No matches found.")

if __name__ == "__main__":
    search_keywords()
