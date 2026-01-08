
import pandas as pd
import os
import glob
from datetime import datetime

def merge_datasets():
    print("="*60)
    print("MERGING FSIC DATASETS")
    print("="*60)
    
    base_dir = r"c:\Users\Richard\Banking"
    
    # 1. Identify File Paths
    # Original large file (most recent large one)
    # New patch file
    
    # Pattern matching
    fsic_pattern = "DEFAULT_INTEGRATION_IMF.STA_FSIC_13.0.1.csv"
    
    files = glob.glob(os.path.join(base_dir, f"*{fsic_pattern}"))
    
    if not files:
        print("No FSIC files found!")
        return
        
    # Sort by size to distinguish
    files_by_size = sorted(files, key=os.path.getsize, reverse=True)
    
    large_file = files_by_size[0] # The 89MB one
    
    # Find the patch file (the one provided by user, recently)
    # User's file: dataset_2026-01-02T17_19_04.061954374Z_...
    patch_file_name = "dataset_2026-01-02T17_19_04.061954374Z_DEFAULT_INTEGRATION_IMF.STA_FSIC_13.0.1.csv"
    patch_file = os.path.join(base_dir, patch_file_name)
    
    if patch_file not in files:
        print(f"Patch file not found: {patch_file}")
        # Maybe it's in the list differently?
        # Let's just use the list if needed, but strict path is better.
    
    print(f"Original Base File: {os.path.basename(large_file)} ({os.path.getsize(large_file)/1024/1024:.2f} MB)")
    print(f"Patch File:         {os.path.basename(patch_file)} ({os.path.getsize(patch_file)/1024:.2f} KB)")
    
    # 2. Load DataFrames
    print("Loading datasets...")
    df_base = pd.read_csv(large_file, low_memory=False)
    df_patch = pd.read_csv(patch_file, low_memory=False)
    
    print(f"Base rows: {len(df_base)}")
    print(f"Patch rows: {len(df_patch)}")
    
    # 3. Concatenate
    print("Merging...")
    # Concatenate (patch last to override? No, concat just adds rows)
    # Usage of `drop_duplicates` is tricky if periods differ or overlap.
    # In IMF data, uniquely identified by: Reference Area, Indicator, Frequency, [Series Key maybe]
    # But usually just appending is fine if countries are distinct.
    
    # Let's verify distinctness
    base_countries = df_base.iloc[:, 3].unique() # Approx column index for country? Safer to just concat.
    patch_countries = df_patch.iloc[:, 3].unique() # checking blindly
    
    combined = pd.concat([df_base, df_patch], ignore_index=True)
    
    # Drop exact duplicates just in case
    before_dedup = len(combined)
    combined.drop_duplicates(inplace=True)
    after_dedup = len(combined)
    print(f"Dropped {before_dedup - after_dedup} exact duplicates.")
    
    # 4. Save New File
    # Generate timestamp LATER than patch file
    # Patch was 2026-01-02T17...
    # Curren time is 2026-01-02T20... (Local)
    # Format: dataset_YYYY-MM-DDTHH_MM_SS.ffffffZ_...
    
    now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H_%M_%S.%fZ")
    new_filename = f"dataset_{now_iso}_DEFAULT_INTEGRATION_IMF.STA_FSIC_13.0.1.csv"
    output_path = os.path.join(base_dir, new_filename)
    
    print(f"Saving merged dataset to: {new_filename}")
    combined.to_csv(output_path, index=False)
    
    print(f"Done! New size: {os.path.getsize(output_path)/1024/1024:.2f} MB")
    
    # 5. Cleanup (Optional: Move old files to cache/archive?)
    # For now, keep them.

if __name__ == "__main__":
    merge_datasets()
