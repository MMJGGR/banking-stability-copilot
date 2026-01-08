
import sys
import os
import time
sys.path.insert(0, os.getcwd())
from src.data_loader import IMFDataLoader

print("Starting loader test...")
start = time.time()
loader = IMFDataLoader()
success = loader.load_from_cache()
print(f"Load from cache success: {success}")
if success:
    print(f"FSIC shape: {loader.load_fsic().shape}")
    print(f"WEO shape: {loader.load_weo().shape}")
    print(f"MFS shape: {loader.load_mfs().shape}")
print(f"Time taken: {time.time() - start:.2f}s")
