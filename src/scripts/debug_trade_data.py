import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_loader import IMFDataLoader

def scan_trade_data():
    print("Scanning WEO for Trade Balance / Exports...")
    loader = IMFDataLoader()
    loader.load_from_cache()
    weo = loader._data_cache.get('WEO')
    
    # Common codes
    # TX: Exports of goods
    # BCA: Current Account
    # BCA_NGDPD: Current Account % GDP
    
    codes = ['TX', 'TX_RP_CIF', 'BCA', 'BCA_NGDPD']
    
    print("\nCoverage:")
    for code in codes:
        count = weo[weo['indicator_code'] == code]['country_code'].nunique()
        print(f"  {code}: {count} countries")

if __name__ == "__main__":
    scan_trade_data()
