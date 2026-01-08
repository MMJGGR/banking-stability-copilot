
import pandas as pd
import sys
import os
from src.data_loader import IMFDataLoader
from src.feature_engineering import CrisisFeatureEngineer

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def inspect_fra_aus():
    print("=" * 70)
    print("INSPECTING DATA FOR FRANCE (FRA) AND AUSTRALIA (AUS)")
    print("=" * 70)
    
    # Load data
    loader = IMFDataLoader()
    loader.load_from_cache()
    
    fsic_df = loader._data_cache.get('FSIC', pd.DataFrame())
    weo_df = loader._data_cache.get('WEO', pd.DataFrame())
    mfs_df = loader._data_cache.get('MFS', pd.DataFrame())
    
    target_countries = ['FRA', 'AUS', 'CAN'] # Added CAN for comparison (Low Risk)
    
    # 1. Inspect Raw WEO Data (Macro)
    print("\n--- 1. Macro Economic Data (WEO) ---")
    weo_indicators = {
        'NGDP_RPCH': 'GDP Growth',
        'GGXWDG_NGDP': 'Govt Debt/GDP', 
        'PCPIPCH': 'Inflation',
        'BCA_NGDPD': 'Current Account/GDP'
    }
    
    for country in target_countries:
        print(f"\n[{country}]")
        country_data = weo_df[weo_df['country_code'] == country]
        if len(country_data) == 0:
            print("  No WEO data found")
            continue
            
        for ind_code, ind_name in weo_indicators.items():
            # Get specific indicator
            rows = country_data[country_data['indicator_code'] == ind_code]
            if len(rows) > 0:
                # Sort by period to get latest
                latest = rows.sort_values('period').iloc[-1]
                print(f"  {ind_name:<20}: {latest['value']:<10.2f} (Year: {latest['period']})")
            else:
                print(f"  {ind_name:<20}: N/A")

    # 2. Inspect Raw FSIC Data (Banking)
    print("\n--- 2. Banking Sector Data (FSIC) ---")
    # key fsic codes often vary, let's look for common ones using FeatureEngineer mapping or raw codes if known
    # Using codes from feature_engineering.py
    fsic_indicators = {
        'FASMB_PA': 'Capital Adequacy',
        'FCNBAD_PA': 'NPL Ratio',
        'FCNROA_PA': 'ROA',
        'FCNROE_PA': 'ROE'
    }
    
    for country in target_countries:
        print(f"\n[{country}]")
        country_data = fsic_df[fsic_df['country_code'] == country]
        
        for ind_code, ind_name in fsic_indicators.items():
            rows = country_data[country_data['indicator_code'] == ind_code]
            if len(rows) > 0:
                latest = rows.sort_values('period').iloc[-1]
                print(f"  {ind_name:<20}: {latest['value']:<10.2f} (Year: {latest['period']})")
            else:
                print(f"  {ind_name:<20}: N/A")
                
    # 3. Inspect Credit to GDP Gap (MFS)
    print("\n--- 3. Credit Cycle (MFS) ---")
    # We need to run the engineer logic to see the GAP
    engineer = CrisisFeatureEngineer()
    credit_gap_df = engineer.compute_credit_to_gdp_gap(mfs_df, weo_df)
    
    for country in target_countries:
        row = credit_gap_df[credit_gap_df['country_code'] == country]
        if len(row) > 0:
            print(f"\n[{country}]")
            print(f"  Credit-to-GDP Ratio : {row.iloc[0]['credit_to_gdp']:.2f}")
            print(f"  Credit-to-GDP Gap   : {row.iloc[0]['credit_to_gdp_gap']:.2f}")
            # Try to infer year? The engineer takes latest common year.
            # We can check raw mfs
            raw_credit = mfs_df[
                (mfs_df['country_code'] == country) & 
                (mfs_df['indicator_code'] == 'FOS_PT') # Credit to private sector
            ]
            if len(raw_credit) > 0:
                latest = raw_credit.sort_values('period').iloc[-1]
                print(f"  Raw Credit Data Year: {latest['period']}")
        else:
             print(f"\n[{country}] No Credit Gap Calculated")

if __name__ == "__main__":
    inspect_fra_aus()
