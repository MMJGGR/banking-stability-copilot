
import pandas as pd
import sys
import os
from src.data_loader import IMFDataLoader
from src.feature_engineering import CrisisFeatureEngineer
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def generate_report():
    print("=" * 70)
    print("GENERATING GLOBAL DATA GAP REPORT")
    print("=" * 70)
    
    # Load data
    loader = IMFDataLoader()
    loader.load_from_cache()
    
    fsic_df = loader._data_cache.get('FSIC', pd.DataFrame())
    weo_df = loader._data_cache.get('WEO', pd.DataFrame())
    mfs_df = loader._data_cache.get('MFS', pd.DataFrame())
    
    engineer = CrisisFeatureEngineer()
    weo_features = engineer.extract_weo_features(weo_df)
    fsic_features = engineer.extract_fsic_features(fsic_df)
    credit_gap = engineer.compute_credit_to_gdp_gap(mfs_df, weo_df)
    
    features = engineer.merge_features(weo_features, fsic_features, credit_gap)
    
    # Define Industry (Banking) Columns
    industry_cols = [
        'capital_adequacy', 
        'npl_ratio', 
        'roe', 
        'roa', 
        'liquid_assets_st_liab', 
        'liquid_assets_total', 
        'customer_deposits_loans', 
        'fx_loan_exposure', 
        'tier1_capital', 
        'npl_provisions'
    ]
    fsic_cols = [c for c in industry_cols if c in features.columns]
    
    # Calculate Coverage
    features['fsic_coverage'] = features[fsic_cols].notna().mean(axis=1)
    
    # Add Names
    names = weo_df[['country_code', 'country_name']].drop_duplicates().set_index('country_code')
    features = features.join(names, on='country_code')
    
    # Sort by GDP
    if 'nominal_gdp' in features.columns:
         features = features.sort_values('nominal_gdp', ascending=False)
    
    # Generate Full CSV Report
    report_data = []
    
    for _, row in features.iterrows():
        gdp_val = row.get('nominal_gdp', 0)
        cov = row.get('fsic_coverage', 0)
        
        status = "CRITICAL (0%)" if cov == 0 else ("PARTIAL" if cov < 0.8 else "GOOD")
        
        report_data.append({
            'Country_Code': row['country_code'],
            'Country_Name': row['country_name'],
            'GDP_USD_Billions': round(gdp_val, 2) if pd.notna(gdp_val) else 0,
            'Banking_Data_Coverage_Pct': round(cov * 100, 1),
            'Data_Status': status
        })
        
    report_df = pd.DataFrame(report_data)
    
    # Filter out aggregates (keep only if GDP < 50,000 Billion? Or filter by known aggregate codes)
    # Aggregates usually have names like "Advanced Economies", "World", "G7".
    # Checking for specific aggregate codes is hard without a list.
    # But usually aggregates don't have detailed FSIC data either?
    # Let's keep them but note they might be aggregates.
    
    csv_path = "missing_data_report.csv"
    report_df.to_csv(csv_path, index=False)
    print(f"\nSaved Global Missing Data Report to: {os.path.abspath(csv_path)}")
    
    print("\nTOP 10 'CRITICAL' BLIND SPOTS (0% Data, Sorted by GDP):")
    print("-" * 80)
    print(f"{'Code':<6} {'Country Name':<25} {'GDP (Billions)':<15} {'Coverage':<10} {'Data Status':<15}")
    print("-" * 80)
    
    # Filter for 0% and non-trivial GDP
    blind_spots = report_df[report_df['Banking_Data_Coverage_Pct'] == 0]
    
    count = 0
    for _, row in blind_spots.iterrows():
        # Heuristic: Skip obvious aggregates if possible (usually start with '1' or '9' or 'G'?)
        # Let's just print.
        print(f"{row['Country_Code']:<6} {str(row['Country_Name'])[:25]:<25} ${row['GDP_USD_Billions']:<15,.1f} {row['Banking_Data_Coverage_Pct']}%      {row['Data_Status']:<15}")
        count += 1
        if count >= 10:
            break

    print("\nSummary:")
    print(f"Total Entities Analysis: {len(report_df)}")
    print(f"Zero coverage: {len(blind_spots)} ({len(blind_spots)/len(report_df):.1%})")
    print(f"Good coverage (>80%): {len(report_df[report_df['Banking_Data_Coverage_Pct'] >= 80])}")

if __name__ == "__main__":
    generate_report()
