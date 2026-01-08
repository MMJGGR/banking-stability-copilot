
import pandas as pd
import os

def verify_new_data():
    base_dir = r"c:\Users\Richard\Banking"
    new_file = "dataset_2026-01-02T17_19_04.061954374Z_DEFAULT_INTEGRATION_IMF.STA_FSIC_13.0.1.csv"
    path = os.path.join(base_dir, new_file)
    
    print(f"Inspecting: {path}")
    if not os.path.exists(path):
        print("File not found!")
        return

    try:
        df = pd.read_csv(path)
        print(f"Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        
        # Check Countries
        country_col = [c for c in df.columns if 'Reference Area' in c or 'Country' in c or 'REF_AREA' in c]
        if country_col:
            countries = df[country_col[0]].unique()
            print(f"Countries found ({len(countries)}): {countries}")
            
            targets = ['DEU', 'GBR', 'USA', 'Germany', 'United Kingdom', 'United States']
            found = [t for t in targets if t in countries or any(t in str(c) for c in countries)]
            print(f"Targets found: {found}")
            
        # Check Indicators
        ind_col = [c for c in df.columns if 'Indicator' in c or 'INDICATOR' in c]
        if ind_col:
            indicators = df[ind_col[0]].unique()
            print(f"Indicators found ({len(indicators)}):")
            for ind in indicators[:10]:
                print(f"  - {ind}")
                
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    verify_new_data()
