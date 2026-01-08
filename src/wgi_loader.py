"""
==============================================================================
World Governance Indicators (WGI) Data Loader
==============================================================================
Source: World Bank Worldwide Governance Indicators
        https://www.worldbank.org/en/publication/worldwide-governance-indicators

This module loads WGI data for integration into the banking risk model,
following S&P BICRA methodology which uses governance factors in both:
- Economic Risk assessment (political stability, voice & accountability)
- Industry Risk - Institutional Framework (rule of law, corruption, regulatory quality)

WGI Dimensions:
- va: Voice and Accountability
- pv: Political Stability and Absence of Violence/Terrorism
- ge: Government Effectiveness
- rq: Regulatory Quality
- rl: Rule of Law
- cc: Control of Corruption

Author: Banking Copilot
Date: 2026-01-03
==============================================================================
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional

# WGI sheet names and their full descriptions
WGI_INDICATORS = {
    'va': 'voice_accountability',      # Voice and Accountability
    'pv': 'political_stability',       # Political Stability
    'ge': 'govt_effectiveness',        # Government Effectiveness
    'rq': 'regulatory_quality',        # Regulatory Quality
    'rl': 'rule_of_law',               # Rule of Law
    'cc': 'control_corruption',        # Control of Corruption
}

# BICRA pillar mapping (per S&P methodology)
BICRA_ECONOMIC_INDICATORS = ['voice_accountability', 'political_stability', 'govt_effectiveness']
BICRA_INDUSTRY_INDICATORS = ['regulatory_quality', 'rule_of_law', 'control_corruption']


class WGILoader:
    """
    Loader for World Governance Indicators dataset.
    
    Extracts governance scores (0-100 scale) for all 6 dimensions
    and maps them to BICRA Economic/Industry pillars.
    """
    
    def __init__(self, wgi_path: str = None):
        """
        Initialize WGI loader.
        
        Args:
            wgi_path: Path to WGI Excel file. If None, searches in project root.
        """
        self.wgi_path = wgi_path or self._find_wgi_file()
        self.data: Optional[pd.DataFrame] = None
        
    def _find_wgi_file(self) -> str:
        """Find WGI Excel file in project directory."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Try common file names
        candidates = [
            'wgidataset_with_sourcedata-2025.xlsx',
            'wgidataset.xlsx',
            'WGI.xlsx',
        ]
        
        for filename in candidates:
            path = os.path.join(project_root, filename)
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            f"WGI dataset not found. Searched for: {candidates} in {project_root}"
        )
    
    def load(self) -> pd.DataFrame:
        """
        Load WGI data from all 6 sheets and merge into single DataFrame.
        
        Returns:
            DataFrame with columns: country_code, year, and all 6 governance scores
        """
        print("\n" + "="*70)
        print("LOADING WORLD GOVERNANCE INDICATORS")
        print("="*70)
        print(f"  Source: {self.wgi_path}")
        
        all_data = []
        
        for sheet_code, feature_name in WGI_INDICATORS.items():
            try:
                df = pd.read_excel(self.wgi_path, sheet_name=sheet_code)
                
                # Extract relevant columns
                # WGI uses 'Economy (code)' for country codes and 'Governance score (0-100)'
                df_clean = df[['Economy (code)', 'Year', 'Governance score (0-100)']].copy()
                df_clean.columns = ['country_code', 'year', feature_name]
                
                # Drop rows with missing scores
                df_clean = df_clean.dropna(subset=[feature_name])
                
                all_data.append(df_clean)
                print(f"  {sheet_code.upper()} ({feature_name}): {len(df_clean)} records")
                
            except Exception as e:
                print(f"  WARNING: Failed to load sheet '{sheet_code}': {e}")
        
        if not all_data:
            raise ValueError("No WGI data could be loaded")
        
        # Merge all indicators
        merged = all_data[0]
        for df in all_data[1:]:
            merged = merged.merge(df, on=['country_code', 'year'], how='outer')
        
        self.data = merged
        print(f"\n  Total: {len(merged)} country-year records, "
              f"{merged['country_code'].nunique()} countries")
        
        return merged
    
    def get_latest_scores(self) -> pd.DataFrame:
        """
        Get the most recent governance scores for each country.
        
        Returns:
            DataFrame with one row per country, containing latest scores
        """
        if self.data is None:
            self.load()
        
        # Get latest year per country
        latest = self.data.sort_values('year').groupby('country_code').last().reset_index()
        
        # Drop the year column (not needed for feature engineering)
        feature_cols = ['country_code'] + list(WGI_INDICATORS.values())
        available_cols = [c for c in feature_cols if c in latest.columns]
        latest = latest[available_cols]
        
        print(f"\n  Latest scores: {len(latest)} countries")
        
        # Show sample
        sample_countries = ['USA', 'DEU', 'GBR', 'CHN', 'IND', 'NGA', 'GHA', 'VEN']
        sample = latest[latest['country_code'].isin(sample_countries)]
        if len(sample) > 0:
            print("\n  Sample Governance Scores (0-100, higher=better):")
            print(sample.to_string(index=False))
        
        return latest
    
    def get_economic_pillar_features(self) -> pd.DataFrame:
        """
        Get WGI features mapped to BICRA Economic pillar.
        
        Returns:
            DataFrame with country_code and economic governance features
        """
        latest = self.get_latest_scores()
        
        cols = ['country_code'] + [c for c in BICRA_ECONOMIC_INDICATORS if c in latest.columns]
        return latest[cols]
    
    def get_industry_pillar_features(self) -> pd.DataFrame:
        """
        Get WGI features mapped to BICRA Industry pillar (Institutional Framework).
        
        Returns:
            DataFrame with country_code and industry governance features
        """
        latest = self.get_latest_scores()
        
        cols = ['country_code'] + [c for c in BICRA_INDUSTRY_INDICATORS if c in latest.columns]
        return latest[cols]


def load_wgi_features() -> pd.DataFrame:
    """
    Convenience function to load WGI features for model training.
    
    Returns:
        DataFrame with country_code and all 6 governance scores
    """
    loader = WGILoader()
    return loader.get_latest_scores()


if __name__ == "__main__":
    # Test loading
    loader = WGILoader()
    scores = loader.get_latest_scores()
    
    print("\n" + "="*70)
    print("WGI FEATURE SUMMARY")
    print("="*70)
    
    print("\nEconomic Pillar Features:")
    econ = loader.get_economic_pillar_features()
    print(econ.describe().round(1))
    
    print("\nIndustry Pillar Features:")
    industry = loader.get_industry_pillar_features()
    print(industry.describe().round(1))
