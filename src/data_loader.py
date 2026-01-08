"""
Data loader for IMF datasets (FSIC, MFS, WEO).
OPTIMIZED: Uses vectorized pandas operations for fast loading.
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.config import (
    BASE_DIR, DATA_DIR, CACHE_DIR, DATASET_PATTERNS,
    FSIC_CORE_INDICATORS, WEO_CORE_INDICATORS, MFS_CORE_INDICATORS
)


class IMFDataLoader:
    """
    Optimized loader for IMF statistical datasets.
    Uses vectorized operations for fast CSV processing.
    """
    
    def __init__(self, data_dir: str = None, cache_dir: str = None):
        self.data_dir = data_dir or DATA_DIR
        self.cache_dir = cache_dir or CACHE_DIR
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def find_dataset_files(self) -> Dict[str, str]:
        """Find IMF dataset files in data directory and base directory."""
        files = {}
        search_paths = [self.data_dir, BASE_DIR]
        
        for search_path in search_paths:
            for pattern_name, pattern in DATASET_PATTERNS.items():
                for ext in ['*.csv', '*.CSV']:
                    file_pattern = os.path.join(search_path, f"*{pattern}*{ext}")
                    matches = glob.glob(file_pattern)
                    if matches and pattern_name not in files:
                        files[pattern_name] = max(matches, key=os.path.getmtime)
                        
        return files
    
    def _identify_time_columns(self, columns: List[str]) -> Tuple[List[str], List[str]]:
        """
        Fast identification of time period columns vs metadata columns.
        """
        time_cols = []
        meta_cols = []
        
        # Pre-compile regex patterns
        year_pattern = re.compile(r'^(19|20)\d{2}$')
        quarter_pattern = re.compile(r'^(Q[1-4]\s*\d{4}|\d{4}[-_]?Q[1-4])$', re.I)
        month_pattern = re.compile(r'^(M\d{2}\s*\d{4}|\d{4}[-_]?\d{2})$', re.I)
        
        for col in columns:
            col_str = str(col).strip()
            if year_pattern.match(col_str) or quarter_pattern.match(col_str) or month_pattern.match(col_str):
                time_cols.append(col)
            else:
                meta_cols.append(col)
                
        return meta_cols, time_cols
    
    def _vectorized_parse_periods(self, period_series: pd.Series) -> pd.Series:
        """Vectorized period parsing."""
        periods = period_series.astype(str).str.strip()
        
        # Initialize result
        result = pd.Series(pd.NaT, index=periods.index)
        
        # Annual: 1980, 2024, etc.
        annual_mask = periods.str.match(r'^(19|20)\d{2}$', na=False)
        if annual_mask.any():
            years = periods[annual_mask].astype(int)
            result[annual_mask] = pd.to_datetime(years.astype(str) + '-12-31')
        
        # Quarterly: Q1 2020
        q1_mask = periods.str.match(r'^Q([1-4])\s*(\d{4})$', case=False, na=False)
        if q1_mask.any():
            q_extract = periods[q1_mask].str.extract(r'^Q([1-4])\s*(\d{4})$', flags=re.I)
            q_extract.columns = ['q', 'year']
            months = q_extract['q'].astype(int) * 3
            result[q1_mask] = pd.to_datetime(
                q_extract['year'] + '-' + months.astype(str).str.zfill(2) + '-01'
            )
        
        # Quarterly: 2020-Q1
        q2_mask = periods.str.match(r'^(\d{4})[-_]?Q([1-4])$', case=False, na=False)
        if q2_mask.any():
            q_extract = periods[q2_mask].str.extract(r'^(\d{4})[-_]?Q([1-4])$', flags=re.I)
            q_extract.columns = ['year', 'q']
            months = q_extract['q'].astype(int) * 3
            result[q2_mask] = pd.to_datetime(
                q_extract['year'] + '-' + months.astype(str).str.zfill(2) + '-01'
            )
        
        return result
    
    def _load_and_melt(self, filepath: str, dataset_name: str) -> pd.DataFrame:
        """
        Load CSV and convert from wide to long format using vectorized melt.
        This is the key optimization - 100x faster than iterrows.
        """
        print(f"Loading {dataset_name} from: {filepath}")
        
        # Read CSV
        df = pd.read_csv(filepath, low_memory=False, encoding='utf-8-sig')
        print(f"  Read {len(df)} rows, {len(df.columns)} columns")
        
        # Identify time vs metadata columns
        meta_cols, time_cols = self._identify_time_columns(df.columns.tolist())
        
        if not time_cols:
            print(f"  Warning: No time columns found, checking column patterns...")
            # Try numeric column detection
            numeric_cols = [c for c in df.columns if str(c).replace('.', '').isdigit()]
            if numeric_cols:
                time_cols = numeric_cols
                meta_cols = [c for c in df.columns if c not in time_cols]
        
        print(f"  {len(time_cols)} time columns, {len(meta_cols)} metadata columns")
        
        if not time_cols:
            print(f"  ERROR: Could not identify time columns")
            return pd.DataFrame()
        
        # Identify key metadata columns
        series_key_col = None
        country_name_col = None
        indicator_name_col = None
        freq_col = None
        unit_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            if 'series_key' in col_lower or col == df.columns[1]:
                series_key_col = col
            elif col_lower == 'country' or col_lower == 'reference area':
                # Prioritize exact match for country/reference area columns
                country_name_col = col
            elif ('reference area' in col_lower or col_lower == 'ref_area') and country_name_col is None:
                # Fallback to partial match only if exact not found, but exclude date columns
                if 'date' not in col_lower:
                    country_name_col = col
            elif col_lower == 'indicator' or col_lower == 'indicator_name':
                indicator_name_col = col
            elif col_lower in ['freq', 'frequency']:
                freq_col = col
            elif 'unit' in col_lower and 'date' not in col_lower:
                unit_col = col
        
        # Extract country/indicator from SERIES_KEY (vectorized)
        if series_key_col and series_key_col in df.columns:
            key_parts = df[series_key_col].astype(str).str.split('.', expand=True)
            df['_country_code'] = key_parts[0].str.upper().str[:3] if 0 in key_parts.columns else ''
            df['_indicator_code'] = key_parts[1] if 1 in key_parts.columns else ''
        else:
            df['_country_code'] = ''
            df['_indicator_code'] = ''
        
        # Get country name
        if country_name_col and country_name_col in df.columns:
            df['_country_name'] = df[country_name_col].fillna('')
        else:
            df['_country_name'] = ''
        
        # Get indicator name  
        if indicator_name_col and indicator_name_col in df.columns:
            df['_indicator_name'] = df[indicator_name_col].fillna('')
        else:
            df['_indicator_name'] = ''
        
        # Get frequency
        if freq_col and freq_col in df.columns:
            df['_frequency'] = df[freq_col].fillna('Q')
        else:
            df['_frequency'] = 'Q' if dataset_name == 'FSIC' else ('A' if dataset_name == 'WEO' else 'M')
        
        # Get unit
        if unit_col and unit_col in df.columns:
            df['_unit'] = df[unit_col].fillna('')
        else:
            df['_unit'] = ''
        
        # Identify columns to keep for ID
        id_vars = ['_country_code', '_country_name', '_indicator_code', '_indicator_name', '_frequency', '_unit']
        
        # MELT: Convert wide to long format (vectorized, fast!)
        print(f"  Melting {len(df)} rows × {len(time_cols)} periods...")
        
        melted = df.melt(
            id_vars=id_vars,
            value_vars=time_cols,
            var_name='period_str',
            value_name='value'
        )
        
        print(f"  Melted to {len(melted)} records")
        
        # Drop empty values (vectorized)
        melted = melted.dropna(subset=['value'])
        melted = melted[melted['value'].astype(str).str.strip() != '']
        melted = melted[~melted['value'].astype(str).str.lower().isin(['nan', 'na', ''])]
        
        # Convert value to numeric (vectorized)
        melted['value'] = pd.to_numeric(melted['value'], errors='coerce')
        melted = melted.dropna(subset=['value'])
        
        print(f"  {len(melted)} non-null records after filtering")
        
        if len(melted) == 0:
            return pd.DataFrame()
        
        # Parse periods (vectorized)
        melted['period'] = self._vectorized_parse_periods(melted['period_str'])
        
        # Rename columns
        melted = melted.rename(columns={
            '_country_code': 'country_code',
            '_country_name': 'country_name',
            '_indicator_code': 'indicator_code',
            '_indicator_name': 'indicator_name',
            '_frequency': 'frequency',
            '_unit': 'unit'
        })
        
        # Add dataset identifier
        melted['dataset'] = dataset_name
        
        # Sort
        melted = melted.sort_values(['country_code', 'indicator_code', 'period'])
        
        print(f"  Final: {len(melted)} records for {melted['country_code'].nunique()} countries")
        
        return melted
    
    def load_fsic(self, filepath: str = None) -> pd.DataFrame:
        """Load Financial Soundness Indicators data."""
        if 'FSIC' in self._data_cache and filepath is None:
            return self._data_cache['FSIC']
            
        if filepath is None:
            files = self.find_dataset_files()
            filepath = files.get('FSIC')
            if not filepath:
                raise FileNotFoundError("FSIC dataset not found")
        
        result = self._load_and_melt(filepath, 'FSIC')
        self._data_cache['FSIC'] = result
        return result
    
    def load_weo(self, filepath: str = None) -> pd.DataFrame:
        """Load World Economic Outlook data."""
        if 'WEO' in self._data_cache and filepath is None:
            return self._data_cache['WEO']
            
        if filepath is None:
            files = self.find_dataset_files()
            filepath = files.get('WEO')
            if not filepath:
                raise FileNotFoundError("WEO dataset not found")
        
        result = self._load_and_melt(filepath, 'WEO')
        self._data_cache['WEO'] = result
        return result
    
    def load_mfs(self, filepath: str = None) -> pd.DataFrame:
        """Load Monetary and Financial Statistics data."""
        if 'MFS' in self._data_cache and filepath is None:
            return self._data_cache['MFS']
            
        if filepath is None:
            files = self.find_dataset_files()
            filepath = files.get('MFS')
            if not filepath:
                raise FileNotFoundError("MFS dataset not found")
        
        result = self._load_and_melt(filepath, 'MFS')
        self._data_cache['MFS'] = result
        return result
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all available IMF datasets."""
        datasets = {}
        
        for name in ['FSIC', 'WEO', 'MFS']:
            try:
                if name == 'FSIC':
                    datasets[name] = self.load_fsic()
                elif name == 'WEO':
                    datasets[name] = self.load_weo()
                elif name == 'MFS':
                    datasets[name] = self.load_mfs()
                print(f"Loaded {name}: {len(datasets[name])} records")
            except FileNotFoundError as e:
                print(f"Warning: {name} dataset not found - {e}")
            except Exception as e:
                print(f"Error loading {name}: {e}")
                import traceback
                traceback.print_exc()
                
        return datasets
    
    def get_countries(self, dataset: str = None) -> pd.DataFrame:
        """Get list of available countries with their coverage stats."""
        all_data = []
        
        if dataset:
            datasets = {dataset: self._data_cache.get(dataset, pd.DataFrame())}
        else:
            datasets = self._data_cache
            
        for name, df in datasets.items():
            if df is not None and len(df) > 0:
                country_stats = df.groupby(['country_code', 'country_name']).agg({
                    'indicator_code': 'nunique',
                    'period': ['min', 'max', 'count']
                }).reset_index()
                country_stats.columns = ['country_code', 'country_name', 
                                        'n_indicators', 'min_period', 'max_period', 'n_observations']
                country_stats['dataset'] = name
                all_data.append(country_stats)
                
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def get_indicators(self, dataset: str) -> pd.DataFrame:
        """Get list of available indicators for a dataset."""
        df = self._data_cache.get(dataset, pd.DataFrame())
        
        if df is not None and len(df) > 0:
            indicator_stats = df.groupby(['indicator_code', 'indicator_name']).agg({
                'country_code': 'nunique',
                'period': ['min', 'max', 'count']
            }).reset_index()
            indicator_stats.columns = ['indicator_code', 'indicator_name',
                                      'n_countries', 'min_period', 'max_period', 'n_observations']
            return indicator_stats
            
        return pd.DataFrame()
    
    def get_country_data(self, country_code: str, dataset: str = None) -> pd.DataFrame:
        """Get all data for a specific country."""
        results = []
        
        datasets = {dataset: self._data_cache.get(dataset)} if dataset else self._data_cache
        
        for name, df in datasets.items():
            if df is not None and len(df) > 0:
                country_data = df[df['country_code'] == country_code.upper()].copy()
                results.append(country_data)
                
        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()
    
    def save_cache(self):
        """Save loaded data to parquet cache for instant loading next time."""
        for name, df in self._data_cache.items():
            if df is not None and len(df) > 0:
                cache_path = os.path.join(self.cache_dir, f"{name}_cache.parquet")
                df.to_parquet(cache_path, index=False)
                print(f"Cached {name} to {cache_path}")
    
    def load_from_cache(self) -> bool:
        """Load data from parquet cache if available."""
        loaded = False
        
        for name in ['FSIC', 'WEO', 'MFS']:
            cache_path = os.path.join(self.cache_dir, f"{name}_cache.parquet")
            if os.path.exists(cache_path):
                try:
                    self._data_cache[name] = pd.read_parquet(cache_path)
                    print(f"Loaded {name} from cache: {len(self._data_cache[name])} records")
                    loaded = True
                except Exception as e:
                    print(f"Error loading {name} cache: {e}")
                    
        return loaded


def create_unified_dataset(loader: IMFDataLoader) -> pd.DataFrame:
    """
    Create a unified country-period matrix combining all datasets.
    """
    datasets = loader.load_all_datasets()
    
    unified_dfs = []
    
    for name, df in datasets.items():
        if df is not None and len(df) > 0:
            pivot = df.pivot_table(
                index=['country_code', 'period'],
                columns='indicator_code',
                values='value',
                aggfunc='mean'
            ).reset_index()
            
            pivot.columns = [f"{name}_{c}" if c not in ['country_code', 'period'] 
                           else c for c in pivot.columns]
            unified_dfs.append(pivot)
    
    if not unified_dfs:
        return pd.DataFrame()
    
    result = unified_dfs[0]
    for df in unified_dfs[1:]:
        result = result.merge(df, on=['country_code', 'period'], how='outer')
    
    return result.sort_values(['country_code', 'period'])
