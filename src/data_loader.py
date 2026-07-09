"""
Data loader for IMF datasets (FSIC, MFS, WEO) and related data sources.
OPTIMIZED: Uses vectorized pandas operations for fast loading.

This module consolidates:
- IMFDataLoader: FSIC, MFS, WEO datasets
- FSIBSISLoader: Balance sheet data from FSIBSIS dataset
- WGILoader: World Governance Indicators
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import (
    BASE_DIR, DATA_DIR, CACHE_DIR, DATASET_PATTERNS,
    FSIC_CORE_INDICATORS, WEO_CORE_INDICATORS, MFS_CORE_INDICATORS
)
from src.lfs_resolver import ensure_lfs_file

TIME_PERIOD_PATTERN = re.compile(
    r"^((19|20)\d{2}|Q[1-4]\s*(19|20)\d{2}|(19|20)\d{2}[-_]?Q[1-4]|"
    r"M(0[1-9]|1[0-2])\s*(19|20)\d{2}|(19|20)\d{2}[-_]?M?(0[1-9]|1[0-2]))$",
    re.IGNORECASE,
)


def parse_period_label(value) -> pd.Timestamp:
    """Parse IMF annual, quarterly, and monthly column labels."""
    label = str(value).strip().upper()

    if re.fullmatch(r"(19|20)\d{2}", label):
        return pd.Timestamp(f"{label}-12-31")

    match = re.fullmatch(r"Q([1-4])\s*((19|20)\d{2})", label)
    if match:
        return pd.Period(
            year=int(match.group(2)), quarter=int(match.group(1)), freq="Q"
        ).end_time.normalize()

    match = re.fullmatch(r"((19|20)\d{2})[-_]?Q([1-4])", label)
    if match:
        return pd.Period(
            year=int(match.group(1)), quarter=int(match.group(3)), freq="Q"
        ).end_time.normalize()

    match = re.fullmatch(r"M(0[1-9]|1[0-2])\s*((19|20)\d{2})", label)
    if match:
        return pd.Period(
            year=int(match.group(2)), month=int(match.group(1)), freq="M"
        ).end_time.normalize()

    match = re.fullmatch(r"((19|20)\d{2})[-_]?M?(0[1-9]|1[0-2])", label)
    if match:
        return pd.Period(
            year=int(match.group(1)), month=int(match.group(3)), freq="M"
        ).end_time.normalize()

    return pd.NaT


def is_time_period_column(value) -> bool:
    return bool(TIME_PERIOD_PATTERN.fullmatch(str(value).strip()))



class IMFDataLoader:
    """
    Optimized loader for IMF statistical datasets.
    Uses vectorized operations for fast CSV processing.
    """
    
    def __init__(self, data_dir: str = None, cache_dir: str = None):
        self.data_dir = data_dir or DATA_DIR
        self.cache_dir = cache_dir or CACHE_DIR
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._country_data_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        
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
        
        for col in columns:
            if is_time_period_column(col):
                time_cols.append(col)
            else:
                meta_cols.append(col)
                
        return meta_cols, time_cols
    
    def _vectorized_parse_periods(self, period_series: pd.Series) -> pd.Series:
        """Vectorized period parsing."""
        periods = period_series.astype(str).str.strip()
        mapping = {
            label: parse_period_label(label)
            for label in periods.dropna().drop_duplicates()
        }
        return periods.map(mapping)
    
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
        latest_actual_col = None
        
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
            elif col_lower == 'latest_actual_annual_data':
                latest_actual_col = col
        
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

        if latest_actual_col and latest_actual_col in df.columns:
            df['_latest_actual_year'] = pd.to_numeric(
                df[latest_actual_col], errors='coerce'
            )
        else:
            df['_latest_actual_year'] = np.nan
        
        # Identify columns to keep for ID
        id_vars = [
            '_country_code', '_country_name', '_indicator_code',
            '_indicator_name', '_frequency', '_unit', '_latest_actual_year'
        ]
        
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
        invalid_periods = int(melted['period'].isna().sum())
        if invalid_periods:
            logger.warning(
                "Dropping %s %s observations with unrecognized periods",
                invalid_periods,
                dataset_name,
            )
            melted = melted.dropna(subset=['period'])
        
        # Rename columns
        melted = melted.rename(columns={
            '_country_code': 'country_code',
            '_country_name': 'country_name',
            '_indicator_code': 'indicator_code',
            '_indicator_name': 'indicator_name',
            '_frequency': 'frequency',
            '_unit': 'unit',
            '_latest_actual_year': 'latest_actual_year',
        })

        observation_year = melted['period'].dt.year
        melted['observation_status'] = 'unknown'
        has_actual_cutoff = melted['latest_actual_year'].notna()
        melted.loc[
            has_actual_cutoff
            & (observation_year <= melted['latest_actual_year']),
            'observation_status',
        ] = 'actual'
        melted.loc[
            has_actual_cutoff
            & (observation_year == melted['latest_actual_year'] + 1),
            'observation_status',
        ] = 'estimate'
        melted.loc[
            has_actual_cutoff
            & (observation_year > melted['latest_actual_year'] + 1),
            'observation_status',
        ] = 'projection'
        
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
    
    def _load_country_from_cache(self, country_code: str, dataset: str) -> pd.DataFrame:
        """Load one country's parquet cache slice without materializing the full table."""
        country_code = country_code.upper()
        cache_key = (dataset, country_code)

        if cache_key in self._country_data_cache:
            return self._country_data_cache[cache_key].copy()

        cache_path = os.path.join(self.cache_dir, f"{dataset}_cache.parquet")
        if not os.path.exists(cache_path):
            return pd.DataFrame()

        try:
            ensure_lfs_file(cache_path)
            country_data = pd.read_parquet(
                cache_path,
                engine="pyarrow",
                filters=[("country_code", "==", country_code)],
            )
        except Exception as e:
            logger.warning(
                "Unable to load %s country slice for %s from cache: %s",
                dataset,
                country_code,
                e,
            )
            return pd.DataFrame()

        # Keep this bounded because Streamlit holds the loader as a cached resource.
        if len(self._country_data_cache) >= 24:
            oldest_key = next(iter(self._country_data_cache))
            self._country_data_cache.pop(oldest_key, None)

        self._country_data_cache[cache_key] = country_data
        return country_data.copy()

    def get_country_data(self, country_code: str, dataset: str = None) -> pd.DataFrame:
        """Get all data for a specific country."""
        results = []

        if dataset:
            dataset_names = [dataset]
        else:
            cached_names = list(self._data_cache.keys())
            default_names = ['FSIC', 'WEO', 'MFS']
            dataset_names = cached_names or default_names
        
        for name in dataset_names:
            df = self._data_cache.get(name)
            if df is not None and len(df) > 0:
                country_data = df[df['country_code'] == country_code.upper()].copy()
                results.append(country_data)
            else:
                country_data = self._load_country_from_cache(country_code, name)
                if country_data is not None and len(country_data) > 0:
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
                    ensure_lfs_file(cache_path)
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


# =============================================================================
# FSIBSIS LOADER - Balance Sheet Data
# =============================================================================

# Country name to ISO code mapping for FSIBSIS
COUNTRY_NAME_TO_ISO = {
    'Afghanistan, Islamic Republic of': 'AFG', 'Albania': 'ALB', 'Algeria': 'DZA',
    'Angola': 'AGO', 'Argentina': 'ARG', 'Armenia, Republic of': 'ARM',
    'Australia': 'AUS', 'Austria': 'AUT', 'Azerbaijan, Republic of': 'AZE',
    'Bangladesh': 'BGD', 'Barbados': 'BRB', 'Belarus, Republic of': 'BLR',
    'Belgium': 'BEL', 'Belize': 'BLZ', 'Bhutan': 'BTN', 'Bolivia': 'BOL',
    'Bosnia and Herzegovina': 'BIH', 'Botswana': 'BWA', 'Brazil': 'BRA',
    'Brunei Darussalam': 'BRN', 'Bulgaria': 'BGR', 'Burundi': 'BDI',
    'Cambodia': 'KHM', 'Cameroon': 'CMR', 'Canada': 'CAN',
    'Central African Republic': 'CAF', 'Chad': 'TCD', 'Chile': 'CHL',
    'China, P.R.: Mainland': 'CHN', 'Colombia': 'COL', 'Comoros, Union of the': 'COM',
    'Congo, Dem. Rep. of the': 'COD', 'Congo, Republic of': 'COG', 'Costa Rica': 'CRI',
    'Croatia': 'HRV', 'Cyprus': 'CYP', 'Czech Republic': 'CZE', 'Denmark': 'DNK',
    'Dominican Republic': 'DOM', 'Ecuador': 'ECU', 'Egypt': 'EGY', 'El Salvador': 'SLV',
    'Equatorial Guinea': 'GNQ', 'Estonia': 'EST', 'Eswatini': 'SWZ', 'Ethiopia': 'ETH',
    'Fiji': 'FJI', 'Finland': 'FIN', 'France': 'FRA', 'Gabon': 'GAB',
    'Gambia, The': 'GMB', 'Georgia': 'GEO', 'Germany': 'DEU', 'Ghana': 'GHA',
    'Greece': 'GRC', 'Guatemala': 'GTM', 'Guinea': 'GIN', 'Guinea-Bissau': 'GNB',
    'Guyana': 'GUY', 'Haiti': 'HTI', 'Honduras': 'HND', 'Hungary': 'HUN',
    'Iceland': 'ISL', 'India': 'IND', 'Indonesia': 'IDN',
    'Iran, Islamic Republic of': 'IRN', 'Iraq': 'IRQ', 'Ireland': 'IRL',
    'Israel': 'ISR', 'Italy': 'ITA', 'Jamaica': 'JAM', 'Japan': 'JPN',
    'Jordan': 'JOR', 'Kazakhstan': 'KAZ', 'Kenya': 'KEN', 'Korea, Republic of': 'KOR',
    'Kosovo': 'XKX', 'Kuwait': 'KWT', 'Kyrgyz Republic': 'KGZ', 'Lao P.D.R.': 'LAO',
    'Latvia': 'LVA', 'Lebanon': 'LBN', 'Lesotho': 'LSO', 'Liberia': 'LBR',
    'Libya': 'LBY', 'Lithuania': 'LTU', 'Luxembourg': 'LUX', 'Madagascar': 'MDG',
    'Malawi': 'MWI', 'Malaysia': 'MYS', 'Maldives': 'MDV', 'Mali': 'MLI',
    'Malta': 'MLT', 'Mauritania': 'MRT', 'Mauritius': 'MUS', 'Mexico': 'MEX',
    'Moldova, Republic of': 'MDA', 'Mongolia': 'MNG', 'Montenegro': 'MNE',
    'Morocco': 'MAR', 'Mozambique': 'MOZ', 'Myanmar': 'MMR', 'Namibia': 'NAM',
    'Nepal': 'NPL', 'Netherlands': 'NLD', 'New Zealand': 'NZL', 'Nicaragua': 'NIC',
    'Niger': 'NER', 'Nigeria': 'NGA', 'North Macedonia': 'MKD', 'Norway': 'NOR',
    'Oman': 'OMN', 'Pakistan': 'PAK', 'Panama': 'PAN', 'Papua New Guinea': 'PNG',
    'Paraguay': 'PRY', 'Peru': 'PER', 'Philippines': 'PHL', 'Poland': 'POL',
    'Portugal': 'PRT', 'Qatar': 'QAT', 'Romania': 'ROU', 'Russian Federation': 'RUS',
    'Rwanda': 'RWA', 'Samoa': 'WSM', 'Saudi Arabia': 'SAU', 'Senegal': 'SEN',
    'Serbia': 'SRB', 'Seychelles': 'SYC', 'Sierra Leone': 'SLE', 'Singapore': 'SGP',
    'Slovak Republic': 'SVK', 'Slovenia': 'SVN', 'Solomon Islands': 'SLB',
    'South Africa': 'ZAF', 'South Sudan': 'SSD', 'Spain': 'ESP', 'Sri Lanka': 'LKA',
    'Sudan': 'SDN', 'Suriname': 'SUR', 'Sweden': 'SWE', 'Switzerland': 'CHE',
    'Syrian Arab Republic': 'SYR', 'Tajikistan': 'TJK',
    'Tanzania, United Republic of': 'TZA', 'Thailand': 'THA', 'Timor-Leste': 'TLS',
    'Togo': 'TGO', 'Tonga': 'TON', 'Trinidad and Tobago': 'TTO', 'Tunisia': 'TUN',
    'Turkey': 'TUR', 'Turkmenistan': 'TKM', 'Uganda': 'UGA', 'Ukraine': 'UKR',
    'United Arab Emirates': 'ARE', 'United Kingdom': 'GBR', 'United States': 'USA',
    'Uruguay': 'URY', 'Uzbekistan': 'UZB', 'Vanuatu': 'VUT', 'Venezuela': 'VEN',
    'Vietnam': 'VNM', 'Yemen, Republic of': 'YEM', 'Zambia': 'ZMB', 'Zimbabwe': 'ZWE',
}


class FSIBSISLoader:
    """
    Loader for IMF FSIBSIS (Financial Soundness Indicators Balance Sheet) dataset.
    Provides balance sheet data for deposit takers (banks).
    """
    
    INDICATOR_MAPPINGS = {
        'interest_income': 'Interest income, Domestic currency',
        'interest_expenses': 'Interest expenses, Domestic currency',
        'net_interest_income': 'Net interest income, Domestic currency',
        'total_assets': 'Total assets, Assets, Domestic currency',
        'gross_loans': 'Gross loans, Assets, Domestic currency',
        'debt_securities': 'Debt securities, Assets, Domestic currency',
        'noninterbank_loans': 'Noninterbank loans, Assets, Domestic currency',
        'govt_claims': 'Noninterbank loans, General government, Assets, Domestic currency',
        'residential_re_loans': 'Residential real estate loans, Domestic currency',
        'total_liabilities': 'Total liabilities, Domestic currency',
        'deposits': 'Currency and deposits, Liabilities, Domestic currency',
        'customer_deposits': 'Customer deposits, Liabilities, Domestic currency',
        'interbank_deposits': 'Interbank deposits, Liabilities, Domestic currency',
        'capital_reserves': 'Capital and reserves, Domestic currency',
        'regulatory_capital': 'Total regulatory capital, Domestic currency',
        'tier1_capital': 'Tier 1 capital less corresponding supervisory deductions, Domestic currency',
        'npl': 'Nonperforming loans, Domestic currency',
        'specific_provisions': 'Specific provisions, Assets, Domestic currency',
        'noninterest_income': 'Noninterest income, Domestic currency',
        'large_exposures': 'Value of large exposures, Domestic currency',
    }
    
    def __init__(self, file_path: str = None):
        self.file_path = file_path
        self.cache_path = os.path.join(CACHE_DIR, 'FSIBSIS_cache.parquet')
        self.data = None
        self.bank_data = None
        self.year_cols = []
        self.period_cols = []
        
    def load(self, file_path: str = None) -> pd.DataFrame:
        """Load and parse FSIBSIS dataset."""
        path = file_path or self.file_path

        if path is None and os.path.exists(self.cache_path):
            ensure_lfs_file(self.cache_path)
            self.bank_data = pd.read_parquet(self.cache_path)
            self.period_cols = [
                col for col in self.bank_data.columns
                if is_time_period_column(col)
            ]
            self.period_cols = sorted(
                self.period_cols, key=lambda col: parse_period_label(col)
            )
            self.year_cols = [
                col for col in self.period_cols
                if re.fullmatch(r'(19|20)\d{2}', str(col))
            ]
            logger.info(
                "Loaded FSIBSIS cache: %s rows, %s countries",
                len(self.bank_data),
                self.bank_data['country_code'].nunique(),
            )
            return self.bank_data
        
        if path is None:
            pattern = os.path.join(BASE_DIR, '*FSIBSIS*.csv')
            files = glob.glob(pattern)
            if files:
                path = files[0]
            else:
                raise FileNotFoundError("No FSIBSIS CSV file found")
        
        logger.info(f"Loading FSIBSIS from: {path}")
        
        chunks = []
        chunk_size = 50000
        first_chunk = pd.read_csv(path, nrows=5)
        all_cols = first_chunk.columns.tolist()
        period_cols = [c for c in all_cols if is_time_period_column(c)]
        cols_to_use = ['COUNTRY', 'SECTOR', 'INDICATOR'] + period_cols
        
        for chunk in pd.read_csv(path, usecols=cols_to_use, chunksize=chunk_size, low_memory=False):
            bank_chunk = chunk[chunk['SECTOR'] == 'Deposit takers']
            if len(bank_chunk) > 0:
                chunks.append(bank_chunk)
        
        self.bank_data = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols_to_use)
        self.period_cols = sorted(
            period_cols, key=lambda col: parse_period_label(col)
        )
        self.year_cols = [
            col for col in self.period_cols if re.fullmatch(r'(19|20)\d{2}', str(col))
        ]
        self.bank_data['country_code'] = self.bank_data['COUNTRY'].map(COUNTRY_NAME_TO_ISO)
        self.bank_data.to_parquet(self.cache_path, index=False)
        
        logger.info(f"FSIBSIS: {len(self.bank_data)} rows, {self.bank_data['country_code'].nunique()} countries")
        return self.bank_data
    
    def get_country_data(self, country_code: str) -> pd.DataFrame:
        """Get time-series data for a country compatible with Data Explorer."""
        if self.bank_data is None:
            self.load()
        country_data = self.bank_data[self.bank_data['country_code'] == country_code].copy()
        if len(country_data) == 0:
            return pd.DataFrame()
        cols_to_keep = ['INDICATOR'] + self.period_cols
        result = country_data[cols_to_keep].copy()
        result['INDICATOR'] = result['INDICATOR'].apply(lambda x: f"[BIS] {x}" if pd.notna(x) else x)
        return result
    
    def get_indicator_data(self, country_code: str, indicator_key: str, year: str = None) -> Tuple[Optional[float], Optional[str]]:
        """Get (value, year) for an indicator."""
        return self.get_indicator_data_as_of(
            country_code=country_code,
            indicator_key=indicator_key,
            year=year,
            as_of_date=None,
        )

    def get_indicator_data_as_of(
        self,
        country_code: str,
        indicator_key: str,
        year: str = None,
        as_of_date=None,
    ) -> Tuple[Optional[float], Optional[str]]:
        """Get (value, period label) for an indicator at or before a cutoff."""
        if self.bank_data is None:
            self.load()
        patterns = self._get_indicator_patterns(indicator_key)
        if not patterns:
            return None, None
        country_data = self.bank_data[self.bank_data['country_code'] == country_code]
        if len(country_data) == 0:
            return None, None
        period_cols = self.period_cols
        if as_of_date is not None:
            cutoff = pd.Timestamp(as_of_date).normalize()
            period_cols = [
                col for col in period_cols
                if parse_period_label(col) <= cutoff
            ]
        for pattern in patterns:
            mask = country_data['INDICATOR'] == pattern
            if not mask.any():
                mask = country_data['INDICATOR'].str.contains(pattern, case=False, na=False, regex=False)
            rows = country_data[mask]
            if len(rows) > 0:
                for yr in reversed(period_cols):
                    for _, row in rows.iterrows():
                        val = row.get(yr)
                        if pd.notnull(val):
                            if year and yr == year:
                                return val, yr
                            elif not year:
                                return val, yr
        return None, None
    
    def _get_indicator_patterns(self, indicator_key: str) -> List[str]:
        """Get list of patterns to try for an indicator."""
        base_patterns = {
            'total_assets': ['Total assets, Assets', 'Total assets'],
            'gross_loans': ['Gross loans, Assets', 'Gross loans'],
            'debt_securities': ['Debt securities, Assets', 'Debt securities'],
            'govt_claims': ['Noninterbank loans, General government, Assets'],
            'deposits': ['Currency and deposits, Liabilities', 'Customer deposits, Liabilities'],
            'customer_deposits': ['Customer deposits, Liabilities', 'Currency and deposits, Liabilities'],
            'interbank_deposits': ['Interbank deposits, Liabilities', 'Interbank deposits'],
            'interest_income': ['Interest income'],
            'interest_expenses': ['Interest expenses'],
            'net_interest_income': ['Net interest income'],
            'noninterest_income': ['Noninterest income'],
            'total_liabilities': ['Total liabilities'],
            'regulatory_capital': ['Total regulatory capital'],
            'tier1_capital': ['Tier 1 capital less corresponding supervisory deductions'],
            'specific_provisions': ['Specific provisions, Assets', 'Provisions (net): Loan loss provisions'],
            'large_exposures': ['Value of large exposures'],
            'residential_re_loans': ['Residential real estate loans'],
            'npl': ['Nonperforming loans'],  # For NPL coverage ratio
        }
        currency_suffixes = ['Domestic currency', 'Euro', 'US dollar']
        base_list = base_patterns.get(indicator_key, [])
        expanded = []
        for base in base_list:
            for suffix in currency_suffixes:
                expanded.append(f"{base}, {suffix}")
        return expanded
    
    def extract_features(self, as_of_date=None) -> pd.DataFrame:
        """Extract all new features from FSIBSIS for all countries as of a cutoff."""
        if self.bank_data is None:
            self.load()
        period_cols = self.period_cols
        if as_of_date is not None:
            cutoff = pd.Timestamp(as_of_date).normalize()
            period_cols = [
                col for col in period_cols
                if parse_period_label(col) <= cutoff
            ]
        if not period_cols:
            return pd.DataFrame()

        bank_data = self.bank_data[
            ['country_code', 'INDICATOR'] + period_cols
        ].dropna(subset=['country_code']).copy()
        period_values = bank_data[period_cols].apply(
            pd.to_numeric,
            errors='coerce',
        )
        present = period_values.notna().to_numpy()
        has_value = present.any(axis=1)
        if not has_value.any():
            return pd.DataFrame()
        reversed_present = present[:, ::-1]
        latest_positions = (
            len(period_cols) - 1 - reversed_present.argmax(axis=1)
        )
        latest_positions[~has_value] = -1
        row_positions = np.arange(len(bank_data))
        latest_value_array = np.full(len(bank_data), np.nan, dtype=float)
        valid_positions = latest_positions >= 0
        latest_value_array[valid_positions] = period_values.to_numpy()[
            row_positions[valid_positions],
            latest_positions[valid_positions],
        ]
        bank_data['_latest_value'] = latest_value_array
        period_labels = np.array(period_cols, dtype=object)
        latest_period_array = np.full(len(bank_data), None, dtype=object)
        latest_period_array[valid_positions] = period_labels[
            latest_positions[valid_positions]
        ]
        bank_data['_latest_period'] = latest_period_array
        period_order = {period: idx for idx, period in enumerate(period_cols)}
        bank_data = bank_data.dropna(subset=['_latest_value', '_latest_period'])
        bank_data['_period_order'] = bank_data['_latest_period'].map(period_order)

        values_by_key = {}
        years_by_key = {}
        for key in self.INDICATOR_MAPPINGS:
            patterns = self._get_indicator_patterns(key)
            if not patterns:
                continue
            mask = pd.Series(False, index=bank_data.index)
            for pattern in patterns:
                mask = mask | bank_data['INDICATOR'].eq(pattern)
            if not mask.any():
                for pattern in patterns:
                    mask = mask | bank_data['INDICATOR'].str.contains(
                        pattern,
                        case=False,
                        na=False,
                        regex=False,
                    )
            matched = bank_data.loc[mask]
            if len(matched) == 0:
                continue
            matched = (
                matched
                .sort_values(['country_code', '_period_order'])
                .drop_duplicates(subset='country_code', keep='last')
            )
            values_by_key[key] = matched.set_index('country_code')['_latest_value']
            years_by_key[key] = matched.set_index('country_code')['_latest_period']

        countries = bank_data['country_code'].dropna().unique()
        features_list = []
        
        for country in countries:
            features = {'country_code': country}
            values, years = {}, {}
            for key in self.INDICATOR_MAPPINGS:
                val = values_by_key.get(key, pd.Series(dtype=float)).get(country)
                yr = years_by_key.get(key, pd.Series(dtype=object)).get(country)
                values[key] = val
                years[key] = yr
            
            # Net Interest Margin
            if values.get('interest_income') and values.get('interest_expenses') and values.get('total_assets'):
                if values['total_assets'] > 0:
                    features['net_interest_margin'] = ((values['interest_income'] - values['interest_expenses']) / values['total_assets']) * 100
                    features['net_interest_margin_year'] = years.get('interest_income')
            
            # Interbank Funding Ratio
            if values.get('interbank_deposits') and values.get('total_liabilities'):
                if values['total_liabilities'] > 0:
                    features['interbank_funding_ratio'] = (values['interbank_deposits'] / values['total_liabilities'] * 100)
                    features['interbank_funding_ratio_year'] = years.get('interbank_deposits')
            
            # Income Diversification
            if values.get('noninterest_income') and values.get('net_interest_income'):
                total_income = values['net_interest_income'] + values['noninterest_income']
                if total_income > 0:
                    features['income_diversification'] = (values['noninterest_income'] / total_income * 100)
                    features['income_diversification_year'] = years.get('noninterest_income')
            
            # Securities-to-Assets
            if values.get('debt_securities') and values.get('total_assets'):
                if values['total_assets'] > 0:
                    features['securities_to_assets'] = (values['debt_securities'] / values['total_assets'] * 100)
                    features['securities_to_assets_year'] = years.get('debt_securities')
            
            # NPL Coverage Ratio (Provisions / NPL) - standard banking metric
            if values.get('specific_provisions') and values.get('npl'):
                if values['npl'] > 0:
                    features['specific_provisions_ratio'] = (values['specific_provisions'] / values['npl'] * 100)
                    features['specific_provisions_ratio_year'] = years.get('specific_provisions')
            
            # Large Exposure Ratio
            if values.get('large_exposures') and values.get('regulatory_capital'):
                if values['regulatory_capital'] > 0:
                    features['large_exposure_ratio'] = (values['large_exposures'] / values['regulatory_capital'] * 100)
                    features['large_exposure_ratio_year'] = years.get('large_exposures')
            
            # Sovereign Exposure
            if values.get('govt_claims') and values.get('gross_loans'):
                if values['gross_loans'] > 0:
                    features['sovereign_exposure_fsibsis'] = (values['govt_claims'] / values['gross_loans'] * 100)
                    features['sovereign_exposure_fsibsis_year'] = years.get('govt_claims')
            
            # Deposit Funding Ratio
            if values.get('deposits') and values.get('total_liabilities'):
                if values['total_liabilities'] > 0:
                    features['deposit_funding_ratio'] = (values['deposits'] / values['total_liabilities'] * 100)
                    features['deposit_funding_ratio_year'] = years.get('deposits')
            
            if len(features) > 1:
                features_list.append(features)
        
        return pd.DataFrame(features_list)


def load_fsibsis_features(file_path: str = None, as_of_date=None) -> pd.DataFrame:
    """Convenience function to load and extract FSIBSIS features."""
    loader = FSIBSISLoader(file_path)
    loader.load()
    return loader.extract_features(as_of_date=as_of_date)


# =============================================================================
# WGI LOADER - World Governance Indicators
# =============================================================================

WGI_INDICATORS = {
    'va': 'voice_accountability',
    'pv': 'political_stability',
    'ge': 'govt_effectiveness',
    'rq': 'regulatory_quality',
    'rl': 'rule_of_law',
    'cc': 'control_corruption',
}

BICRA_ECONOMIC_INDICATORS = ['voice_accountability', 'political_stability', 'govt_effectiveness']
BICRA_INDUSTRY_INDICATORS = ['regulatory_quality', 'rule_of_law', 'control_corruption']


class WGILoader:
    """
    Loader for World Governance Indicators dataset.
    Extracts governance scores (0-100 scale) for all 6 dimensions.
    """
    
    def __init__(self, wgi_path: str = None):
        self.wgi_path = wgi_path or self._find_wgi_file()
        self.cache_path = os.path.join(CACHE_DIR, 'WGI_cache.parquet')
        self.data: Optional[pd.DataFrame] = None
        
    def _find_wgi_file(self) -> str:
        """Find WGI Excel file in project directory."""
        candidates = [
            'wgidataset_with_sourcedata-2025.xlsx',
            'wgidataset.xlsx',
            'WGI.xlsx',
        ]
        for filename in candidates:
            path = os.path.join(BASE_DIR, filename)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"WGI dataset not found. Searched for: {candidates} in {BASE_DIR}")
    
    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        """Load WGI data from all 6 sheets and merge into single DataFrame."""
        if not force_refresh and os.path.exists(self.cache_path):
            ensure_lfs_file(self.cache_path)
            self.data = pd.read_parquet(self.cache_path)
            print(
                f"Loaded WGI cache: {len(self.data)} records, "
                f"{self.data['country_code'].nunique()} countries"
            )
            return self.data

        print("\n" + "="*70)
        print("LOADING WORLD GOVERNANCE INDICATORS")
        print("="*70)
        print(f"  Source: {self.wgi_path}")
        
        all_data = []
        for sheet_code, feature_name in WGI_INDICATORS.items():
            try:
                df = pd.read_excel(self.wgi_path, sheet_name=sheet_code)
                df_clean = df[['Economy (code)', 'Year', 'Governance score (0-100)']].copy()
                df_clean.columns = ['country_code', 'year', feature_name]
                df_clean = df_clean.dropna(subset=[feature_name])
                all_data.append(df_clean)
                print(f"  {sheet_code.upper()} ({feature_name}): {len(df_clean)} records")
            except Exception as e:
                print(f"  WARNING: Failed to load sheet '{sheet_code}': {e}")
        
        if not all_data:
            raise ValueError("No WGI data could be loaded")
        
        merged = all_data[0]
        for df in all_data[1:]:
            merged = merged.merge(df, on=['country_code', 'year'], how='outer')
        
        self.data = merged
        self.data.to_parquet(self.cache_path, index=False)
        print(f"\n  Total: {len(merged)} country-year records, {merged['country_code'].nunique()} countries")
        return merged
    
    def get_latest_scores(self, as_of_date=None) -> pd.DataFrame:
        """Get the most recent governance scores for each country as of a cutoff."""
        if self.data is None:
            self.load()
        data = self.data.copy()
        if as_of_date is not None:
            cutoff_year = pd.Timestamp(as_of_date).year
            data = data[pd.to_numeric(data['year'], errors='coerce') <= cutoff_year]
        latest = data.sort_values('year').groupby('country_code').last().reset_index()
        
        # Add year columns for each feature so dashboard can display it
        feature_cols = ['country_code']
        for col in WGI_INDICATORS.values():
            if col in latest.columns:
                feature_cols.append(col)
                latest[f'{col}_year'] = latest['year']
                feature_cols.append(f'{col}_year')
        
        return latest[feature_cols]
    
    def get_economic_pillar_features(self) -> pd.DataFrame:
        """Get WGI features mapped to BICRA Economic pillar."""
        latest = self.get_latest_scores()
        cols = ['country_code'] + [c for c in BICRA_ECONOMIC_INDICATORS if c in latest.columns]
        return latest[cols]
    
    def get_industry_pillar_features(self) -> pd.DataFrame:
        """Get WGI features mapped to BICRA Industry pillar."""
        latest = self.get_latest_scores()
        cols = ['country_code'] + [c for c in BICRA_INDUSTRY_INDICATORS if c in latest.columns]
        return latest[cols]


def load_wgi_features() -> pd.DataFrame:
    """Convenience function to load WGI features for model training."""
    loader = WGILoader()
    return loader.get_latest_scores()

