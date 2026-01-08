"""
==============================================================================
FSIBSIS Data Loader - IMF Financial Soundness Indicators Balance Sheet Data
==============================================================================
Loads the IMF.STA:FSIBSIS dataset which contains detailed banking sector
balance sheet data for calculating:
- Net Interest Margin
- Interbank Funding Ratio  
- Income Diversification
- Securities-to-Assets Ratio
- Specific Provisions Ratio
- Large Exposure Concentration
- Sovereign Exposure (enhanced)

Author: Banking Copilot
Date: 2026-01-07
==============================================================================
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Country name to ISO code mapping for FSIBSIS
COUNTRY_NAME_TO_ISO = {
    'Afghanistan, Islamic Republic of': 'AFG',
    'Albania': 'ALB',
    'Algeria': 'DZA',
    'Angola': 'AGO',
    'Argentina': 'ARG',
    'Armenia, Republic of': 'ARM',
    'Australia': 'AUS',
    'Austria': 'AUT',
    'Azerbaijan, Republic of': 'AZE',
    'Bangladesh': 'BGD',
    'Barbados': 'BRB',
    'Belarus, Republic of': 'BLR',
    'Belgium': 'BEL',
    'Belize': 'BLZ',
    'Bhutan': 'BTN',
    'Bolivia': 'BOL',
    'Bosnia and Herzegovina': 'BIH',
    'Botswana': 'BWA',
    'Brazil': 'BRA',
    'Brunei Darussalam': 'BRN',
    'Bulgaria': 'BGR',
    'Burundi': 'BDI',
    'Cambodia': 'KHM',
    'Cameroon': 'CMR',
    'Canada': 'CAN',
    'Central African Republic': 'CAF',
    'Chad': 'TCD',
    'Chile': 'CHL',
    'China, P.R.: Mainland': 'CHN',
    'Colombia': 'COL',
    'Comoros, Union of the': 'COM',
    'Congo, Dem. Rep. of the': 'COD',
    'Congo, Republic of': 'COG',
    'Costa Rica': 'CRI',
    'Croatia': 'HRV',
    'Cyprus': 'CYP',
    'Czech Republic': 'CZE',
    'Denmark': 'DNK',
    'Dominican Republic': 'DOM',
    'Ecuador': 'ECU',
    'Egypt': 'EGY',
    'El Salvador': 'SLV',
    'Equatorial Guinea': 'GNQ',
    'Estonia': 'EST',
    'Eswatini': 'SWZ',
    'Ethiopia': 'ETH',
    'Fiji': 'FJI',
    'Finland': 'FIN',
    'France': 'FRA',
    'Gabon': 'GAB',
    'Gambia, The': 'GMB',
    'Georgia': 'GEO',
    'Germany': 'DEU',
    'Ghana': 'GHA',
    'Greece': 'GRC',
    'Guatemala': 'GTM',
    'Guinea': 'GIN',
    'Guinea-Bissau': 'GNB',
    'Guyana': 'GUY',
    'Haiti': 'HTI',
    'Honduras': 'HND',
    'Hungary': 'HUN',
    'Iceland': 'ISL',
    'India': 'IND',
    'Indonesia': 'IDN',
    'Iran, Islamic Republic of': 'IRN',
    'Iraq': 'IRQ',
    'Ireland': 'IRL',
    'Israel': 'ISR',
    'Italy': 'ITA',
    'Jamaica': 'JAM',
    'Japan': 'JPN',
    'Jordan': 'JOR',
    'Kazakhstan': 'KAZ',
    'Kenya': 'KEN',
    'Korea, Republic of': 'KOR',
    'Kosovo': 'XKX',
    'Kuwait': 'KWT',
    'Kyrgyz Republic': 'KGZ',
    'Lao P.D.R.': 'LAO',
    'Latvia': 'LVA',
    'Lebanon': 'LBN',
    'Lesotho': 'LSO',
    'Liberia': 'LBR',
    'Libya': 'LBY',
    'Lithuania': 'LTU',
    'Luxembourg': 'LUX',
    'Madagascar': 'MDG',
    'Malawi': 'MWI',
    'Malaysia': 'MYS',
    'Maldives': 'MDV',
    'Mali': 'MLI',
    'Malta': 'MLT',
    'Mauritania': 'MRT',
    'Mauritius': 'MUS',
    'Mexico': 'MEX',
    'Moldova, Republic of': 'MDA',
    'Mongolia': 'MNG',
    'Montenegro': 'MNE',
    'Morocco': 'MAR',
    'Mozambique': 'MOZ',
    'Myanmar': 'MMR',
    'Namibia': 'NAM',
    'Nepal': 'NPL',
    'Netherlands': 'NLD',
    'New Zealand': 'NZL',
    'Nicaragua': 'NIC',
    'Niger': 'NER',
    'Nigeria': 'NGA',
    'North Macedonia': 'MKD',
    'Norway': 'NOR',
    'Oman': 'OMN',
    'Pakistan': 'PAK',
    'Panama': 'PAN',
    'Papua New Guinea': 'PNG',
    'Paraguay': 'PRY',
    'Peru': 'PER',
    'Philippines': 'PHL',
    'Poland': 'POL',
    'Portugal': 'PRT',
    'Qatar': 'QAT',
    'Romania': 'ROU',
    'Russian Federation': 'RUS',
    'Rwanda': 'RWA',
    'Samoa': 'WSM',
    'Saudi Arabia': 'SAU',
    'Senegal': 'SEN',
    'Serbia': 'SRB',
    'Seychelles': 'SYC',
    'Sierra Leone': 'SLE',
    'Singapore': 'SGP',
    'Slovak Republic': 'SVK',
    'Slovenia': 'SVN',
    'Solomon Islands': 'SLB',
    'South Africa': 'ZAF',
    'South Sudan': 'SSD',
    'Spain': 'ESP',
    'Sri Lanka': 'LKA',
    'Sudan': 'SDN',
    'Suriname': 'SUR',
    'Sweden': 'SWE',
    'Switzerland': 'CHE',
    'Syrian Arab Republic': 'SYR',
    'Tajikistan': 'TJK',
    'Tanzania, United Republic of': 'TZA',
    'Thailand': 'THA',
    'Timor-Leste': 'TLS',
    'Togo': 'TGO',
    'Tonga': 'TON',
    'Trinidad and Tobago': 'TTO',
    'Tunisia': 'TUN',
    'Turkey': 'TUR',
    'Turkmenistan': 'TKM',
    'Uganda': 'UGA',
    'Ukraine': 'UKR',
    'United Arab Emirates': 'ARE',
    'United Kingdom': 'GBR',
    'United States': 'USA',
    'Uruguay': 'URY',
    'Uzbekistan': 'UZB',
    'Vanuatu': 'VUT',
    'Venezuela': 'VEN',
    'Vietnam': 'VNM',
    'Yemen, Republic of': 'YEM',
    'Zambia': 'ZMB',
    'Zimbabwe': 'ZWE',
}


class FSIBSISLoader:
    """
    Loader for IMF FSIBSIS (Financial Soundness Indicators Balance Sheet) dataset.
    
    This dataset provides detailed balance sheet data for deposit takers (banks)
    including income statements, assets, liabilities, and capital components.
    """
    
    # Indicator mappings for feature calculation
    # NOTE: Names must match EXACTLY as they appear in FSIBSIS
    INDICATOR_MAPPINGS = {
        # Interest margin components
        'interest_income': 'Interest income, Domestic currency',
        'interest_expenses': 'Interest expenses, Domestic currency',
        'net_interest_income': 'Net interest income, Domestic currency',
        
        # Assets - NOTE: "Total assets, Assets, Domestic currency" is the full name
        'total_assets': 'Total assets, Assets, Domestic currency',
        'gross_loans': 'Gross loans, Assets, Domestic currency',
        'debt_securities': 'Debt securities, Assets, Domestic currency',
        'noninterbank_loans': 'Noninterbank loans, Assets, Domestic currency',
        'govt_claims': 'Noninterbank loans, General government, Assets, Domestic currency',
        'residential_re_loans': 'Residential real estate loans, Domestic currency',
        'credit_to_private': 'Credit to the private sector, Domestic currency',
        
        # Liabilities
        'total_liabilities': 'Total liabilities, Domestic currency',
        'deposits': 'Currency and deposits, Liabilities, Domestic currency',
        'customer_deposits': 'Customer deposits, Liabilities, Domestic currency',
        'interbank_deposits': 'Interbank deposits, Liabilities, Domestic currency',
        'short_term_liab': 'Short term liabilities, Domestic currency',
        
        # Capital
        'capital_reserves': 'Capital and reserves, Domestic currency',
        'regulatory_capital': 'Total regulatory capital, Domestic currency',
        'rwa': 'Risk-weighted assets, Domestic currency',
        'tier1_capital': 'Tier 1 capital less corresponding supervisory deductions, Domestic currency',
        
        # Provisions & NPL
        'npl': 'Nonperforming loans, Domestic currency',
        'specific_provisions': 'Specific provisions, Assets, Domestic currency',
        'loan_loss_provisions': 'Provisions (net): Loan loss provisions, Domestic currency',
        
        # Income
        'net_income': 'Net income after taxes, Domestic currency',
        'noninterest_income': 'Noninterest income, Domestic currency',
        
        # Concentration
        'large_exposures': 'Value of large exposures, Domestic currency',
        'loan_concentration': 'Loan concentration by economic activity, Domestic currency',
        
        # Basel III liquidity
        'available_stable_funding': 'Available amount of stable funding, Domestic currency',
        'required_stable_funding': 'Required amount of stable funding, Domestic currency',
        'cash_outflows_30d': 'Total net cash outflows over the next 30 calendar days, Domestic currency',
    }
    
    def __init__(self, file_path: str = None):
        """
        Initialize FSIBSIS loader.
        
        Args:
            file_path: Path to FSIBSIS CSV file. If None, searches for it.
        """
        self.file_path = file_path
        self.data = None
        self.bank_data = None
        self.year_cols = []
        
    def load(self, file_path: str = None) -> pd.DataFrame:
        """Load and parse FSIBSIS dataset."""
        path = file_path or self.file_path
        
        if path is None:
            # Search for FSIBSIS file in Banking directory
            import glob
            pattern = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   '*FSIBSIS*.csv')
            files = glob.glob(pattern)
            if files:
                path = files[0]
                logger.info(f"Found FSIBSIS file: {path}")
            else:
                raise FileNotFoundError("No FSIBSIS CSV file found")
        
        logger.info(f"Loading FSIBSIS from: {path}")
        self.data = pd.read_csv(path, low_memory=False)
        
        # Filter to Deposit Takers (banks)
        self.bank_data = self.data[self.data['SECTOR'] == 'Deposit takers'].copy()
        
        # Identify year columns (annual data preferred)
        self.year_cols = sorted([c for c in self.data.columns 
                                 if len(c) == 4 and c.startswith('20')])
        
        # Add ISO country codes
        self.bank_data['country_code'] = self.bank_data['COUNTRY'].map(COUNTRY_NAME_TO_ISO)
        
        logger.info(f"Loaded {len(self.bank_data)} rows for {self.bank_data['country_code'].nunique()} countries")
        logger.info(f"Year columns: {self.year_cols[0]} to {self.year_cols[-1]}")
        
        return self.bank_data
    
    def get_country_data(self, country_code: str) -> pd.DataFrame:
        """
        Get time-series data for a country in a format compatible with Data Explorer.
        
        Returns DataFrame with columns: INDICATOR, year columns (2000, 2001, etc.)
        This allows FSIBSIS data to be visualized alongside FSIC data.
        """
        if self.bank_data is None:
            self.load()
        
        country_data = self.bank_data[self.bank_data['country_code'] == country_code].copy()
        if len(country_data) == 0:
            return pd.DataFrame()
        
        # Select indicator name and year columns
        cols_to_keep = ['INDICATOR'] + self.year_cols
        result = country_data[cols_to_keep].copy()
        
        # Add a prefix to distinguish from FSIC
        result['INDICATOR'] = result['INDICATOR'].apply(lambda x: f"[BIS] {x}" if pd.notna(x) else x)
        
        return result
    
    def get_indicator_value(self, country_code: str, indicator_key: str, 
                            year: str = None) -> Optional[float]:
        """
        Get the value of an indicator for a specific country.
        Uses flexible matching to handle variations in indicator names across countries.
        
        Args:
            country_code: ISO country code
            indicator_key: Key from INDICATOR_MAPPINGS
            year: Specific year, or None for most recent
            
        Returns:
            Value or None if not available
        """
        if self.bank_data is None:
            self.load()
        
        # Use pattern-based matching for flexibility across countries
        patterns = self._get_indicator_patterns(indicator_key)
        if not patterns:
            return None
        
        # Filter to country
        country_data = self.bank_data[self.bank_data['country_code'] == country_code]
        if len(country_data) == 0:
            return None
        
        # Try each pattern in order
        for pattern in patterns:
            if pattern.startswith('^'):
                # Regex pattern
                mask = country_data['INDICATOR'].str.match(pattern, case=False, na=False)
            else:
                # Exact match first, then contains
                mask = country_data['INDICATOR'] == pattern
                if not mask.any():
                    mask = country_data['INDICATOR'].str.contains(pattern, case=False, na=False, regex=False)
            
            rows = country_data[mask]
            if len(rows) > 0:
                # If multiple rows, take the one with most recent data
                for yr in reversed(self.year_cols):
                    for _, row in rows.iterrows():
                        val = row.get(yr)
                        if pd.notnull(val):
                            if year and yr == year:
                                return val
                            elif not year:
                                return val
        
        return None

    def get_indicator_data(self, country_code: str, indicator_key: str, year: str = None) -> Tuple[Optional[float], Optional[str]]:
        """
        Get (value, year) for an indicator.
        """
        if self.bank_data is None:
            self.load()
        
        patterns = self._get_indicator_patterns(indicator_key)
        if not patterns:
            return None, None
        
        country_data = self.bank_data[self.bank_data['country_code'] == country_code]
        if len(country_data) == 0:
            return None, None
        
        for pattern in patterns:
            if pattern.startswith('^'):
                mask = country_data['INDICATOR'].str.match(pattern, case=False, na=False)
            else:
                mask = country_data['INDICATOR'] == pattern
                if not mask.any():
                    mask = country_data['INDICATOR'].str.contains(pattern, case=False, na=False, regex=False)
            
            rows = country_data[mask]
            if len(rows) > 0:
                for yr in reversed(self.year_cols):
                    for _, row in rows.iterrows():
                        val = row.get(yr)
                        if pd.notnull(val):
                            if year and yr == year:
                                return val, yr
                            elif not year:
                                return val, yr
        return None, None
    
    def _get_indicator_patterns(self, indicator_key: str) -> List[str]:
        """
        Get list of patterns to try for an indicator, in priority order.
        This handles variations in naming across different countries.
        
        NOTE: Some countries report in their national currency:
        - Eurozone countries use "Euro" suffix
        - USA uses "US dollar" suffix
        - Most emerging markets use "Domestic currency" suffix
        
        We try all three to maximize coverage.
        """
        # Base patterns for each indicator - we'll expand with currency suffixes
        base_patterns = {
            'total_assets': [
                'Total assets, Assets',
                'Total assets',
                'Average total assets',
            ],
            'gross_loans': [
                'Gross loans, Assets',
                'Gross loans',
                'Credit to the private sector',
            ],
            'debt_securities': [
                'Debt securities, Assets',
                'Debt securities',
            ],
            'noninterbank_loans': [
                'Noninterbank loans, Assets',
                'Noninterbank loans',
            ],
            'govt_claims': [
                'Noninterbank loans, General government, Assets',
                'Noninterbank loans to General government, Assets',
            ],
            'residential_re_loans': [
                'Residential real estate loans',
            ],
            'total_liabilities': [
                'Total liabilities',
                'Total liabilities and capital and reserves',
            ],
            'deposits': [
                'Currency and deposits, Liabilities',
                'Customer deposits, Liabilities',
                'Deposits',
            ],
            'customer_deposits': [
                'Customer deposits, Liabilities',
                'Currency and deposits, Liabilities',
            ],
            'interbank_deposits': [
                'Interbank deposits, Liabilities',
                'Interbank deposits',
            ],
            'short_term_liab': [
                'Short term liabilities',
            ],
            'interest_income': [
                'Interest income',
                'Interest income: Gross interest income',
            ],
            'interest_expenses': [
                'Interest expenses',
            ],
            'net_interest_income': [
                'Net interest income',
            ],
            'noninterest_income': [
                'Noninterest income',
            ],
            'net_income': [
                'Net income after taxes',
                'Net income before taxes',
            ],
            'capital_reserves': [
                'Capital and reserves',
            ],
            'regulatory_capital': [
                'Total regulatory capital',
            ],
            'rwa': [
                'Risk-weighted assets',
            ],
            'tier1_capital': [
                'Tier 1 capital less corresponding supervisory deductions',
            ],
            'npl': [
                'Nonperforming loans',
            ],
            'specific_provisions': [
                'Specific provisions, Assets',
                'General and specific provisions, Liabilities',
                'Provisions (net): Loan loss provisions',
            ],
            'loan_loss_provisions': [
                'Provisions (net): Loan loss provisions',
            ],
            'large_exposures': [
                'Value of large exposures',
            ],
            'loan_concentration': [
                'Loan concentration by economic activity',
            ],
            'available_stable_funding': [
                'Available amount of stable funding',
            ],
            'required_stable_funding': [
                'Required amount of stable funding',
            ],
            'cash_outflows_30d': [
                'Total net cash outflows over the next 30 calendar days',
            ],
        }
        
        # Currency suffixes to try - order matters: Domestic first (most common), then Euro, then USD
        currency_suffixes = ['Domestic currency', 'Euro', 'US dollar']
        
        base_list = base_patterns.get(indicator_key, [])
        if not base_list:
            return []
        
        # Expand each base pattern with all currency suffixes
        expanded_patterns = []
        for base in base_list:
            for suffix in currency_suffixes:
                expanded_patterns.append(f"{base}, {suffix}")
        
        return expanded_patterns
    
    
    def extract_features(self) -> pd.DataFrame:
        """
        Extract all new features from FSIBSIS for all countries.
        
        Returns:
            DataFrame with country_code and calculated features
        """
        if self.bank_data is None:
            self.load()
            
        countries = self.bank_data['country_code'].dropna().unique()
        features_list = []
        
        for country in countries:
            features = {'country_code': country}
            
            # Get raw values and years for most recent data
            values = {}
            years = {}
            for key in self.INDICATOR_MAPPINGS:
                val, yr = self.get_indicator_data(country, key)
                values[key] = val
                years[key] = yr
            
            # 1. Net Interest Margin
            if values['interest_income'] and values['interest_expenses'] and values['total_assets']:
                if values['total_assets'] > 0:
                    nim = ((values['interest_income'] - values['interest_expenses']) / 
                           values['total_assets']) * 100
                    features['net_interest_margin'] = nim
                    features['net_interest_margin_year'] = years.get('interest_income')
            
            # 2. Interbank Funding Ratio
            if values['interbank_deposits'] and values['total_liabilities']:
                if values['total_liabilities'] > 0:
                    features['interbank_funding_ratio'] = (
                        values['interbank_deposits'] / values['total_liabilities'] * 100
                    )
                    features['interbank_funding_ratio_year'] = years.get('interbank_deposits')
            
            # 3. Income Diversification
            if values['noninterest_income'] and values['net_interest_income']:
                total_income = values['net_interest_income'] + values['noninterest_income']
                if total_income > 0:
                    features['income_diversification'] = (
                        values['noninterest_income'] / total_income * 100
                    )
                    features['income_diversification_year'] = years.get('noninterest_income')
            
            # 4. Securities-to-Assets
            if values['debt_securities'] and values['total_assets']:
                if values['total_assets'] > 0:
                    features['securities_to_assets'] = (
                        values['debt_securities'] / values['total_assets'] * 100
                    )
                    features['securities_to_assets_year'] = years.get('debt_securities')
            
            # 5. Specific Provisions Ratio
            if values['specific_provisions'] and values['gross_loans']:
                if values['gross_loans'] > 0:
                    features['specific_provisions_ratio'] = (
                        values['specific_provisions'] / values['gross_loans'] * 100
                    )
                    features['specific_provisions_ratio_year'] = years.get('specific_provisions')
            
            # 6. Large Exposure Concentration
            # UPDATED: Use Regulatory Capital (Tier 1+2) for denominator
            if values['large_exposures'] and values.get('regulatory_capital'):
                if values['regulatory_capital'] > 0:
                    features['large_exposure_ratio'] = (
                        values['large_exposures'] / values['regulatory_capital'] * 100
                    )
                    features['large_exposure_ratio_year'] = years.get('large_exposures')
            
            # BONUS: Enhanced sovereign exposure
            if values['govt_claims'] and values['gross_loans']:
                if values['gross_loans'] > 0:
                    features['sovereign_exposure_fsibsis'] = (
                        values['govt_claims'] / values['gross_loans'] * 100
                    )
                    features['sovereign_exposure_fsibsis_year'] = years.get('govt_claims')
            
            # BONUS: Deposit Funding Ratio
            if values['deposits'] and values['total_liabilities']:
                if values['total_liabilities'] > 0:
                    features['deposit_funding_ratio'] = (
                        values['deposits'] / values['total_liabilities'] * 100
                    )
                    features['deposit_funding_ratio_year'] = years.get('deposits')
            
            # BONUS: Real Estate Exposure = RE Loans / Gross Loans * 100
            if values['residential_re_loans'] and values['gross_loans']:
                if values['gross_loans'] > 0:
                    features['real_estate_loans_fsibsis'] = (
                        values['residential_re_loans'] / values['gross_loans'] * 100
                    )
            
            if len(features) > 1:  # More than just country_code
                features_list.append(features)
        
        result = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(result.columns)-1} features for {len(result)} countries")
        
        return result


def load_fsibsis_features(file_path: str = None) -> pd.DataFrame:
    """
    Convenience function to load and extract FSIBSIS features.
    
    Args:
        file_path: Path to FSIBSIS CSV file
        
    Returns:
        DataFrame with country_code and calculated features
    """
    loader = FSIBSISLoader(file_path)
    loader.load()
    return loader.extract_features()


if __name__ == "__main__":
    # Test loading
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Find FSIBSIS file
    import glob
    files = glob.glob(r"C:\Users\Richard\Banking\*FSIBSIS*.csv")
    if files:
        features = load_fsibsis_features(files[0])
        print("\n=== FSIBSIS Features Extracted ===")
        print(f"Countries: {len(features)}")
        print(f"Features: {list(features.columns)}")
        print("\nSample (Kenya, Nigeria, Germany):")
        sample = features[features['country_code'].isin(['KEN', 'NGA', 'DEU'])]
        print(sample.T)
    else:
        print("No FSIBSIS file found")
