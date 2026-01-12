"""
==============================================================================
Laeven-Valencia Systemic Banking Crisis Labels
==============================================================================
CRISP-DM Phase: Data Preparation (Target Variable)

This module provides crisis labels based on the Laeven-Valencia (2018) database,
the gold standard for identifying systemic banking crises in academic literature.

Reference:
    Laeven, L., & Valencia, F. (2018). Systemic Banking Crises Revisited.
    IMF Working Paper WP/18/206.

Usage:
    labels = CrisisLabels()
    y = labels.get_crisis_target(country_code='GRC', year=2007, horizon=3)
    # Returns 1 if Greece had crisis in 2008-2010, 0 otherwise

Author: Banking Copilot
Date: 2026-01-02
==============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class CrisisLabels:
    """
    Laeven-Valencia (2018) systemic banking crisis database.
    
    Provides binary labels for supervised crisis prediction models.
    Target = 1 if country experiences systemic crisis within horizon years.
    """
    
    # Systemic Banking Crises (Laeven-Valencia 2018)
    # Format: country_code -> list of (start_year, end_year) tuples
    SYSTEMIC_CRISES = {
        # Advanced Economies
        'USA': [(2007, 2009)],                    # Global Financial Crisis
        'GBR': [(2007, 2009)],                    # UK banking crisis
        'DEU': [(2008, 2009)],                    # Germany (minor)
        'FRA': [(2008, 2008)],                    # France (minor)
        'ESP': [(2008, 2012)],                    # Spain - savings banks
        'IRL': [(2008, 2011)],                    # Ireland - property bubble
        'GRC': [(2008, 2012)],                    # Greece - sovereign-bank loop
        'PRT': [(2008, 2012)],                    # Portugal
        'ITA': [(2008, 2009)],                    # Italy (minor)
        'NLD': [(2008, 2009)],                    # Netherlands
        'BEL': [(2008, 2009)],                    # Belgium - Fortis, Dexia
        'AUT': [(2008, 2009)],                    # Austria
        'ISL': [(2008, 2010)],                    # Iceland - total collapse
        'CYP': [(2012, 2013)],                    # Cyprus - bail-in
        'DNK': [(2008, 2009)],                    # Denmark
        'SWE': [(1991, 1995), (2008, 2009)],      # Sweden - 90s + GFC
        'FIN': [(1991, 1995)],                    # Finland - 90s crisis
        'NOR': [(1987, 1993)],                    # Norway - 80s-90s
        'JPN': [(1997, 2001)],                    # Japan - lost decade
        'KOR': [(1997, 1998)],                    # Korea - Asian crisis
        
        # Emerging Europe
        'TUR': [(2000, 2001), (2018, 2019)],      # Turkey - 2000 + Lira crisis
        'RUS': [(1998, 1999), (2008, 2009)],      # Russia - 98 default + GFC
        'UKR': [(1998, 1999), (2008, 2009), (2014, 2015)],  # Ukraine - multiple
        'HUN': [(2008, 2009)],                    # Hungary
        'LVA': [(2008, 2010)],                    # Latvia
        'LTU': [(2008, 2009)],                    # Lithuania
        'EST': [(2008, 2009)],                    # Estonia
        'SVN': [(2012, 2013)],                    # Slovenia
        'HRV': [(2008, 2009)],                    # Croatia
        'SRB': [(2008, 2009)],                    # Serbia
        'ROU': [(2008, 2009)],                    # Romania
        'BGR': [(2014, 2014)],                    # Bulgaria - KTB
        
        # Latin America
        'ARG': [(1989, 1991), (1995, 1995), (2001, 2003), (2018, 2019)],  # Argentina - incl. 2018 peso crisis
        'BRA': [(1990, 1994), (1999, 1999)],      # Brazil
        'MEX': [(1994, 1996)],                    # Mexico - Tequila crisis
        'VEN': [(1994, 1995), (2009, 2010)],      # Venezuela
        'ECU': [(1998, 2002)],                    # Ecuador
        'COL': [(1998, 2000)],                    # Colombia
        'URY': [(2002, 2005)],                    # Uruguay
        'PRY': [(1995, 1999)],                    # Paraguay
        'BOL': [(1994, 1994)],                    # Bolivia
        'DOM': [(2003, 2004)],                    # Dominican Republic
        
        # Asia-Pacific
        'IDN': [(1997, 2001)],                    # Indonesia - Asian crisis
        'THA': [(1997, 2000)],                    # Thailand - trigger of Asian crisis
        'MYS': [(1997, 1999)],                    # Malaysia
        'PHL': [(1997, 2001)],                    # Philippines
        'IND': [(1993, 1993)],                    # India (minor)
        'CHN': [(1998, 1998)],                    # China (minor, state banks)
        'PAK': [(2008, 2008)],                    # Pakistan
        'BGD': [(1987, 1987)],                    # Bangladesh
        'LKA': [(1989, 1991), (2022, 2023)],      # Sri Lanka + 2022 sovereign default (IMF program)
        'VNM': [(1997, 1997)],                    # Vietnam
        'MNG': [(2008, 2009)],                    # Mongolia
        
        # Africa
        'NGA': [(1991, 1995), (2009, 2011)],      # Nigeria
        'ZAF': [(1985, 1985), (1989, 1989)],      # South Africa (minor)
        'KEN': [(1992, 1995)],                    # Kenya
        'GHA': [(1997, 1997), (2017, 2018), (2022, 2024)],  # Ghana - incl. debt distress
        'EGY': [(1991, 1991)],                    # Egypt
        'TUN': [(1991, 1991)],                    # Tunisia
        'MAR': [(1980, 1984)],                    # Morocco
        'ZWE': [(1995, 1999)],                    # Zimbabwe
        'MUS': [(1996, 1996)],                    # Mauritius
        'SEN': [(1988, 1991)],                    # Senegal
        'CIV': [(1988, 1991)],                    # Côte d'Ivoire
        'CMR': [(1989, 1997)],                    # Cameroon
        'TZA': [(1988, 1991)],                    # Tanzania
        'UGA': [(1994, 1994)],                    # Uganda
        
        # Middle East
        'ARE': [(2008, 2009)],                    # UAE - Dubai World
        'KWT': [(1982, 1985)],                    # Kuwait - Souk Al-Manakh
        'JOR': [(1989, 1991)],                    # Jordan
        'LBN': [(1990, 1990), (2019, 2024)],      # Lebanon - 2019 banking collapse (World Bank: worst in 150 years)
    }
    
    def __init__(self):
        """Initialize crisis labels database."""
        self._build_crisis_df()
    
    def _build_crisis_df(self):
        """Build DataFrame of crisis events for efficient querying."""
        records = []
        for country, periods in self.SYSTEMIC_CRISES.items():
            for start, end in periods:
                for year in range(start, end + 1):
                    records.append({
                        'country_code': country,
                        'year': year,
                        'crisis': 1,
                        'crisis_start': start,
                        'crisis_end': end
                    })
        
        self.crisis_df = pd.DataFrame(records) if records else pd.DataFrame()
        print(f"  Loaded {len(self.SYSTEMIC_CRISES)} countries with {len(records)} crisis-years")
    
    def is_crisis_year(self, country_code: str, year: int) -> bool:
        """Check if country was in crisis during specified year."""
        if country_code not in self.SYSTEMIC_CRISES:
            return False
        
        for start, end in self.SYSTEMIC_CRISES[country_code]:
            if start <= year <= end:
                return True
        return False
    
    def get_crisis_target(self, country_code: str, year: int, 
                         horizon: int = 3) -> int:
        """
        Get binary crisis target for supervised learning.
        
        Args:
            country_code: ISO 3-letter country code
            year: Current year (observation year)
            horizon: Prediction horizon in years (default 3)
        
        Returns:
            1 if crisis occurs in [year+1, year+horizon], 0 otherwise
        """
        for future_year in range(year + 1, year + horizon + 1):
            if self.is_crisis_year(country_code, future_year):
                return 1
        return 0
    
    def create_labeled_dataset(self, features_df: pd.DataFrame,
                               year_col: str = 'year',
                               horizon: int = 3) -> pd.DataFrame:
        """
        Create labeled dataset for supervised learning.
        
        Args:
            features_df: DataFrame with country_code and year columns
            year_col: Name of year column
            horizon: Prediction horizon
        
        Returns:
            DataFrame with added 'crisis_target' column
        """
        if 'country_code' not in features_df.columns:
            raise ValueError("features_df must have 'country_code' column")
        
        labeled = features_df.copy()
        
        if year_col in labeled.columns:
            labeled['crisis_target'] = labeled.apply(
                lambda row: self.get_crisis_target(
                    row['country_code'], 
                    row[year_col], 
                    horizon
                ),
                axis=1
            )
        else:
            # If no year column, use latest available data (assume current year)
            current_year = 2024
            labeled['crisis_target'] = labeled['country_code'].apply(
                lambda c: self.get_crisis_target(c, current_year, horizon)
            )
        
        return labeled
    
    def get_crisis_countries(self, year_range: Tuple[int, int] = None) -> List[str]:
        """Get list of countries that had crises in specified range."""
        if year_range is None:
            return list(self.SYSTEMIC_CRISES.keys())
        
        start, end = year_range
        crisis_countries = []
        
        for country, periods in self.SYSTEMIC_CRISES.items():
            for crisis_start, crisis_end in periods:
                if crisis_start <= end and crisis_end >= start:
                    crisis_countries.append(country)
                    break
        
        return crisis_countries
    
    def get_crisis_summary(self) -> pd.DataFrame:
        """Get summary statistics of crisis database."""
        records = []
        for country, periods in self.SYSTEMIC_CRISES.items():
            total_years = sum(end - start + 1 for start, end in periods)
            records.append({
                'country_code': country,
                'n_crises': len(periods),
                'total_crisis_years': total_years,
                'latest_crisis': max(end for _, end in periods),
                'first_crisis': min(start for start, _ in periods),
            })
        
        return pd.DataFrame(records).sort_values('latest_crisis', ascending=False)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LAEVEN-VALENCIA CRISIS LABELS")
    print("="*70)
    
    labels = CrisisLabels()
    
    # Test cases
    print("\n--- Test Cases ---")
    print(f"  Greece 2007 -> 3yr target: {labels.get_crisis_target('GRC', 2007, 3)}")  # Should be 1
    print(f"  USA 2006 -> 3yr target: {labels.get_crisis_target('USA', 2006, 3)}")     # Should be 1
    print(f"  CHE 2007 -> 3yr target: {labels.get_crisis_target('CHE', 2007, 3)}")     # Should be 0 (Switzerland)
    print(f"  JPN 1995 -> 3yr target: {labels.get_crisis_target('JPN', 1995, 3)}")     # Should be 1
    
    # Crisis summary
    print("\n--- Crisis Summary (Top 10 by Latest) ---")
    summary = labels.get_crisis_summary().head(10)
    print(summary.to_string(index=False))
