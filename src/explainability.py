"""
Explainability engine for the Risk Scorer.
Provides "Glass Box" transparency by calculating the contribution of each indicator to the final score.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

class RiskExplainer:
    """
    Decomposes risk scores into component contributions.
    """
    
    def __init__(self):
        # Contribution weights (approximate based on RiskScorer logic)
        # In a real model, these would be exact Shapley values or coefficients.
        # Here we use the heuristic weights from the Scorer classes.
        self.ECON_WEIGHTS = {
            'NGDP_RPCH': 1.0,      # GDP Growth
            'NGDPDPC': 0.8,        # GDP per Capita
            'PCPIPCH': -0.7,       # Inflation
            'BCA_NGDPD': 0.6,      # Current Account
            'GGXWDG_NGDP': -0.5,   # Govt Debt
            'LUR': -0.4            # Unemployment
        }
        
        self.IND_WEIGHTS = {
            'RCAR': 1.0,           # Capital
            'NPLGL': -1.0,         # NPLs
            'ROE': 0.8,            # Profitability
            'LASTL': 0.7           # Liquidity
        }
    
    def explain_score(self, df: pd.DataFrame, country_code: str) -> Dict[str, Any]:
        """
        Explain why a country got its score.
        Returns a breakdown of positive and negative factors.
        """
        if df.empty:
            return {}
            
        country_data = df[df['country_code'] == country_code]
        if country_data.empty:
            return {}
            
        # Get latest values for all relevant indicators
        explanation = {
            'positive_factors': [],
            'negative_factors': [],
            'neutral_factors': []
        }
        
        # Helper to process indicators
        def process_indicator(indicator_code, weight, readable_name):
            # Find matching rows
            matches = country_data[
                country_data['indicator_code'].str.contains(indicator_code, case=False, na=False)
            ]
            if matches.empty:
                return
            
            val = matches.sort_values('period').iloc[-1]['value']
            
            # Determine contribution (heuristic)
            # We compare to a "neutral" baseline
            contribution = 0
            
            if indicator_code == 'NGDP_RPCH': # GDP
                contribution = (val - 2.0) * weight # Baseline 2% growth
            elif indicator_code == 'NPLGL': # NPL
                contribution = (5.0 - val) * abs(weight) # Baseline 5% NPL
            elif indicator_code == 'RCAR': # Capital
                contribution = (val - 12.0) * weight # Baseline 12% Capital
            elif indicator_code == 'PCPIPCH': # Inflation
                contribution = (3.0 - val) * abs(weight) # Baseline 3% Inflation
            elif indicator_code == 'ROE': # ROE
                contribution = (val - 8.0) * weight # Baseline 8% ROE
            
            # Categorize
            factor = {
                'name': readable_name,
                'value': val,
                'contribution': contribution,
                'weight': weight
            }
            
            if contribution > 0.5:
                explanation['positive_factors'].append(factor)
            elif contribution < -0.5:
                explanation['negative_factors'].append(factor)
            else:
                explanation['neutral_factors'].append(factor)

        # Process Key Indicators
        process_indicator('NGDP_RPCH', 1.0, 'GDP Growth')
        process_indicator('NPLGL', -1.0, 'NPL Ratio')
        process_indicator('RCAR', 1.0, 'Capital Adequacy')
        process_indicator('PCPIPCH', -0.7, 'Inflation')
        process_indicator('ROE', 0.8, 'Profitability')
        
        # Sort by magnitude of contribution
        explanation['positive_factors'].sort(key=lambda x: x['contribution'], reverse=True)
        explanation['negative_factors'].sort(key=lambda x: x['contribution']) # Most negative first
        
        return explanation
