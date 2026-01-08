"""
Rule-based insight generator for credit analyst copilot.
Produces natural language insights without requiring an LLM API.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.config import FSIC_CORE_INDICATORS, WEO_CORE_INDICATORS, RISK_COLORS


class InsightGenerator:
    """
    Template-based natural language insight generator.
    Produces analyst-ready commentary on banking sector conditions.
    """
    
    # Threshold definitions for risk assessment
    THRESHOLDS = {
        'NPLGL': {'critical': 10.0, 'warning': 5.0, 'good': 3.0},
        'RCAR': {'critical': 8.0, 'warning': 10.5, 'good': 14.0},
        'T1RWA': {'critical': 6.0, 'warning': 8.0, 'good': 12.0},
        'ROE': {'critical': 0.0, 'warning': 5.0, 'good': 12.0},
        'ROA': {'critical': 0.0, 'warning': 0.5, 'good': 1.0},
        'LASTL': {'critical': 15.0, 'warning': 25.0, 'good': 40.0},
        'NGDP_RPCH': {'critical': -2.0, 'warning': 1.0, 'good': 3.0},
        'PCPIPCH': {'critical': 15.0, 'warning': 7.0, 'good': 3.0},
    }
    
    # Templates for different insight types
    TEMPLATES = {
        'capital_strong': "Capital position is **strong** with a regulatory capital ratio of {value:.1f}%, well above the {threshold}% minimum. This provides a robust buffer against potential losses.",
        'capital_adequate': "Capital adequacy ratio of {value:.1f}% meets regulatory requirements but offers limited buffer against stress scenarios.",
        'capital_weak': "⚠️ **Capital concerns**: Regulatory capital ratio of {value:.1f}% is below the recommended {threshold}% threshold, signaling potential vulnerability.",
        
        'npl_low': "Asset quality is **strong** with NPLs at {value:.1f}% of gross loans, indicating effective credit risk management.",
        'npl_moderate': "NPL ratio of {value:.1f}% is manageable but warrants monitoring, particularly in the current economic environment.",
        'npl_high': "⚠️ **Elevated credit risk**: NPL ratio of {value:.1f}% exceeds the {threshold}% warning threshold. Provisioning coverage should be assessed.",
        
        'roe_strong': "Profitability is **healthy** with ROE of {value:.1f}%, supporting internal capital generation.",
        'roe_weak': "Profitability under pressure with ROE of {value:.1f}%. Limited capacity for internal capital building.",
        'roe_negative': "🔴 **Negative profitability**: ROE of {value:.1f}% indicates the banking sector is currently loss-making.",
        
        'liquidity_strong': "Liquidity position is **robust** with liquid assets covering {value:.1f}% of short-term liabilities.",
        'liquidity_adequate': "Liquidity coverage of {value:.1f}% is adequate but provides limited buffer in stress conditions.",
        'liquidity_weak': "⚠️ **Liquidity risk**: Liquid assets to short-term liabilities of {value:.1f}% is below the {threshold}% prudent threshold.",
        
        'trend_improving': "📈 **Positive trend**: {indicator} has improved by {change:.1f}% over the past {periods} periods.",
        'trend_deteriorating': "📉 **Deteriorating trend**: {indicator} has declined by {change:.1f}% over the past {periods} periods.",
        'trend_stable': "{indicator} has remained relatively stable over the analysis period.",
        
        'peer_above': "{country} ranks in the **{percentile}th percentile** among peers for {indicator}, indicating relative strength.",
        'peer_below': "{country} ranks in the **{percentile}th percentile** for {indicator}, below the peer median of {median:.1f}%.",
        
        'gdp_strong': "Economic backdrop is **supportive** with GDP growth of {value:.1f}%, providing favorable operating conditions for the banking sector.",
        'gdp_weak': "Economic headwinds with GDP growth of {value:.1f}% may pressure bank asset quality and profitability.",
        'gdp_recession': "🔴 **Recessionary conditions**: GDP contraction of {value:.1f}% poses significant risks to the banking sector.",
        
        'inflation_low': "Inflation at {value:.1f}% is well-contained, supporting real returns and financial stability.",
        'inflation_high': "⚠️ Elevated inflation of {value:.1f}% may erode real returns and complicate monetary policy.",
        
        'overall_strong': "The banking system demonstrates **strong fundamentals** across capital, asset quality, and profitability metrics.",
        'overall_adequate': "The banking system shows **adequate** fundamentals with some areas requiring attention.",
        'overall_weak': "The banking system exhibits **material vulnerabilities** that warrant careful credit assessment.",
    }
    
    def __init__(self):
        self.insights_cache: Dict[str, List[str]] = {}
    
    def _get_indicator_value(self, df: pd.DataFrame, 
                            country_code: str, 
                            indicator_pattern: str) -> Optional[float]:
        """Get most recent value for an indicator."""
        country_data = df[df['country_code'] == country_code]
        if country_data.empty:
            return None
        
        # Find matching indicator
        matching = country_data[
            country_data['indicator_code'].str.contains(indicator_pattern, case=False, na=False)
        ]
        
        if matching.empty:
            return None
        
        # Get most recent
        most_recent = matching.sort_values('period').iloc[-1]
        return most_recent['value']
    
    def generate_capital_insight(self, value: float) -> str:
        """Generate insight for capital adequacy."""
        thresholds = self.THRESHOLDS.get('RCAR', {})
        
        if value >= thresholds.get('good', 14.0):
            return self.TEMPLATES['capital_strong'].format(value=value, threshold=thresholds.get('good', 14))
        elif value >= thresholds.get('warning', 10.5):
            return self.TEMPLATES['capital_adequate'].format(value=value)
        else:
            return self.TEMPLATES['capital_weak'].format(value=value, threshold=thresholds.get('warning', 10.5))
    
    def generate_npl_insight(self, value: float) -> str:
        """Generate insight for NPL ratio."""
        thresholds = self.THRESHOLDS.get('NPLGL', {})
        
        if value <= thresholds.get('good', 3.0):
            return self.TEMPLATES['npl_low'].format(value=value)
        elif value <= thresholds.get('warning', 5.0):
            return self.TEMPLATES['npl_moderate'].format(value=value)
        else:
            return self.TEMPLATES['npl_high'].format(value=value, threshold=thresholds.get('warning', 5.0))
    
    def generate_roe_insight(self, value: float) -> str:
        """Generate insight for return on equity."""
        if value >= 10.0:
            return self.TEMPLATES['roe_strong'].format(value=value)
        elif value >= 0:
            return self.TEMPLATES['roe_weak'].format(value=value)
        else:
            return self.TEMPLATES['roe_negative'].format(value=value)
    
    def generate_liquidity_insight(self, value: float) -> str:
        """Generate insight for liquidity."""
        thresholds = self.THRESHOLDS.get('LASTL', {})
        
        if value >= thresholds.get('good', 40.0):
            return self.TEMPLATES['liquidity_strong'].format(value=value)
        elif value >= thresholds.get('warning', 25.0):
            return self.TEMPLATES['liquidity_adequate'].format(value=value)
        else:
            return self.TEMPLATES['liquidity_weak'].format(value=value, threshold=thresholds.get('warning', 25.0))
    
    def generate_gdp_insight(self, value: float) -> str:
        """Generate insight for GDP growth."""
        if value >= 3.0:
            return self.TEMPLATES['gdp_strong'].format(value=value)
        elif value >= 0:
            return self.TEMPLATES['gdp_weak'].format(value=value)
        else:
            return self.TEMPLATES['gdp_recession'].format(value=value)
    
    def generate_trend_insight(self, indicator_name: str, 
                              slope_pct: float, 
                              n_periods: int) -> str:
        """Generate insight for trend."""
        if slope_pct > 2:
            return self.TEMPLATES['trend_improving'].format(
                indicator=indicator_name, change=abs(slope_pct), periods=n_periods
            )
        elif slope_pct < -2:
            return self.TEMPLATES['trend_deteriorating'].format(
                indicator=indicator_name, change=abs(slope_pct), periods=n_periods
            )
        else:
            return self.TEMPLATES['trend_stable'].format(indicator=indicator_name)
    
    def generate_peer_insight(self, country: str, indicator: str,
                             percentile: float, median: float) -> str:
        """Generate peer comparison insight."""
        if percentile >= 60:
            return self.TEMPLATES['peer_above'].format(
                country=country, percentile=int(percentile), indicator=indicator
            )
        else:
            return self.TEMPLATES['peer_below'].format(
                country=country, percentile=int(percentile), indicator=indicator, median=median
            )
    
    def generate_country_summary(self, 
                                country_code: str,
                                country_name: str,
                                fsic_df: pd.DataFrame,
                                weo_df: pd.DataFrame,
                                risk_score: float = None,
                                risk_tier: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive country summary with insights.
        
        Returns:
            Dict with structured insights and overall assessment.
        """
        insights = {
            'country_code': country_code,
            'country_name': country_name,
            'generated_at': datetime.now().isoformat(),
            'banking_sector': [],
            'macro_context': [],
            'key_risks': [],
            'key_strengths': [],
            'overall_assessment': '',
            'risk_score': risk_score,
            'risk_tier': risk_tier
        }
        
        risk_count = 0
        strength_count = 0
        
        # Banking sector insights
        # Capital
        capital = self._get_indicator_value(fsic_df, country_code, 'RCAR|T1RWA|CAR')
        if capital is not None:
            insight = self.generate_capital_insight(capital)
            insights['banking_sector'].append(insight)
            if capital >= 14.0:
                strength_count += 1
                insights['key_strengths'].append(f"Strong capital buffer ({capital:.1f}%)")
            elif capital < 10.5:
                risk_count += 1
                insights['key_risks'].append(f"Capital adequacy concerns ({capital:.1f}%)")
        
        # NPLs
        npl = self._get_indicator_value(fsic_df, country_code, 'NPLGL')
        if npl is not None:
            insight = self.generate_npl_insight(npl)
            insights['banking_sector'].append(insight)
            if npl <= 3.0:
                strength_count += 1
                insights['key_strengths'].append(f"Strong asset quality (NPL: {npl:.1f}%)")
            elif npl > 5.0:
                risk_count += 1
                insights['key_risks'].append(f"Elevated NPLs ({npl:.1f}%)")
        
        # Profitability
        roe = self._get_indicator_value(fsic_df, country_code, 'ROE')
        if roe is not None:
            insight = self.generate_roe_insight(roe)
            insights['banking_sector'].append(insight)
            if roe >= 12.0:
                strength_count += 1
                insights['key_strengths'].append(f"Healthy profitability (ROE: {roe:.1f}%)")
            elif roe < 5.0:
                risk_count += 1
                insights['key_risks'].append(f"Weak profitability (ROE: {roe:.1f}%)")
        
        # Liquidity
        liquidity = self._get_indicator_value(fsic_df, country_code, 'LASTL')
        if liquidity is not None:
            insight = self.generate_liquidity_insight(liquidity)
            insights['banking_sector'].append(insight)
            if liquidity >= 40.0:
                strength_count += 1
                insights['key_strengths'].append(f"Robust liquidity ({liquidity:.1f}%)")
            elif liquidity < 25.0:
                risk_count += 1
                insights['key_risks'].append(f"Liquidity constraints ({liquidity:.1f}%)")
        
        # Macro context
        gdp = self._get_indicator_value(weo_df, country_code, 'NGDP_RPCH')
        if gdp is not None:
            insight = self.generate_gdp_insight(gdp)
            insights['macro_context'].append(insight)
            if gdp < 0:
                risk_count += 1
                insights['key_risks'].append(f"Recessionary environment ({gdp:.1f}% GDP)")
        
        inflation = self._get_indicator_value(weo_df, country_code, 'PCPIPCH')
        if inflation is not None:
            if inflation <= 3.0:
                insights['macro_context'].append(
                    self.TEMPLATES['inflation_low'].format(value=inflation)
                )
            elif inflation > 7.0:
                insights['macro_context'].append(
                    self.TEMPLATES['inflation_high'].format(value=inflation)
                )
                risk_count += 1
                insights['key_risks'].append(f"High inflation ({inflation:.1f}%)")
        
        # Overall assessment
        if risk_count == 0 and strength_count >= 2:
            insights['overall_assessment'] = self.TEMPLATES['overall_strong']
        elif risk_count >= 2:
            insights['overall_assessment'] = self.TEMPLATES['overall_weak']
        else:
            insights['overall_assessment'] = self.TEMPLATES['overall_adequate']
        
        return insights
    
    def generate_comparison_insights(self, 
                                    countries: List[str],
                                    df: pd.DataFrame,
                                    indicator_code: str) -> List[str]:
        """
        Generate comparative insights across countries.
        """
        insights = []
        
        # Get latest values for each country
        values = {}
        for country in countries:
            val = self._get_indicator_value(df, country, indicator_code)
            if val is not None:
                values[country] = val
        
        if len(values) < 2:
            return ["Insufficient data for comparison."]
        
        # Rank countries
        sorted_countries = sorted(values.items(), key=lambda x: x[1], reverse=True)
        best = sorted_countries[0]
        worst = sorted_countries[-1]
        
        indicator_name = indicator_code.replace('_', ' ')
        
        insights.append(
            f"**{best[0]}** leads with the highest {indicator_name} at {best[1]:.1f}%."
        )
        insights.append(
            f"**{worst[0]}** trails with {indicator_name} at {worst[1]:.1f}%."
        )
        
        # Calculate spread
        spread = best[1] - worst[1]
        insights.append(
            f"The range across selected countries is {spread:.1f} percentage points."
        )
        
        return insights
    
    def format_executive_summary(self, insights: Dict[str, Any]) -> str:
        """
        Format insights into an executive summary markdown.
        """
        summary_parts = [
            f"# Banking Sector Assessment: {insights['country_name']}",
            f"*Generated: {insights['generated_at'][:10]}*",
            ""
        ]
        
        if insights['risk_score'] is not None:
            summary_parts.append(f"**Risk Score**: {insights['risk_score']:.0f}/100")
        if insights['risk_tier']:
            summary_parts.append(f"**Risk Tier**: {insights['risk_tier']}")
        
        summary_parts.append("")
        summary_parts.append("## Overall Assessment")
        summary_parts.append(insights['overall_assessment'])
        summary_parts.append("")
        
        if insights['key_strengths']:
            summary_parts.append("### Key Strengths")
            for s in insights['key_strengths']:
                summary_parts.append(f"- ✅ {s}")
            summary_parts.append("")
        
        if insights['key_risks']:
            summary_parts.append("### Key Risks")
            for r in insights['key_risks']:
                summary_parts.append(f"- ⚠️ {r}")
            summary_parts.append("")
        
        if insights['banking_sector']:
            summary_parts.append("## Banking Sector Analysis")
            for insight in insights['banking_sector']:
                summary_parts.append(f"\n{insight}")
            summary_parts.append("")
        
        if insights['macro_context']:
            summary_parts.append("## Macroeconomic Context")
            for insight in insights['macro_context']:
                summary_parts.append(f"\n{insight}")
        
        return "\n".join(summary_parts)
