"""
Trend analysis and anomaly detection for banking indicators.
Provides time-series analytics for identifying deteriorating conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings('ignore')


class TrendAnalyzer:
    """
    Time-series trend analysis for banking and economic indicators.
    """
    
    def __init__(self, min_periods: int = 4):
        self.min_periods = min_periods
    
    def calculate_trend(self, series: pd.Series) -> Dict[str, Any]:
        """
        Calculate trend statistics for a single time series.
        
        Returns:
            Dict with trend direction, slope, r-squared, volatility, etc.
        """
        series = series.dropna()
        
        if len(series) < self.min_periods:
            return {
                'trend': 'insufficient_data',
                'slope': None,
                'slope_pct': None,
                'r_squared': None,
                'volatility': None,
                'n_periods': len(series)
            }
        
        values = series.values
        n = len(values)
        x = np.arange(n)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        r_squared = r_value ** 2
        
        # Slope as percentage of mean
        mean_val = np.mean(values)
        slope_pct = (slope / abs(mean_val) * 100) if mean_val != 0 else 0
        
        # Volatility (coefficient of variation)
        volatility = (np.std(values) / abs(mean_val) * 100) if mean_val != 0 else 0
        
        # Trend classification
        if abs(slope_pct) < 2:
            trend = 'stable'
        elif slope > 0:
            trend = 'improving' if slope_pct > 5 else 'slightly_improving'
        else:
            trend = 'deteriorating' if slope_pct < -5 else 'slightly_deteriorating'
        
        # Recent momentum (last 3 vs previous 3 periods)
        if n >= 6:
            recent_avg = np.mean(values[-3:])
            prior_avg = np.mean(values[-6:-3])
            momentum = ((recent_avg - prior_avg) / abs(prior_avg) * 100) if prior_avg != 0 else 0
        else:
            momentum = None
        
        return {
            'trend': trend,
            'slope': round(slope, 4),
            'slope_pct': round(slope_pct, 2),
            'r_squared': round(r_squared, 3),
            'p_value': round(p_value, 4),
            'volatility': round(volatility, 2),
            'momentum': round(momentum, 2) if momentum else None,
            'n_periods': n,
            'latest_value': values[-1],
            'earliest_value': values[0],
            'min_value': np.min(values),
            'max_value': np.max(values),
            'mean_value': round(mean_val, 2)
        }
    
    def detect_anomalies(self, series: pd.Series, 
                        z_threshold: float = 2.5) -> pd.DataFrame:
        """
        Detect anomalous values using z-score method.
        """
        series = series.dropna()
        
        if len(series) < self.min_periods:
            return pd.DataFrame()
        
        values = series.values
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return pd.DataFrame()
        
        z_scores = (values - mean_val) / std_val
        
        anomalies = []
        for i, (idx, val) in enumerate(series.items()):
            if abs(z_scores[i]) > z_threshold:
                anomalies.append({
                    'period': idx,
                    'value': val,
                    'z_score': round(z_scores[i], 2),
                    'type': 'high' if z_scores[i] > 0 else 'low'
                })
        
        return pd.DataFrame(anomalies)
    
    def analyze_country(self, df: pd.DataFrame, 
                       country_code: str) -> Dict[str, Dict]:
        """
        Analyze all indicators for a specific country.
        
        Args:
            df: Long-format DataFrame
            country_code: Country to analyze
            
        Returns:
            Dict mapping indicator_code to trend analysis results
        """
        country_data = df[df['country_code'] == country_code].copy()
        
        if len(country_data) == 0:
            return {}
        
        results = {}
        
        for indicator in country_data['indicator_code'].unique():
            ind_data = country_data[country_data['indicator_code'] == indicator]
            ind_data = ind_data.sort_values('period').set_index('period')
            
            if 'value' in ind_data.columns:
                series = ind_data['value']
                trend_result = self.calculate_trend(series)
                anomalies = self.detect_anomalies(series)
                
                trend_result['anomalies'] = anomalies.to_dict('records')
                results[indicator] = trend_result
        
        return results
    
    def compare_to_peers(self, df: pd.DataFrame, 
                        country_code: str,
                        indicator_code: str,
                        period: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare a country's indicator to peer group.
        """
        if period:
            period_data = df[df['period'] == period]
        else:
            # Use most recent period
            period_data = df.sort_values('period').groupby(
                ['country_code', 'indicator_code']
            ).last().reset_index()
        
        ind_data = period_data[period_data['indicator_code'] == indicator_code]
        
        if len(ind_data) == 0:
            return {}
        
        all_values = ind_data['value'].dropna()
        country_row = ind_data[ind_data['country_code'] == country_code]
        
        if len(country_row) == 0:
            return {}
        
        country_value = country_row['value'].iloc[0]
        
        percentile = stats.percentileofscore(all_values, country_value)
        
        return {
            'value': country_value,
            'percentile': round(percentile, 1),
            'peer_count': len(all_values),
            'peer_min': all_values.min(),
            'peer_max': all_values.max(),
            'peer_median': all_values.median(),
            'peer_mean': round(all_values.mean(), 2),
            'vs_median': round(country_value - all_values.median(), 2)
        }
    
    def get_deteriorating_indicators(self, df: pd.DataFrame,
                                     country_code: str) -> List[Dict]:
        """
        Get list of deteriorating indicators for a country,
        sorted by severity.
        """
        trends = self.analyze_country(df, country_code)
        
        deteriorating = []
        for indicator, trend_data in trends.items():
            if trend_data['trend'] in ['deteriorating', 'slightly_deteriorating']:
                deteriorating.append({
                    'indicator_code': indicator,
                    **trend_data
                })
        
        # Sort by absolute slope (most severe first)
        deteriorating.sort(key=lambda x: abs(x.get('slope_pct', 0)), reverse=True)
        
        return deteriorating


class MovingAverageAnalyzer:
    """
    Moving average analysis for smoothing and trend identification.
    """
    
    @staticmethod
    def calculate_ma(series: pd.Series, window: int = 4) -> pd.Series:
        """Calculate simple moving average."""
        return series.rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    def calculate_ema(series: pd.Series, span: int = 4) -> pd.Series:
        """Calculate exponential moving average."""
        return series.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def detect_crossovers(series: pd.Series, 
                         short_window: int = 3,
                         long_window: int = 8) -> pd.DataFrame:
        """
        Detect moving average crossovers (bullish/bearish signals).
        """
        short_ma = series.rolling(window=short_window, min_periods=1).mean()
        long_ma = series.rolling(window=long_window, min_periods=1).mean()
        
        crossovers = []
        prev_diff = None
        
        for i, (idx, val) in enumerate(series.items()):
            if i >= long_window:
                diff = short_ma.iloc[i] - long_ma.iloc[i]
                
                if prev_diff is not None:
                    if prev_diff < 0 and diff >= 0:
                        crossovers.append({
                            'period': idx,
                            'type': 'bullish',
                            'short_ma': short_ma.iloc[i],
                            'long_ma': long_ma.iloc[i]
                        })
                    elif prev_diff > 0 and diff <= 0:
                        crossovers.append({
                            'period': idx,
                            'type': 'bearish',
                            'short_ma': short_ma.iloc[i],
                            'long_ma': long_ma.iloc[i]
                        })
                
                prev_diff = diff
        
        return pd.DataFrame(crossovers)


def smooth_series(series: pd.Series, method: str = 'savgol') -> pd.Series:
    """
    Smooth a time series for visualization.
    """
    series = series.dropna()
    
    if len(series) < 5:
        return series
    
    values = series.values
    
    if method == 'savgol':
        # Savitzky-Golay filter
        window = min(len(values) - 1, 7)
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            smoothed = savgol_filter(values, window, polyorder=2)
        else:
            smoothed = values
    elif method == 'ewm':
        smoothed = pd.Series(values).ewm(span=3).mean().values
    else:
        smoothed = values
    
    return pd.Series(smoothed, index=series.index)
