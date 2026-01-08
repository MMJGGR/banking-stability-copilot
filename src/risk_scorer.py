"""
ML-based risk scoring module.
Unsupervised learning approach inspired by S&P BICRA methodology.
Uses clustering and dimensionality reduction for risk tier assignment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import warnings

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

warnings.filterwarnings('ignore')


class RiskScorer:
    """
    Unsupervised risk scoring system for banking sector analysis.
    Inspired by S&P BICRA (Banking Industry Country Risk Assessment).
    """
    
    # Indicators where higher is better (inverse risk)
    HIGHER_IS_BETTER = [
        'RCAR', 'T1RWA', 'CAR',  # Capital adequacy
        'ROA', 'ROE',            # Profitability  
        'LASTL', 'LATA',         # Liquidity
        'NGDP_RPCH', 'NGDPDPC',  # GDP growth/level
    ]
    
    # Indicators where lower is better
    LOWER_IS_BETTER = [
        'NPLGL', 'NPLNP',        # NPL ratios
        'NEIGIE',                # Cost inefficiency
        'PCPIPCH', 'LUR',        # Inflation, unemployment
        'GGXWDG_NGDP',           # Debt
    ]
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.pca = None
        self.cluster_model = None
        self.feature_importance_: Dict[str, float] = {}
        self.cluster_risk_mapping_: Dict[int, float] = {}
        self.fitted_ = False
        
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare feature matrix from long-format data.
        Handles indicator direction (higher/lower is better) for consistent risk interpretation.
        """
        # Pivot to wide format: countries x indicators (latest values)
        if 'period' in df.columns:
            # Get most recent data for each country-indicator pair
            latest = df.sort_values('period').groupby(
                ['country_code', 'indicator_code']
            ).last().reset_index()
        else:
            latest = df.copy()
        
        wide = latest.pivot_table(
            index='country_code',
            columns='indicator_code',
            values='value',
            aggfunc='mean'
        )
        
        # Flip sign for "lower is better" indicators
        for col in wide.columns:
            indicator_base = str(col).split('_')[0] if '_' in str(col) else str(col)
            if any(lib in indicator_base for lib in self.LOWER_IS_BETTER):
                wide[col] = -wide[col]  # Now higher = better for all
        
        # Drop columns with too many missing values (>50%)
        threshold = len(wide) * 0.5
        valid_cols = wide.columns[wide.notna().sum() > threshold]
        wide = wide[valid_cols]
        
        # Drop rows with too many missing values
        threshold = len(wide.columns) * 0.3
        wide = wide[wide.notna().sum(axis=1) > threshold]
        
        return wide, list(wide.columns)
    
    def _compute_feature_importance(self, X: np.ndarray, 
                                   cluster_labels: np.ndarray,
                                   feature_names: List[str]) -> Dict[str, float]:
        """
        Compute feature importance for risk differentiation.
        Uses mutual information between features and cluster assignments.
        """
        try:
            # Impute remaining NaNs for MI calculation
            X_filled = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
            
            mi_scores = mutual_info_classif(
                X_filled, cluster_labels, 
                discrete_features=False,
                random_state=self.random_state
            )
            
            # Normalize to 0-1
            mi_scores = mi_scores / (mi_scores.max() + 1e-10)
            
            importance = dict(zip(feature_names, mi_scores))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
        except Exception:
            return {name: 1.0/len(feature_names) for name in feature_names}
    
    def fit(self, df: pd.DataFrame) -> 'RiskScorer':
        """
        Fit the risk scoring model on historical data.
        
        Args:
            df: Long-format DataFrame with country-indicator-period-value structure
        """
        wide, feature_names = self._prepare_features(df)
        
        if len(wide) < self.n_clusters:
            raise ValueError(f"Need at least {self.n_clusters} countries to fit model")
        
        # Fill remaining NaNs with column medians
        X = wide.fillna(wide.median()).values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA for dimensionality reduction
        n_components = min(10, X_scaled.shape[1] - 1, X_scaled.shape[0] - 1)
        if n_components < 1:
            n_components = 1
            
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Clustering
        if HAS_HDBSCAN and len(wide) > 20:
            # HDBSCAN for more robust clustering
            self.cluster_model = hdbscan.HDBSCAN(
                min_cluster_size=max(3, len(wide) // 10),
                min_samples=2,
                metric='euclidean'
            )
            cluster_labels = self.cluster_model.fit_predict(X_pca)
            
            # Re-assign outliers (-1) to nearest cluster
            if -1 in cluster_labels:
                outlier_mask = cluster_labels == -1
                if not outlier_mask.all():
                    # Use KMeans for outliers
                    n_clusters_found = len(set(cluster_labels)) - 1
                    if n_clusters_found > 0:
                        km = KMeans(n_clusters=n_clusters_found, random_state=self.random_state)
                        km.fit(X_pca[~outlier_mask])
                        cluster_labels[outlier_mask] = km.predict(X_pca[outlier_mask])
        else:
            # Standard KMeans
            self.cluster_model = KMeans(
                n_clusters=min(self.n_clusters, len(wide)),
                random_state=self.random_state,
                n_init=10
            )
            cluster_labels = self.cluster_model.fit_predict(X_pca)
        
        # Map clusters to risk levels (0-100 scale)
        # Higher average feature value = lower risk = higher score
        cluster_means = {}
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            cluster_means[cluster_id] = X_scaled[mask].mean()
        
        # Sort clusters by mean (higher mean = stronger = higher score)
        sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1], reverse=True)
        n_clusters = len(sorted_clusters)
        
        self.cluster_risk_mapping_ = {}
        for rank, (cluster_id, _) in enumerate(sorted_clusters):
            # Map to 0-100 score (rank 0 = best = 100, last = worst = lower score)
            score = 100 - (rank / max(1, n_clusters - 1)) * 70  # Range: 30-100
            self.cluster_risk_mapping_[cluster_id] = score
        
        # Compute feature importance
        self.feature_importance_ = self._compute_feature_importance(
            X_scaled, cluster_labels, feature_names
        )
        
        # Store for prediction
        self._wide_template = wide
        self._feature_names = feature_names
        self.fitted_ = True
        
        return self
    
    def score_countries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score countries based on fitted model.
        
        Returns DataFrame with columns:
        [country_code, risk_score, risk_tier, cluster_id, confidence]
        """
        if not self.fitted_:
            self.fit(df)
        
        wide, _ = self._prepare_features(df)
        
        # Align to training features
        for col in self._feature_names:
            if col not in wide.columns:
                wide[col] = np.nan
        wide = wide[self._feature_names]
        
        # Fill and scale
        X = wide.fillna(wide.median()).values
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        # Predict clusters
        if hasattr(self.cluster_model, 'predict'):
            cluster_labels = self.cluster_model.predict(X_pca)
        else:
            # For HDBSCAN, use approximate prediction
            cluster_labels = self.cluster_model.fit_predict(X_pca)
        
        # Calculate composite scores
        results = []
        for i, country in enumerate(wide.index):
            cluster_id = cluster_labels[i]
            
            # Base score from cluster
            base_score = self.cluster_risk_mapping_.get(cluster_id, 50.0)
            
            # Adjust based on feature-weighted distance from cluster center
            # Countries closer to "good" indicators get bonus
            feature_weighted_adj = 0
            for j, feat in enumerate(self._feature_names):
                if feat in self.feature_importance_:
                    feat_val = X_scaled[i, j]
                    importance = self.feature_importance_[feat]
                    feature_weighted_adj += importance * feat_val * 5  # Scale adjustment
            
            final_score = np.clip(base_score + feature_weighted_adj, 0, 100)
            
            # Confidence based on data completeness
            data_completeness = wide.loc[country].notna().mean()
            confidence = data_completeness
            
            # Risk tier assignment
            if final_score >= 80:
                tier = "Very Strong"
            elif final_score >= 65:
                tier = "Strong"
            elif final_score >= 50:
                tier = "Adequate"
            elif final_score >= 35:
                tier = "Weak"
            else:
                tier = "Very Weak"
            
            results.append({
                'country_code': country,
                'risk_score': round(final_score, 1),
                'risk_tier': tier,
                'cluster_id': int(cluster_id),
                'confidence': round(confidence, 3)
            })
        
        return pd.DataFrame(results)
    
    def get_key_risk_drivers(self, country_code: str, df: pd.DataFrame, 
                            top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Identify key risk drivers for a specific country.
        Returns indicators most contributing to risk score.
        """
        if not self.fitted_:
            self.fit(df)
        
        wide, _ = self._prepare_features(df)
        
        if country_code not in wide.index:
            return []
        
        country_data = wide.loc[country_code]
        
        # Compare to median
        median_values = wide.median()
        
        drivers = []
        for feat in self._feature_names[:20]:  # Top 20 features
            if pd.notna(country_data.get(feat)):
                value = country_data[feat]
                median_val = median_values[feat]
                
                # Z-score deviation
                std = wide[feat].std()
                if std > 0:
                    z_score = (value - median_val) / std
                else:
                    z_score = 0
                
                importance = self.feature_importance_.get(feat, 0)
                
                # Determine if this is a risk or strength
                is_risk = z_score < -0.5  # Below median
                
                drivers.append({
                    'indicator': feat,
                    'value': value,
                    'peer_median': median_val,
                    'z_score': round(z_score, 2),
                    'importance': round(importance, 3),
                    'impact': 'risk' if is_risk else 'strength'
                })
        
        # Sort by importance * |z_score| to get most impactful
        drivers.sort(key=lambda x: x['importance'] * abs(x['z_score']), reverse=True)
        
        return drivers[:top_n]
    
    def detect_deterioration(self, df: pd.DataFrame, 
                            lookback_periods: int = 4) -> pd.DataFrame:
        """
        Detect countries showing signs of deterioration.
        Key for early warning on credit risk.
        
        Returns DataFrame with deterioration signals.
        """
        if 'period' not in df.columns:
            return pd.DataFrame()
        
        # Get time-ordered data
        df_sorted = df.sort_values('period')
        
        # Group by country and indicator
        results = []
        
        for country in df_sorted['country_code'].unique():
            country_data = df_sorted[df_sorted['country_code'] == country]
            
            for indicator in country_data['indicator_code'].unique():
                ind_data = country_data[country_data['indicator_code'] == indicator]
                
                if len(ind_data) >= lookback_periods:
                    recent = ind_data.tail(lookback_periods)
                    values = recent['value'].values
                    
                    # Calculate trend
                    if len(values) > 1:
                        x = np.arange(len(values))
                        slope = np.polyfit(x, values, 1)[0]
                        
                        # Determine if deterioration (depends on indicator type)
                        indicator_base = str(indicator).split('_')[0]
                        
                        is_higher_better = any(
                            hb in indicator_base for hb in self.HIGHER_IS_BETTER
                        )
                        
                        if is_higher_better:
                            is_deteriorating = slope < 0
                        else:
                            is_deteriorating = slope > 0
                        
                        if is_deteriorating and abs(slope) > values.std() * 0.1:
                            results.append({
                                'country_code': country,
                                'indicator_code': indicator,
                                'current_value': values[-1],
                                'trend_slope': round(slope, 4),
                                'periods_analyzed': len(values),
                                'deterioration_severity': 'High' if abs(slope) > values.std() else 'Moderate'
                            })
        
        return pd.DataFrame(results)


class EconomicResilienceScorer:
    """
    Scores economic resilience component (inspired by BICRA).
    Uses WEO macroeconomic data.
    """
    
    # Key indicators for economic resilience
    RESILIENCE_INDICATORS = {
        'NGDP_RPCH': ('growth', 1.0),      # GDP growth - higher better
        'NGDPDPC': ('level', 0.8),          # GDP per capita - higher better
        'PCPIPCH': ('stability', -0.7),     # Inflation - lower better
        'BCA_NGDPD': ('external', 0.6),     # Current account - higher better
        'GGXWDG_NGDP': ('fiscal', -0.5),    # Govt debt - lower better
        'LUR': ('labor', -0.4),             # Unemployment - lower better
    }
    
    def score(self, weo_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate economic resilience scores from WEO data.
        """
        # Get latest values per country-indicator
        if 'period' in weo_data.columns:
            latest = weo_data.sort_values('period').groupby(
                ['country_code', 'indicator_code']
            ).last().reset_index()
        else:
            latest = weo_data
        
        # Pivot
        wide = latest.pivot_table(
            index='country_code',
            columns='indicator_code',
            values='value'
        )
        
        # Normalize each indicator to 0-100 percentile
        normalized = pd.DataFrame(index=wide.index)
        
        for ind, (category, weight) in self.RESILIENCE_INDICATORS.items():
            matching_cols = [c for c in wide.columns if ind in c]
            if matching_cols:
                col = matching_cols[0]
                values = wide[col]
                
                if weight > 0:
                    # Higher is better
                    percentiles = values.rank(pct=True) * 100
                else:
                    # Lower is better
                    percentiles = (1 - values.rank(pct=True)) * 100
                
                normalized[f'{ind}_score'] = percentiles * abs(weight)
        
        if normalized.empty:
            return pd.DataFrame()
        
        # Composite score
        scores = pd.DataFrame({
            'country_code': normalized.index,
            'economic_resilience_score': normalized.mean(axis=1).values
        })
        
        return scores


class IndustryRiskScorer:
    """
    Scores banking industry risk component (inspired by BICRA).
    Uses FSIC banking sector data.
    """
    
    # Key indicators for industry risk
    INDUSTRY_INDICATORS = {
        'RCAR': ('capital', 1.0),           # Regulatory capital ratio
        'NPLGL': ('asset_quality', -1.0),   # NPL ratio
        'ROE': ('profitability', 0.8),      # Return on equity
        'LASTL': ('liquidity', 0.7),        # Liquid assets/ST liabilities
    }
    
    def score(self, fsic_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate industry risk scores from FSIC data.
        """
        if 'period' in fsic_data.columns:
            latest = fsic_data.sort_values('period').groupby(
                ['country_code', 'indicator_code']
            ).last().reset_index()
        else:
            latest = fsic_data
        
        wide = latest.pivot_table(
            index='country_code',
            columns='indicator_code',
            values='value'
        )
        
        normalized = pd.DataFrame(index=wide.index)
        
        for ind, (category, weight) in self.INDUSTRY_INDICATORS.items():
            matching_cols = [c for c in wide.columns if ind in c]
            if matching_cols:
                col = matching_cols[0]
                values = wide[col]
                
                if weight > 0:
                    percentiles = values.rank(pct=True) * 100
                else:
                    percentiles = (1 - values.rank(pct=True)) * 100
                
                normalized[f'{ind}_score'] = percentiles * abs(weight)
        
        if normalized.empty:
            return pd.DataFrame()
        
        scores = pd.DataFrame({
            'country_code': normalized.index,
            'industry_risk_score': normalized.mean(axis=1).values
        })
        
        return scores


def compute_composite_bicra_score(
    fsic_data: pd.DataFrame,
    weo_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute composite BICRA-inspired score combining:
    - Economic Resilience (from WEO)
    - Industry Risk (from FSIC)
    
    Returns DataFrame with country-level composite scores.
    """
    econ_scorer = EconomicResilienceScorer()
    industry_scorer = IndustryRiskScorer()
    
    econ_scores = econ_scorer.score(weo_data)
    industry_scores = industry_scorer.score(fsic_data)
    
    if econ_scores.empty and industry_scores.empty:
        return pd.DataFrame()
    
    # Merge scores
    if not econ_scores.empty and not industry_scores.empty:
        combined = econ_scores.merge(industry_scores, on='country_code', how='outer')
    elif not econ_scores.empty:
        combined = econ_scores.copy()
        combined['industry_risk_score'] = 50  # Default
    else:
        combined = industry_scores.copy()
        combined['economic_resilience_score'] = 50
    
    # Fill missing
    combined = combined.fillna(50)
    
    # Composite (equal weight)
    combined['composite_score'] = (
        combined['economic_resilience_score'] * 0.5 +
        combined['industry_risk_score'] * 0.5
    )
    
    # Assign tiers
    def assign_tier(score):
        if score >= 80:
            return "1 (Very Strong)"
        elif score >= 65:
            return "2 (Strong)"
        elif score >= 50:
            return "3 (Adequate)"
        elif score >= 35:
            return "4 (Weak)"
        else:
            return "5 (Very Weak)"
    
    combined['bicra_tier'] = combined['composite_score'].apply(assign_tier)
    
    return combined.round(1)
