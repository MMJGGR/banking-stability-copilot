"""
Gap imputation module for handling missing data in IMF datasets.
Uses temporal interpolation, KNN, and matrix factorization techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy.sparse.linalg import svds
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class GapImputer:
    """
    Multi-strategy imputation for sparse country-indicator time series data.
    """
    
    def __init__(self, n_neighbors: int = 5, svd_components: int = 10):
        self.n_neighbors = n_neighbors
        self.svd_components = svd_components
        self.scaler = StandardScaler()
        self._imputation_log: List[Dict] = []
        
    def impute_temporal(self, series: pd.Series, method: str = 'linear') -> Tuple[pd.Series, pd.Series]:
        """
        Temporal interpolation for a single time series.
        
        Args:
            series: Time-indexed pandas Series with potential gaps
            method: Interpolation method ('linear', 'spline', 'nearest')
            
        Returns:
            Tuple of (imputed_series, confidence_scores)
        """
        if len(series) == 0 or series.isna().all():
            return series, pd.Series(index=series.index, data=0.0)
        
        # Create copy to avoid modifying original
        result = series.copy()
        confidence = pd.Series(index=series.index, data=1.0)
        
        # Mark missing values
        missing_mask = series.isna()
        
        if not missing_mask.any():
            return result, confidence
        
        # Get valid data points
        valid_idx = series.dropna().index
        valid_values = series.dropna().values
        
        if len(valid_values) < 2:
            # Not enough data for interpolation
            result.fillna(series.mean() if len(valid_values) > 0 else 0, inplace=True)
            confidence[missing_mask] = 0.1
            return result, confidence
        
        try:
            # Convert index to numeric for interpolation
            if hasattr(series.index, 'year'):
                numeric_idx = np.array([d.year + d.month/12 for d in series.index])
                valid_numeric = np.array([d.year + d.month/12 for d in valid_idx])
            else:
                numeric_idx = np.arange(len(series))
                valid_numeric = np.array([series.index.get_loc(i) for i in valid_idx])
            
            # Interpolate
            if method == 'spline' and len(valid_values) >= 4:
                interp_func = interp1d(valid_numeric, valid_values, kind='cubic',
                                      fill_value='extrapolate', bounds_error=False)
            else:
                interp_func = interp1d(valid_numeric, valid_values, kind='linear',
                                      fill_value='extrapolate', bounds_error=False)
            
            # Fill missing values
            for i, idx in enumerate(series.index):
                if missing_mask[idx]:
                    result[idx] = interp_func(numeric_idx[i])
                    
                    # Confidence based on distance to nearest known point
                    distances = np.abs(valid_numeric - numeric_idx[i])
                    min_distance = distances.min()
                    confidence[idx] = max(0.1, 1.0 - min_distance / 5.0)
                    
        except Exception as e:
            # Fallback to forward/backward fill
            result = result.ffill().bfill()
            confidence[missing_mask] = 0.3
            
        return result, confidence
    
    def impute_knn_cross_country(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        KNN imputation using similar countries.
        
        Uses the FULL feature set to find similar countries, then imputes
        missing values from k nearest neighbors based on non-missing features.
        
        Args:
            df: DataFrame with countries as rows and indicators as columns
            
        Returns:
            Tuple of (imputed_df, confidence_df)
        """
        if df.empty:
            return df, pd.DataFrame(index=df.index, columns=df.columns, data=0.0)
        
        # Track original missing positions
        missing_mask = df.isna()
        
        # Initialize confidence
        confidence = pd.DataFrame(index=df.index, columns=df.columns, data=1.0)
        confidence[missing_mask] = 0.0
        
        # Columns with any valid data
        valid_cols = df.columns[df.notna().any()]
        
        if len(valid_cols) == 0:
            return df, confidence
        
        df_subset = df[valid_cols].copy()
        
        # KNN imputation - let sklearn handle the NaN values directly
        # KNNImputer finds neighbors based on non-missing features, 
        # then imputes from neighbors' actual values
        try:
            # Don't pre-fill with mean! Let KNNImputer handle NaN properly
            imputer = KNNImputer(
                n_neighbors=min(self.n_neighbors, len(df) - 1),
                weights='distance'  # Weight by inverse distance for better accuracy
            )
            imputed_data = imputer.fit_transform(df_subset)
            
            result = pd.DataFrame(imputed_data, index=df_subset.index, columns=df_subset.columns)
            
            # Update confidence for imputed values
            for col in result.columns:
                was_missing = missing_mask.loc[result.index, col] if col in missing_mask.columns else pd.Series(False, index=result.index)
                confidence.loc[result.index, col] = confidence.loc[result.index, col].where(~was_missing, 0.7)
            
            # Merge back with original columns
            full_result = df.copy()
            full_result[result.columns] = result
            
            return full_result, confidence
            
        except Exception as e:
            # Fallback to median imputation (not mean - more robust to outliers)
            print(f"    KNN imputation failed ({e}), falling back to median")
            result = df.fillna(df.median())
            confidence[missing_mask] = 0.3
            return result, confidence
    
    def impute_matrix_factorization(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        SVD-based matrix factorization for sparse matrices.
        Good for finding latent patterns across countries and indicators.
        
        Args:
            df: DataFrame with countries as rows and indicators as columns
            
        Returns:
            Tuple of (imputed_df, confidence_df)
        """
        if df.empty:
            return df, pd.DataFrame(index=df.index, columns=df.columns, data=0.0)
        
        missing_mask = df.isna()
        confidence = pd.DataFrame(index=df.index, columns=df.columns, data=1.0)
        
        # Fill with column means for SVD
        df_filled = df.fillna(df.mean())
        
        # Handle remaining NaNs (columns that were all NaN)
        df_filled = df_filled.fillna(0)
        
        try:
            # Normalize
            row_means = df_filled.mean(axis=1)
            df_centered = df_filled.sub(row_means, axis=0)
            
            # SVD decomposition
            n_components = min(self.svd_components, min(df_centered.shape) - 1)
            if n_components < 1:
                n_components = 1
                
            U, sigma, Vt = svds(df_centered.values.astype(float), k=n_components)
            
            # Reconstruct
            reconstructed = U @ np.diag(sigma) @ Vt
            reconstructed = reconstructed + row_means.values.reshape(-1, 1)
            
            result = pd.DataFrame(reconstructed, index=df.index, columns=df.columns)
            
            # Use reconstructed values only for missing entries
            final_result = df.copy()
            final_result[missing_mask] = result[missing_mask]
            
            # Confidence based on reconstruction quality
            confidence[missing_mask] = 0.6
            
            return final_result, confidence
            
        except Exception as e:
            return df.fillna(df.mean()), confidence.where(~missing_mask, 0.3)
    
    def impute_dataset(self, df: pd.DataFrame, 
                       strategy: str = 'hybrid') -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Full imputation pipeline for a dataset.
        
        Args:
            df: Long-format DataFrame with columns [country_code, indicator_code, period, value]
            strategy: 'temporal', 'knn', 'svd', or 'hybrid'
            
        Returns:
            Tuple of (imputed_df, confidence_df, stats)
        """
        stats = {
            'original_missing': 0,
            'imputed_count': 0,
            'methods_used': [],
            'avg_confidence': 0.0
        }
        
        if df.empty:
            return df, pd.DataFrame(), stats
        
        # Pivot to wide format for cross-sectional imputation
        wide = df.pivot_table(
            index='country_code',
            columns=['indicator_code', 'period'],
            values='value'
        )
        
        stats['original_missing'] = wide.isna().sum().sum()
        
        if strategy == 'temporal' or strategy == 'hybrid':
            # First: temporal interpolation within each country-indicator pair
            imputed_temporal = wide.copy()
            conf_temporal = pd.DataFrame(index=wide.index, columns=wide.columns, data=1.0)
            
            for country in wide.index:
                indicators = wide.columns.get_level_values(0).unique()
                for indicator in indicators:
                    try:
                        series = wide.loc[country, indicator]
                        if isinstance(series, pd.Series) and len(series) > 1:
                            imp_series, conf_series = self.impute_temporal(series)
                            imputed_temporal.loc[country, indicator] = imp_series.values
                            conf_temporal.loc[country, indicator] = conf_series.values
                    except Exception:
                        continue
            
            stats['methods_used'].append('temporal')
            wide = imputed_temporal
            confidence = conf_temporal
        else:
            confidence = pd.DataFrame(index=wide.index, columns=wide.columns, data=1.0)
        
        if strategy == 'knn' or (strategy == 'hybrid' and wide.isna().any().any()):
            # Then: KNN for remaining gaps
            wide, conf_knn = self.impute_knn_cross_country(wide)
            confidence = confidence.combine(conf_knn, min)
            stats['methods_used'].append('knn')
        
        if strategy == 'svd' or (strategy == 'hybrid' and wide.isna().any().any()):
            # Finally: matrix factorization
            wide, conf_svd = self.impute_matrix_factorization(wide)
            confidence = confidence.combine(conf_svd, min)
            stats['methods_used'].append('svd')
        
        # Convert back to long format
        imputed_long = wide.stack(level=[0, 1]).reset_index()
        imputed_long.columns = ['country_code', 'indicator_code', 'period', 'value']
        
        conf_long = confidence.stack(level=[0, 1]).reset_index()
        conf_long.columns = ['country_code', 'indicator_code', 'period', 'confidence']
        
        # Merge confidence
        imputed_long = imputed_long.merge(conf_long, on=['country_code', 'indicator_code', 'period'])
        
        stats['imputed_count'] = stats['original_missing'] - imputed_long['value'].isna().sum()
        stats['avg_confidence'] = imputed_long['confidence'].mean()
        
        return imputed_long, confidence, stats


def impute_time_series(series: pd.Series, method: str = 'linear') -> pd.Series:
    """Convenience function for single series imputation."""
    imputer = GapImputer()
    result, _ = imputer.impute_temporal(series, method)
    return result
