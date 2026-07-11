"""
Data-Driven Banking System Risk Model

This script:
1. Explores ALL available indicators in FSIC, WEO, MFS datasets
2. Uses ML (PCA, correlation) to identify meaningful feature groupings
3. Anchors with GDP per capita for logical soundness
4. Builds a two-pillar model: Economic Risk + Industry Risk
5. Uses WEIGHTED HYBRID IMPUTATION: KNN + confidence-weighted scoring

The model learns from the data rather than hardcoding weights.
Uncertain (imputed) values are discounted toward median risk.
"""

import os
import sys
import gc
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import IMFDataLoader
from src.config import CACHE_DIR
from src.country_names import country_name_from_code
from src.feature_engineering import CrisisFeatureEngineer
from src.crisis_classifier import (
    CrisisClassifier,
    train_crisis_model,
    _compute_lag_features,
)
from src.imputation import GapImputer
from src.data_loader import WGILoader
from src.pillar_pipeline import PillarInferencePipeline

MODEL_PATH = os.path.join(CACHE_DIR, "risk_model.pkl")
INFERENCE_PIPELINE_PATH = os.path.join(CACHE_DIR, "inference_pipeline.pkl")


def explore_all_indicators(fsic_df, weo_df, mfs_df):
    """
    Explore all available indicators across datasets.
    Returns wide-format DataFrame with all indicators.
    """
    print("\n" + "=" * 70)
    print("EXPLORING ALL AVAILABLE INDICATORS")
    print("=" * 70)
    
    def get_indicator_summary(df, dataset_name):
        """Summarize indicators in a dataset."""
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        summary = df.groupby('indicator_code').agg({
            'country_code': 'nunique',
            'value': ['mean', 'std', 'min', 'max'],
            'period': 'max'
        }).reset_index()
        summary.columns = ['indicator', 'n_countries', 'mean', 'std', 'min', 'max', 'latest_period']
        summary['dataset'] = dataset_name
        return summary
    
    # Summarize each dataset
    fsic_summary = get_indicator_summary(fsic_df, 'FSIC')
    weo_summary = get_indicator_summary(weo_df, 'WEO')
    mfs_summary = get_indicator_summary(mfs_df, 'MFS')
    
    print(f"\nFSIC: {len(fsic_summary)} indicators")
    print(f"WEO:  {len(weo_summary)} indicators")
    print(f"MFS:  {len(mfs_summary)} indicators")
    
    # Show most common FSIC indicators (banking sector)
    if len(fsic_summary) > 0:
        print("\nTop FSIC indicators by country coverage:")
        top_fsic = fsic_summary.nlargest(20, 'n_countries')
        for _, row in top_fsic.iterrows():
            print(f"  {row['indicator']}: {row['n_countries']} countries, "
                  f"mean={row['mean']:.2f}")
    
    # Show most common WEO indicators (macro)
    if len(weo_summary) > 0:
        print("\nTop WEO indicators by country coverage:")
        top_weo = weo_summary.nlargest(15, 'n_countries')
        for _, row in top_weo.iterrows():
            print(f"  {row['indicator']}: {row['n_countries']} countries")
    
    return pd.concat([fsic_summary, weo_summary, mfs_summary], ignore_index=True)


def build_country_feature_matrix(fsic_df, weo_df, mfs_df, min_countries=30):
    """
    Build a wide-format country x indicator matrix.
    Uses most recent values, filters to indicators with good coverage.
    """
    print("\n" + "=" * 70)
    print("BUILDING COUNTRY FEATURE MATRIX")
    print("=" * 70)
    
    def get_latest_wide(df, dataset_prefix, min_countries=30):
        """Convert long to wide format with latest values."""
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        # Get most recent per country-indicator
        latest = df.sort_values('period').groupby(
            ['country_code', 'indicator_code']
        ).agg({'value': 'last', 'country_name': 'first'}).reset_index()
        
        # Filter to indicators with good coverage
        indicator_counts = latest.groupby('indicator_code')['country_code'].nunique()
        good_indicators = indicator_counts[indicator_counts >= min_countries].index
        latest = latest[latest['indicator_code'].isin(good_indicators)]
        
        if len(latest) == 0:
            return pd.DataFrame()
        
        # Pivot
        wide = latest.pivot_table(
            index='country_code',
            columns='indicator_code',
            values='value'
        )
        
        # Add dataset prefix
        wide.columns = [f"{dataset_prefix}_{c}" for c in wide.columns]
        
        print(f"  {dataset_prefix}: {len(wide.columns)} indicators, {len(wide)} countries")
        
        return wide
    
    # Get country names mapping
    country_names = {}
    for df in [fsic_df, weo_df, mfs_df]:
        if df is not None and len(df) > 0:
            names = df.groupby('country_code')['country_name'].first().to_dict()
            country_names.update(names)
    
    # Build wide format for each
    fsic_wide = get_latest_wide(fsic_df, 'FSIC', min_countries)
    weo_wide = get_latest_wide(weo_df, 'WEO', min_countries)
    # MFS has too many indicators - sample top ones
    mfs_wide = get_latest_wide(mfs_df, 'MFS', min_countries=50)
    
    # Combine
    all_wide = [df for df in [fsic_wide, weo_wide, mfs_wide] if len(df) > 0]
    
    if not all_wide:
        return pd.DataFrame(), {}
    
    combined = all_wide[0]
    for df in all_wide[1:]:
        combined = combined.join(df, how='outer')
    
    print(f"\nCombined: {len(combined.columns)} indicators, {len(combined)} countries")
    
    return combined, country_names


def identify_anchor_indicator(features_df, weo_df):
    """
    Find GDP per capita to use as development anchor.
    This ensures Japan > Nigeria in development, which correlates with lower banking risk.
    
    Returns a Series indexed by country_code.
    """
    print("\n" + "=" * 70)
    print("IDENTIFYING DEVELOPMENT ANCHOR")
    print("=" * 70)
    
    # PRIORITY: Look for GDP per capita specifically (not GDP growth!)
    # gdp_per_capita is the key indicator for development level
    if 'gdp_per_capita' in features_df.columns and 'country_code' in features_df.columns:
        # Create Series indexed by country_code
        anchor_values = features_df.set_index('country_code')['gdp_per_capita']
        print(f"Using gdp_per_capita as development anchor ({len(anchor_values.dropna())} countries)")
        return anchor_values
    
    # If features_df already has country_code as index
    if 'gdp_per_capita' in features_df.columns:
        anchor_values = features_df['gdp_per_capita']
        print(f"Using gdp_per_capita as development anchor ({len(anchor_values.dropna())} countries)")
        return anchor_values
    
    # Fallback: Look for NGDPDPC (IMF GDP per capita code)
    gdp_pc_cols = [c for c in features_df.columns if 'NGDPDPC' in c.upper() or 'per_capita' in c.lower()]
    if gdp_pc_cols:
        anchor_col = gdp_pc_cols[0]
        if 'country_code' in features_df.columns:
            anchor_values = features_df.set_index('country_code')[anchor_col]
        else:
            anchor_values = features_df[anchor_col]
        print(f"Using {anchor_col} as development anchor ({len(anchor_values.dropna())} countries)")
        return anchor_values
    
    # Alternative: try to get from WEO directly
    if weo_df is not None and len(weo_df) > 0:
        gdp_data = weo_df[weo_df['indicator_code'].str.contains('NGDPDPC', case=False, na=False)]
        if len(gdp_data) > 0:
            latest = gdp_data.sort_values('period').groupby('country_code')['value'].last()
            print(f"Using NGDPDPC from WEO as anchor ({len(latest)} countries)")
            return latest
    
    print("WARNING: No GDP per capita found - model may not be properly anchored")
    return None


def build_two_pillar_model_legacy(features_df, anchor_series, country_names):
    """
    Build the two-pillar risk model:
    1. Economic Risk Pillar - macro fundamentals
    2. Industry Risk Pillar - banking sector health
    
    Uses PCA to extract meaningful components, then anchors with development level.
    """
    print("\n" + "=" * 70)
    print("BUILDING TWO-PILLAR RISK MODEL")
    print("=" * 70)
    
    # Separate banking (FSIC) from macro (WEO) indicators
    # Define pillar mapping matching FeatureEngineer output
    # NOTE: gdp_per_capita RE-INCLUDED in PCA to ensure the primary component captures
    # development level (Wealth = Resilience). However, it remains EXCLUDED from the 
    # supervised CrisisClassifier to prevent wealth-bias in the crisis probability component.
    # Ref: Borio & Lowe (2002), Drehmann & Juselius (2014)
    economic_cols = [
        'gdp_growth', 'inflation', 'current_account_gdp', 'gdp_per_capita',
        'govt_debt_gdp', 'fiscal_balance_gdp', 'unemployment',
        'credit_to_gdp', 'credit_to_gdp_relative', 'debt_service_gdp', 'external_debt_gdp',
        'sovereign_liability_to_reserves',
        # Literature gap features (2026-02)
        'inflation_differential_3yr',   # REER proxy (Gourinchas & Obstfeld 2012)
        'interest_cost_gdp',            # Debt sustainability (fiscal - primary balance)
        'interest_cost_trend_3yr',      # Interest cost trend
        'credit_growth_3yr',            # Credit boom (Schularick & Taylor 2012)
        'm2_to_reserves',               # Twin crises (Kaminsky & Reinhart 1999)
        'ca_deficit_severity',          # Capital inflow surge proxy
        'tot_deterioration_3yr',        # Terms of trade shock
        # WGI governance (BICRA: Economic Risk)
        'voice_accountability', 'political_stability', 'govt_effectiveness'
    ]
    
    industry_cols = [
        'capital_adequacy', 'npl_ratio', 'roe', 'roa', 'liquid_assets_st_liab', 
        'liquid_assets_total', 'customer_deposits_loans', 'fx_loan_exposure', 
        'tier1_capital', 'npl_provisions', 'loan_concentration',
        'real_estate_loans', 'sovereign_exposure_ratio',
        'bank_liability_to_nfa', # Banking Sector FX Risk
        'real_estate_credit_growth_3yr',  # Housing boom proxy (Reinhart & Rogoff 2009)
        # WGI governance (BICRA: Institutional Framework)
        'regulatory_quality', 'rule_of_law', 'control_corruption'
    ]
    
    # Filter to columns present in df
    weo_cols = [c for c in economic_cols if c in features_df.columns]
    fsic_cols = [c for c in industry_cols if c in features_df.columns]
    
    print(f"Economic (Macro) indicators: {len(weo_cols)}")
    print(f"Industry (Banking) indicators: {len(fsic_cols)}")
    
    # Get countries with reasonable data coverage
    min_data_pct = 0.2  # At least 20% of indicators
    
    # Set country_code as index for proper filtering
    if 'country_code' in features_df.columns:
        features_indexed = features_df.set_index('country_code')
    else:
        features_indexed = features_df
    
    data_coverage = features_indexed.select_dtypes(include=[np.number]).notna().mean(axis=1)
    good_countries = data_coverage[data_coverage >= min_data_pct].index
    
    print(f"Countries with sufficient data: {len(good_countries)}")
    
    features_filtered = features_indexed.loc[good_countries].copy()
    
    # Separate numeric columns from non-numeric (like country_code)
    numeric_cols = features_filtered.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in features_filtered.columns if c not in numeric_cols]
    
    # Work only with numeric columns for median/scaling
    features_numeric = features_filtered[numeric_cols].copy()
    
    # =========================================================================
    # WEIGHTED HYBRID IMPUTATION STRATEGY
    # =========================================================================
    # 1. Track which values are real vs imputed (for confidence weighting)
    # 2. Use KNN imputation (similar countries fill gaps)
    # 3. Weight final scores by data confidence (imputed values → median risk)
    # =========================================================================
    
    print("\n--- Weighted Hybrid Imputation ---")
    
    # Track original missing positions for confidence weighting
    original_missing_mask = features_numeric.isna()
    
    # Calculate imputation confidence per country (% of real data)
    imputation_confidence = 1 - original_missing_mask.mean(axis=1)
    print(f"  Data completeness range: {imputation_confidence.min():.0%} - {imputation_confidence.max():.0%}")
    
    # Apply KNN imputation - uses similar countries to fill gaps
    try:
        imputer = GapImputer(n_neighbors=5)
        # imputer returns (imputed_df, confidence_df)
        features_imputed, knn_confidence = imputer.impute_knn_cross_country(features_numeric)
        print(f"  KNN imputation applied ({original_missing_mask.sum().sum()} values imputed)")
    except Exception as e:
        # Fallback to median if KNN fails
        print(f"  KNN failed, falling back to median: {e}")
        features_imputed = features_numeric.fillna(features_numeric.median()).fillna(0)
    
    # Use imputed features for scaling
    # Post-imputation safety: fill any remaining NaN with column median
    # This handles very sparse features (e.g., tot_deterioration_3yr with <10 countries)
    # where KNN can't find enough neighbors with data
    remaining_nans = features_imputed.isna().sum().sum()
    if remaining_nans > 0:
        features_imputed = features_imputed.fillna(features_imputed.median()).fillna(0)
        print(f"  Post-KNN median fallback: {remaining_nans} remaining NaN values filled")
    features_numeric = features_imputed
    
    # Save imputed features for dashboard display (to show asterisks correctly)
    try:
        from src.config import CACHE_DIR
        imputed_path = os.path.join(CACHE_DIR, 'imputed_features.parquet')
        
        # Merge year columns back from features_filtered (they were excluded from imputation)
        year_cols = [c for c in features_filtered.columns if c.endswith('_year')]
        imputed_with_years = features_imputed.copy()
        for col in year_cols:
            if col in features_filtered.columns:
                imputed_with_years[col] = features_filtered[col]
        
        # Reset index to make country_code a column
        imputed_with_years.reset_index().to_parquet(imputed_path, index=False)
        print(f"  Saved imputed features to: {imputed_path} (with {len(year_cols)} year columns)")
    except Exception as e:
        print(f"  WARNING: Could not save imputed features: {e}")
    
    # --- PREPROCESSING: LOG TRANSFORM SKEWED FEATURES ---
    # Fix single-feature dominance in PCA by compressing scale of high-variance features
    # NOTE: nominal_gdp and gdp_per_capita removed from skewed_features
    # since they are no longer in the PCA model inputs
    skewed_features = [
        'inflation',            # Range: -5% to 1000% -> Log handles hyperinflation outliers
        'sovereign_exposure_ratio' # Range: 0% to 50% -> Log smooths high exposure
    ]
    
    # Loan concentration is inverted (negative), so handle separately
    # It ranges from -100 to 0. We want to compress the magnitude.
    # Logic: -log1p(abs(x)) -> Preserves negative direction but compresses scale
    if 'loan_concentration' in features_numeric.columns:
        # Convert to positive, log transform, then invert back
        # Use abs() to handle the negative values
        abs_conc = features_numeric['loan_concentration'].abs()
        features_numeric['loan_concentration'] = -np.log1p(abs_conc)
        print(f"  Applied log transform to loan_concentration")

    for col in skewed_features:
        if col in features_numeric.columns:
            # Handle negative values (like deflation) by shifting
            min_val = features_numeric[col].min()
            if min_val < 0:
                features_numeric[col] = np.log1p(features_numeric[col] - min_val)
            else:
                features_numeric[col] = np.log1p(features_numeric[col])
            print(f"  Applied log transform to {col}")
    
    # Scale features (in-place where possible)
    # Use MinMaxScaler to force all features to strict [0, 1] range
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features_numeric),
        index=features_numeric.index,
        columns=features_numeric.columns
    )
    del features_filtered, features_numeric  # Free memory
    gc.collect()
    
    # --- PILLAR 1: ECONOMIC RISK (from WEO) ---
    print("\n--- Pillar 1: Economic Risk ---")
    
    if len(weo_cols) > 0:
        weo_data = features_scaled[weo_cols]
        
        # PCA to extract main economic factors
        n_components = min(5, len(weo_cols) - 1, len(good_countries) - 1)
        if n_components > 0:
            pca_econ = PCA(n_components=n_components)
            econ_components = pca_econ.fit_transform(weo_data)
            
            print(f"  PCA variance explained: {pca_econ.explained_variance_ratio_.sum():.1%}")
            
            # First component typically captures overall economic strength
            economic_score_raw = econ_components[:, 0]
            
            # Store loadings
            econ_loadings = dict(zip(weo_cols, pca_econ.components_[0]))
        else:
            economic_score_raw = np.zeros(len(good_countries))
            econ_loadings = {}
    else:
        economic_score_raw = np.zeros(len(good_countries))
        econ_loadings = {}
    
    # --- PILLAR 2: INDUSTRY RISK (from FSIC) ---
    print("\n--- Pillar 2: Industry Risk ---")
    
    if len(fsic_cols) > 0:
        fsic_data = features_scaled[fsic_cols]
        
        n_components = min(5, len(fsic_cols) - 1, len(good_countries) - 1)
        if n_components > 0:
            pca_industry = PCA(n_components=n_components)
            industry_components = pca_industry.fit_transform(fsic_data)
            
            print(f"  PCA variance explained: {pca_industry.explained_variance_ratio_.sum():.1%}")
            
            industry_score_raw = industry_components[:, 0]
            
            # Store loadings
            ind_loadings = dict(zip(fsic_cols, pca_industry.components_[0]))
        else:
            industry_score_raw = np.zeros(len(good_countries))
            ind_loadings = {}
    else:
        industry_score_raw = np.zeros(len(good_countries))
        ind_loadings = {}
    
    # --- COMBINED PCA (All Features Together) ---
    print("\n--- Combined PCA (Economic + Industry) ---")
    
    all_cols = weo_cols + fsic_cols
    if len(all_cols) > 0:
        combined_data = features_scaled[all_cols]
        
        n_components_combined = min(5, len(all_cols) - 1, len(good_countries) - 1)
        if n_components_combined > 0:
            pca_combined = PCA(n_components=n_components_combined)
            combined_components = pca_combined.fit_transform(combined_data)
            
            print(f"  PCA variance explained: {pca_combined.explained_variance_ratio_.sum():.1%}")
            print(f"  PC1 variance: {pca_combined.explained_variance_ratio_[0]:.1%}")
            
            combined_score_raw = combined_components[:, 0]
            
            # Compare combined vs separate approach
            separate_avg = 0.5 * economic_score_raw + 0.5 * industry_score_raw
            correlation = np.corrcoef(combined_score_raw, separate_avg)[0, 1]
            print(f"  Correlation (Combined vs Separate Pillars): {correlation:.3f}")
            
            if abs(correlation) > 0.8:
                print("  [INFO] High correlation - Combined & Separate approaches are consistent")
            else:
                print("  [NOTE] Lower correlation - Combined PCA captures different patterns")
        else:
            combined_score_raw = np.zeros(len(good_countries))
    else:
        combined_score_raw = np.zeros(len(good_countries))
    
    # --- ANCHOR ADJUSTMENT (Crisis-Frequency Based) ---
    # REVERTED: Crisis Frequency Anchor caused model inversion (Advanced Economies have more GFC crises).
    # Using GDP per Capita (Development) as the robust anchor for structural safety.
    # High GDP -> High Pillar Score (Safe)
    print("\n--- PCA Direction Anchor (GDP/Development Based) ---")
    
    if anchor_series is not None:
        anchor_aligned_gdp = anchor_series.reindex(good_countries)
        anchor_filled = anchor_aligned_gdp.fillna(anchor_aligned_gdp.median())
        # Use log GDP to reduce outlier impact
        anchor_log = np.log10(anchor_filled.clip(lower=100))
        anchor_sc = (anchor_log - anchor_log.mean()) / anchor_log.std()
        
        # Economic Pillar
        corr_econ = np.corrcoef(anchor_sc, economic_score_raw)[0,1]
        print(f"  GDP correlation with Economic PC1: {corr_econ:.2f}")
        if corr_econ < 0:
            economic_score_raw = -economic_score_raw
            print("  Flipped economic score (High GDP = High Score/Safe)")
            
        # Industry Pillar
        corr_ind = np.corrcoef(anchor_sc, industry_score_raw)[0,1]
        print(f"  GDP correlation with Industry PC1: {corr_ind:.2f}")
        if corr_ind < 0:
            industry_score_raw = -industry_score_raw
            print("  Flipped industry score (High GDP = High Score/Safe)")
            
        # Combined Pillar
        corr_comb = np.corrcoef(anchor_sc, combined_score_raw)[0,1]
        print(f"  GDP correlation with Combined PC1: {corr_comb:.2f}")
        if corr_comb < 0:
            combined_score_raw = -combined_score_raw
            print("  Flipped combined score (High GDP = High Score/Safe)")
    
    # Pure pillar-based scoring — no anchor weight
    anchor_weight = 0.0
    anchor_scaled = np.zeros(len(good_countries))
    
    # --- COMBINE INTO FINAL SCORE ---
    print("\n--- Computing Final Risk Scores ---")
    
    # Get anchor values as numpy array (handle both Series and ndarray)
    anchor_values = anchor_scaled.values if hasattr(anchor_scaled, 'values') else anchor_scaled
    
    # Combine: pillars are weighted equally (anchor weight is 0)
    # Higher combined = better/safer
    combined_raw = (
        anchor_weight * anchor_values +  # GDP per capita contribution (currently 0)
        (1 - anchor_weight) * 0.5 * economic_score_raw +
        (1 - anchor_weight) * 0.5 * industry_score_raw
    )
    
    # Higher combined_raw = better = lower risk number
    percentiles = pd.Series(combined_raw, index=good_countries).rank(pct=True)
    
    # Map percentiles to 1-10: top percentile = 1 (lowest risk), bottom = 10
    score_raw = 1 + 9 * (1 - percentiles)
    
    # --- APPLY CONFIDENCE WEIGHTING ---
    # Countries with low confidence (highly imputed) regress toward MEDIAN risk (5.5)
    # This prevents countries with no data from getting extreme good/bad scores
    median_risk = 5.5
    
    # Align confidence scores
    confidence_aligned = imputation_confidence.reindex(good_countries).fillna(0.0)
    
    # Relax penalty: Use square root of confidence
    # If confidence is 43% -> Weight is sqrt(0.43) = 65% KNN, 35% Median
    # This trusting KNN more because we verified valid donors exist (FRA, CAN, AUS)
    weight = np.sqrt(confidence_aligned)
    
    # Weighted average: weight * calculated_score + (1-weight) * median_risk
    risk_scores = (weight * score_raw + (1 - weight) * median_risk)
    
    # --- PILLAR-SPECIFIC COVERAGE & CONFIDENCE FLOOR ---
    # Track coverage per pillar (how many real vs imputed values)
    print("\n--- Confidence-Based Risk Caps ---")
    
    # Calculate pillar-specific coverage using original missing mask
    economic_coverage = pd.Series(index=good_countries, dtype=float)
    industry_coverage = pd.Series(index=good_countries, dtype=float)
    
    available_weo_cols = [c for c in weo_cols if c in original_missing_mask.columns]
    available_fsic_cols = [c for c in fsic_cols if c in original_missing_mask.columns]
    
    if available_weo_cols:
        economic_coverage = 1 - original_missing_mask.loc[good_countries, available_weo_cols].mean(axis=1)
    if available_fsic_cols:
        industry_coverage = 1 - original_missing_mask.loc[good_countries, available_fsic_cols].mean(axis=1)
    
    # Apply confidence-based risk FLOOR (countries with poor data can't be rated "Very Low Risk")
    # Coverage < 50%: Risk score floored at 6.0 (can't be better than "Moderate Risk")
    # Coverage 50-70%: Risk score floored at 4.0 (can't be "Very Low Risk")
    risk_floor = pd.Series(1.0, index=good_countries)  # Default: no floor
    risk_floor[confidence_aligned < 0.50] = 6.0  # Very low confidence -> Moderate Risk minimum
    risk_floor[(confidence_aligned >= 0.50) & (confidence_aligned < 0.70)] = 4.0  # Low-Medium -> Low Risk minimum
    
    # Also apply floor if EITHER pillar has very poor coverage (< 30%)
    low_econ_coverage = economic_coverage < 0.30
    low_ind_coverage = industry_coverage < 0.30
    risk_floor[low_econ_coverage | low_ind_coverage] = np.maximum(
        risk_floor[low_econ_coverage | low_ind_coverage], 5.0
    )
    
    # Apply floor: risk score can't go below the floor
    # Track the score BEFORE floor application for the flag
    score_before_floor = risk_scores.copy()
    risk_scores = np.maximum(risk_scores, risk_floor)
    risk_scores = risk_scores.round(1)
    
    # risk_floor_applied is True only when the floor actually raised the score
    # Round both to same precision before comparing to avoid false positives
    floor_was_applied = risk_scores > score_before_floor.round(1)
    
    # Count how many countries were affected by the floor
    floor_applied = floor_was_applied.sum()
    print(f"  Risk floor applied to {floor_applied} countries")
    print(f"  Coverage thresholds: <50% -> floor 6.0, 50-70% -> floor 4.0")
    
    # Build results DataFrame
    # Get anchor values as numpy array (handle both Series and ndarray)
    anchor_for_df = anchor_scaled.values if hasattr(anchor_scaled, 'values') else anchor_scaled
    
    results = pd.DataFrame({
        'country_code': good_countries,
        'risk_score': risk_scores.values,
        'economic_pillar': pd.Series(economic_score_raw, index=good_countries).rank(pct=True).values * 10,
        'industry_pillar': pd.Series(industry_score_raw, index=good_countries).rank(pct=True).values * 10,
        'combined_pillar': pd.Series(combined_score_raw, index=good_countries).rank(pct=True).values * 10,
        'development_level': anchor_for_df if anchor_series is not None else 0,
        'data_coverage': data_coverage.loc[good_countries].values,
        'economic_coverage': economic_coverage.values,
        'industry_coverage': industry_coverage.values,
        'risk_floor_applied': floor_was_applied.values
    })

    
    # Add risk category
    def score_to_category(score):
        if score <= 2:
            return "1-2: Very Low Risk"
        elif score <= 4:
            return "3-4: Low Risk"
        elif score <= 6:
            return "5-6: Moderate Risk"
        elif score <= 8:
            return "7-8: High Risk"
        else:
            return "9-10: Very High Risk"
    
    results['risk_category'] = results['risk_score'].apply(score_to_category)
    
    # Add country names
    results['country_name'] = results['country_code'].map(country_names).fillna('')
    
    return results.sort_values('risk_score'), {'economic_loadings': econ_loadings, 'industry_loadings': ind_loadings}


def build_two_pillar_model(
    features_df,
    anchor_series,
    country_names,
    return_pipeline=False,
):
    """Fit the persisted pillar pipeline and score the training snapshot."""
    pipeline = PillarInferencePipeline().fit(features_df, anchor_series)
    results = pipeline.transform(features_df, country_names)
    if return_pipeline:
        return results, pipeline.loadings(), pipeline
    return results, pipeline.loadings()


GOVERNANCE_BIAS_CONTROLS = [
    'voice_accountability', 'political_stability', 'govt_effectiveness',
    'regulatory_quality', 'rule_of_law', 'control_corruption',
]


def _partial_correlation(x, y, controls):
    """Correlation of x and y after removing a linear fit on the controls."""
    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(controls), axis=1)
    if mask.sum() < len(controls[0]) + 3:
        return np.nan
    design = np.column_stack([np.ones(mask.sum()), controls[mask]])
    x_res = x[mask] - design @ np.linalg.lstsq(design, x[mask], rcond=None)[0]
    y_res = y[mask] - design @ np.linalg.lstsq(design, y[mask], rcond=None)[0]
    if x_res.std() == 0 or y_res.std() == 0:
        return np.nan
    return float(np.corrcoef(x_res, y_res)[0, 1])


def validate_model(results_df, features_df=None):
    """
    Validate model produces logically sound results.
    """
    print("\n" + "=" * 70)
    print("MODEL VALIDATION")
    print("=" * 70)

    # Data Quality Validation Checks ensuring model isn't producing garbage.
    # Bias-gate policy (approved 2026-07-10, docs/GOVERNANCE.md section 2):
    # the gate measures AVAILABILITY bias specifically, so it uses the
    # pre-policy score (critical-field penalty and crisis uplift removed;
    # both are disclosed adjustments that correlate with coverage by design)
    # and, when the feature matrix is provided, the PARTIAL correlation with
    # coverage after controlling for development level (log GDP per capita
    # and governance scores). Sparse-data countries genuinely have weaker
    # fundamentals; that gradient belongs in the score, availability bias
    # does not. Raw correlations are reported as informational.
    coverage_corr = results_df['data_coverage'].corr(results_df['risk_score'])
    base_score = results_df['risk_score'].copy()
    for policy_column in ('critical_penalty', 'crisis_uplift'):
        if policy_column in results_df.columns:
            base_score = base_score - results_df[policy_column].fillna(0)
    base_corr = results_df['data_coverage'].corr(base_score)
    print(f"\n  Data coverage correlation with final score (incl. policy penalties): {coverage_corr:.3f}")
    print(f"  Data coverage correlation with pre-penalty score: {base_corr:.3f}")

    gate_corr = base_corr
    gate_label = 'pre-penalty correlation with coverage'
    if features_df is not None and 'gdp_per_capita' in features_df.columns:
        merged = results_df[['country_code', 'data_coverage']].copy()
        merged['base_score'] = base_score.values
        control_columns = ['gdp_per_capita'] + [
            column for column in GOVERNANCE_BIAS_CONTROLS
            if column in features_df.columns
        ]
        merged = merged.merge(
            features_df[['country_code'] + control_columns],
            on='country_code', how='left',
        )
        controls = np.column_stack([
            np.log10(pd.to_numeric(merged['gdp_per_capita'], errors='coerce')
                     .clip(lower=100)
                     .fillna(merged['gdp_per_capita'].median())),
        ] + [
            pd.to_numeric(merged[column], errors='coerce')
            .fillna(pd.to_numeric(merged[column], errors='coerce').median())
            for column in control_columns[1:]
        ])
        partial = _partial_correlation(
            merged['data_coverage'].to_numpy(dtype=float),
            merged['base_score'].to_numpy(dtype=float),
            controls,
        )
        if np.isfinite(partial):
            gate_corr = partial
            gate_label = 'partial correlation with coverage (controls: development, governance)'
        print(f"  Partial correlation controlling for development/governance (bias gate): {partial:.3f}")
    print(f"  (Low gate correlation is GOOD - scores are not biased by data availability)")

    validation_checks = [
        ('Score Range', lambda df: df['risk_score'].between(1, 10).all(), 'All scores 1-10'),
        ('Score Distribution', lambda df: 1.5 < df['risk_score'].std() < 4.0, 'Std dev reasonable (1.5-4.0)'),
        ('Data Confidence', lambda df: abs(gate_corr) < 0.4, f'{gate_label}: {abs(gate_corr):.2f} < 0.4'),
    ]
    
    passed = 0
    failed = 0
    
    print("  Running data quality checks...")
    for name, check_func, description in validation_checks:
        try:
            if check_func(results_df):
                 print(f"  [PASS] {name}: {description}")
                 passed += 1
            else:
                 print(f"  [FAIL] {name}: {description}")
                 failed += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            failed += 1

    # Informational Geographical checks (no longer pass/fail criteria)
    print("\n  Informational: Regional consistency checks (non-blocking)")
    geo_pairs = [
        ('USA', 'NGA'), ('JPN', 'NGA'), ('DEU', 'ARG'), 
        ('CHE', 'VEN'), ('GBR', 'PAK'), ('CAN', 'TUR')
    ]
    for c1, c2 in geo_pairs:
         r1 = results_df[results_df['country_code'] == c1]
         r2 = results_df[results_df['country_code'] == c2]
         if len(r1) > 0 and len(r2) > 0:
             s1, s2 = r1.iloc[0]['risk_score'], r2.iloc[0]['risk_score']
             rel = "<" if s1 < s2 else ">"
             print(f"         {c1} ({s1:.1f}) {rel} {c2} ({s2:.1f})")
    
    print(f"\nValidation: {passed} passed, {failed} failed")
    
    return passed, failed


class BankingRiskModel:
    """
    Trained banking risk model that can be saved and loaded.
    """
    
    def __init__(self):
        self.country_scores = pd.DataFrame()
        self.trained = False
        self.training_date = None
        self.countries_trained = 0
        self.feature_values = None  # Raw feature values per country for comparison
        self.pca_info = {}  # PCA loadings for explainability
        self.pillar_pipeline = None
        
    def train(
        self,
        fsic_df,
        weo_df,
        mfs_df,
        as_of_date=None,
        retrain_classifier=True,
        extra_features=None,
    ):
        """
        Train the hybrid risk model.
        
        Steps:
        1. Run feature engineering pipeline (Clean, Extract, Merge)
        2. Train/Load Supervision Crisis Classifier
        3. Build Unsupervised Two-Pillar Model (PCA)
        4. Combine into final Hybrid Risk Score
        """
        print("\n" + "="*70)
        print("TRAINING HYBRID BANKING RISK MODEL")
        print("="*70)
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        cutoff = pd.Timestamp(
            as_of_date or f"{pd.Timestamp.today().year - 1}-12-31"
        )

        def filter_to_cutoff(df):
            if df is None or len(df) == 0 or 'period' not in df.columns:
                return df
            periods = pd.to_datetime(df['period'], errors='coerce')
            return df.loc[periods.notna() & (periods <= cutoff)].copy()

        fsic_df = filter_to_cutoff(fsic_df)
        weo_df = filter_to_cutoff(weo_df)
        mfs_df = filter_to_cutoff(mfs_df)

        print(f"  Snapshot cutoff: {cutoff.date()}")
        print(f"  Input data: FSIC ({len(fsic_df):,} records), WEO ({len(weo_df):,} records), MFS ({len(mfs_df):,} records)")
        
        # --- 1. FEATURE ENGINEERING ---
        print("\n" + "-"*70)
        print("[Step 1/4] FEATURE ENGINEERING")
        print("-"*70)
        engineer = CrisisFeatureEngineer()
        
        classifier = CrisisClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.1,
            use_smote=True,
            ensemble=True
        )
        # Extract features from each dataset
        print("  [1a] Extracting core IMF features (WEO, FSIC, MFS)...")
        weo_features = engineer.extract_weo_features(
            weo_df,
            as_of_date=cutoff,
            include_estimates=True,
            include_projections=False,
        )
        fsic_features = engineer.extract_fsic_features(fsic_df, as_of_date=cutoff)
        credit_gap = engineer.compute_credit_to_gdp_relative(mfs_df, weo_df, as_of_date=cutoff)
        sovereign_nexus = engineer.compute_sovereign_bank_nexus(mfs_df, weo_df, as_of_date=cutoff)

        # Compute literature-identified gap features (2026-02)
        print("\n  [1a+] Computing literature gap features...")
        lit_gap_features = engineer.compute_literature_gap_features(
            weo_df=weo_df, mfs_df=mfs_df, fsic_df=fsic_df, weo_features=weo_features,
            as_of_date=cutoff,
        )
        
        # Load WGI governance features
        print("\n  [1b] Loading WGI governance indicators...")
        try:
            wgi_loader = WGILoader()
            wgi_features = wgi_loader.get_latest_scores(as_of_date=cutoff)
            
            # Print WGI feature summary
            if wgi_features is not None and len(wgi_features) > 0:
                print(f"\n  WGI GOVERNANCE INDICATORS LOADED:")
                wgi_cols = [c for c in wgi_features.columns if c != 'country_code']
                for col in wgi_cols:
                    non_null = wgi_features[col].notna().sum()
                    mean_val = wgi_features[col].mean()
                    print(f"    - {col}: {non_null} countries, mean={mean_val:.1f}")
        except Exception as e:
            print(f"  WARNING: Could not load WGI data: {e}")
            wgi_features = None
        
        # Load FSIBSIS features (balance sheet funding stability indicators)
        print("\n  [1c] Loading FSIBSIS balance sheet indicators...")
        try:
            from src.data_loader import load_fsibsis_features
            fsibsis_features = load_fsibsis_features(as_of_date=cutoff)
            if fsibsis_features is not None and len(fsibsis_features) > 0:
                fsibsis_cols = [c for c in fsibsis_features.columns if c != 'country_code' and not c.endswith('_year')]
                print(f"        Countries: {len(fsibsis_features)}")
                print(f"        Features:  {len(fsibsis_cols)} indicators")
                for col in fsibsis_cols[:6]:  # Show first 6
                    non_null = fsibsis_features[col].notna().sum()
                    print(f"          - {col}: {non_null} countries")
                if len(fsibsis_cols) > 6:
                    print(f"          ... and {len(fsibsis_cols) - 6} more")
        except Exception as e:
            print(f"        WARNING: Could not load FSIBSIS data: {e}")
            fsibsis_features = None
        
        # Merge all features (using keyword args for clarity)
        features = engineer.merge_features(
            weo_features=weo_features, 
            fsic_features=fsic_features, 
            credit_gap=credit_gap, 
            sovereign_nexus=sovereign_nexus,
            wgi_features=wgi_features, 
            fsibsis_features=fsibsis_features,
            lit_gap_features=lit_gap_features
        )
        
        # Free intermediate DataFrames
        del weo_features, fsic_features, credit_gap
        gc.collect()
        
        if len(features) == 0:
            raise ValueError("Feature engineering failed: No data produced")

        if (
            'fiscal_space' not in features.columns
            and 'fiscal_balance_gdp' in features.columns
            and 'govt_debt_gdp' in features.columns
        ):
            features['fiscal_space'] = (
                features['fiscal_balance_gdp'] - features['govt_debt_gdp'] / 100
            )

        lag_feature_cols = [
            'gdp_growth_3yr_avg',
            'inflation_acceleration',
            'debt_buildup_3yr',
            'ca_deterioration_3yr',
        ]
        missing_lag_cols = [
            column for column in lag_feature_cols
            if column not in features.columns
        ]
        if missing_lag_cols and weo_df is not None and len(weo_df) > 0:
            latest_weo_year = int(pd.to_datetime(weo_df['period']).dt.year.max())
            latest_lags = _compute_lag_features(
                weo_df,
                latest_weo_year,
                features['country_code'].tolist(),
            )
            if len(latest_lags) > 0:
                features = features.merge(
                    latest_lags,
                    on='country_code',
                    how='left',
                )
        
        # Structural crisis-history feature: recently resolved systemic
        # crises leave recapitalized, artificially clean balance sheets, so
        # the pillar needs memory of the episode itself (challenger review
        # concern (c), 2026-07-10). Capped so "never in the database" and
        # "a generation ago" are equivalent.
        from src.crisis_labels import CrisisLabels as _CrisisLabels
        crisis_history = _CrisisLabels()
        recency_cap = 25.0
        cutoff_year = cutoff.year

        def _years_since_crisis(country_code):
            periods = crisis_history.crises.get(country_code)
            if not periods:
                return recency_cap
            last_end = max(end for _, end in periods)
            return float(min(recency_cap, max(0, cutoff_year - last_end)))

        features['years_since_banking_crisis'] = (
            features['country_code'].map(_years_since_crisis)
        )

        # Staged liquidity challenger features (2026-07-11): optional external /
        # government liquidity columns merged by country. Left-merged so the
        # country universe is unchanged; production runs pass nothing and are
        # unaffected. The pillar pipeline only consumes columns declared in
        # ECONOMIC_FEATURES/INDUSTRY_FEATURES + FEATURE_RISK_DIRECTIONS.
        if extra_features is not None and len(extra_features) > 0:
            extra = extra_features.copy()
            extra['country_code'] = extra['country_code'].astype(str).str.upper()
            extra = extra.drop_duplicates('country_code')
            overlap = [
                c for c in extra.columns
                if c != 'country_code' and c in features.columns
            ]
            if overlap:
                extra = extra.drop(columns=overlap)
            features = features.merge(extra, on='country_code', how='left')
            print(
                f"  Merged {len(extra.columns) - 1} staged liquidity feature(s) "
                f"for {extra['country_code'].nunique()} countries"
            )

        # Print organized feature summary matching README structure
        print("\n  " + "-"*50)
        print("  FEATURE MATRIX SUMMARY")
        print("  " + "-"*50)
        
        # Define feature categories
        economic_features = [
            'gdp_per_capita', 'gdp_growth', 'inflation', 'unemployment', 'nominal_gdp',
            'current_account_gdp', 'govt_debt_gdp', 'fiscal_balance_gdp',
            'external_debt_gdp', 'debt_service_gdp', 'credit_to_gdp', 'credit_to_gdp_relative',
            'inflation_differential_3yr', 'interest_cost_gdp', 'interest_cost_trend_3yr',
            'credit_growth_3yr', 'm2_to_reserves', 'ca_deficit_severity', 'tot_deterioration_3yr',
            'voice_accountability', 'political_stability', 'govt_effectiveness'
        ]
        industry_features = [
            'capital_adequacy', 'tier1_capital', 'capital_quality', 'npl_ratio', 'npl_provisions',
            'roe', 'roa', 'liquid_assets_st_liab', 'liquid_assets_total',
            'customer_deposits_loans', 'fx_loan_exposure', 'loan_concentration',
            'real_estate_loans', 'sovereign_exposure_ratio',
            'real_estate_credit_growth_3yr', 'years_since_banking_crisis',
            'regulatory_quality', 'rule_of_law', 'control_corruption'
        ]
        fsibsis_features = [
            'net_interest_margin', 'interbank_funding_ratio', 'income_diversification',
            'securities_to_assets', 'specific_provisions_ratio', 'large_exposure_ratio',
            'deposit_funding_ratio'
        ]
        
        feature_cols = [c for c in features.columns if c != 'country_code' and not c.endswith('_period')]
        
        # Count by category
        econ_present = [f for f in economic_features if f in feature_cols]
        ind_present = [f for f in industry_features if f in feature_cols]
        fsib_present = [f for f in fsibsis_features if f in feature_cols]
        
        print(f"\n  Countries: {len(features)}")
        print(f"  Total Features: {len(feature_cols)}")
        print(f"\n  Economic Pillar ({len(econ_present)}/{len(economic_features)} features):")
        for f in econ_present:
            coverage = features[f].notna().sum()
            print(f"    - {f}: {coverage} countries")
        missing_econ = set(economic_features) - set(econ_present)
        if missing_econ:
            print(f"    [MISSING: {', '.join(missing_econ)}]")
        
        print(f"\n  Industry Pillar ({len(ind_present)}/{len(industry_features)} features):")
        for f in ind_present:
            coverage = features[f].notna().sum()
            print(f"    - {f}: {coverage} countries")
        missing_ind = set(industry_features) - set(ind_present)
        if missing_ind:
            print(f"    [MISSING: {', '.join(missing_ind)}]")
        
        print(f"\n  FSIBSIS Computed ({len(fsib_present)}/{len(fsibsis_features)} features):")
        for f in fsib_present:
            coverage = features[f].notna().sum()
            print(f"    - {f}: {coverage} countries")
        print("  " + "-"*50)
        
        # Save features to parquet (including WGI) for dashboard consistency
        from src.config import CACHE_DIR
        features_path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
        features.to_parquet(features_path, index=False)
        print(f"\n  Saved features to: {features_path}")

        # Persist the per-country data-heuristic flags collected during
        # feature engineering so silent adjustments are auditable.
        try:
            import json as _json
            from collections import Counter as _Counter
            flags = list(getattr(engineer, 'heuristic_flags', []))
            heuristics_payload = {
                'schema_version': 1,
                'flag_count': len(flags),
                'counts_by_heuristic': dict(
                    _Counter(flag['heuristic'] for flag in flags)
                ),
                'flags': flags,
            }
            heuristics_path = os.path.join(
                os.path.dirname(CACHE_DIR), 'artifacts', 'feature_heuristics.json'
            )
            os.makedirs(os.path.dirname(heuristics_path), exist_ok=True)
            with open(heuristics_path, 'w', encoding='utf-8') as sink:
                _json.dump(heuristics_payload, sink, indent=2, sort_keys=True)
            print(
                f"  Saved {len(flags)} data-heuristic flags to: {heuristics_path}"
            )
        except Exception as exc:
            print(f"  WARNING: could not write feature heuristics: {exc}")
        
        # --- 1b. PLOT CORRELATION MATRIX ---
        try:
             import seaborn as sns
             import matplotlib.pyplot as plt
             
             # Select pillars cols
             cols_to_plot = econ_present + ind_present + fsib_present
             cols_available = [c for c in cols_to_plot if c in features.columns]
             
             if len(cols_available) > 1:
                 print("\n  Generating Correlation Matrix Plot...")
                 plt.figure(figsize=(16, 14))
                 corr = features[cols_available].corr()
                 
                 # Mask upper triangle
                 mask = np.triu(np.ones_like(corr, dtype=bool))
                 
                 sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                             square=True, linewidths=.5, cbar_kws={"shrink": .5},
                             annot=False) # Annot extraction is too crowded
                 
                 plt.title('Feature Correlation Matrix (Economic & Industry Pillars)', fontsize=16)
                 plt.tight_layout()
                 
                 corr_path = os.path.join(CACHE_DIR, 'eda', 'feature_correlations.png')
                 os.makedirs(os.path.dirname(corr_path), exist_ok=True)
                 plt.savefig(corr_path, dpi=150)
                 plt.close()
                 print(f"  Saved Correlation Plot: {corr_path}")
        except Exception as e:
            print(f"  WARNING: Could not save correlation plot: {e}")

        
        # --- 2. SUPERVISED CRISIS CLASSIFIER ---
        print("\n" + "-"*70)
        print("[Step 2/4] CRISIS CLASSIFIER")
        print("-"*70)
        if retrain_classifier:
            classifier, metrics = train_crisis_model(
                weo_df=weo_df, fsic_df=fsic_df, as_of_date=cutoff,
            )
        else:
            try:
                classifier = CrisisClassifier.load()
                metrics = {"cached_classifier": True}
                print("  Loaded cached crisis classifier for snapshot scoring.")
            except Exception as e:
                print(
                    "  Cached crisis classifier unavailable; retraining "
                    f"for this snapshot: {e}"
                )
                classifier, metrics = train_crisis_model(
                    weo_df=weo_df,
                    fsic_df=fsic_df,
                    as_of_date=cutoff,
                )
        
        # Get crisis probabilities
        print("  Generating crisis probabilities...")
        feature_cols = [c for c in features.columns 
                       if c not in ['country_code', 'country_name', 'crisis_target'] 
                       and not c.endswith('_period')]
        
        # Ensure we have all necessary columns (fill missing with median)
        X = features[feature_cols].copy()
        # Cached classifiers trained before the credit_to_gdp_relative rename
        # still expect the legacy feature name.
        legacy_feature_aliases = {'credit_to_gdp_gap': 'credit_to_gdp_relative'}
        for col in classifier.feature_names_:
            if col not in X.columns:
                alias = legacy_feature_aliases.get(col)
                if alias is not None and alias in X.columns:
                    X[col] = X[alias]
                else:
                    X[col] = np.nan
        if not retrain_classifier:
            all_null_classifier_inputs = [
                col for col in classifier.feature_names_
                if X[col].isna().all()
            ]
            if all_null_classifier_inputs:
                fill_values = getattr(classifier, 'feature_fill_values_', None)
                fallback_fills = {}
                for col in all_null_classifier_inputs:
                    trained_fill = (
                        fill_values.get(col)
                        if fill_values is not None and col in fill_values
                        else np.nan
                    )
                    fallback_fills[col] = (
                        trained_fill if pd.notna(trained_fill) else 0.0
                    )
                    X[col] = fallback_fills[col]
                print(
                    "  Cached classifier fallback fills for all-null "
                    f"features: {fallback_fills}"
                )
        
        # Predict probability of crisis within 3 years
        # Note: input columns must match classifier training columns
        X_aligned = X[classifier.feature_names_]
        crisis_probs = classifier.predict_proba(X_aligned)
        
        features['crisis_prob'] = crisis_probs
        print(f"  Crisis probability: mean={crisis_probs.mean():.1%}, std={crisis_probs.std():.1%}")
        
        # --- 3. UNSUPERVISED PILLARS (Economic + Industry) ---
        print("\n" + "-"*70)
        print("[Step 3/4] PCA PILLAR CONSTRUCTION")
        print("-"*70)
        
        # Load country names mapping for display (deduplicate to avoid InvalidIndexError)
        country_names = weo_df[['country_code', 'country_name']].drop_duplicates(subset='country_code').set_index('country_code')['country_name']
        # Official SDMX pulls may not carry display names; derive them from codes.
        country_names = country_names.fillna('').astype(str).str.strip()
        blank_names = country_names == ''
        if blank_names.any():
            country_names.loc[blank_names] = [
                country_name_from_code(code) for code in country_names.index[blank_names]
            ]
        
        # Get development anchor (GDP per capita)
        anchor = identify_anchor_indicator(features, None) # None passed as features already has gdp_per_capita
        
        # Build two-pillar model (PCA-based)
        # Build two-pillar model (PCA-based)
        # We pass the engineered features directly
        (
            pillar_scores,
            pca_loadings,
            self.pillar_pipeline,
        ) = build_two_pillar_model(
            features,
            anchor,
            country_names,
            return_pipeline=True,
        )

        # Persist the imputed-feature sidecar the dashboard uses to label
        # imputed values. It must be rebuilt with every model build so it
        # always matches the current feature matrix; the legacy pillar path
        # used to own this file and left it stale after the pipeline switch.
        try:
            imputed_matrix = self.pillar_pipeline.impute(features)
            features_indexed = (
                features.set_index('country_code')
                if 'country_code' in features.columns else features
            )
            sidecar = imputed_matrix.copy()
            for column in features_indexed.columns:
                if column.endswith('_year'):
                    sidecar[column] = features_indexed[column].reindex(sidecar.index)
            sidecar.index.name = 'country_code'
            sidecar.reset_index().to_parquet(
                os.path.join(CACHE_DIR, 'imputed_features.parquet'),
                index=False,
            )
            print(f"  Saved imputed feature sidecar ({len(sidecar)} countries)")
        except Exception as exc:
            print(f"  WARNING: could not write imputed feature sidecar: {exc}")

        # --- 4. HYBRID RISK SCORE ---
        print("\n" + "-"*70)
        print("[Step 4/4] FINAL RISK SCORING")
        print("-"*70)
        
        # Merge pillar scores with crisis probabilities
        final_df = pillar_scores.merge(
            features[['country_code', 'crisis_prob']], 
            on='country_code', 
            how='left'
        )
        
        # The classifier acts as an upward-only overlay on the pillar score:
        # when the classifier's implied score (1 + 9 * probability) exceeds
        # the pillar score, 10% of that gap is added. The overlay is monotone
        # in crisis probability and can never lower a high pillar-based risk
        # score (a weak classifier signal must not de-risk a country).
        classifier_implied = 1 + 9 * final_df['crisis_prob']
        final_df['crisis_uplift'] = (
            0.1 * (classifier_implied - final_df['risk_score'])
        ).clip(lower=0)
        final_df['hybrid_risk_score'] = (
            final_df['risk_score'] + final_df['crisis_uplift']
        ).clip(1, 10)
        
        # Update risk category based on new hybrid score
        def score_to_category(score):
            if score <= 2: return "1-2: Very Low Risk"
            elif score <= 4: return "3-4: Low Risk"
            elif score <= 6: return "5-6: Moderate Risk"
            elif score <= 8: return "7-8: High Risk"
            else: return "9-10: Very High Risk"
            
        final_df['risk_score'] = final_df['hybrid_risk_score']  # Update final score
        final_df['risk_category'] = final_df['risk_score'].apply(score_to_category)
        
        self.country_scores = final_df.sort_values('risk_score')
        self.trained = True
        self.training_date = datetime.now().isoformat()
        self.countries_trained = len(self.country_scores)
        
        # Store raw feature values for dashboard comparison
        self.feature_values = features.copy()
        
        # Store PCA explanation (weights are 50% Economic + 50% Industry)
        self.pca_info = {
            'training_date': self.training_date,
            'snapshot_date': cutoff.date().isoformat(),
            'economic_weight': 0.50,
            'industry_weight': 0.50,
            'note': 'PCA reduces features to principal components. First PC captures most variance.',
            'economic_loadings': pca_loadings.get('economic_loadings', {}),
            'industry_loadings': pca_loadings.get('industry_loadings', {})
        }
        
        return self.country_scores

    def score_engineered_features(
        self,
        features,
        crisis_classifier,
        country_names=None,
    ):
        """Score a future engineered snapshot with fitted transforms only."""
        if self.pillar_pipeline is None:
            raise ValueError(
                "Persisted pillar pipeline is unavailable; retraining is "
                "required before comparable inference"
            )
        if country_names is None:
            country_names = pd.Series(dtype=object)

        classifier_input = features.copy()
        for column in crisis_classifier.feature_names_:
            if column not in classifier_input.columns:
                classifier_input[column] = np.nan
        probabilities = crisis_classifier.predict_proba(
            classifier_input[crisis_classifier.feature_names_]
        )
        probability_frame = pd.DataFrame(
            {
                "country_code": features["country_code"].values,
                "crisis_prob": probabilities,
            }
        )

        pillars = self.pillar_pipeline.transform(features, country_names)
        scored = pillars.merge(
            probability_frame,
            on="country_code",
            how="left",
            validate="one_to_one",
        )
        scored["crisis_uplift"] = (
            0.1 * ((1 + 9 * scored["crisis_prob"]) - scored["risk_score"])
        ).clip(lower=0)
        scored["hybrid_risk_score"] = (
            scored["risk_score"] + scored["crisis_uplift"]
        ).clip(1, 10)
        scored["risk_score"] = scored["hybrid_risk_score"]
        scored["risk_category"] = scored["risk_score"].apply(
            lambda score: (
                "1-2: Very Low Risk" if score <= 2 else
                "3-4: Low Risk" if score <= 4 else
                "5-6: Moderate Risk" if score <= 6 else
                "7-8: High Risk" if score <= 8 else
                "9-10: Very High Risk"
            )
        )
        return scored.sort_values("risk_score")
    
    def get_score(self, country_code):
        """Get risk score for a country."""
        if not self.trained:
            raise ValueError("Model not trained")
        
        row = self.country_scores[
            self.country_scores['country_code'] == country_code.upper()
        ]
        
        if len(row) == 0:
            return None
        
        return row.iloc[0].to_dict()
    
    def get_all_scores(self):
        """Get all country scores."""
        return self.country_scores.copy()
    
    def save(self, path=None):
        """Save model to disk."""
        path = path or MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'country_scores': self.country_scores,
                'trained': self.trained,
                'training_date': self.training_date,
                'countries_trained': self.countries_trained,
                'feature_values': self.feature_values,
                'pca_info': self.pca_info,
            }, f)

        if self.pillar_pipeline is not None:
            pipeline_path = (
                INFERENCE_PIPELINE_PATH
                if path == MODEL_PATH
                else os.path.join(
                    os.path.dirname(path),
                    f"{os.path.splitext(os.path.basename(path))[0]}"
                    "_inference_pipeline.pkl",
                )
            )
            with open(pipeline_path, "wb") as pipeline_file:
                pickle.dump(
                    {
                        "schema_version": 1,
                        "snapshot_date": self.pca_info.get("snapshot_date"),
                        "pillar_pipeline": self.pillar_pipeline,
                    },
                    pipeline_file,
                )
            print(f"Inference pipeline saved to: {pipeline_path}")
        
        print(f"\nModel saved to: {path}")
    
    @classmethod
    def load(cls, path=None):
        """Load model from disk."""
        path = path or MODEL_PATH
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        
        model = cls()
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model.country_scores = data['country_scores']
        model.trained = data['trained']
        model.training_date = data['training_date']
        model.countries_trained = data['countries_trained']
        model.feature_values = data.get('feature_values')  # New: for comparison
        model.pca_info = data.get('pca_info', {})  # New: for explainability
        pipeline_path = (
            INFERENCE_PIPELINE_PATH
            if path == MODEL_PATH
            else os.path.join(
                os.path.dirname(path),
                f"{os.path.splitext(os.path.basename(path))[0]}"
                "_inference_pipeline.pkl",
            )
        )
        if os.path.exists(pipeline_path):
            with open(pipeline_path, "rb") as pipeline_file:
                pipeline_artifact = pickle.load(pipeline_file)
            model.pillar_pipeline = pipeline_artifact.get("pillar_pipeline")
        
        return model


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("DATA-DRIVEN BANKING SYSTEM RISK MODEL")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load IMF data (FSIC, WEO, MFS)
    print("\n" + "-" * 70)
    print("DATA LOADING")
    print("-" * 70)
    
    loader = IMFDataLoader()
    
    if not loader.load_from_cache():
        print("  Loading from CSV files...")
        loader.load_all_datasets()
        loader.save_cache()
    
    fsic_df = loader._data_cache.get('FSIC', pd.DataFrame())
    weo_df = loader._data_cache.get('WEO', pd.DataFrame())
    mfs_df = loader._data_cache.get('MFS', pd.DataFrame())
    
    # Load FSIBSIS for summary (also loaded in train())
    from src.data_loader import FSIBSISLoader, WGILoader
    
    print("\n  Loading FSIBSIS balance sheet data...")
    try:
        fsibsis_loader = FSIBSISLoader()
        fsibsis_loader.load()
        fsibsis_records = len(fsibsis_loader.bank_data) if fsibsis_loader.bank_data is not None else 0
        fsibsis_countries = fsibsis_loader.bank_data['country_code'].nunique() if fsibsis_loader.bank_data is not None else 0
    except Exception as e:
        print(f"    Warning: {e}")
        fsibsis_records = 0
        fsibsis_countries = 0
    
    print("  Loading WGI governance data...")
    try:
        wgi_loader = WGILoader()
        wgi_data = wgi_loader.load()
        wgi_records = len(wgi_data) if wgi_data is not None else 0
        wgi_countries = wgi_data['country_code'].nunique() if wgi_data is not None else 0
    except Exception as e:
        print(f"    Warning: {e}")
        wgi_records = 0
        wgi_countries = 0
    
    # Print comprehensive data summary
    print("\n" + "-" * 70)
    print("DATA SOURCES SUMMARY")
    print("-" * 70)
    print(f"  {'Source':<12} {'Records':>12} {'Countries':>12} {'Description'}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*30}")
    print(f"  {'FSIC':<12} {len(fsic_df):>12,} {fsic_df['country_code'].nunique() if len(fsic_df) > 0 else 0:>12} Financial Soundness Indicators")
    print(f"  {'WEO':<12} {len(weo_df):>12,} {weo_df['country_code'].nunique() if len(weo_df) > 0 else 0:>12} World Economic Outlook")
    print(f"  {'MFS':<12} {len(mfs_df):>12,} {mfs_df['country_code'].nunique() if len(mfs_df) > 0 else 0:>12} Monetary & Financial Stats")
    print(f"  {'FSIBSIS':<12} {fsibsis_records:>12,} {fsibsis_countries:>12} Balance Sheet Data")
    print(f"  {'WGI':<12} {wgi_records:>12,} {wgi_countries:>12} World Governance Indicators")
    print(f"  {'-'*12} {'-'*12} {'-'*12}")
    total_records = len(fsic_df) + len(weo_df) + len(mfs_df) + fsibsis_records + wgi_records
    print(f"  {'TOTAL':<12} {total_records:>12,}")
    
    # Explore all indicators
    explore_all_indicators(fsic_df, weo_df, mfs_df)
    
    # Train model
    model = BankingRiskModel()
    as_of_date = os.getenv("MODEL_AS_OF_DATE")
    results = model.train(fsic_df, weo_df, mfs_df, as_of_date=as_of_date)
    
    # Validate
    validate_model(results, features_df=model.feature_values)
    
    # Save
    model.save()
    
    # Show results
    print("\n" + "=" * 70)
    print("FINAL RISK RANKINGS")
    print("=" * 70)
    
    print("\nTOP 20 LOWEST RISK:")
    print(results[['country_code', 'country_name', 'risk_score', 'risk_category']].head(20).to_string(index=False))
    
    print("\nTOP 20 HIGHEST RISK:")
    print(results[['country_code', 'country_name', 'risk_score', 'risk_category']].tail(20).to_string(index=False))
    
    # Specific countries of interest
    print("\n" + "=" * 70)
    print("KEY COUNTRIES")
    print("=" * 70)
    
    key_countries = ['USA', 'JPN', 'DEU', 'GBR', 'CHE', 'FRA', 'CAN', 'AUS',
                     'CHN', 'IND', 'BRA', 'MEX', 'IDN', 'TUR', 'ZAF',
                     'NGA', 'PAK', 'ARG', 'EGY', 'VEN', 'KEN']
    
    key_results = results[results['country_code'].isin(key_countries)].copy()
    key_results = key_results.sort_values('risk_score')
    
    # Calculate separate pillars average and variance from combined
    key_results['separate_avg'] = (key_results['economic_pillar'] + key_results['industry_pillar']) / 2
    key_results['variance'] = (key_results['combined_pillar'] - key_results['separate_avg']).round(2)
    
    print(key_results[['country_code', 'country_name', 'risk_score', 
                       'economic_pillar', 'industry_pillar', 'combined_pillar', 'variance']].to_string(index=False))
    
    # Summary statistics
    print(f"\n  Variance Statistics:")
    print(f"    Mean Variance:     {key_results['variance'].mean():.2f}")
    print(f"    Std Dev Variance:  {key_results['variance'].std():.2f}")
    print(f"    Max Deviation:     {key_results['variance'].abs().max():.2f}")
    
    # Identify countries where combined PCA differs significantly
    significant_diff = key_results[key_results['variance'].abs() > 1.5]
    if len(significant_diff) > 0:
        print(f"\n  Countries with significant variance (|diff| > 1.5):")
        for _, row in significant_diff.iterrows():
            direction = "higher" if row['variance'] > 0 else "lower"
            print(f"    {row['country_code']}: Combined {direction} by {abs(row['variance']):.1f} points")
    
    return model


if __name__ == "__main__":
    main()
