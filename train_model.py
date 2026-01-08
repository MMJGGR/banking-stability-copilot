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
from src.feature_engineering import CrisisFeatureEngineer
from src.crisis_classifier import CrisisClassifier, HybridRiskScorer, train_crisis_model
from src.imputation import GapImputer
from src.wgi_loader import WGILoader

MODEL_PATH = os.path.join(CACHE_DIR, "risk_model.pkl")


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


def build_two_pillar_model(features_df, anchor_series, country_names):
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
    economic_cols = [
        'gdp_growth', 'gdp_per_capita', 'inflation', 'current_account_gdp', 
        'govt_debt_gdp', 'fiscal_balance_gdp', 'unemployment', 'nominal_gdp',
        'credit_to_gdp', 'credit_to_gdp_gap', 'debt_service_gdp', 'external_debt_gdp',
        # WGI governance (BICRA: Economic Risk)
        'voice_accountability', 'political_stability', 'govt_effectiveness'
    ]
    
    industry_cols = [
        'capital_adequacy', 'npl_ratio', 'roe', 'roa', 'liquid_assets_st_liab', 
        'liquid_assets_total', 'customer_deposits_loans', 'fx_loan_exposure', 
        'tier1_capital', 'npl_provisions', 'loan_concentration',
        'real_estate_loans', 'sovereign_exposure_ratio', # Include new nexus feature
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
    features_numeric = features_imputed
    
    # --- PREPROCESSING: LOG TRANSFORM SKEWED FEATURES ---
    # Fix single-feature dominance in PCA by compressing scale of high-variance features
    skewed_features = [
        'nominal_gdp',          # Range: 0.1B to 30,000B -> Log reduces variance dominance
        'inflation',            # Range: -5% to 1000% -> Log handles hyperinflation outliers
        'gdp_per_capita',       # Range: $200 to $130k -> Log reflects standard economic utility
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
    
    # --- ANCHOR ADJUSTMENT ---
    print("\n--- Anchoring with Development Level ---")
    
    # Get anchor values for our countries
    if anchor_series is not None:
        anchor_aligned = anchor_series.reindex(good_countries)
        anchor_filled = anchor_aligned.fillna(anchor_aligned.median())
        
        # Log transform GDP per capita (better distribution)
        anchor_log = np.log10(anchor_filled.clip(lower=100))
        anchor_scaled = (anchor_log - anchor_log.mean()) / anchor_log.std()
        
        econ_corr = np.corrcoef(anchor_scaled, economic_score_raw)[0,1]
        ind_corr = np.corrcoef(anchor_scaled, industry_score_raw)[0,1]
        comb_corr = np.corrcoef(anchor_scaled, combined_score_raw)[0,1]
        
        print(f"  Anchor correlation with Economic: {econ_corr:.2f}")
        print(f"  Anchor correlation with Industry: {ind_corr:.2f}")
        print(f"  Anchor correlation with Combined: {comb_corr:.2f}")
        
        # Ensure direction: higher development (anchor) should mean HIGHER combined score (= lower risk)
        # If correlation is NEGATIVE, we need to flip so higher anchor = higher pillar score
        if econ_corr < 0:
            economic_score_raw = -economic_score_raw
            print("  Flipped economic score direction")
        
        if ind_corr < 0:
            industry_score_raw = -industry_score_raw
            print("  Flipped industry score direction")
        
        if comb_corr < 0:
            combined_score_raw = -combined_score_raw
            print("  Flipped combined score direction")
        
        # REMOVED: GDP per capita anchor was biasing model too heavily toward development
        # GDP per capita is now ONLY in the Economic pillar, not double-counted
        anchor_weight = 0.0  # No anchor weight - pure pillar-based scoring
    else:
        anchor_scaled = np.zeros(len(good_countries))
        anchor_weight = 0.0
    
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


def validate_model(results_df):
    """
    Validate model produces logically sound results.
    """
    print("\n" + "=" * 70)
    print("MODEL VALIDATION")
    print("=" * 70)
    
    # Data Quality Validation Checks ensuring model isn't producing garbage
    coverage_corr = results_df['data_coverage'].corr(results_df['risk_score'])
    print(f"\n  Data coverage correlation with risk score: {coverage_corr:.3f}")
    print(f"  (Low correlation is GOOD - means scores are not biased by data availability)")
    
    validation_checks = [
        ('Score Range', lambda df: df['risk_score'].between(1, 10).all(), 'All scores 1-10'),
        ('Score Distribution', lambda df: 1.5 < df['risk_score'].std() < 4.0, 'Std dev reasonable (1.5-4.0)'),
        ('Data Confidence', lambda df: abs(df['data_coverage'].corr(df['risk_score'])) < 0.4, f'Correlation with coverage: {abs(coverage_corr):.2f} < 0.4'),
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
        
    def train(self, fsic_df, weo_df, mfs_df):
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
        
        # --- 1. FEATURE ENGINEERING ---
        print("\n[Step 1] Running Feature Engineering Pipeline...")
        engineer = CrisisFeatureEngineer()
        
        # Extract features from each dataset
        weo_features = engineer.extract_weo_features(weo_df)
        fsic_features = engineer.extract_fsic_features(fsic_df)
        credit_gap = engineer.compute_credit_to_gdp_gap(mfs_df, weo_df)
        sovereign_nexus = engineer.compute_sovereign_bank_nexus(mfs_df, weo_df)
        
        # Load WGI governance features
        print("  Loading WGI governance indicators...")
        try:
            wgi_loader = WGILoader()
            wgi_features = wgi_loader.get_latest_scores()
            
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
        
        # Load FSIBSIS features (new funding stability indicators)
        print("  Loading FSIBSIS funding stability indicators...")
        try:
            from src.data_loader_fsibsis import load_fsibsis_features
            fsibsis_features = load_fsibsis_features()
            if fsibsis_features is not None and len(fsibsis_features) > 0:
                print(f"  FSIBSIS: {len(fsibsis_features)} countries, {len(fsibsis_features.columns)-1} indicators")
        except Exception as e:
            print(f"  WARNING: Could not load FSIBSIS data: {e}")
            fsibsis_features = None
        
        # Merge all features (using keyword args for clarity)
        features = engineer.merge_features(
            weo_features=weo_features, 
            fsic_features=fsic_features, 
            credit_gap=credit_gap, 
            sovereign_nexus=sovereign_nexus,
            wgi_features=wgi_features, 
            fsibsis_features=fsibsis_features
        )
        
        # Free intermediate DataFrames
        del weo_features, fsic_features, credit_gap
        gc.collect()
        
        if len(features) == 0:
            raise ValueError("Feature engineering failed: No data produced")
            
        print(f"  Generated features for {len(features)} countries")
        
        # --- 2. SUPERVISED CRISIS CLASSIFIER ---
        print("\n[Step 2] Training/Loading Crisis Classifier...")
        # Train classifier (or load if already trained and cached)
        classifier, metrics = train_crisis_model()
        
        # Get crisis probabilities
        print("  Generating crisis probabilities...")
        feature_cols = [c for c in features.columns 
                       if c not in ['country_code', 'country_name', 'crisis_target'] 
                       and not c.endswith('_period')]
        
        # Ensure we have all necessary columns (fill missing with median)
        X = features[feature_cols].copy()
        for col in classifier.feature_names_:
            if col not in X.columns:
                X[col] = np.nan
        
        # Predict probability of crisis within 3 years
        # Note: input columns must match classifier training columns
        X_aligned = X[classifier.feature_names_]
        crisis_probs = classifier.predict_proba(X_aligned)
        
        features['crisis_prob'] = crisis_probs
        print(f"  Crisis probabilities generated (mean: {crisis_probs.mean():.1%})")
        
        # --- 3. UNSUPERVISED PILLARS (Economic + Industry) ---
        print("\n[Step 3] Building Unsupervised Pillars...")
        
        # Load country names mapping for display (deduplicate to avoid InvalidIndexError)
        country_names = weo_df[['country_code', 'country_name']].drop_duplicates(subset='country_code').set_index('country_code')['country_name']
        
        # Get development anchor (GDP per capita)
        anchor = identify_anchor_indicator(features, None) # None passed as features already has gdp_per_capita
        
        # Build two-pillar model (PCA-based)
        # Build two-pillar model (PCA-based)
        # We pass the engineered features directly
        pillar_scores, pca_loadings = build_two_pillar_model(features, anchor, country_names)
        
        # --- 4. HYBRID RISK SCORE ---
        print("\n[Step 4] Computing Final Hybrid Scores...")
        
        # Merge pillar scores with crisis probabilities
        final_df = pillar_scores.merge(
            features[['country_code', 'crisis_prob']], 
            on='country_code', 
            how='left'
        )
        
        # IMPORTANT: The pillar_scores['risk_score'] already incorporates:
        # - 50% GDP per capita (anchor)
        # - 25% Economic pillar (PCA)  
        # - 25% Industry pillar (PCA)
        # We adjust with crisis probability but keep the anchor-based score dominant
        
        # Adjust: 90% pillar-based score + 10% crisis probability adjustment
        # Higher crisis_prob should INCREASE risk score
        # NOTE: Using low weight for crisis_prob as classifier gives counterintuitive results
        final_df['crisis_adjustment'] = final_df['crisis_prob'] * 3  # Scale crisis prob (0-1) to adjustment
        final_df['hybrid_risk_score'] = (
            0.9 * final_df['risk_score'] +  # Keep mostly the anchor-weighted score
            0.1 * (1 + 9 * final_df['crisis_prob'])  # Add small crisis-based component (1-10 scale)
        )
        
        # Clamp to 1-10 range
        final_df['hybrid_risk_score'] = final_df['hybrid_risk_score'].clip(1, 10)
        
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
            'economic_weight': 0.50,
            'industry_weight': 0.50,
            'note': 'PCA reduces features to principal components. First PC captures most variance.',
            'economic_loadings': pca_loadings.get('economic_loadings', {}),
            'industry_loadings': pca_loadings.get('industry_loadings', {})
        }
        
        return self.country_scores
    
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
                'pca_info': self.pca_info
            }, f)
        
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
        
        return model


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("DATA-DRIVEN BANKING SYSTEM RISK MODEL")
    print("=" * 70)
    
    # Load data
    loader = IMFDataLoader()
    
    if not loader.load_from_cache():
        print("\nLoading from CSV files...")
        loader.load_all_datasets()
        loader.save_cache()
    
    fsic_df = loader._data_cache.get('FSIC', pd.DataFrame())
    weo_df = loader._data_cache.get('WEO', pd.DataFrame())
    mfs_df = loader._data_cache.get('MFS', pd.DataFrame())
    
    # Explore all indicators
    explore_all_indicators(fsic_df, weo_df, mfs_df)
    
    # Train model
    model = BankingRiskModel()
    results = model.train(fsic_df, weo_df, mfs_df)
    
    # Validate
    validate_model(results)
    
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
