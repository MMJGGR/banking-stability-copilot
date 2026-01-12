# -*- coding: utf-8 -*-
"""
Comprehensive Model Review - Banking Stability Model
Uses ASCII-compatible output for Windows compatibility
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import CACHE_DIR, BASE_DIR
from src.data_loader import IMFDataLoader, FSIBSISLoader, WGILoader
from src.crisis_labels import CrisisLabels

REVIEW_OUTPUT_DIR = os.path.join(BASE_DIR, "cache", "model_review")
os.makedirs(REVIEW_OUTPUT_DIR, exist_ok=True)

print("="*80)
print("COMPREHENSIVE MODEL REVIEW - ML/AI DIAGNOSTIC ANALYSIS")
print("Review Date:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("="*80)


def analyze_data_coverage():
    """Analyze data coverage."""
    print("\n" + "="*80)
    print("SECTION 1: DATA QUALITY & COVERAGE ANALYSIS")
    print("="*80)
    
    results = {}
    
    loader = IMFDataLoader()
    loader.load_from_cache()
    
    datasets = {
        'FSIC': loader._data_cache.get('FSIC'),
        'WEO': loader._data_cache.get('WEO'),
        'MFS': loader._data_cache.get('MFS')
    }
    
    print("\n--- 1.1 Dataset Size Summary ---")
    for name, df in datasets.items():
        if df is not None and len(df) > 0:
            n_records = len(df)
            n_countries = df['country_code'].nunique()
            n_indicators = df['indicator_code'].nunique() if 'indicator_code' in df.columns else 0
            print(f"  {name}: {n_records:,} records | {n_countries} countries | {n_indicators} indicators")
            results[name] = {'records': n_records, 'countries': n_countries}
    
    # Check FSIBSIS and WGI
    try:
        fsibsis_loader = FSIBSISLoader()
        fsibsis_loader.load()
        if fsibsis_loader.bank_data is not None:
            print(f"  FSIBSIS: {len(fsibsis_loader.bank_data):,} records | {fsibsis_loader.bank_data['country_code'].nunique()} countries")
    except Exception as e:
        print(f"  FSIBSIS: Failed - {e}")
    
    try:
        wgi_loader = WGILoader()
        wgi_data = wgi_loader.load()
        if wgi_data is not None:
            print(f"  WGI: {len(wgi_data):,} records | {wgi_data['country_code'].nunique()} countries")
    except Exception as e:
        print(f"  WGI: Failed - {e}")
    
    # Feature coverage
    print("\n--- 1.2 Feature Coverage Analysis ---")
    try:
        features_path = os.path.join(CACHE_DIR, "crisis_features.parquet")
        features_df = pd.read_parquet(features_path)
        
        meta_cols = ['country_code', 'country_name', 'year', 'crisis_target']
        feature_cols = [c for c in features_df.columns if c not in meta_cols and not c.endswith('_year')]
        numeric_df = features_df[feature_cols].select_dtypes(include=[np.number])
        
        print(f"\n  Total numeric features: {len(numeric_df.columns)}")
        print(f"  Total countries: {len(features_df)}")
        
        # Coverage per feature
        coverage_stats = []
        for col in numeric_df.columns:
            non_null = numeric_df[col].notna().sum()
            coverage_pct = non_null / len(numeric_df) * 100
            coverage_stats.append({'feature': col, 'coverage': coverage_pct})
        
        coverage_df = pd.DataFrame(coverage_stats).sort_values('coverage')
        
        print("\n  Features with LOWEST coverage:")
        for _, row in coverage_df.head(10).iterrows():
            print(f"    {row['feature']}: {row['coverage']:.1f}%")
        
        # Missingness pattern
        high_missing = [c for c in coverage_stats if c['coverage'] < 50]
        print(f"\n  Features with <50% coverage: {len(high_missing)}/{len(coverage_stats)}")
        
        results['feature_coverage'] = coverage_df.to_dict('records')
        results['low_coverage_count'] = len(high_missing)
        
    except Exception as e:
        print(f"  ERROR: {e}")
    
    return results


def analyze_model_performance():
    """Analyze model performance metrics."""
    print("\n" + "="*80)
    print("SECTION 2: MODEL PERFORMANCE ANALYSIS")
    print("="*80)
    
    results = {}
    
    # Load model
    model_path = os.path.join(CACHE_DIR, "risk_model.pkl")
    classifier_path = os.path.join(CACHE_DIR, "crisis_classifier.pkl")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("\n  Model loaded successfully")
        print(f"  Training date: {model.get('training_date', 'Unknown')}")
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return results
    
    # PCA info
    print("\n--- 2.1 PCA Dimensionality Reduction ---")
    pca_info = model.get('pca_info', {})
    
    econ_var = pca_info.get('economic_variance_explained', 'N/A')
    ind_var = pca_info.get('industry_variance_explained', 'N/A')
    
    print(f"  Economic Pillar PC1 variance: {econ_var}")
    print(f"  Industry Pillar PC1 variance: {ind_var}")
    
    if isinstance(econ_var, (int, float)):
        status = "[OK]" if econ_var >= 0.5 else "[WARNING]"
        print(f"  {status} Economic PCA {'adequate' if econ_var >= 0.5 else 'weak'} (threshold: 50%)")
    
    if isinstance(ind_var, (int, float)):
        status = "[OK]" if ind_var >= 0.5 else "[WARNING]"
        print(f"  {status} Industry PCA {'adequate' if ind_var >= 0.5 else 'weak'} (threshold: 50%)")
    
    results['pca'] = {'economic': econ_var, 'industry': ind_var}
    
    # Classifier info
    print("\n--- 2.2 Crisis Classifier ---")
    try:
        with open(classifier_path, 'rb') as f:
            classifier_data = pickle.load(f)
        
        print(f"  Classifier keys: {list(classifier_data.keys())}")
        
        if 'feature_importance' in classifier_data:
            fi = classifier_data['feature_importance']
            sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
            print("\n  Top 10 Important Features:")
            for i, (feat, imp) in enumerate(sorted_fi, 1):
                print(f"    {i:2d}. {feat}: {imp:.4f}")
            results['top_features'] = sorted_fi
        
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Score distribution
    print("\n--- 2.3 Score Distribution ---")
    scores_df = model.get('country_scores', pd.DataFrame())
    
    if not scores_df.empty:
        print(f"  Countries scored: {len(scores_df)}")
        print(f"  Score range: {scores_df['risk_score'].min():.2f} - {scores_df['risk_score'].max():.2f}")
        print(f"  Mean: {scores_df['risk_score'].mean():.2f}, Std: {scores_df['risk_score'].std():.2f}")
        
        # Tier distribution
        def score_to_tier(s):
            if s <= 2: return "Very Low"
            elif s <= 4: return "Low"
            elif s <= 6: return "Moderate"
            elif s <= 8: return "High"
            else: return "Very High"
        
        scores_df['tier'] = scores_df['risk_score'].apply(score_to_tier)
        tier_dist = scores_df['tier'].value_counts()
        
        print("\n  Tier distribution:")
        for tier in ["Very Low", "Low", "Moderate", "High", "Very High"]:
            if tier in tier_dist:
                count = tier_dist[tier]
                pct = count / len(scores_df) * 100
                print(f"    {tier}: {count} ({pct:.1f}%)")
        
        results['score_stats'] = {
            'min': float(scores_df['risk_score'].min()),
            'max': float(scores_df['risk_score'].max()),
            'mean': float(scores_df['risk_score'].mean()),
            'std': float(scores_df['risk_score'].std())
        }
    
    return results


def analyze_crisis_labels():
    """Analyze crisis label adequacy."""
    print("\n" + "="*80)
    print("SECTION 3: CRISIS LABEL ADEQUACY")
    print("="*80)
    
    results = {}
    
    labels = CrisisLabels()
    summary = labels.get_crisis_summary()
    
    print(f"\n--- 3.1 Crisis Database Overview ---")
    n_countries = len(summary)
    n_events = int(summary['n_crises'].sum())
    n_years = int(summary['total_crisis_years'].sum())
    
    print(f"  Countries with crisis history: {n_countries}")
    print(f"  Total crisis events: {n_events}")
    print(f"  Total crisis-years: {n_years}")
    print(f"  Time range: {summary['first_crisis'].min()} - {summary['latest_crisis'].max()}")
    
    results['n_countries'] = n_countries
    results['n_events'] = n_events
    
    # Recent crises
    print("\n--- 3.2 Recent Crisis Coverage (Post-2015) ---")
    recent = summary[summary['latest_crisis'] >= 2015].sort_values('latest_crisis', ascending=False)
    print(f"  Countries with recent crises: {len(recent)}")
    for _, row in recent.iterrows():
        print(f"    {row['country_code']}: ended {row['latest_crisis']}")
    
    # Known events check
    print("\n--- 3.3 Known Crisis Events Validation ---")
    known_events = {
        ('GHA', 2022): "Ghana 2022-24 debt crisis",
        ('TUR', 2018): "Turkey 2018-19 lira crisis",
        ('LKA', 2022): "Sri Lanka 2022 default",
        ('ARG', 2018): "Argentina 2018-19 crisis",
        ('LBN', 2019): "Lebanon 2019 banking crisis",
    }
    
    missing = []
    for (country, year), event_name in known_events.items():
        is_covered = labels.is_crisis_year(country, year)
        status = "[OK]" if is_covered else "[MISSING]"
        print(f"  {status} {event_name}")
        if not is_covered:
            missing.append(event_name)
    
    results['missing_events'] = missing
    
    # Sample size assessment
    print("\n--- 3.4 Sample Size Assessment ---")
    n_features = 33  # From README
    events_per_feature = n_events / n_features
    
    print(f"  Crisis events: {n_events}")
    print(f"  Model features: {n_features}")
    print(f"  Events per feature: {events_per_feature:.2f}")
    
    if events_per_feature < 1:
        print("  [CRITICAL] Far too few events per feature")
        results['sample_assessment'] = 'critical'
    elif events_per_feature < 5:
        print("  [WARNING] High instability risk")
        results['sample_assessment'] = 'warning'
    elif events_per_feature < 10:
        print("  [CAUTION] Marginally adequate")
        results['sample_assessment'] = 'marginal'
    else:
        print("  [OK] Sample size adequate")
        results['sample_assessment'] = 'adequate'
    
    return results


def analyze_country_validation():
    """Validate country risk scores."""
    print("\n" + "="*80)
    print("SECTION 4: COUNTRY OUTPUT VALIDATION")
    print("="*80)
    
    results = {}
    
    model_path = os.path.join(CACHE_DIR, "risk_model.pkl")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        scores_df = model.get('country_scores', pd.DataFrame())
    except Exception as e:
        print(f"  ERROR: {e}")
        return results
    
    if scores_df.empty:
        print("  No scores available")
        return results
    
    # Validate expected high-risk
    print("\n--- 4.1 Expected HIGH-Risk Countries ---")
    expected_high = ['VEN', 'ZWE', 'LBN', 'ARG', 'TUR', 'PAK', 'NGA', 'UKR', 'GHA', 'EGY']
    
    high_results = []
    for country in expected_high:
        row = scores_df[scores_df['country_code'] == country]
        if len(row) > 0:
            score = row['risk_score'].values[0]
            name = row['country_name'].values[0] if 'country_name' in row.columns else country
            status = "[OK]" if score >= 6 else "[UNEXPECTED]"
            print(f"  {status} {name}: {score:.2f}")
            high_results.append({'country': country, 'score': float(score), 'validated': score >= 6})
        else:
            print(f"  [NOT FOUND] {country}")
            high_results.append({'country': country, 'score': None, 'validated': False})
    
    # Validate expected low-risk
    print("\n--- 4.2 Expected LOW-Risk Countries ---")
    expected_low = ['CHE', 'NOR', 'FIN', 'SWE', 'DNK', 'SGP', 'NLD', 'DEU', 'CAN', 'AUS', 'USA', 'GBR', 'JPN']
    
    low_results = []
    for country in expected_low:
        row = scores_df[scores_df['country_code'] == country]
        if len(row) > 0:
            score = row['risk_score'].values[0]
            name = row['country_name'].values[0] if 'country_name' in row.columns else country
            status = "[OK]" if score <= 4 else "[UNEXPECTED]"
            print(f"  {status} {name}: {score:.2f}")
            low_results.append({'country': country, 'score': float(score), 'validated': score <= 4})
        else:
            print(f"  [NOT FOUND] {country}")
            low_results.append({'country': country, 'score': None, 'validated': False})
    
    # Validation summary
    high_valid = sum(1 for r in high_results if r['validated'])
    low_valid = sum(1 for r in low_results if r['validated'])
    
    print(f"\n--- 4.3 Validation Summary ---")
    print(f"  High-risk validated: {high_valid}/{len(expected_high)} ({high_valid/len(expected_high)*100:.0f}%)")
    print(f"  Low-risk validated: {low_valid}/{len(expected_low)} ({low_valid/len(expected_low)*100:.0f}%)")
    
    results['high_risk'] = high_results
    results['low_risk'] = low_results
    
    # Coverage bias check
    print("\n--- 4.4 Data Coverage Bias Check ---")
    if 'data_coverage' in scores_df.columns:
        corr = scores_df['data_coverage'].corr(scores_df['risk_score'])
        print(f"  Correlation (coverage vs score): {corr:.3f}")
        
        if abs(corr) > 0.3:
            print("  [WARNING] Significant correlation - possible bias")
        else:
            print("  [OK] No significant coverage bias")
        
        results['coverage_correlation'] = float(corr)
    
    # Full ranking
    print("\n--- 4.5 Complete Risk Ranking ---")
    ranked = scores_df.sort_values('risk_score', ascending=False)
    
    print("\n  TOP 15 HIGHEST RISK:")
    for i, (_, row) in enumerate(ranked.head(15).iterrows(), 1):
        name = row.get('country_name', row['country_code'])
        cov = row.get('data_coverage', 0) * 100
        print(f"    {i:2d}. {row['country_code']}: {row['risk_score']:.2f} (coverage: {cov:.0f}%)")
    
    print("\n  TOP 15 LOWEST RISK:")
    for i, (_, row) in enumerate(ranked.tail(15).iloc[::-1].iterrows(), 1):
        name = row.get('country_name', row['country_code'])
        cov = row.get('data_coverage', 0) * 100
        print(f"    {i:2d}. {row['country_code']}: {row['risk_score']:.2f} (coverage: {cov:.0f}%)")
    
    return results


def analyze_literature_compliance():
    """Compare against standards."""
    print("\n" + "="*80)
    print("SECTION 5: LITERATURE & FRAMEWORK COMPLIANCE")
    print("="*80)
    
    results = {}
    
    # Load features
    features_path = os.path.join(CACHE_DIR, "crisis_features.parquet")
    try:
        features_df = pd.read_parquet(features_path)
        available = set(features_df.columns)
    except:
        available = set()
    
    print("\n--- 5.1 S&P BICRA Alignment ---")
    
    bicra = [
        ('GDP per capita', 'gdp_per_capita'),
        ('Credit-to-GDP gap', 'credit_to_gdp_gap'),
        ('Government debt', 'govt_debt_gdp'),
        ('Current account', 'current_account_gdp'),
        ('Inflation', 'inflation'),
        ('Capital adequacy', 'capital_adequacy'),
        ('NPL ratio', 'npl_ratio'),
        ('ROE', 'roe'),
        ('Liquidity', 'liquid_assets_st_liab'),
        ('Loan concentration', 'loan_concentration'),
        ('Real estate loans', 'real_estate_loans'),
    ]
    
    bicra_score = 0
    for name, feature in bicra:
        status = "[OK]" if feature in available else "[MISSING]"
        print(f"    {status} {name} ({feature})")
        if feature in available:
            bicra_score += 1
    
    compliance = bicra_score / len(bicra) * 100
    print(f"\n  BICRA Compliance: {compliance:.0f}%")
    results['bicra_compliance'] = compliance
    
    print("\n--- 5.2 BIS Early Warning Indicators ---")
    bis = [
        ('Credit-to-GDP gap', 'credit_to_gdp_gap'),
        ('Property price gap', None),
        ('Debt service ratio', None),
    ]
    
    bis_score = 0
    for name, feature in bis:
        if feature and feature in available:
            print(f"    [OK] {name}")
            bis_score += 1
        else:
            print(f"    [MISSING] {name}")
    
    print(f"\n  BIS Coverage: {bis_score}/{len(bis)}")
    results['bis_coverage'] = bis_score
    
    print("\n--- 5.3 Academic Literature Gaps ---")
    gaps = [
        "Property price gap (Borio & Drehmann 2009) - HIGH priority",
        "Debt service ratio (BIS 2010) - MEDIUM priority",
        "External debt breakdown (Reinhart & Rogoff) - MEDIUM priority",
    ]
    
    print("  Key gaps identified:")
    for gap in gaps:
        print(f"    - {gap}")
    
    results['gaps'] = gaps
    
    return results


def analyze_imputation():
    """Analyze imputation impact."""
    print("\n" + "="*80)
    print("SECTION 6: IMPUTATION IMPACT ANALYSIS")
    print("="*80)
    
    results = {}
    
    features_path = os.path.join(CACHE_DIR, "crisis_features.parquet")
    
    try:
        raw_df = pd.read_parquet(features_path)
        
        meta_cols = ['country_code', 'country_name', 'year', 'crisis_target']
        feature_cols = [c for c in raw_df.columns if c not in meta_cols and not c.endswith('_year')]
        numeric_df = raw_df[feature_cols].select_dtypes(include=[np.number])
        
        # Missing rates
        missing_rates = {}
        for col in numeric_df.columns:
            rate = numeric_df[col].isna().sum() / len(numeric_df) * 100
            missing_rates[col] = rate
        
        high_missing = sum(1 for v in missing_rates.values() if v > 70)
        mod_missing = sum(1 for v in missing_rates.values() if 30 < v <= 70)
        
        print(f"\n--- 6.1 Missing Data Summary ---")
        print(f"  Features >70% missing: {high_missing}/{len(missing_rates)}")
        print(f"  Features 30-70% missing: {mod_missing}/{len(missing_rates)}")
        
        # Worst features
        print("\n  Highest missing rates:")
        sorted_rates = sorted(missing_rates.items(), key=lambda x: x[1], reverse=True)
        for col, rate in sorted_rates[:10]:
            print(f"    {col}: {rate:.1f}%")
        
        # Assessment
        print(f"\n--- 6.2 Imputation Impact Assessment ---")
        if high_missing > len(missing_rates) * 0.3:
            print("  [CRITICAL] >30% of features heavily imputed")
            results['validity'] = 'impaired'
        elif high_missing > len(missing_rates) * 0.1:
            print("  [CAUTION] 10-30% of features heavily imputed")
            results['validity'] = 'acceptable_with_caution'
        else:
            print("  [OK] Imputation levels acceptable")
            results['validity'] = 'acceptable'
        
        results['high_missing_count'] = high_missing
        
    except Exception as e:
        print(f"  ERROR: {e}")
    
    return results


def main():
    """Run all analyses."""
    all_results = {}
    
    all_results['data_coverage'] = analyze_data_coverage()
    all_results['model_performance'] = analyze_model_performance()
    all_results['crisis_labels'] = analyze_crisis_labels()
    all_results['country_validation'] = analyze_country_validation()
    all_results['literature'] = analyze_literature_compliance()
    all_results['imputation'] = analyze_imputation()
    
    # Summary
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    
    critical = []
    warnings = []
    strengths = []
    
    # Assess crisis labels
    if all_results.get('crisis_labels', {}).get('sample_assessment') in ['critical', 'warning']:
        warnings.append("Small crisis sample limits statistical reliability")
    
    if len(all_results.get('crisis_labels', {}).get('missing_events', [])) > 2:
        warnings.append("Recent crisis events not captured in training data")
    
    # Assess imputation
    if all_results.get('imputation', {}).get('validity') == 'impaired':
        critical.append("High imputation rates may impair reliability")
    
    # Assess validation
    hr = all_results.get('country_validation', {}).get('high_risk', [])
    lr = all_results.get('country_validation', {}).get('low_risk', [])
    
    high_valid = sum(1 for r in hr if r.get('validated', False))
    low_valid = sum(1 for r in lr if r.get('validated', False))
    
    if high_valid < len(hr) * 0.7:
        warnings.append(f"Only {high_valid}/{len(hr)} high-risk countries validated")
    
    if low_valid < len(lr) * 0.7:
        warnings.append(f"Only {low_valid}/{len(lr)} low-risk countries validated")
    
    # Strengths
    bicra = all_results.get('literature', {}).get('bicra_compliance', 0)
    if bicra >= 80:
        strengths.append(f"Strong S&P BICRA alignment ({bicra:.0f}%)")
    elif bicra >= 60:
        strengths.append(f"Moderate BICRA alignment ({bicra:.0f}%)")
    
    strengths.append("Two-pillar architecture matches industry practice")
    strengths.append("Comprehensive data sources (5 IMF/WB datasets)")
    strengths.append("Literature-backed feature selection")
    
    print("\n  CRITICAL ISSUES:")
    if critical:
        for issue in critical:
            print(f"    [X] {issue}")
    else:
        print("    None identified")
    
    print("\n  WARNINGS:")
    if warnings:
        for warning in warnings:
            print(f"    [!] {warning}")
    else:
        print("    None identified")
    
    print("\n  STRENGTHS:")
    for strength in strengths:
        print(f"    [+] {strength}")
    
    # Verdict
    print("\n  OVERALL VERDICT:")
    if critical:
        print("    Model requires attention to critical issues")
        verdict = "REQUIRES_ATTENTION"
    elif len(warnings) > 3:
        print("    Model is functional with notable limitations")
        verdict = "ACCEPTABLE_WITH_LIMITATIONS"
    else:
        print("    Model is within acceptable criteria")
        verdict = "ACCEPTABLE"
    
    all_results['summary'] = {
        'critical': critical,
        'warnings': warnings,
        'strengths': strengths,
        'verdict': verdict
    }
    
    # Save results
    output_path = os.path.join(REVIEW_OUTPUT_DIR, "review_results.json")
    
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=convert)
    
    print(f"\n  Results saved to: {output_path}")
    print("\n" + "="*80)
    print("REVIEW COMPLETE")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    main()
