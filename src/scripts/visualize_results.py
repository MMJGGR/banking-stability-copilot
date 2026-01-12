"""
Banking Risk Model - Visualization Script

Generates visualizations for the trained risk model outputs:
1. Risk score distribution histogram
2. Top/Bottom risk countries bar chart
3. Economic vs Industry pillar scatter
4. Key countries comparison
5. World map heatmap (if geopandas available)
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import CACHE_DIR

# Output directory for visualizations
OUTPUT_DIR = os.path.join(CACHE_DIR, 'visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_model_results():
    """Load trained model results."""
    model_path = os.path.join(CACHE_DIR, "risk_model.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['country_scores']


def plot_risk_distribution(df: pd.DataFrame):
    """Plot distribution of risk scores."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram with KDE
    sns.histplot(df['risk_score'], bins=20, kde=True, ax=ax, color='steelblue', alpha=0.7)
    
    # Add risk category boundaries
    boundaries = [2, 4, 6, 8]
    colors = ['green', 'yellowgreen', 'orange', 'red']
    labels = ['Very Low', 'Low', 'Moderate', 'High']
    
    for bound, color, label in zip(boundaries, colors, labels):
        ax.axvline(bound, color=color, linestyle='--', linewidth=2, alpha=0.7)
        ax.text(bound + 0.1, ax.get_ylim()[1] * 0.9, label, fontsize=9, color=color)
    
    ax.set_xlabel('Risk Score (1 = Lowest, 10 = Highest)', fontsize=12)
    ax.set_ylabel('Number of Countries', fontsize=12)
    ax.set_title('Distribution of Banking System Risk Scores', fontsize=14, fontweight='bold')
    
    # Add statistics
    stats_text = f"Mean: {df['risk_score'].mean():.2f}\nMedian: {df['risk_score'].median():.2f}\nStd: {df['risk_score'].std():.2f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'risk_distribution.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_top_bottom_countries(df: pd.DataFrame, n=15):
    """Plot top and bottom countries by risk."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Lowest risk (best)
    lowest = df.nsmallest(n, 'risk_score')
    colors_low = plt.cm.RdYlGn(np.linspace(0.8, 0.5, n))
    
    axes[0].barh(range(n), lowest['risk_score'].values, color=colors_low)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([f"{row['country_code']} - {row['country_name'][:20]}" 
                             for _, row in lowest.iterrows()], fontsize=10)
    axes[0].set_xlabel('Risk Score', fontsize=12)
    axes[0].set_title(f'Top {n} Lowest Risk Countries', fontsize=14, fontweight='bold', color='green')
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 10)
    
    # Add score labels
    for i, (_, row) in enumerate(lowest.iterrows()):
        axes[0].text(row['risk_score'] + 0.1, i, f"{row['risk_score']:.1f}", va='center', fontsize=9)
    
    # Highest risk (worst)
    highest = df.nlargest(n, 'risk_score')
    colors_high = plt.cm.RdYlGn(np.linspace(0.2, 0.0, n))
    
    axes[1].barh(range(n), highest['risk_score'].values, color=colors_high)
    axes[1].set_yticks(range(n))
    axes[1].set_yticklabels([f"{row['country_code']} - {row['country_name'][:20]}" 
                             for _, row in highest.iterrows()], fontsize=10)
    axes[1].set_xlabel('Risk Score', fontsize=12)
    axes[1].set_title(f'Top {n} Highest Risk Countries', fontsize=14, fontweight='bold', color='red')
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 10)
    
    # Add score labels
    for i, (_, row) in enumerate(highest.iterrows()):
        axes[1].text(row['risk_score'] + 0.1, i, f"{row['risk_score']:.1f}", va='center', fontsize=9)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'top_bottom_countries.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_pillar_comparison(df: pd.DataFrame):
    """Scatter plot comparing Economic vs Industry pillars."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by risk score
    scatter = ax.scatter(
        df['economic_pillar'], 
        df['industry_pillar'],
        c=df['risk_score'],
        cmap='RdYlGn_r',
        s=80,
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Risk Score', fontsize=12)
    
    # Label key countries
    key_countries = ['USA', 'CHN', 'JPN', 'DEU', 'GBR', 'IND', 'BRA', 'RUS', 'VEN', 'ARG', 'NGA', 'ZAF']
    for _, row in df[df['country_code'].isin(key_countries)].iterrows():
        ax.annotate(
            row['country_code'],
            (row['economic_pillar'], row['industry_pillar']),
            fontsize=9,
            fontweight='bold',
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    ax.set_xlabel('Economic Pillar Score (0-10)', fontsize=12)
    ax.set_ylabel('Industry Pillar Score (0-10)', fontsize=12)
    ax.set_title('Economic vs Industry Risk Pillars', fontsize=14, fontweight='bold')
    
    # Add quadrant labels
    ax.axhline(5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(5, color='gray', linestyle='--', alpha=0.5)
    ax.text(8, 8, 'Strong Both', fontsize=10, alpha=0.7)
    ax.text(2, 8, 'Strong Industry\nWeak Economy', fontsize=10, alpha=0.7)
    ax.text(8, 2, 'Strong Economy\nWeak Industry', fontsize=10, alpha=0.7)
    ax.text(2, 2, 'Weak Both', fontsize=10, alpha=0.7)
    
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'pillar_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_key_countries_breakdown(df: pd.DataFrame):
    """Detailed breakdown for key countries."""
    key_countries = ['USA', 'CHN', 'JPN', 'DEU', 'GBR', 'FRA', 'IND', 'BRA', 
                     'CAN', 'AUS', 'MEX', 'TUR', 'ZAF', 'NGA', 'ARG', 'VEN']
    
    key_df = df[df['country_code'].isin(key_countries)].copy()
    key_df = key_df.sort_values('risk_score')
    
    if len(key_df) == 0:
        print("  No key countries found in data")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(key_df))
    width = 0.25
    
    # Economic pillar
    bars1 = ax.bar(x - width, key_df['economic_pillar'], width, label='Economic Pillar', color='steelblue', alpha=0.8)
    
    # Industry pillar
    bars2 = ax.bar(x, key_df['industry_pillar'], width, label='Industry Pillar', color='coral', alpha=0.8)
    
    # Final risk score (inverted for visual comparison)
    bars3 = ax.bar(x + width, 10 - key_df['risk_score'], width, label='Safety Score (10 - Risk)', color='green', alpha=0.8)
    
    ax.set_xlabel('Country', fontsize=12)
    ax.set_ylabel('Score (0-10)', fontsize=12)
    ax.set_title('Key Countries: Pillar Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(key_df['country_code'], fontsize=10, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 11)
    
    # Add risk score labels on top
    for i, (_, row) in enumerate(key_df.iterrows()):
        ax.text(i, 10.3, f"Risk: {row['risk_score']:.1f}", ha='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'key_countries_breakdown.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_risk_category_pie(df: pd.DataFrame):
    """Pie chart of risk category distribution."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    category_counts = df['risk_category'].value_counts()
    
    colors = {
        '1-2: Very Low Risk': '#2ecc71',
        '3-4: Low Risk': '#f1c40f', 
        '5-6: Moderate Risk': '#e67e22',
        '7-8: High Risk': '#e74c3c',
        '9-10: Very High Risk': '#8e44ad'
    }
    
    pie_colors = [colors.get(cat, 'gray') for cat in category_counts.index]
    
    wedges, texts, autotexts = ax.pie(
        category_counts.values,
        labels=category_counts.index,
        autopct='%1.1f%%',
        colors=pie_colors,
        explode=[0.02] * len(category_counts),
        shadow=True,
        startangle=90
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    ax.set_title('Distribution of Countries by Risk Category', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'risk_category_pie.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_correlation_matrix(df: pd.DataFrame):
    """Correlation matrix of model components."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Select numeric columns for correlation
    numeric_cols = ['risk_score', 'economic_pillar', 'industry_pillar', 'data_coverage']
    if 'crisis_prob' in df.columns:
        numeric_cols.append('crisis_prob')
    if 'hybrid_risk_score' in df.columns:
        numeric_cols.append('hybrid_risk_score')
    
    corr_df = df[numeric_cols].dropna()
    corr = corr_df.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, ax=ax, square=True, linewidths=1)
    
    ax.set_title('Correlation Matrix of Risk Components', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'correlation_matrix.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("BANKING RISK MODEL VISUALIZATION")
    print("=" * 70)
    
    # Load data
    print("\nLoading model results...")
    df = load_model_results()
    print(f"  Loaded {len(df)} countries")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_risk_distribution(df)
    plot_top_bottom_countries(df)
    plot_pillar_comparison(df)
    plot_key_countries_breakdown(df)
    plot_risk_category_pie(df)
    plot_correlation_matrix(df)
    
    print(f"\n[OK] All visualizations saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
