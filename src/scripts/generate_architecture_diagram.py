"""
Generate the Process Architecture diagram as a static PNG image.
This replaces the unreliable Mermaid.js rendering on Streamlit Cloud.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

def create_architecture_diagram():
    """Create the banking risk model architecture diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    
    # Color scheme
    colors = {
        'data_source': '#3498db',      # Blue
        'processing': '#9b59b6',       # Purple
        'pillar': '#2ecc71',           # Green
        'pca': '#f39c12',              # Orange
        'ml': '#e74c3c',               # Red
        'output': '#1abc9c',           # Teal
        'text': 'white',
        'arrow': '#bdc3c7'
    }
    
    def draw_box(x, y, width, height, label, color, fontsize=9):
        """Draw a rounded box with label."""
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.85
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
                color='white', fontweight='bold', wrap=True)
        return (x, y)
    
    def draw_arrow(start, end, color='#bdc3c7'):
        """Draw an arrow between two points."""
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    
    # Title
    ax.text(7, 9.5, 'Banking Risk Model Architecture', ha='center', va='center',
            fontsize=16, color='white', fontweight='bold')
    
    # === DATA SOURCES (Top Row) ===
    y_data = 8.0
    fsic = draw_box(2, y_data, 2.2, 0.7, 'FSIC\n(Banking FSI)', colors['data_source'])
    mfs = draw_box(5, y_data, 2.2, 0.7, 'MFS\n(Balance Sheets)', colors['data_source'])
    weo = draw_box(8, y_data, 2.2, 0.7, 'WEO\n(Macro Context)', colors['data_source'])
    wgi = draw_box(11, y_data, 2.2, 0.7, 'WGI\n(Governance)', colors['data_source'])
    
    # === PROCESSING (Second Row) ===
    y_proc = 6.3
    clean = draw_box(6.5, y_proc, 3, 0.7, 'Cleaning & Imputation', colors['processing'])
    
    # Arrows from data sources to cleaning
    for src in [fsic, mfs, weo, wgi]:
        draw_arrow((src[0], src[1] - 0.35), (6.5, y_proc + 0.35))
    
    # Feature Calculation
    y_calc = 5.0
    calc = draw_box(6.5, y_calc, 3, 0.7, 'Feature Calculation', colors['processing'])
    draw_arrow((6.5, y_proc - 0.35), (6.5, y_calc + 0.35))
    
    # === PILLARS (Third Row) ===
    y_pillar = 3.5
    econ = draw_box(4, y_pillar, 2.8, 0.7, 'Economic Pillar\n(12 Features)', colors['pillar'])
    ind = draw_box(9, y_pillar, 2.8, 0.7, 'Industry Pillar\n(14 Features)', colors['pillar'])
    
    draw_arrow((5.5, y_calc - 0.35), (4, y_pillar + 0.35))
    draw_arrow((7.5, y_calc - 0.35), (9, y_pillar + 0.35))
    
    # === PCA Components ===
    y_pca = 2.3
    pca_e = draw_box(4, y_pca, 2.5, 0.6, 'PCA Component 1\n(Economic)', colors['pca'], fontsize=8)
    pca_i = draw_box(9, y_pca, 2.5, 0.6, 'PCA Component 1\n(Industry)', colors['pca'], fontsize=8)
    
    draw_arrow((4, y_pillar - 0.35), (4, y_pca + 0.3))
    draw_arrow((9, y_pillar - 0.35), (9, y_pca + 0.3))
    
    # === ML CLASSIFIER (Side Branch) ===
    y_ml = 3.5
    xgb = draw_box(12, y_ml, 2, 0.7, 'XGBoost\nClassifier', colors['ml'])
    draw_arrow((8, y_calc - 0.2), (11, y_ml + 0.35), color='#95a5a6')  # Dashed conceptually
    
    y_prob = 2.3
    prob = draw_box(12, y_prob, 2, 0.6, 'Crisis\nProbability', colors['ml'], fontsize=8)
    draw_arrow((12, y_ml - 0.35), (12, y_prob + 0.3))
    
    # === FINAL OUTPUT ===
    y_risk = 1.0
    risk = draw_box(7, y_risk, 3.2, 0.8, 'Final Risk Score\n(1-10)', colors['output'], fontsize=11)
    
    draw_arrow((4, y_pca - 0.3), (5.8, y_risk + 0.4))
    draw_arrow((9, y_pca - 0.3), (8.2, y_risk + 0.4))
    draw_arrow((12, y_prob - 0.3), (8.5, y_risk + 0.2))
    
    # === DASHBOARD OUTPUT ===
    dash = draw_box(11.5, y_risk, 2.2, 0.7, 'Streamlit\nDashboard', colors['output'])
    draw_arrow((8.6, y_risk), (10.4, y_risk))
    
    # Legend
    legend_y = 0.3
    legend_items = [
        ('Data Sources', colors['data_source']),
        ('Processing', colors['processing']),
        ('Risk Pillars', colors['pillar']),
        ('PCA/ML', colors['pca']),
        ('Output', colors['output'])
    ]
    for i, (label, color) in enumerate(legend_items):
        x_pos = 1.5 + i * 2.5
        rect = plt.Rectangle((x_pos - 0.2, legend_y - 0.15), 0.4, 0.3, 
                             facecolor=color, edgecolor='white', linewidth=1)
        ax.add_patch(rect)
        ax.text(x_pos + 0.4, legend_y, label, color='white', fontsize=8, va='center')
    
    plt.tight_layout()
    
    # Save to cache/eda directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'cache', 'eda')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'architecture_diagram.png')
    
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', edgecolor='none', 
                bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print(f"Architecture diagram saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    create_architecture_diagram()
