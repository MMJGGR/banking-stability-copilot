"""
Configuration settings for the Banking System Stability Copilot.
Defines indicator categories, thresholds, and risk tier definitions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Dataset identifiers
DATASET_PATTERNS = {
    "FSIC": "IMF.STA_FSIC",
    "MFS": "IMF.STA_MFS",
    "WEO": "IMF.RES_WEO"
}

# Core Financial Soundness Indicators (FSIC) - Key banking health metrics
FSIC_CORE_INDICATORS = {
    # Capital Adequacy
    "RCAR_PT": "Regulatory capital to risk-weighted assets (%)",
    "T1RWA_PT": "Tier 1 capital to risk-weighted assets (%)",
    "CAR_PT": "Capital to assets (%)",
    
    # Asset Quality  
    "NPLGL_PT": "Non-performing loans to total gross loans (%)",
    "NPLNP_PT": "Non-performing loans net of provisions to capital (%)",
    
    # Earnings & Profitability
    "ROA_PT": "Return on assets (%)",
    "ROE_PT": "Return on equity (%)",
    "IMNII_PT": "Interest margin to gross income (%)",
    "NEIGIE_PT": "Non-interest expenses to gross income (%)",
    
    # Liquidity
    "LASTL_PT": "Liquid assets to short-term liabilities (%)",
    "LATA_PT": "Liquid assets to total assets (%)",
    
    # Sensitivity to Market Risk
    "NOPK_PT": "Net open position in FX to capital (%)"
}

# WEO Macroeconomic Indicators
WEO_CORE_INDICATORS = {
    "NGDP_RPCH": "Real GDP growth (%)",
    "NGDPDPC": "GDP per capita (current USD)",
    "PCPIPCH": "Inflation rate (%)",
    "BCA_NGDPD": "Current account balance (% of GDP)",
    "GGXWDG_NGDP": "Government debt (% of GDP)",
    "GGXCNL_NGDP": "Fiscal balance (% of GDP)",
    "LUR": "Unemployment rate (%)",
    "TM_RPCH": "Import volume change (%)",
    "TX_RPCH": "Export volume change (%)"
}

# MFS Monetary Indicators (key patterns to match)
MFS_CORE_INDICATORS = {
    "DC_ODCORP": "Credit to private sector",
    "DC_S1M1": "Broad money (M2)",
    "RA_FA": "Reserve assets",
    "DC_S11": "Central bank claims",
    "DC_A_ACO": "Total assets"
}

# Risk Tier Definitions (inspired by S&P BICRA)
@dataclass
class RiskThresholds:
    """Thresholds for categorizing indicator risk levels."""
    
    # Capital adequacy thresholds (higher is better)
    capital_adequacy: Dict[str, Tuple[float, float, float]] = field(default_factory=lambda: {
        "very_strong": (15.0, float('inf')),   # AAA-AA
        "strong": (12.0, 15.0),                 # A
        "adequate": (10.0, 12.0),               # BBB
        "weak": (8.0, 10.0),                    # BB
        "very_weak": (0.0, 8.0)                 # B and below
    })
    
    # NPL ratio thresholds (lower is better)
    npl_ratio: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "very_strong": (0.0, 2.0),
        "strong": (2.0, 4.0),
        "adequate": (4.0, 7.0),
        "weak": (7.0, 10.0),
        "very_weak": (10.0, float('inf'))
    })
    
    # ROE thresholds (higher is better)
    roe: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "very_strong": (15.0, float('inf')),
        "strong": (10.0, 15.0),
        "adequate": (5.0, 10.0),
        "weak": (0.0, 5.0),
        "very_weak": (float('-inf'), 0.0)
    })
    
    # Liquidity ratio thresholds (higher is better)
    liquidity: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "very_strong": (40.0, float('inf')),
        "strong": (30.0, 40.0),
        "adequate": (20.0, 30.0),
        "weak": (15.0, 20.0),
        "very_weak": (0.0, 15.0)
    })

# Risk category colors
RISK_COLORS = {
    "very_strong": "#2E7D32",  # Dark green
    "strong": "#66BB6A",       # Light green
    "adequate": "#FFC107",     # Amber
    "weak": "#FF7043",         # Orange
    "very_weak": "#D32F2F",    # Red
    "unknown": "#9E9E9E"       # Gray
}

# Score to rating mapping (0-100 scale)
SCORE_TO_RATING = [
    (90, 100, "AAA", "Extremely Strong"),
    (80, 90, "AA", "Very Strong"),
    (70, 80, "A", "Strong"),
    (60, 70, "BBB", "Adequate"),
    (50, 60, "BB", "Moderate"),
    (40, 50, "B", "Weak"),
    (30, 40, "CCC", "Very Weak"),
    (0, 30, "CC/C", "Extremely Weak")
]

def get_rating_from_score(score: float) -> Tuple[str, str]:
    """Convert numeric score (0-100) to rating category."""
    for low, high, rating, description in SCORE_TO_RATING:
        if low <= score < high:
            return rating, description
    return "NR", "Not Rated"

# Streamlit theme
STREAMLIT_THEME = {
    "primaryColor": "#1E88E5",
    "backgroundColor": "#0E1117",
    "secondaryBackgroundColor": "#262730",
    "textColor": "#FAFAFA"
}
