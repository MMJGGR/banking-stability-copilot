"""
CapitalIQ-inspired CSS Stylings for Banking Stability Copilot.
Focus on data density, clean lines, and professional contrast.
"""

COLORS = {
    'background': '#0E1117',
    'surface': '#161B22', 
    'border': '#30363D',
    'text_primary': '#E6E6E6',
    'text_secondary': '#8B949E',
    'accent': '#2F81F7',
    'success': '#238636',
    'warning': '#D29922',
    'danger': '#DA3633',
}

# Tier Colors map (1=best, 5=worst)
RISK_COLORS = {
    1: '#238636',  # Very Strong (Green)
    2: '#2EA043',  # Strong
    3: '#D29922',  # Adequate (Amber)
    4: '#D95C33',  # Weak
    5: '#DA3633',  # Very Weak (Red)
}

STYLES = """
<style>
    /* Import premium font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Apply font globally */
    html, body, .stApp, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    /* =========================================
       LAYOUT & SPACING
       ========================================= */
    h1, h2, h3 {
        padding-bottom: 0px !important;
        margin-bottom: 0.5rem !important;
        font-weight: 600 !important;
    }
    
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* =========================================
       METRICS / KPI CARDS
       ========================================= */
    div[data-testid="stMetricValue"] {
        font-family: 'Inter', 'Roboto Mono', monospace;
        font-weight: 700;
        font-size: 1.5rem !important;
    }
    
    /* =========================================
       SUMMARY CARDS
       ========================================= */
    .summary-box {
        background: linear-gradient(135deg, #161B22 0%, #1a1f26 100%);
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    
    .summary-header {
        color: #8B949E;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    
    .summary-value {
        color: #E6E6E6;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* =========================================
       DATA SNAPSHOT TABLE
       ========================================= */
    .snapshot-row {
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
        border-bottom: 1px solid #21262D;
    }
    
    .snapshot-row:last-child {
        border-bottom: none;
    }
    
    .snapshot-label {
        color: #8B949E;
        font-size: 0.85rem;
    }
    
    .snapshot-value {
        color: #E6E6E6;
        font-weight: 500;
        font-family: 'Consolas', 'Roboto Mono', monospace;
    }
    
    .snapshot-value.missing {
        color: #484F58;
    }
    
    .snapshot-value.imputed {
        color: #D29922;  /* Amber/warning color for imputed values */
        font-style: italic;
    }

    
    /* =========================================
       PREDICTION CARDS
       ========================================= */
    .prediction-card {
        background: linear-gradient(135deg, #161B22 0%, #21262D 100%);
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
    }
    
    .prediction-label {
        color: #8B949E;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .prediction-score {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'Consolas', 'Roboto Mono', monospace;
    }
    
    .prediction-tier {
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .prediction-category {
        color: #E6E6E6;
        font-size: 1rem;
    }
    
    /* =========================================
       TABLES
       ========================================= */
    div[data-testid="stDataFrame"] div[class*="stDataFrame"] {
        font-size: 0.85rem;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* =========================================
       TABS
       ========================================= */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #161B22;
        border-radius: 6px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        border-radius: 4px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #21262D;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #30363D !important;
    }
    
    /* =========================================
       EXPANDERS
       ========================================= */
    .streamlit-expanderHeader {
        background-color: #161B22;
        border-radius: 6px;
    }
    
    /* =========================================
       INPUTS
       ========================================= */
    .stNumberInput > div > div > input {
        background-color: #161B22;
        border: 1px solid #30363D;
        color: #E6E6E6;
    }
    
    .stSelectbox > div > div {
        background-color: #161B22;
    }
    
    /* =========================================
       CHARTS
       ========================================= */
    .chart-container {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 1rem;
    }
</style>
"""


def get_risk_color_hex(tier_num: int) -> str:
    """Get hex color for a risk tier (1-5)."""
    return RISK_COLORS.get(tier_num, '#8B949E')


def get_risk_label(tier_num: int) -> str:
    """Get text label for a risk tier (1-5)."""
    labels = {
        1: "Very Strong",
        2: "Strong",
        3: "Adequate",
        4: "Weak",
        5: "Very Weak"
    }
    return labels.get(tier_num, "N/A")


def score_to_tier(score: float) -> int:
    """Convert a 1-10 risk score to 1-5 tier."""
    if score <= 2:
        return 1
    elif score <= 4:
        return 2
    elif score <= 6:
        return 3
    elif score <= 8:
        return 4
    else:
        return 5


def score_to_category(score: float) -> str:
    """Convert a 1-10 risk score to category string."""
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
