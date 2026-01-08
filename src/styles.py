"""
Shared CSS and Styling constants for the application.
Design Philosophy: Premium, Data-Dense, "Terminal/Bloomberg" Aesthetic.
No emojis. High contrast.
"""

# Color Palette
COLORS = {
    'background': '#0E1117',
    'secondary_bg': '#161B22',
    'text': '#E6E6E6',
    'muted': '#8B949E',
    'border': '#30363D',
    'accent_blue': '#58A6FF',
    'risk_very_low': '#238636',  # Green
    'risk_low': '#2EA043',       # Light Green
    'risk_medium': '#D29922',    # Amber
    'risk_high': '#D95C33',      # Orange
    'risk_very_high': '#DA3633', # Red
    'white': '#FFFFFF',
    'black': '#000000',
}

# Main CSS Injection
MAIN_STYLES = """
<style>
    /* Global Cleanups */
    .stApp {
        background-color: #0E1117;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #E6E6E6 !important;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        font-weight: 600;
    }
    
    p, li, label, .stMarkdown {
        color: #C9D1D9 !important;
    }
    
    /* Remove default Streamlit padding/margins where excessive */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Custom Card Style */
    .data-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 6px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .data-card-header {
        color: #8B949E;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .data-card-value {
        color: #E6E6E6;
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'Consolas', 'Monaco', monospace; /* Terminal feel for numbers */
    }
    
    .data-card-delta {
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    /* Risk Badge */
    .risk-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Navigation / Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #0E1117;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161B22;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        border: 1px solid #30363D;
        border-bottom: none;
        color: #8B949E;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0E1117;
        color: #58A6FF;
        border-top: 2px solid #58A6FF;
    }
    
    /* Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #30363D;
        border-radius: 4px;
    }

</style>
"""

# Risk Tier Styling Helper
def get_risk_color(score: float) -> str:
    """Return hex color for a given risk score (0-100, where 100 is best/safest)."""
    if score >= 80: return COLORS['risk_very_low']
    if score >= 65: return COLORS['risk_low']
    if score >= 50: return COLORS['risk_medium']
    if score >= 35: return COLORS['risk_high']
    return COLORS['risk_very_high']

def get_risk_label(score: float) -> str:
    if score >= 80: return "VERY STRONG"
    if score >= 65: return "STRONG"
    if score >= 50: return "ADEQUATE"
    if score >= 35: return "WEAK"
    return "VERY WEAK"
