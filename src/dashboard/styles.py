"""
CSS styling for BankEnv.
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
    /*
     * BankEnv presentation tokens deliberately derive from Streamlit's active
     * theme variables. This keeps an in-app theme override in sync even when it
     * differs from the operating-system color preference.
     */
    .stApp {
        --bankenv-font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI",
            Roboto, Helvetica, Arial, sans-serif;
        --bankenv-font-mono: ui-monospace, "SFMono-Regular", Consolas,
            "Liberation Mono", monospace;
        --bankenv-background: var(--background-color, #FFFFFF);
        --bankenv-surface: var(--secondary-background-color, #F4F6F8);
        --bankenv-text: var(--text-color, #111827);
        --bankenv-primary: var(--primary-color, #0B74DE);
        --bankenv-muted-text: #4B5563;
        --bankenv-muted-text: color-mix(
            in srgb,
            var(--bankenv-text) 72%,
            var(--bankenv-background)
        );
        --bankenv-border: rgba(128, 128, 128, 0.35);
        --bankenv-border: color-mix(
            in srgb,
            var(--bankenv-text) 28%,
            var(--bankenv-background)
        );
        --bankenv-subtle: rgba(128, 128, 128, 0.14);
        --bankenv-subtle: color-mix(
            in srgb,
            var(--bankenv-text) 12%,
            var(--bankenv-background)
        );
        --bankenv-warning-text: #7A4B00;
        --bankenv-warning-text: color-mix(
            in srgb,
            #B45309 66%,
            var(--bankenv-text)
        );
        --bankenv-danger-text: #B42318;
        --bankenv-danger-text: color-mix(
            in srgb,
            #B42318 72%,
            var(--bankenv-text)
        );
        --bankenv-success-text: #147D39;
        --bankenv-success-text: color-mix(
            in srgb,
            #147D39 72%,
            var(--bankenv-text)
        );
        --bankenv-focus-ring: var(--bankenv-primary);
        --bankenv-radius-sm: 6px;
        --bankenv-radius-md: 8px;
        --bankenv-space-1: 0.25rem;
        --bankenv-space-2: 0.5rem;
        --bankenv-space-3: 0.75rem;
        --bankenv-space-4: 1rem;
        --bankenv-space-5: 1.25rem;
    }

    html, body, .stApp {
        font-family: var(--bankenv-font-sans) !important;
    }
    
    /* =========================================
       LAYOUT & SPACING
       ========================================= */
    h1, h2, h3 {
        padding-bottom: 0px !important;
        margin-bottom: 0.5rem !important;
        font-weight: 600 !important;
        line-height: 1.2 !important;
        overflow-wrap: anywhere;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1180px;
    }

    .bankenv-brand {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        margin: 0 0 0.35rem 0;
        line-height: 1;
    }

    .bankenv-brand-mark {
        /* Follow the active Streamlit theme, not prefers-color-scheme. */
        --bankenv-tile-bg: var(--bankenv-text);
        --bankenv-main-stroke: var(--bankenv-background);
        --bankenv-muted-stroke: #64748B;
        --bankenv-muted-stroke: color-mix(
            in srgb,
            var(--bankenv-background) 68%,
            var(--bankenv-text)
        );
        --bankenv-accent-stroke: #0284C7;
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: var(--bankenv-tile-bg);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: 1px solid var(--bankenv-border);
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.18);
    }

    .bankenv-brand-mark svg {
        width: 32px;
        height: 32px;
        display: block;
    }

    .bankenv-main-stroke {
        stroke: var(--bankenv-main-stroke);
    }

    .bankenv-muted-stroke {
        stroke: var(--bankenv-muted-stroke);
    }

    .bankenv-accent-stroke {
        stroke: var(--bankenv-accent-stroke);
    }

    .bankenv-brand-name {
        color: var(--bankenv-text);
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    /* A visible, wrapping replacement for hover-only title attributes. */
    .bankenv-full-label {
        color: var(--bankenv-muted-text);
        font-size: 0.875rem;
        line-height: 1.45;
        margin: -0.2rem 0 0.45rem;
        white-space: normal;
        overflow-wrap: anywhere;
    }

    .bankenv-full-label__prefix {
        color: var(--bankenv-text);
        font-weight: 600;
    }

    /* Strong, consistent keyboard focus without adding focusable elements. */
    :where(
        a,
        button,
        input,
        textarea,
        select,
        [role="tab"],
        [role="option"],
        [role="checkbox"],
        [role="switch"],
        [tabindex]:not([tabindex="-1"])
    ):focus-visible {
        outline: 3px solid var(--bankenv-focus-ring) !important;
        outline-offset: 2px !important;
        border-radius: var(--bankenv-radius-sm);
    }
    
    /* =========================================
       METRICS / KPI CARDS
       ========================================= */
    div[data-testid="stMetricValue"] {
        font-family: var(--bankenv-font-sans);
        font-weight: 700;
        font-size: 1.42rem !important;
        font-variant-numeric: tabular-nums;
    }

    /* Opt-in primitive for UI-native/custom metric groups. */
    .bankenv-kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
        gap: var(--bankenv-space-3);
        align-items: stretch;
    }

    .bankenv-kpi-card {
        min-width: 0;
        padding: var(--bankenv-space-3);
        border: 1px solid var(--bankenv-border);
        border-radius: var(--bankenv-radius-md);
        background: var(--bankenv-surface);
    }
    
    /* =========================================
       SUMMARY CARDS
       ========================================= */
    .summary-box {
        background: var(--bankenv-surface);
        border: 1px solid var(--bankenv-border);
        border-radius: var(--bankenv-radius-md);
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    
    .summary-header {
        color: var(--bankenv-muted-text);
        font-size: 0.8125rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    
    .summary-value {
        color: var(--bankenv-text);
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* =========================================
       DATA SNAPSHOT TABLE
       ========================================= */
    .snapshot-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: baseline;
        gap: var(--bankenv-space-3);
        padding: 0.35rem 0;
        border-bottom: 1px solid var(--bankenv-border);
    }
    
    .snapshot-row:last-child {
        border-bottom: none;
    }
    
    .snapshot-label {
        color: var(--bankenv-muted-text);
        font-size: 0.875rem;
        overflow-wrap: anywhere;
    }
    
    .snapshot-value {
        color: var(--bankenv-text);
        font-weight: 500;
        font-family: var(--bankenv-font-mono);
        font-variant-numeric: tabular-nums;
        text-align: right;
    }
    
    .snapshot-value.missing {
        color: var(--bankenv-muted-text);
    }
    
    .snapshot-value.imputed {
        color: var(--bankenv-warning-text);
        font-style: italic;
    }

    
    /* =========================================
       PREDICTION CARDS
       ========================================= */
    .prediction-card {
        background: var(--bankenv-surface);
        border: 1px solid var(--bankenv-border);
        border-radius: var(--bankenv-radius-md);
        padding: 1.25rem;
        text-align: center;
    }
    
    .prediction-label {
        color: var(--bankenv-muted-text);
        font-size: 0.8125rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .prediction-score {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: var(--bankenv-font-mono);
        font-variant-numeric: tabular-nums;
    }
    
    .prediction-tier {
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .prediction-category {
        color: var(--bankenv-text);
        font-size: 1rem;
    }
    
    /* =========================================
       TABLES
       ========================================= */
    div[data-testid="stDataFrame"] {
        max-width: 100%;
        min-width: 0;
        overflow: hidden;
        border-radius: var(--bankenv-radius-sm);
    }

    div[data-testid="stDataFrame"] div[class*="stDataFrame"] {
        font-size: 0.875rem;
        font-family: var(--bankenv-font-sans);
    }

    .bankenv-table-scroll {
        max-width: 100%;
        overflow-x: auto;
        overscroll-behavior-inline: contain;
        -webkit-overflow-scrolling: touch;
        scrollbar-gutter: stable;
    }
    
    /* =========================================
       TABS
       ========================================= */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: var(--bankenv-surface);
        border-radius: var(--bankenv-radius-sm);
        padding: 4px;
        max-width: 100%;
        overflow-x: auto;
        overscroll-behavior-inline: contain;
        scrollbar-width: thin;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.65rem 1rem;
        font-weight: 500;
        border-radius: 4px;
        flex: 0 0 auto;
        min-height: 44px;
    }

    @media (max-width: 640px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            /*
             * Clear Streamlit's fixed mobile toolbar. Keep a plain fallback
             * before the safe-area expression: some hosted WebViews reject
             * the whole max()/env() declaration even though the media query
             * itself matches.
             */
            padding-top: 4.5rem !important;
            padding-top: max(
                4.5rem,
                calc(3.5rem + env(safe-area-inset-top))
            ) !important;
        }

        .bankenv-brand-mark {
            width: 40px;
            height: 40px;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 0.55rem 0.7rem;
            font-size: 0.92rem;
        }

        /* Streamlit metric rows become a compact two-column grid on mobile. */
        div[data-testid="stHorizontalBlock"]:has(
            > div[data-testid="stColumn"] div[data-testid="stMetric"]
        ) {
            flex-wrap: wrap !important;
            gap: var(--bankenv-space-3) !important;
        }

        div[data-testid="stHorizontalBlock"]:has(
            > div[data-testid="stColumn"] div[data-testid="stMetric"]
        ) > div[data-testid="stColumn"] {
            flex: 1 1 calc(50% - var(--bankenv-space-3)) !important;
            width: calc(50% - var(--bankenv-space-3)) !important;
            min-width: 8.5rem !important;
        }

        .bankenv-kpi-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .bankenv-chart-shell {
            min-height: 18rem;
        }

        .snapshot-row {
            gap: var(--bankenv-space-2);
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.32rem !important;
        }
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--bankenv-subtle);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--bankenv-subtle) !important;
    }
    
    /* =========================================
       EXPANDERS
       ========================================= */
    .streamlit-expanderHeader {
        background-color: var(--bankenv-surface);
        border-radius: var(--bankenv-radius-sm);
    }
    
    /* =========================================
       INPUTS
       ========================================= */
    .stNumberInput > div > div > input {
        background-color: var(--bankenv-surface);
        border: 1px solid var(--bankenv-border);
        color: var(--bankenv-text);
    }
    
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span {
        background-color: var(--bankenv-surface) !important;
        color: var(--bankenv-text) !important;
        -webkit-text-fill-color: var(--bankenv-text) !important;
    }

    div[data-baseweb="select"] > div {
        border-color: var(--bankenv-border) !important;
    }

    div[data-baseweb="popover"],
    ul[data-testid="stVirtualDropdown"],
    div[role="listbox"] {
        background-color: var(--bankenv-background) !important;
        color: var(--bankenv-text) !important;
    }

    li[role="option"],
    div[role="option"] {
        background-color: var(--bankenv-background) !important;
        color: var(--bankenv-text) !important;
    }

    li[role="option"]:hover,
    div[role="option"]:hover {
        background-color: var(--bankenv-surface) !important;
    }

    div[data-baseweb="tag"] {
        background-color: rgba(47, 129, 247, 0.18) !important;
        color: var(--bankenv-text) !important;
    }
    
    /* =========================================
       CHARTS
       ========================================= */
    .chart-container {
        background-color: var(--bankenv-surface);
        border: 1px solid var(--bankenv-border);
        border-radius: var(--bankenv-radius-md);
        padding: 1rem;
    }

    .bankenv-chart-shell,
    div[data-testid="stPlotlyChart"] {
        width: 100%;
        max-width: 100%;
        min-width: 0;
    }

    div[data-testid="stPlotlyChart"] .modebar-btn:focus-visible {
        outline: 3px solid var(--bankenv-focus-ring) !important;
        outline-offset: 1px;
    }

    @media (max-width: 380px) {
        .bankenv-kpi-grid {
            grid-template-columns: 1fr;
        }

        div[data-testid="stHorizontalBlock"]:has(
            > div[data-testid="stColumn"] div[data-testid="stMetric"]
        ) > div[data-testid="stColumn"] {
            min-width: 100% !important;
            width: 100% !important;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }

    @media (forced-colors: active) {
        :where(a, button, input, textarea, select, [role="tab"]):focus-visible {
            outline-color: Highlight !important;
        }

        .bankenv-brand-mark,
        .bankenv-kpi-card,
        .summary-box,
        .prediction-card,
        .chart-container {
            border: 1px solid CanvasText;
        }
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
