"""
Reusable UI components for the Banking Copilot application.
Adheres to the minimal, premium "Terminal" design system.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Optional, Union

from src.styles import COLORS, get_risk_color, get_risk_label

def render_metric_card(
    label: str,
    value: Union[str, float],
    delta: Optional[str] = None,
    delta_color: str = "normal",
    tooltip: Optional[str] = None
):
    """
    Render a premium metric card.
    
    Args:
        label: Title of the metric
        value: The main value to display
        delta: Optional change indicator string (e.g. "+1.2%")
        delta_color: "normal" (green=good), "inverse" (red=good), or "off"
    """
    # Determine Color for Delta
    # In CSS, we'll manually handle colors based on the passed delta string logic if needed,
    # but simple inline styles work best for dynamic values.
    
    delta_html = ""
    if delta:
        # Simple parsing for color
        is_positive = delta.startswith("+") or (not delta.startswith("-") and float(delta.strip('%')) > 0 )
        
        # Logic: if delta_color is normal, positive=green. If inverse, positive=red.
        if delta_color == "inverse":
            color = COLORS['risk_high'] if is_positive else COLORS['risk_low']
        else:
            color = COLORS['risk_low'] if is_positive else COLORS['risk_high']
            
        # Arrow icon (CSS shape or simple text arrow)
        arrow = "↑" if is_positive else "↓"
        delta_html = f'<span style="color: {color};">{arrow} {delta}</span>'

    # Format value if float
    display_value = f"{value:.1f}" if isinstance(value, float) else str(value)

    st.markdown(f"""
    <div class="data-card">
        <div class="data-card-header" title="{tooltip or ''}">{label}</div>
        <div class="data-card-value">{display_value}</div>
        <div class="data-card-delta">{delta_html}</div>
    </div>
    """, unsafe_allow_html=True)


def render_risk_gauge(score: float, title: str = "Composite Risk Score"):
    """
    Render a prominent, high-contrast risk gauge using Plotly.
    Designed to be the visual centerpiece.
    """
    color = get_risk_color(score)
    label = get_risk_label(score)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            'suffix': "/100", 
            'font': {'size': 40, 'color': COLORS['text'], 'family': "Consolas, monospace"},
        },
        title={
            'text': title.upper(), 
            'font': {'size': 14, 'color': COLORS['muted'], 'family': "Segoe UI, sans-serif"}
        },
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLORS['border']},
            'bar': {'color': color, 'thickness': 0.8}, # The filled part
            'bgcolor': COLORS['secondary_bg'],
            'borderwidth': 0,
            'bordercolor': COLORS['border'],
            'steps': [
                # Subtle background track
                {'range': [0, 100], 'color': '#1f242d'} 
            ],
            'threshold': {
                'line': {'color': COLORS['white'], 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=260,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Text label below gauge
    st.markdown(f"""
    <div style="text-align: center; margin-top: -10px;">
        <span class="risk-badge" style="background-color: {color}20; color: {color}; border: 1px solid {color};">
            {label}
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_score_breakdown(econ_score: float, ind_score: float):
    """
    Render the two pillar scores as simple horizontal progress bars.
    """
    st.markdown("### Risk Pillars")
    
    for label, score in [("Economic Resilience", econ_score), ("Industry Risk", ind_score)]:
        color = get_risk_color(score)
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="color: {COLORS['muted']}; font-size: 0.9rem;">{label}</span>
                <span style="color: {COLORS['text']}; font-weight: bold; font-family: monospace;">{score:.0f}</span>
            </div>
            <div style="width: 100%; height: 6px; background-color: {COLORS['border']}; border-radius: 3px; overflow: hidden;">
                <div style="width: {score}%; height: 100%; background-color: {color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_header(title: str, subtitle: Optional[str] = None):
    """
    Render a standard page header.
    """
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(f"<p style='color: {COLORS['muted']}; margin-top: -10px; margin-bottom: 2rem;'>{subtitle}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
