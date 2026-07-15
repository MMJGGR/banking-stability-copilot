"""Shared presentation helpers for the BankEnv Streamlit interface.

The functions in this module are intentionally model- and data-agnostic. They
provide a single integration point for stylesheet injection, accessible label
readback, editorial display names, and responsive Plotly defaults.
"""

from __future__ import annotations

from copy import deepcopy
import html
import re
from typing import Any, Mapping

import streamlit as st

from src.dashboard.styles import STYLES


SYSTEM_FONT_CSS = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, '
    'Arial, sans-serif'
)

PILLAR_DISPLAY_NAMES = {
    "economic": "Operating Environment",
    "industry": "Banking System",
    "combined": "Combined",
}

_DISPLAY_TOKENS = {
    "gdp": "GDP",
    "gfn": "GFN",
    "gni": "GNI",
    "imf": "IMF",
    "iip": "IIP",
    "mfs": "MFS",
    "nfa": "NFA",
    "nim": "NIM",
    "npl": "NPL",
    "ppg": "PPG",
    "roa": "ROA",
    "roe": "ROE",
    "usd": "USD",
    "weo": "WEO",
    "wgi": "WGI",
    "fx": "FX",
    "govt": "Government",
}

_PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "showTips": True,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "toggleSpikelines",
    ],
}


def render_dashboard_styles() -> None:
    """Inject the style-only payload without creating a focusable UI control.

    ``st.html`` treats a payload containing only ``<style>`` tags specially and
    does not add a visible layout block. The fallback supports older local
    Streamlit installations while the deployed dependency floor provides
    ``st.html``.
    """

    html_renderer = getattr(st, "html", None)
    if callable(html_renderer):
        html_renderer(STYLES)
    else:  # pragma: no cover - serving requirements include st.html
        st.markdown(STYLES, unsafe_allow_html=True)


def full_label_html(label: Any, prefix: str = "Selected item") -> str:
    """Return escaped, wrapping readback markup for a selected long label."""

    text = str(label or "").strip()
    if not text:
        return ""
    safe_label = html.escape(text, quote=True)
    safe_prefix = html.escape(str(prefix or "Selected item").strip(), quote=True)
    return (
        '<p class="bankenv-full-label">'
        f'<span class="bankenv-full-label__prefix">{safe_prefix}:</span> '
        f"{safe_label}</p>"
    )


def render_full_label(label: Any, prefix: str = "Selected item") -> None:
    """Render a full label that works with touch, keyboard, and narrow screens."""

    markup = full_label_html(label, prefix)
    if not markup:
        return
    html_renderer = getattr(st, "html", None)
    if callable(html_renderer):
        html_renderer(markup)
    else:  # pragma: no cover - serving requirements include st.html
        st.markdown(markup, unsafe_allow_html=True)


def format_identifier(
    value: Any,
    overrides: Mapping[str, str] | None = None,
) -> str:
    """Convert a technical identifier to a stable analyst-facing label.

    Callers can pass their canonical feature registry through ``overrides``;
    otherwise common financial acronyms are preserved instead of title-cased.
    """

    raw = str(value or "").strip()
    if not raw:
        return ""
    if overrides and raw in overrides:
        return str(overrides[raw])
    tokens = [token for token in re.split(r"[_\s-]+", raw) if token]
    return " ".join(
        _DISPLAY_TOKENS.get(token.lower(), token.capitalize())
        for token in tokens
    )


def format_pillar_label(value: Any) -> str:
    """Return the app's canonical display name for a model pillar."""

    raw = str(value or "").strip().lower()
    return PILLAR_DISPLAY_NAMES.get(raw, format_identifier(raw))


def accessible_plotly_config() -> dict[str, Any]:
    """Return an independent responsive, low-clutter Plotly configuration."""

    return deepcopy(_PLOTLY_CONFIG)


def apply_responsive_chart_layout(
    figure: Any,
    *,
    title: str | None = None,
    showlegend: bool | None = None,
    yaxis_title: str | None = None,
) -> Any:
    """Apply theme-neutral responsive layout defaults to a Plotly figure.

    Colors remain owned by Streamlit's active chart theme. The horizontal legend
    and automatic margins keep long country/indicator labels out of the plot on
    narrow displays. The function mutates and returns ``figure`` like Plotly's
    native ``update_layout`` API.
    """

    layout: dict[str, Any] = {
        "autosize": True,
        "font": {"family": SYSTEM_FONT_CSS},
        "margin": {"l": 48, "r": 24, "t": 56 if title else 24, "b": 56},
        "hoverlabel": {"align": "left"},
        "xaxis": {"automargin": True},
        "yaxis": {"automargin": True},
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "left",
            "x": 0,
            "title_text": "",
        },
    }
    if title is not None:
        layout["title"] = {
            "text": str(title),
            "x": 0,
            "xanchor": "left",
            "font": {"size": 16},
        }
    if showlegend is not None:
        layout["showlegend"] = bool(showlegend)
    if yaxis_title is not None:
        layout["yaxis"]["title"] = {"text": str(yaxis_title)}
    figure.update_layout(**layout)
    return figure

