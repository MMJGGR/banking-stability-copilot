from __future__ import annotations

import plotly.graph_objects as go

from src.dashboard import presentation
from src.dashboard.styles import STYLES


def test_stylesheet_is_style_only_and_has_no_focusable_markup():
    payload = STYLES.strip()

    assert payload.startswith("<style>")
    assert payload.endswith("</style>")
    assert "</style><" not in payload.replace(" ", "").lower()
    for interactive_tag in ("<a ", "<button", "<input", "<select", "<textarea"):
        assert interactive_tag not in payload.lower()


def test_styles_use_streamlit_theme_tokens_not_os_theme_or_remote_font():
    assert "fonts.googleapis.com" not in STYLES
    assert "@import" not in STYLES
    assert "@media (prefers-color-scheme" not in STYLES
    assert "--bankenv-background: Canvas" in STYLES
    assert "--bankenv-text: CanvasText" in STYLES
    assert "--bankenv-surface: color-mix(in srgb, CanvasText" in STYLES
    assert "--bankenv-tile-bg: var(--bankenv-text)" in STYLES
    assert "--bankenv-main-stroke: var(--bankenv-background)" in STYLES


def test_styles_include_accessible_focus_and_responsive_primitives():
    assert "):focus-visible" in STYLES
    assert "outline: 3px solid var(--bankenv-focus-ring)" in STYLES
    assert ".bankenv-kpi-grid" in STYLES
    assert ".bankenv-table-scroll" in STYLES
    assert ".bankenv-chart-shell" in STYLES
    assert "div[data-testid=\"stPlotlyChart\"]" in STYLES
    assert "@media (prefers-reduced-motion: reduce)" in STYLES
    assert "@media (forced-colors: active)" in STYLES


def test_desktop_and_mobile_layouts_clear_the_fixed_streamlit_toolbar():
    desktop_styles, mobile_styles = STYLES.split(
        "@media (max-width: 640px)",
        maxsplit=1,
    )

    assert "padding-top: 4.5rem !important;" in desktop_styles
    assert "padding-top: 4.5rem !important;" in mobile_styles
    assert "env(safe-area-inset-top)" in mobile_styles


def test_dashboard_styles_prefer_style_only_streamlit_renderer(monkeypatch):
    rendered = []

    monkeypatch.setattr(presentation.st, "html", rendered.append, raising=False)
    monkeypatch.setattr(
        presentation.st,
        "markdown",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("style-only payload should use st.html")
        ),
    )

    presentation.render_dashboard_styles()

    assert rendered == [STYLES]


def test_time_boundary_accepts_pandas_timestamp_with_annotation():
    import pandas as pd

    figure = go.Figure()
    presentation.add_time_boundary(
        figure,
        pd.Timestamp("2026-01-01"),
        label="Projection begins",
    )

    assert len(figure.layout.shapes) == 1
    assert figure.layout.shapes[0].x0 == pd.Timestamp("2026-01-01").to_pydatetime()
    assert figure.layout.annotations[0].text == "Projection begins"


def test_full_label_markup_is_escaped_visible_and_wrapping():
    markup = presentation.full_label_html(
        '<script>alert("x")</script> Very long indicator',
        'Numerator <A>',
    )

    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup
    assert "Numerator &lt;A&gt;" in markup
    assert 'class="bankenv-full-label"' in markup
    assert "title=" not in markup
    assert "nowrap" not in markup


def test_editorial_formatters_preserve_financial_acronyms_and_pillar_names():
    assert presentation.format_identifier("npl_ratio_to_gdp") == "NPL Ratio To GDP"
    assert presentation.format_identifier(
        "npl_ratio", {"npl_ratio": "Nonperforming loan ratio"}
    ) == "Nonperforming loan ratio"
    assert presentation.format_pillar_label("economic") == "Operating Environment"
    assert presentation.format_pillar_label("industry") == "Banking System"


def test_accessible_plotly_config_is_responsive_and_independent():
    first = presentation.accessible_plotly_config()
    second = presentation.accessible_plotly_config()

    assert first["responsive"] is True
    assert first["displaylogo"] is False
    first["modeBarButtonsToRemove"].append("zoom2d")
    assert "zoom2d" not in second["modeBarButtonsToRemove"]


def test_responsive_chart_layout_uses_automatic_margins_and_bottom_legend():
    figure = go.Figure(go.Scatter(x=[1, 2], y=[2, 3], name="Series"))

    returned = presentation.apply_responsive_chart_layout(
        figure,
        title="A long chart title",
        showlegend=True,
        yaxis_title="Risk score",
    )

    assert returned is figure
    assert figure.layout.autosize is True
    assert figure.layout.title.text == "A long chart title"
    assert figure.layout.xaxis.automargin is True
    assert figure.layout.yaxis.automargin is True
    assert figure.layout.yaxis.title.text == "Risk score"
    assert figure.layout.legend.orientation == "h"
    assert figure.layout.legend.y < 0
    assert figure.layout.showlegend is True
