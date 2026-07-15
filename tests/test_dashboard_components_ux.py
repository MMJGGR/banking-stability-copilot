from __future__ import annotations

import ast
import inspect

import pandas as pd

from src.dashboard import components


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_source_context_and_history_export_include_complete_metadata():
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-03-31", "2025-06-30"]),
            "value": [4.25, 4.75],
            "unit": ["Percent", "Percent"],
            "frequency": ["Q", "Q"],
            "observation_status": ["actual", "estimate"],
            "latest_actual_year": [2025, 2025],
            "indicator_code": ["NGDP_RPCH", "NGDP_RPCH"],
        }
    )

    context = components._source_history_context(
        "WEO", history, "NGDP_RPCH", "Real GDP growth"
    )
    exported = components._history_display_table(history, "Q", "WEO")

    assert context["Source"] == "IMF World Economic Outlook"
    assert context["Unit"] == "Percent"
    assert context["Frequency"] == "Quarterly"
    assert context["Observation status"] == "actual, estimate"
    assert list(exported.columns) == [
        "Period",
        "Value",
        "Unit",
        "Frequency",
        "Status",
        "Source",
    ]
    assert exported["Period"].tolist() == ["2025-Q2", "2025-Q1"]
    assert exported["Value"].dtype.kind in "fi"
    assert exported["Status"].tolist() == ["Estimate", "Actual"]
    assert exported["Source"].nunique() == 1


def test_components_copy_has_no_staged_source_language_or_decorative_emoji():
    source = inspect.getsource(components)

    assert "staged" not in source.lower()
    for decorative_character in ("📋", "⚠", "🔮"):
        assert decorative_character not in source
    assert "Risk score raised to a minimum floor" in source
    assert "except:\n" not in source
    assert "except Exception:\n            pass" not in source


def test_every_plotly_render_uses_shared_accessible_config_and_layout():
    tree = ast.parse(inspect.getsource(components))
    plotly_calls = []
    responsive_layout_calls = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "st"
            and node.func.attr == "plotly_chart"
        ):
            plotly_calls.append(node)
        if isinstance(node.func, ast.Name) and node.func.id == "apply_responsive_chart_layout":
            responsive_layout_calls.append(node)

    assert len(plotly_calls) == 3
    assert len(responsive_layout_calls) == 3
    for call in plotly_calls:
        config_keywords = [keyword for keyword in call.keywords if keyword.arg == "config"]
        assert len(config_keywords) == 1
        config_value = config_keywords[0].value
        assert isinstance(config_value, ast.Call)
        assert isinstance(config_value.func, ast.Name)
        assert config_value.func.id == "accessible_plotly_config"


def test_time_series_renders_responsive_chart_and_downloadable_metadata(monkeypatch):
    history = pd.DataFrame(
        {
            "indicator_code": ["NGDP_RPCH", "NGDP_RPCH"],
            "indicator_name": ["Real GDP growth", "Real GDP growth"],
            "period": ["2024-Q4", "2025-Q1"],
            "value": [3.1, 3.4],
            "unit": ["Percent", "Percent"],
            "frequency": ["Q", "Q"],
            "observation_status": ["actual", "estimate"],
        }
    )
    rendered = {}

    monkeypatch.setattr(components.st, "columns", lambda *args, **kwargs: [_Context(), _Context()])
    monkeypatch.setattr(components.st, "expander", lambda *args, **kwargs: _Context())
    monkeypatch.setattr(components, "render_full_label", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        components.st,
        "selectbox",
        lambda label, options, **kwargs: options[0] if "Indicator" in label else "All Data",
    )
    monkeypatch.setattr(components.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(components.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(components.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(components.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        components.st,
        "plotly_chart",
        lambda figure, **kwargs: rendered.update(figure=figure, chart_kwargs=kwargs),
    )
    monkeypatch.setattr(
        components.st,
        "download_button",
        lambda label, **kwargs: rendered.update(download_label=label, download_kwargs=kwargs),
    )

    components.render_time_series_deep_dive(history, "WEO", "KEN")

    assert rendered["figure"].layout.autosize is True
    assert rendered["chart_kwargs"]["config"]["responsive"] is True
    assert rendered["download_label"] == "Download displayed data (CSV)"
    csv_payload = rendered["download_kwargs"]["data"].decode("utf-8")
    assert "Period,Value,Unit,Frequency,Status,Source" in csv_payload
    assert "IMF World Economic Outlook" in csv_payload


def test_driver_chart_uses_text_labels_in_addition_to_color(monkeypatch):
    rendered = {}
    monkeypatch.setattr(
        components.st,
        "plotly_chart",
        lambda figure, **kwargs: rendered.update(figure=figure, kwargs=kwargs),
    )
    monkeypatch.setattr(components.st, "warning", lambda *args, **kwargs: None)

    components.render_drivers_chart(
        [
            {"indicator": "NPL ratio", "z_score": 1.3, "impact": "risk"},
            {"indicator": "Capital", "z_score": -0.8, "impact": "strength"},
        ]
    )

    trace = rendered["figure"].data[0]
    assert set(trace.text) == {"Raises risk", "Lowers risk"}
    assert len(set(trace.marker.color)) == 2
    assert rendered["figure"].layout.autosize is True
    assert rendered["kwargs"]["config"]["responsive"] is True
