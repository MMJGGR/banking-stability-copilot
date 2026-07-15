"""Rendered Streamlit contracts for the four public BankEnv workspaces."""

from __future__ import annotations

import pytest
import streamlit as st
from packaging.version import Version

from streamlit.testing.v1 import AppTest


requires_supported_widget_testing = pytest.mark.skipif(
    Version(st.__version__) < Version("1.45"),
    reason=(
        "Streamlit before 1.45 cannot rerun AppTest selectboxes whose stored "
        "values use format_func labels; production requirements start at 1.45."
    ),
)


def _run_app(*, view: str, tool: str | None = None, section: str | None = None) -> AppTest:
    app = AppTest.from_file("app.py", default_timeout=180)
    app.query_params["view"] = view
    if tool is not None:
        app.query_params["tool"] = tool
    if section is not None:
        app.session_state["methodology_section"] = section
    app.run(timeout=180)
    assert not app.exception, [str(item.value) for item in app.exception]
    assert not app.error, [str(item.value) for item in app.error]
    return app


@pytest.mark.parametrize(
    ("view", "tool", "section"),
    [
        ("global", None, None),
        ("country", None, None),
        ("explorer", "compare", None),
        ("explorer", "calculate", None),
        ("explorer", "source inspector", None),
        ("methodology", None, "How the Score Works"),
        ("methodology", None, "Data and Coverage"),
        ("methodology", None, "Validation and Release"),
    ],
)
def test_public_workspace_loads_without_streamlit_errors(view, tool, section):
    _run_app(view=view, tool=tool, section=section)


def test_stale_country_navigation_and_peer_state_are_sanitized():
    app = AppTest.from_file("app.py", default_timeout=180)
    app.query_params.update({"view": "country", "country": "USA"})
    app.session_state["primary_view"] = "Removed page"
    app.session_state["profile_country_code"] = "ZZZ"
    app.session_state["explorer_focus_country"] = "ZZZ"
    app.session_state["custom_peer_codes_USA"] = ["USA", "ZZZ", "FRA", "FRA"]

    app.run(timeout=180)

    assert not app.exception, [str(item.value) for item in app.exception]
    assert app.session_state["primary_view"] == "Country"
    assert app.session_state["profile_country_code"] == "USA"
    assert app.session_state["explorer_focus_country"] == "USA"
    assert app.session_state["custom_peer_codes_USA"] == ["FRA"]


@requires_supported_widget_testing
def test_source_inspector_remains_loaded_after_inner_control_rerun():
    app = _run_app(view="explorer", tool="source inspector")
    load_button = next(button for button in app.button if button.label == "Load Source")
    load_button.click().run(timeout=180)
    assert not app.exception, [str(item.value) for item in app.exception]

    range_control = next(
        item for item in app.selectbox if item.label == "Time Range"
    )
    range_control.select("5 Years").run(timeout=180)

    assert not app.exception, [str(item.value) for item in app.exception]
    assert any(
        item.label == "Select Indicator to Visualize" for item in app.selectbox
    )


def test_validation_appendix_does_not_nest_expanders():
    app = _run_app(
        view="methodology",
        section="Validation and Release",
    )
    appendix = next(
        item
        for item in app.checkbox
        if item.label == "Show candidate monitoring appendix"
    )
    appendix.check().run(timeout=180)
    assert not app.exception, [str(item.value) for item in app.exception]


@requires_supported_widget_testing
def test_country_driver_attribution_loads_on_demand():
    app = _run_app(view="country")
    driver_button = next(
        button for button in app.button if button.label == "Load score drivers"
    )
    driver_button.click().run(timeout=180)
    assert not app.exception, [str(item.value) for item in app.exception]
    assert any(
        "Main Risk-Raising Drivers" in str(item.value) for item in app.markdown
    )
