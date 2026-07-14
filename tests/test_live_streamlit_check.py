from src.scripts import check_live_streamlit as live_check


def _observation(status=200, body="", final_url="https://bankenv.streamlit.app/"):
    return live_check.HttpObservation(
        requested_url="https://bankenv.streamlit.app/",
        final_url=final_url,
        status=status,
        body=body,
    )


def test_public_root_is_reachable():
    classification, reachable, login_gated, _ = live_check.classify_observations(
        _observation(status=200, body="Streamlit"),
        _observation(status=200, body="ok"),
    )
    assert classification == "public_http_reachable"
    assert reachable is True
    assert login_gated is False


def test_access_control_is_reachable_but_login_gated():
    classification, reachable, login_gated, _ = live_check.classify_observations(
        _observation(status=403, body="Forbidden"),
        None,
    )
    assert classification == "login_gated_reachable"
    assert reachable is True
    assert login_gated is True


def test_login_redirect_or_private_page_is_not_misreported_as_public():
    classification, reachable, login_gated, _ = live_check.classify_observations(
        _observation(
            status=200,
            body="Sign in to Streamlit",
            final_url="https://share.streamlit.io/login?next=private-app",
        ),
        _observation(status=200, body="ok"),
    )
    assert classification == "login_gated_reachable"
    assert reachable is True
    assert login_gated is True


def test_streamlit_auth_location_is_detected_when_redirects_loop():
    classification, reachable, login_gated, _ = live_check.classify_observations(
        live_check.HttpObservation(
            requested_url="https://bankenv.streamlit.app/",
            final_url="https://bankenv.streamlit.app/",
            status=303,
            body="See Other",
            location="https://share.streamlit.io/-/auth/app?redirect_uri=private",
        ),
        None,
    )
    assert classification == "login_gated_reachable"
    assert reachable is True
    assert login_gated is True


def test_health_endpoint_can_establish_reachability_when_root_is_unavailable():
    classification, reachable, login_gated, _ = live_check.classify_observations(
        _observation(status=503, body="starting"),
        _observation(status=200, body="ok"),
    )
    assert classification == "public_health_reachable"
    assert reachable is True
    assert login_gated is False


def test_sanitize_url_removes_credentials_query_and_fragment():
    sanitized = live_check.sanitize_url(
        "https://user:secret@bankenv.streamlit.app/path?token=secret#fragment"
    )
    assert sanitized == "https://bankenv.streamlit.app/path"
    assert "secret" not in sanitized


def test_run_check_retries_only_until_reachable(monkeypatch):
    reports = iter(
        [
            {"classification": "unavailable", "reachable": False, "login_gated": False},
            {
                "classification": "public_http_reachable",
                "reachable": True,
                "login_gated": False,
            },
        ]
    )
    monkeypatch.setattr(live_check, "probe_once", lambda *_args, **_kwargs: next(reports))
    sleeps = []

    report = live_check.run_check(
        "https://bankenv.streamlit.app",
        attempts=3,
        interval_seconds=0.25,
        sleep=sleeps.append,
    )

    assert report["classification"] == "public_http_reachable"
    assert report["attempt"] == 2
    assert sleeps == [0.25]
