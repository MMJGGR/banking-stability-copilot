"""Classify the reachability of a deployed Streamlit application.

This is deliberately a transport-level check.  It reports a public endpoint
separately from an access-controlled/login-gated endpoint and does not claim
that a particular Git revision is live.  Login-gated is a successful
reachability result because an unauthenticated GitHub runner cannot inspect a
private Streamlit application.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


AUTH_URL_MARKERS = (
    "/login",
    "/-/login",
    "/signin",
    "/sign-in",
    "/oauth",
    "/auth",
    "/-/auth",
)
AUTH_BODY_MARKERS = (
    "sign in to streamlit",
    "log in to streamlit",
    "this app is private",
    "request access",
    "you do not have access",
    "authentication required",
)
MAX_BODY_BYTES = 256_000


class _NoRedirectHandler(HTTPRedirectHandler):
    """Keep the first response so auth redirects cannot loop in CI."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


NO_REDIRECT_OPENER = build_opener(_NoRedirectHandler())


@dataclass(frozen=True)
class HttpObservation:
    """One bounded HTTP observation used by the reachability classifier."""

    requested_url: str
    final_url: str
    status: int
    body: str
    location: str | None = None

    def public_dict(self) -> dict:
        result = {
            "status": int(self.status),
            "final_url": sanitize_url(self.final_url),
        }
        if self.location:
            result["redirect_url"] = sanitize_url(self.location)
        return result


def sanitize_url(value: str) -> str:
    """Remove credentials, query strings, and fragments from reported URLs."""
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if parsed.port is not None:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))


def normalize_app_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("App URL must be an absolute http(s) URL")
    path = parsed.path.rstrip("/")
    hostname = parsed.hostname
    if parsed.port is not None:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, path, "", ""))


def health_url(app_url: str) -> str:
    return f"{normalize_app_url(app_url)}/_stcore/health"


def fetch_http(url: str, timeout: float = 15.0) -> HttpObservation:
    """Fetch a bounded response, retaining HTTP errors for classification."""
    request = Request(
        url,
        headers={
            "User-Agent": "BankEnv-live-check/1.0",
            "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1",
        },
    )
    try:
        with NO_REDIRECT_OPENER.open(request, timeout=timeout) as response:  # noqa: S310
            body = response.read(MAX_BODY_BYTES).decode("utf-8", errors="replace")
            return HttpObservation(
                requested_url=url,
                final_url=response.geturl(),
                status=int(response.getcode()),
                body=body,
                location=response.headers.get("Location"),
            )
    except HTTPError as exc:
        body = exc.read(MAX_BODY_BYTES).decode("utf-8", errors="replace")
        return HttpObservation(
            requested_url=url,
            final_url=exc.geturl(),
            status=int(exc.code),
            body=body,
            location=exc.headers.get("Location"),
        )


def _looks_access_controlled(observation: HttpObservation | None) -> bool:
    if observation is None:
        return False
    if observation.status in {401, 403}:
        return True
    observed_urls = [observation.final_url]
    if observation.location:
        observed_urls.append(observation.location)
    body = observation.body.lower()
    return any(
        marker in urlsplit(url).path.lower()
        for url in observed_urls
        for marker in AUTH_URL_MARKERS
    ) or any(
        marker in body for marker in AUTH_BODY_MARKERS
    )


def classify_observations(
    root: HttpObservation | None,
    health: HttpObservation | None,
) -> tuple[str, bool, bool, str]:
    """Return classification, reachable, login-gated, and an explanation."""
    if _looks_access_controlled(root):
        return (
            "login_gated_reachable",
            True,
            True,
            "The application endpoint responded with an authentication or access-control surface.",
        )
    if root is not None and 200 <= root.status < 400:
        return (
            "public_http_reachable",
            True,
            False,
            "The public application endpoint returned a successful HTTP response.",
        )
    if _looks_access_controlled(health):
        return (
            "login_gated_reachable",
            True,
            True,
            "The Streamlit health endpoint is protected by authentication or access control.",
        )
    if health is not None and health.status == 200 and health.body.strip().lower().startswith("ok"):
        return (
            "public_health_reachable",
            True,
            False,
            "The Streamlit health endpoint returned OK.",
        )
    return (
        "unavailable",
        False,
        False,
        "Neither the application endpoint nor the Streamlit health endpoint was reachable successfully.",
    )


def probe_once(
    app_url: str,
    *,
    timeout: float = 15.0,
    fetcher: Callable[[str, float], HttpObservation] = fetch_http,
) -> dict:
    root_url = normalize_app_url(app_url)
    observations: dict[str, HttpObservation | None] = {"root": None, "health": None}
    errors: dict[str, str] = {}
    for name, url in (("root", root_url), ("health", health_url(root_url))):
        try:
            observations[name] = fetcher(url, timeout)
        except (OSError, TimeoutError, URLError) as exc:
            # Do not serialize exception messages: they can echo a URL query or
            # proxy detail.  The exception class is sufficient for operations.
            errors[name] = type(exc).__name__

    classification, reachable, login_gated, detail = classify_observations(
        observations["root"], observations["health"]
    )
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "app_url": sanitize_url(root_url),
        "classification": classification,
        "reachable": reachable,
        "login_gated": login_gated,
        "detail": detail,
        "root": observations["root"].public_dict() if observations["root"] else None,
        "health": observations["health"].public_dict() if observations["health"] else None,
        "errors": errors,
    }


def run_check(
    app_url: str,
    *,
    attempts: int = 6,
    interval_seconds: float = 30.0,
    timeout: float = 15.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    last_report = None
    for attempt in range(1, attempts + 1):
        last_report = probe_once(app_url, timeout=timeout)
        last_report["attempt"] = attempt
        last_report["attempts_allowed"] = attempts
        if last_report["reachable"]:
            return last_report
        if attempt < attempts:
            sleep(interval_seconds)
    assert last_report is not None
    return last_report


def _append_github_metadata(report: dict) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as handle:
            handle.write(f"classification={report['classification']}\n")
            handle.write(f"reachable={str(report['reachable']).lower()}\n")
            handle.write(f"login_gated={str(report['login_gated']).lower()}\n")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        access = "login-gated/access-controlled" if report["login_gated"] else "public"
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write("## Streamlit reachability check\n\n")
            handle.write(f"- Classification: `{report['classification']}`\n")
            handle.write(f"- Access surface: {access}\n")
            handle.write(f"- Attempts used: {report['attempt']}\n")
            handle.write(f"- Detail: {report['detail']}\n\n")
            handle.write(
                "This check proves endpoint reachability only; it does not attest "
                "that a specific Git commit is deployed.\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Streamlit application URL")
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--initial-delay", type=float, default=0.0)
    parser.add_argument("--output", default="artifacts/live_app_check.json")
    args = parser.parse_args()

    if args.initial_delay > 0:
        time.sleep(args.initial_delay)
    report = run_check(
        args.url,
        attempts=args.attempts,
        interval_seconds=args.interval_seconds,
        timeout=args.timeout,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _append_github_metadata(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["reachable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
