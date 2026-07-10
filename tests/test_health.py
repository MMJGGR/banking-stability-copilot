from datetime import datetime, timezone

from src.health import build_health_report, parse_period


NOW = datetime(2026, 7, 10, tzinfo=timezone.utc)


def test_parse_period_formats():
    assert parse_period("2026-05-31").month == 5
    assert parse_period("2026-Q1").month == 3
    assert parse_period("2026-M04").month == 4
    assert parse_period("2026-04").month == 4
    assert parse_period("2024").month == 12
    assert parse_period(None) is None
    assert parse_period("not a date") is None


def _manifest(latest="2026-05-31"):
    return {
        "snapshot_id": "2026-06-30",
        "snapshot_status": "verified",
        "generated_at": "2026-07-09T21:35:53+00:00",
        "sources": {"MFS": {"latest_observation": latest}},
    }


def test_healthy_active_manifest_reports_ok():
    report = build_health_report(_manifest(), {"mode": "active"}, now=NOW)
    assert report["overall"] == "ok"
    assert report["serving_mode"] == "active"
    assert report["sources"][0]["status"] == "ok"


def test_stale_source_is_flagged():
    report = build_health_report(_manifest("2024-01-31"), {"mode": "active"}, now=NOW)
    assert report["overall"] == "stale"
    assert report["sources"][0]["status"] == "stale"
    assert any("SLA" in note for note in report["notes"])


def test_fallback_mode_reports_degraded():
    report = build_health_report(
        _manifest(),
        {
            "mode": "fallback",
            "fallback_snapshot": "2026-06-30-official-api",
            "active_error": "ValueError: checksum mismatch",
        },
        now=NOW,
    )
    assert report["overall"] == "degraded"
    assert report["fallback_snapshot"] == "2026-06-30-official-api"
    assert any("last-known-good" in note for note in report["notes"])


def test_missing_manifest_reports_unknown():
    report = build_health_report({}, {"mode": "active"}, now=NOW)
    assert report["overall"] == "unknown"
