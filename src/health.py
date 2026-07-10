"""Serving health, freshness, and degraded-mode status from the manifest.

The dashboard renders this report so users can tell whether the data are
stale, whether the app is serving a fallback artifact, and when the snapshot
was generated. Thresholds are the proposed source-staleness SLAs from
docs/GOVERNANCE.md (pending owner approval; see section 21.5 items 30-31).
"""

from datetime import datetime, timezone
import re

# Maximum acceptable age, in days, of each source's latest observation at
# serving time. Derived from the section 9.4 cadence table plus typical
# publication lag; owner approval tracked in docs/GOVERNANCE.md.
SOURCE_FRESHNESS_SLA_DAYS = {
    "WEO": 420,       # two vintages per year, annual observations
    "FSIC": 240,      # rolling monthly/quarterly reporting with country lag
    "MFS": 240,       # rolling monthly reporting with country lag
    "FSIBSIS": 365,   # quarterly/annual balance-sheet reporting
    "WGI": 900,       # annual release with ~18-month reference lag
}

# The manifest itself should be regenerated at least quarterly.
SNAPSHOT_MAX_AGE_DAYS = 190


def parse_period(value):
    """Parse manifest period labels (YYYY, YYYY-MM, YYYY-MM-DD, YYYY-Qn,
    YYYY-Mnn) into a timezone-aware datetime at the period end."""
    if value is None:
        return None
    text = str(value).strip()
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return datetime(int(match[1]), int(match[2]), int(match[3]), tzinfo=timezone.utc)
    match = re.fullmatch(r"(\d{4})-Q([1-4])", text, flags=re.IGNORECASE)
    if match:
        return datetime(int(match[1]), int(match[2]) * 3, 28, tzinfo=timezone.utc)
    match = re.fullmatch(r"(\d{4})-M?(\d{1,2})", text, flags=re.IGNORECASE)
    if match and 1 <= int(match[2]) <= 12:
        return datetime(int(match[1]), int(match[2]), 28, tzinfo=timezone.utc)
    match = re.fullmatch(r"(\d{4})", text)
    if match:
        return datetime(int(match[1]), 12, 31, tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _age_days(moment, now):
    return None if moment is None else max(0, int((now - moment).days))


def build_health_report(manifest: dict, serving_status: dict = None, now=None) -> dict:
    """Summarize serving health from the manifest and loader status."""
    now = now or datetime.now(timezone.utc)
    serving_status = serving_status or {"mode": "active"}
    manifest = manifest or {}

    generated = parse_period(manifest.get("generated_at"))
    generated_age = _age_days(generated, now)

    sources = []
    any_stale = False
    for name, details in sorted(manifest.get("sources", {}).items()):
        latest = details.get("latest_observation") or details.get("latest_period_label")
        age = _age_days(parse_period(latest), now)
        sla = SOURCE_FRESHNESS_SLA_DAYS.get(name)
        if age is None:
            status = "unknown"
        elif sla is not None and age > sla:
            status = "stale"
            any_stale = True
        else:
            status = "ok"
        sources.append({
            "source": name,
            "latest_observation": latest,
            "age_days": age,
            "sla_days": sla,
            "status": status,
        })

    notes = []
    fallback_active = serving_status.get("mode") == "fallback"
    if fallback_active:
        notes.append(
            "Serving the archived last-known-good bundle "
            f"'{serving_status.get('fallback_snapshot')}' because the active "
            f"artifact failed to load: {serving_status.get('active_error')}"
        )
    if not manifest:
        notes.append("No serving manifest is available; freshness is unknown.")
    if generated_age is not None and generated_age > SNAPSHOT_MAX_AGE_DAYS:
        any_stale = True
        notes.append(
            f"Snapshot was generated {generated_age} days ago, beyond the "
            f"{SNAPSHOT_MAX_AGE_DAYS}-day refresh expectation."
        )
    for source in sources:
        if source["status"] == "stale":
            notes.append(
                f"{source['source']} latest observation "
                f"({source['latest_observation']}) is {source['age_days']} days "
                f"old, beyond its {source['sla_days']}-day SLA."
            )

    if fallback_active:
        overall = "degraded"
    elif not manifest:
        overall = "unknown"
    elif any_stale:
        overall = "stale"
    else:
        overall = "ok"

    return {
        "overall": overall,
        "serving_mode": serving_status.get("mode", "active"),
        "fallback_snapshot": serving_status.get("fallback_snapshot"),
        "snapshot_id": manifest.get("snapshot_id"),
        "snapshot_status": manifest.get("snapshot_status"),
        "source_mode": manifest.get("source_mode"),
        "generated_at": manifest.get("generated_at"),
        "generated_age_days": generated_age,
        "sources": sources,
        "notes": notes,
    }
