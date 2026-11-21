"""Daily stress monitor: free, keyless, no human in the loop.

Pulls fast-moving public series that cover frontier and least-developed
economies (IMF SDMX monthly/quarterly releases plus daily official FX), turns
them into rule-based watch flags, and records which IMF releases carry new
country periods. Watch flags never change the governed risk score.

Run: python -m src.monitor [--out data/monitor]
"""
from __future__ import annotations

import argparse
import io
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.sources.sdmx import IMF_SDMX_BASE, _retrying_get

OUT = Path("data/monitor")
CURRENCIES = Path("data/reference/country_currency.csv")
SDMX_CSV = {"Accept": "application/vnd.sdmx.data+csv;version=2.0.0"}
FX_URLS = (
    "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{d}/v1/currencies/usd.min.json",
    "https://{d}.currency-api.pages.dev/v1/currencies/usd.min.json",
)
FX_HISTORY_DAYS = 400

# name: (agency, dataflow, key, frequency). Multi-indicator keys are summed.
IMF_SERIES = {
    "fx_official": ("IMF.STA", "ER", "*.XDC_USD.EOP_RT.M", "M"),
    "reserves_usd": ("IMF.STA", "IL", "*.RXF11_REVS.USD.M", "M"),
    "imf_credit_usd": ("IMF.STA", "FA", "*.CLIMF_USD.M", "M"),
    "deposits_lcy": ("IMF.STA", "MFS_DC", "*.ODCORP_L_F22_IBM+ODCORP_L_F29_IBM.XDC.M", "M"),
    "unrest_index": ("IMF.WHD", "RSUI", "*.EVENT.M", "M"),
    "npl_ratio": ("IMF.STA", "FSIC", "*.*.AQ12_CFSI_PT.Q", "Q"),
}
# Series whose new country periods mean the governed snapshot is behind.
CORE_RELEASES = ("npl_ratio", "deposits_lcy")

# ponytail: fixed heuristic thresholds, not validated early-warning cut-offs;
# calibrate against crisis labels before letting them influence any score.
RULES = {
    "fx_depreciation_90d": ("> 10% local-currency depreciation vs USD in 90 days", lambda v: v > 0.10),
    "fx_depreciation_12m": ("> 20% official depreciation vs USD in 12 months", lambda v: v > 0.20),
    "reserves_change_3m": ("> 15% fall in FX reserves in 3 months", lambda v: v < -0.15),
    "imf_credit_change_3m": ("New IMF money: credit outstanding up > 5% in 3 months", lambda v: v > 0.05),
    "deposits_change_3m": ("> 5% nominal fall in bank deposits in 3 months", lambda v: v < -0.05),
    "unrest_zscore": ("Social unrest > 2 s.d. above its own 2-year mean", lambda v: v > 2),
    "npl_change_4q": ("NPL ratio up > 3 percentage points in 4 quarters", lambda v: v > 3),
}


def _period_end(label: str) -> pd.Timestamp:
    text = str(label).replace("-M", "-")
    freq = "Q" if "-Q" in text else "M"
    return pd.Period(text, freq=freq).end_time.normalize()


def fetch_imf(name: str, today: date) -> pd.DataFrame:
    agency, flow, key, freq = IMF_SERIES[name]
    start = today - timedelta(days=3 * 365)
    start_label = f"{start.year}-Q{(start.month - 1) // 3 + 1}" if freq == "Q" else f"{start.year}-M{start.month:02d}"
    url = f"{IMF_SDMX_BASE}/data/dataflow/{agency}/{flow}/+/{key}"
    response = _retrying_get(
        url, headers=SDMX_CSV, params={"c[TIME_PERIOD]": f"ge:{start_label}"}, timeout=300
    )
    raw = pd.read_csv(io.StringIO(response.text), low_memory=False)
    raw = raw.dropna(subset=["TIME_PERIOD", "OBS_VALUE"])
    frame = pd.DataFrame({
        "country_code": raw.COUNTRY.astype(str),
        "period": raw.TIME_PERIOD.map(_period_end),
        "value": pd.to_numeric(raw.OBS_VALUE, errors="coerce"),
    }).dropna()
    return frame.groupby(["country_code", "period"], as_index=False).value.sum()


def _fx_day(day: date) -> dict | None:
    for template in FX_URLS:
        try:
            payload = _retrying_get(template.format(d=day.isoformat()), timeout=30, retries=2).json()
            return {k.upper(): v for k, v in payload["usd"].items()}
        except Exception:
            continue
    return None


def update_fx(path: Path, today: date) -> pd.DataFrame:
    """Append daily USD cross rates for mapped currencies; backfill on first run."""
    currencies = sorted(set(pd.read_csv(CURRENCIES).currency))
    history = pd.read_csv(path, parse_dates=["date"]) if path.exists() else pd.DataFrame(columns=["date"])
    have = set(history.date.dt.date) if len(history) else set()
    first = today - timedelta(days=FX_HISTORY_DAYS if not have else 7)
    rows = []
    for offset in range((today - first).days + 1):
        day = first + timedelta(days=offset)
        if day in have:
            continue
        rates = _fx_day(day)
        if rates:
            rows.append({"date": pd.Timestamp(day), **{c: rates.get(c) for c in currencies}})
    if rows:
        history = pd.concat([h for h in (history, pd.DataFrame(rows)) if len(h)], ignore_index=True)
    history = history.sort_values("date").drop_duplicates("date", keep="last")
    history = history[history.date >= pd.Timestamp(today - timedelta(days=FX_HISTORY_DAYS))]
    return history.reset_index(drop=True)


def _change(series: pd.DataFrame, months: int, diff: bool = False) -> pd.DataFrame:
    """Latest value vs the value `months` earlier, per country (ratio or difference)."""
    out = []
    for country, group in series.sort_values("period").groupby("country_code"):
        latest = group.iloc[-1]
        base = group[group.period <= latest.period - pd.DateOffset(months=months) + pd.Timedelta(days=3)]
        if base.empty or (not diff and base.iloc[-1].value == 0):
            continue
        old = base.iloc[-1].value
        out.append((country, latest.period, latest.value - old if diff else latest.value / old - 1))
    return pd.DataFrame(out, columns=["country_code", "period", "value"])


def compute_signals(imf: dict[str, pd.DataFrame], fx: pd.DataFrame, today: date) -> pd.DataFrame:
    parts = []

    def add(signal, frame):
        if len(frame):
            parts.append(frame.assign(signal=signal))

    mapping = pd.read_csv(CURRENCIES)
    if len(fx) > 1:
        fx = fx.set_index("date").sort_index().astype(float)
        latest = fx.iloc[-1]
        base = fx[fx.index <= fx.index[-1] - pd.Timedelta(days=90)]
        if len(base):
            change = (latest / base.iloc[-1] - 1).rename("value")
            frame = mapping.join(change, on="currency").dropna(subset=["value"])
            add("fx_depreciation_90d", frame.assign(period=fx.index[-1])[["country_code", "period", "value"]])
    for name, signal, months in (
        ("fx_official", "fx_depreciation_12m", 12),
        ("reserves_usd", "reserves_change_3m", 3),
        ("imf_credit_usd", "imf_credit_change_3m", 3),
        ("deposits_lcy", "deposits_change_3m", 3),
    ):
        if len(imf.get(name, [])):
            add(signal, _change(imf[name], months))
    if len(imf.get("unrest_index", [])):
        rows = []
        for country, group in imf["unrest_index"].sort_values("period").groupby("country_code"):
            window = group.tail(24)
            sd = window.value.std()
            if len(window) >= 12 and sd > 0:
                rows.append((country, window.period.iloc[-1], (window.value.iloc[-1] - window.value.mean()) / sd))
        add("unrest_zscore", pd.DataFrame(rows, columns=["country_code", "period", "value"]))
    if len(imf.get("npl_ratio", [])):
        add("npl_change_4q", _change(imf["npl_ratio"], 12, diff=True))
    if not parts:
        return pd.DataFrame(columns=["country_code", "signal", "value", "period", "age_days", "flag", "rule"])
    signals = pd.concat(parts, ignore_index=True)
    # Economies only (drop IMF group aggregates such as G759).
    signals = signals[signals.country_code.str.fullmatch(r"[A-Z]{3}")]
    # ponytail: a >10x FX move is treated as a redenomination/data break, not
    # stress; a per-currency break log is the upgrade if hyperinflation matters.
    signals = signals[~(signals.signal.str.startswith("fx_") & (signals.value > 9))]
    signals["age_days"] = (pd.Timestamp(today) - pd.to_datetime(signals.period)).dt.days.clip(lower=0)
    signals["flag"] = [RULES[s][1](v) for s, v in zip(signals.signal, signals.value)]
    signals["rule"] = signals.signal.map(lambda s: RULES[s][0])
    signals["period"] = pd.to_datetime(signals.period).dt.date
    signals["value"] = signals.value.round(4)
    return signals.sort_values(["country_code", "signal"])[
        ["country_code", "signal", "value", "period", "age_days", "flag", "rule"]
    ]


def new_releases(imf: dict[str, pd.DataFrame], previous: dict) -> dict[str, list[str]]:
    """Countries whose latest period advanced since the previous run."""
    fresh = {}
    for name in CORE_RELEASES:
        if not len(imf.get(name, [])):
            continue
        latest = imf[name].groupby("country_code").period.max().dt.date.astype(str)
        before = previous.get("latest_periods", {}).get(name, {})
        if before:
            fresh[name] = sorted(c for c, p in latest.items() if p > before.get(c, ""))
    return fresh


def run(out: Path = OUT, today: date | None = None) -> dict:
    today = today or datetime.now(timezone.utc).date()
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}

    imf, errors = {}, {}
    for name in IMF_SERIES:
        try:
            imf[name] = fetch_imf(name, today)
        except Exception as error:  # one failed dataflow must not blank the others
            errors[name] = str(error)[:300]
    fx = update_fx(out / "fx_daily.csv", today)
    fx.to_csv(out / "fx_daily.csv", index=False, date_format="%Y-%m-%d")

    signals = compute_signals(imf, fx, today)
    signals.to_csv(out / "signals.csv", index=False)

    fresh = new_releases(imf, previous)
    status = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "errors": errors,
        "rows": {name: len(frame) for name, frame in imf.items()},
        "countries_flagged": int(signals.loc[signals.flag, "country_code"].nunique()),
        "new_core_releases": fresh,
        "refresh_needed": any(fresh.values()),
        "latest_periods": {
            name: imf[name].groupby("country_code").period.max().dt.date.astype(str).to_dict()
            if name in imf else previous.get("latest_periods", {}).get(name, {})
            for name in IMF_SERIES
        },
        "fx_last_date": str(fx.date.max().date()) if len(fx) else None,
    }
    status_path.write_text(json.dumps(status, indent=1, sort_keys=True) + "\n")
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    result = run(parser.parse_args().out)
    print(json.dumps({k: v for k, v in result.items() if k != "latest_periods"}, indent=1))
