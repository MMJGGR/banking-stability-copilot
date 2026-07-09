"""SDMX 3.0 retrieval clients for the official IMF and World Bank APIs.

Endpoints, key structures, and timings were verified live on 2026-07-09; see
section 9.5 of BANKING_COPILOT_COMPREHENSIVE_REMEDIATION_AND_IMPLEMENTATION_PLAN.md.

The IMF API is public (no key), returns long-format CSV
(one row per country-indicator-period), and supports time filtering via
``c[TIME_PERIOD]=ge:<period>``. Wildcards use ``*`` per dimension.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time

import pandas as pd
import requests

from src.sources.base import SourceResult, SourceUnavailableError

IMF_SDMX_BASE = "https://api.imf.org/external/sdmx/3.0"
WORLD_BANK_BASE = "https://api.worldbank.org/v2"
SDMX_CSV_ACCEPT = "application/vnd.sdmx.data+csv"


def _parse_sdmx_period(value) -> pd.Timestamp:
    label = str(value).strip().upper()
    if label.endswith(".0") and label[:-2].isdigit():
        label = label[:-2]
    if not label or label == "NAN":
        return pd.NaT
    if len(label) == 4 and label.isdigit():
        return pd.Timestamp(f"{label}-12-31")
    if "-Q" in label:
        year, quarter = label.split("-Q", 1)
        if year.isdigit() and quarter[:1].isdigit():
            return pd.Period(
                year=int(year),
                quarter=int(quarter[:1]),
                freq="Q",
            ).end_time.normalize()
    if "-M" in label:
        year, month = label.split("-M", 1)
        if year.isdigit() and month[:2].isdigit():
            return pd.Period(
                year=int(year),
                month=int(month[:2]),
                freq="M",
            ).end_time.normalize()
    return pd.NaT


def _filter_csv_by_period(source: Path, destination: Path, start_period=None,
                          end_period=None, chunksize=250_000) -> None:
    """Client-side period filtering because the IMF API ignores period params."""
    start_ts = _parse_sdmx_period(start_period) if start_period else None
    end_ts = _parse_sdmx_period(end_period) if end_period else None
    first = True
    for chunk in pd.read_csv(source, chunksize=chunksize, low_memory=False):
        periods = chunk["TIME_PERIOD"].map(_parse_sdmx_period)
        mask = periods.notna()
        if start_ts is not None:
            mask &= periods >= start_ts
        if end_ts is not None:
            mask &= periods <= end_ts
        chunk.loc[mask].to_csv(
            destination,
            mode="w" if first else "a",
            header=first,
            index=False,
        )
        first = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _retrying_get(url, headers=None, params=None, timeout=300, retries=3,
                  backoff_seconds=5, stream=False):
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=timeout,
                stream=stream,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(backoff_seconds * (attempt + 1))
    raise SourceUnavailableError(f"GET {url} failed after {retries} attempts: {last_error}")


@dataclass
class SdmxDataflowSource:
    """One IMF dataflow retrievable over SDMX 3.0.

    ``dimensions`` documents the discovered key order (excluding TIME_PERIOD)
    so wildcard keys are built with the correct number of segments.
    """

    name: str
    agency: str
    dataflow_id: str
    dimensions: tuple

    @property
    def wildcard_key(self) -> str:
        return ".".join("*" for _ in self.dimensions)

    def data_url(self) -> str:
        return (
            f"{IMF_SDMX_BASE}/data/dataflow/"
            f"{self.agency}/{self.dataflow_id}/+/{self.wildcard_key}"
        )

    def structure_url(self) -> str:
        return (
            f"{IMF_SDMX_BASE}/structure/dataflow/"
            f"{self.agency}/{self.dataflow_id}/+"
        )

    def check_version(self, timeout=60) -> dict:
        """Return the latest dataflow version from the structure endpoint."""
        response = _retrying_get(
            self.structure_url(),
            headers={"Accept": "application/vnd.sdmx.structure+json"},
            timeout=timeout,
        )
        payload = response.json()
        flows = payload.get("data", {}).get("dataflows", [])
        structures = payload.get("data", {}).get("dataStructures", [])
        return {
            "source": self.name,
            "agency": self.agency,
            "dataflow_id": self.dataflow_id,
            "dataflow_version": flows[0].get("version") if flows else None,
            "structure_version": (
                structures[0].get("version") if structures else None
            ),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    def fetch(self, destination_dir, start_period=None, end_period=None,
              timeout=480, retries=3) -> SourceResult:
        """Download long-format CSV observations to ``destination_dir``.

        The IMF endpoint currently ignores standard period filter parameters
        for this API shape, so period filtering is performed client-side after
        download when ``start_period`` or ``end_period`` is supplied.
        """
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        response = _retrying_get(
            self.data_url(),
            headers={"Accept": SDMX_CSV_ACCEPT},
            timeout=timeout,
            retries=retries,
            stream=True,
        )

        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        if start_period and end_period:
            suffix = f"_from_{start_period}_to_{end_period}"
        elif start_period:
            suffix = f"_from_{start_period}"
        elif end_period:
            suffix = f"_to_{end_period}"
        else:
            suffix = "_full"
        destination = destination_dir / f"sdmx_{self.name}{suffix}_{stamp}.csv"
        raw_destination = destination
        if start_period or end_period:
            raw_destination = destination_dir / f".raw_{destination.name}"
        with raw_destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
        if raw_destination != destination:
            destination.unlink(missing_ok=True)
            _filter_csv_by_period(
                raw_destination,
                destination,
                start_period=start_period,
                end_period=end_period,
            )
            raw_destination.unlink(missing_ok=True)

        version = self._version_from_csv(destination)
        return SourceResult(
            source=self.name,
            retrieval_method="sdmx_api",
            path=str(destination.resolve()),
            bytes=destination.stat().st_size,
            sha256=_sha256_file(destination),
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            remote_version=version,
        )

    @staticmethod
    def _version_from_csv(path: Path):
        """Read the dataflow version from the first data row's STRUCTURE_ID."""
        try:
            with path.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                first = next(reader, None)
                if first:
                    return first.get("STRUCTURE_ID")
        except (OSError, csv.Error):
            return None
        return None


@dataclass
class WorldBankIndicatorSource:
    """WGI governance scores from the World Bank indicators API.

    The Worldwide Governance Indicators live in source database 3; the
    ``GOV_WGI_*.SC`` series are the 0-100 governance scores used by the model
    (same measure as "Governance score (0-100)" in the published workbook).
    """

    name: str = "WGI"
    source_id: int = 3
    indicators: tuple = (
        ("GOV_WGI_VA.SC", "voice_accountability"),
        ("GOV_WGI_PV.SC", "political_stability"),
        ("GOV_WGI_GE.SC", "govt_effectiveness"),
        ("GOV_WGI_RQ.SC", "regulatory_quality"),
        ("GOV_WGI_RL.SC", "rule_of_law"),
        ("GOV_WGI_CC.SC", "control_corruption"),
    )

    def indicator_url(self, indicator_code: str) -> str:
        return (
            f"{WORLD_BANK_BASE}/country/all/indicator/{indicator_code}"
        )

    def fetch(self, destination_dir, start_year=None, timeout=120,
              retries=3) -> SourceResult:
        """Download all six governance-score series to one long-format CSV."""
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for indicator_code, feature_name in self.indicators:
            page = 1
            while True:
                params = {
                    "format": "json",
                    "source": self.source_id,
                    "per_page": 2000,
                    "page": page,
                }
                if start_year:
                    params["date"] = f"{start_year}:{datetime.now().year}"
                response = _retrying_get(
                    self.indicator_url(indicator_code),
                    params=params,
                    timeout=timeout,
                    retries=retries,
                )
                payload = response.json()
                if not isinstance(payload, list) or len(payload) < 2:
                    raise SourceUnavailableError(
                        f"WGI response for {indicator_code} is not paged data: "
                        f"{str(payload)[:200]}"
                    )
                meta, records = payload[0], payload[1] or []
                for record in records:
                    if record.get("value") is None:
                        continue
                    rows.append(
                        {
                            "indicator_code": indicator_code,
                            "feature_name": feature_name,
                            "country_code": record.get("countryiso3code"),
                            "country_name": (record.get("country") or {}).get("value"),
                            "year": record.get("date"),
                            "value": record.get("value"),
                        }
                    )
                if page >= int(meta.get("pages", 1)):
                    break
                page += 1

        if not rows:
            raise SourceUnavailableError("WGI retrieval returned no observations")

        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        destination = destination_dir / f"worldbank_WGI_{stamp}.csv"
        fieldnames = [
            "indicator_code", "feature_name", "country_code",
            "country_name", "year", "value",
        ]
        with destination.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return SourceResult(
            source=self.name,
            retrieval_method="worldbank_api",
            path=str(destination.resolve()),
            bytes=destination.stat().st_size,
            sha256=_sha256_file(destination),
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            remote_version=None,
        )

    def check_version(self, timeout=60) -> dict:
        """Return the most recent year with data as a freshness signal."""
        current_year = datetime.now().year
        latest_year = None
        for indicator_code, _ in self.indicators:
            response = _retrying_get(
                self.indicator_url(indicator_code),
                params={
                    "format": "json",
                    "source": self.source_id,
                    "per_page": 2000,
                    "date": f"2000:{current_year}",
                },
                timeout=timeout,
            )
            payload = response.json()
            if isinstance(payload, list) and len(payload) > 1:
                years = [
                    int(record.get("date"))
                    for record in (payload[1] or [])
                    if record.get("value") is not None
                    and str(record.get("date", "")).isdigit()
                ]
                if years:
                    latest_year = max(latest_year or 0, max(years))
        return {
            "source": self.name,
            "latest_year_with_data": str(latest_year) if latest_year else None,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }


def build_sdmx_sources() -> dict:
    """Registry of official API sources with verified key structures."""
    return {
        "WEO": SdmxDataflowSource(
            name="WEO",
            agency="IMF.RES",
            dataflow_id="WEO",
            dimensions=("COUNTRY", "INDICATOR", "FREQUENCY"),
        ),
        "FSIC": SdmxDataflowSource(
            name="FSIC",
            agency="IMF.STA",
            dataflow_id="FSIC",
            dimensions=("COUNTRY", "SECTOR", "INDICATOR", "FREQUENCY"),
        ),
        "MFS": SdmxDataflowSource(
            name="MFS",
            agency="IMF.STA",
            dataflow_id="MFS_DC",
            dimensions=(
                "COUNTRY", "INDICATOR", "TYPE_OF_TRANSFORMATION", "FREQUENCY",
            ),
        ),
        "FSIBSIS": SdmxDataflowSource(
            name="FSIBSIS",
            agency="IMF.STA",
            dataflow_id="FSIBSIS",
            dimensions=("COUNTRY", "SECTOR", "INDICATOR", "FREQUENCY"),
        ),
        "WGI": WorldBankIndicatorSource(),
    }
