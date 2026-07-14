"""Official BIS financial-history inputs for crisis-model research.

This module is deliberately a build-time adapter.  It downloads and normalises
official BIS bulk files only when one of its functions is called; importing it
from the Streamlit application performs no network access.

The output follows the long-form contract accepted by ``crisis_panel`` while
retaining source-vintage and series-key metadata for audit.  All published BIS
observations remain direct observations.  In particular,
``bis_private_credit_to_gdp_gap`` is emitted only from the official
``WS_CREDIT_GAP`` data set; this module does not relabel a locally calculated
credit deviation as the BIS credit gap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Iterable, Mapping
import zipfile

import numpy as np
import pandas as pd
import pycountry
import requests

from src.config import CACHE_DIR


DEFAULT_CACHE_PATH = Path(CACHE_DIR) / "BIS_FINANCIAL_cache.parquet"
DEFAULT_MANIFEST_PATH = Path(CACHE_DIR) / "BIS_FINANCIAL_manifest.json"


@dataclass(frozen=True)
class BisBulkSpec:
    """One official BIS bulk-download data set."""

    name: str
    dataset_id: str
    version: str
    url: str
    archive_name: str
    required_dimensions: tuple[str, ...]


BIS_BULK_SPECS: dict[str, BisBulkSpec] = {
    "total_credit": BisBulkSpec(
        name="total_credit",
        dataset_id="WS_TC",
        version="2.0",
        url="https://data.bis.org/static/bulk/WS_TC_csv_flat.zip",
        archive_name="WS_TC_csv_flat.zip",
        required_dimensions=(
            "FREQ",
            "BORROWERS_CTY",
            "TC_BORROWERS",
            "TC_LENDERS",
            "VALUATION",
            "UNIT_TYPE",
            "TC_ADJUST",
            "TIME_PERIOD",
            "OBS_VALUE",
            "OBS_STATUS",
        ),
    ),
    "credit_gap": BisBulkSpec(
        name="credit_gap",
        dataset_id="WS_CREDIT_GAP",
        version="1.0",
        url="https://data.bis.org/static/bulk/WS_CREDIT_GAP_csv_flat.zip",
        archive_name="WS_CREDIT_GAP_csv_flat.zip",
        required_dimensions=(
            "FREQ",
            "BORROWERS_CTY",
            "TC_BORROWERS",
            "TC_LENDERS",
            "CG_DTYPE",
            "TIME_PERIOD",
            "OBS_VALUE",
            "OBS_STATUS",
        ),
    ),
    "debt_service": BisBulkSpec(
        name="debt_service",
        dataset_id="WS_DSR",
        version="1.0",
        url="https://data.bis.org/static/bulk/WS_DSR_csv_flat.zip",
        archive_name="WS_DSR_csv_flat.zip",
        required_dimensions=(
            "FREQ",
            "BORROWERS_CTY",
            "DSR_BORROWERS",
            "TIME_PERIOD",
            "OBS_VALUE",
            "OBS_STATUS",
        ),
    ),
    "selected_property_prices": BisBulkSpec(
        name="selected_property_prices",
        dataset_id="WS_SPP",
        version="1.0",
        url="https://data.bis.org/static/bulk/WS_SPP_csv_flat.zip",
        archive_name="WS_SPP_csv_flat.zip",
        required_dimensions=(
            "FREQ",
            "REF_AREA",
            "VALUE",
            "UNIT_MEASURE",
            "TIME_PERIOD",
            "OBS_VALUE",
            "OBS_STATUS",
        ),
    ),
}


@dataclass(frozen=True)
class BisDownload:
    """Auditable metadata for one downloaded or supplied bulk file."""

    dataset: str
    dataset_id: str
    path: str
    source_url: str
    retrieved_at: str
    source_vintage: str | None
    etag: str | None
    bytes: int
    sha256: str
    retrieval_method: str = "bulk_download"

    def to_dict(self) -> dict[str, str | int | None]:
        return asdict(self)


FEATURE_METADATA: dict[str, dict[str, str | int]] = {
    "bis_private_credit_gdp": {
        "label": "BIS total credit to the private non-financial sector (% GDP)",
        "family": "credit_cycle",
        "risk_direction": 1,
    },
    "bis_bank_credit_gdp": {
        "label": "BIS bank credit to the private non-financial sector (% GDP)",
        "family": "credit_cycle",
        "risk_direction": 1,
    },
    "bis_private_credit_to_gdp_gap": {
        "label": "BIS private-sector credit-to-GDP gap (percentage points)",
        "family": "credit_cycle",
        "risk_direction": 1,
    },
    "bis_private_debt_service_ratio": {
        "label": "BIS private non-financial sector debt-service ratio (%)",
        "family": "debt_service_pressure",
        "risk_direction": 1,
    },
    "bis_household_debt_service_ratio": {
        "label": "BIS household debt-service ratio (%)",
        "family": "debt_service_pressure",
        "risk_direction": 1,
    },
    "bis_corporate_debt_service_ratio": {
        "label": "BIS non-financial corporate debt-service ratio (%)",
        "family": "debt_service_pressure",
        "risk_direction": 1,
    },
    "bis_real_house_price_growth_yoy": {
        "label": "BIS selected real residential property-price growth (YoY %)",
        "family": "property_cycle",
        "risk_direction": 1,
    },
    "bis_nominal_house_price_growth_yoy": {
        "label": "BIS selected nominal residential property-price growth (YoY %)",
        "family": "property_cycle",
        "risk_direction": 1,
    },
    "bis_real_house_price_index": {
        "label": "BIS selected real residential property-price index",
        "family": "property_cycle",
        "risk_direction": 1,
    },
    "bis_nominal_house_price_index": {
        "label": "BIS selected nominal residential property-price index",
        "family": "property_cycle",
        "risk_direction": 1,
    },
}


OUTPUT_COLUMNS = (
    "country_code",
    "country_name",
    "indicator_code",
    "indicator_name",
    "feature",
    "family",
    "risk_direction",
    "period",
    "year",
    "value",
    "frequency",
    "unit",
    "observation_status",
    "source_observation_status",
    "is_direct",
    "source_id",
    "source_dataset_id",
    "source_dataset_version",
    "source_structure_id",
    "source_series_key",
    "source_url",
    "source_vintage",
    "retrieved_at",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _dimension_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Map SDMX dimension ids to their labelled flat-CSV columns."""

    result: dict[str, str] = {}
    for column in frame.columns:
        dimension = str(column).split(":", 1)[0].strip().upper()
        result.setdefault(dimension, str(column))
    return result


def _require_dimensions(frame: pd.DataFrame, spec: BisBulkSpec) -> dict[str, str]:
    columns = _dimension_columns(frame)
    missing = sorted(set(spec.required_dimensions).difference(columns))
    if missing:
        raise ValueError(
            f"BIS {spec.dataset_id} bulk file is missing dimensions: {missing}"
        )
    return columns


def _code(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.split(":", n=1)
        .str[0]
        .str.strip()
        .str.upper()
    )


def _label(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return text.str.split(":", n=1).str[1].fillna(text).str.strip()


def _iso2_to_iso3(value: object) -> str | None:
    code = str(value or "").strip().upper()
    if len(code) != 2 or not code.isalpha():
        return None
    country = pycountry.countries.get(alpha_2=code)
    return country.alpha_3 if country is not None else None


def _parse_period(series: pd.Series) -> pd.Series:
    """Parse BIS quarterly/monthly/annual SDMX periods to period-end dates."""

    text = series.astype("string").str.strip()
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    quarterly = text.str.fullmatch(r"\d{4}-Q[1-4]", na=False)
    if quarterly.any():
        result.loc[quarterly] = (
            pd.PeriodIndex(text.loc[quarterly], freq="Q")
            .to_timestamp(how="end")
            .normalize()
            .to_numpy()
        )

    monthly = text.str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", na=False)
    if monthly.any():
        result.loc[monthly] = (
            pd.PeriodIndex(text.loc[monthly], freq="M")
            .to_timestamp(how="end")
            .normalize()
            .to_numpy()
        )

    annual = text.str.fullmatch(r"\d{4}", na=False)
    if annual.any():
        result.loc[annual] = pd.to_datetime(
            text.loc[annual] + "-12-31", errors="coerce"
        ).to_numpy()
    return result


def _normalised_status(series: pd.Series) -> pd.Series:
    codes = _code(series)
    # A is a normal observation; B marks a series break but the value remains
    # reported.  Provisional/estimated observations are retained explicitly.
    return codes.map(
        {
            "A": "actual",
            "B": "actual",
            "E": "estimate",
            "P": "estimate",
        }
    ).fillna("unknown")


def _empty_observations() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _metadata_value(
    metadata: BisDownload | Mapping[str, object] | None,
    name: str,
    default: object = None,
) -> object:
    if metadata is None:
        return default
    if isinstance(metadata, BisDownload):
        return getattr(metadata, name, default)
    return metadata.get(name, default)


def _finalise(
    selected: pd.DataFrame,
    *,
    dataset_name: str,
    metadata: BisDownload | Mapping[str, object] | None,
) -> pd.DataFrame:
    spec = BIS_BULK_SPECS[dataset_name]
    if selected.empty:
        return _empty_observations()

    result = selected.copy()
    result["country_code"] = result["country_alpha2"].map(_iso2_to_iso3)
    result["period"] = _parse_period(result["source_period"])
    result["value"] = pd.to_numeric(result["source_value"], errors="coerce")
    result = result.dropna(subset=["country_code", "period", "value", "feature"])
    if result.empty:
        return _empty_observations()

    result["country_name"] = result["country_label"]
    result["indicator_code"] = result["feature"]
    result["indicator_name"] = result["feature"].map(
        {feature: values["label"] for feature, values in FEATURE_METADATA.items()}
    )
    result["family"] = result["feature"].map(
        {feature: values["family"] for feature, values in FEATURE_METADATA.items()}
    )
    result["risk_direction"] = result["feature"].map(
        {
            feature: int(values["risk_direction"])
            for feature, values in FEATURE_METADATA.items()
        }
    )
    result["year"] = result["period"].dt.year.astype("int16")
    result["frequency"] = "Q"
    result["observation_status"] = _normalised_status(result["source_status"])
    result["source_observation_status"] = result["source_status"].astype("string")
    result["is_direct"] = True
    result["source_id"] = "BIS"
    result["source_dataset_id"] = spec.dataset_id
    result["source_dataset_version"] = spec.version
    result["source_structure_id"] = result.get(
        "source_structure_id", pd.Series(pd.NA, index=result.index)
    )
    result["source_url"] = str(
        _metadata_value(metadata, "source_url", spec.url)
    )
    result["source_vintage"] = _metadata_value(metadata, "source_vintage")
    result["retrieved_at"] = str(
        _metadata_value(metadata, "retrieved_at", _utc_now())
    )

    keys = ["country_code", "indicator_code", "period"]
    conflicting = (
        result.groupby(keys, dropna=False)["value"].nunique(dropna=True).gt(1)
    )
    if conflicting.any():
        examples = list(conflicting[conflicting].index[:5])
        raise ValueError(
            f"BIS {spec.dataset_id} contains conflicting selected observations: "
            f"{examples}"
        )
    result = result.drop_duplicates(keys, keep="last")
    return (
        result.loc[:, OUTPUT_COLUMNS]
        .sort_values(["country_code", "indicator_code", "period"])
        .reset_index(drop=True)
    )


def normalise_bis_total_credit(
    frame: pd.DataFrame,
    *,
    metadata: BisDownload | Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Select comparable private credit-to-GDP series from ``WS_TC``.

    Two official reported series are retained: credit from all sectors and
    credit supplied by domestic banks.  Both use market valuation, percentage
    of GDP, and the BIS break-adjusted series.
    """

    spec = BIS_BULK_SPECS["total_credit"]
    columns = _require_dimensions(frame, spec)
    freq = _code(frame[columns["FREQ"]])
    country = _code(frame[columns["BORROWERS_CTY"]])
    borrower = _code(frame[columns["TC_BORROWERS"]])
    lender = _code(frame[columns["TC_LENDERS"]])
    valuation = _code(frame[columns["VALUATION"]])
    unit_type = _code(frame[columns["UNIT_TYPE"]])
    adjustment = _code(frame[columns["TC_ADJUST"]])
    selected_mask = (
        freq.eq("Q")
        & borrower.eq("P")
        & lender.isin(["A", "B"])
        & valuation.eq("M")
        & unit_type.eq("770")
        & adjustment.eq("A")
    )
    selected = frame.loc[selected_mask].copy()
    if selected.empty:
        return _empty_observations()

    selected_lender = lender.loc[selected.index]
    selected["feature"] = selected_lender.map(
        {"A": "bis_private_credit_gdp", "B": "bis_bank_credit_gdp"}
    )
    selected["country_alpha2"] = country.loc[selected.index]
    selected["country_label"] = _label(
        selected[columns["BORROWERS_CTY"]]
    )
    selected["source_period"] = selected[columns["TIME_PERIOD"]]
    selected["source_value"] = selected[columns["OBS_VALUE"]]
    selected["source_status"] = selected[columns["OBS_STATUS"]]
    selected["unit"] = "% GDP"
    selected["source_series_key"] = (
        "Q."
        + selected["country_alpha2"].astype("string")
        + ".P."
        + selected_lender.astype("string")
        + ".M.770.A"
    )
    structure = _dimension_columns(frame).get("STRUCTURE_ID")
    selected["source_structure_id"] = (
        selected[structure] if structure else f"BIS:{spec.dataset_id}({spec.version})"
    )
    return _finalise(selected, dataset_name="total_credit", metadata=metadata)


def normalise_bis_credit_gap(
    frame: pd.DataFrame,
    *,
    metadata: BisDownload | Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Select the official BIS private-sector credit-to-GDP gap.

    ``CG_DTYPE=C`` is the BIS published actual-minus-HP-trend gap.  No locally
    derived deviation is accepted by this normaliser.
    """

    spec = BIS_BULK_SPECS["credit_gap"]
    columns = _require_dimensions(frame, spec)
    freq = _code(frame[columns["FREQ"]])
    country = _code(frame[columns["BORROWERS_CTY"]])
    borrower = _code(frame[columns["TC_BORROWERS"]])
    lender = _code(frame[columns["TC_LENDERS"]])
    data_type = _code(frame[columns["CG_DTYPE"]])
    selected_mask = (
        freq.eq("Q")
        & borrower.eq("P")
        & lender.eq("A")
        & data_type.eq("C")
    )
    selected = frame.loc[selected_mask].copy()
    if selected.empty:
        return _empty_observations()

    selected["feature"] = "bis_private_credit_to_gdp_gap"
    selected["country_alpha2"] = country.loc[selected.index]
    selected["country_label"] = _label(
        selected[columns["BORROWERS_CTY"]]
    )
    selected["source_period"] = selected[columns["TIME_PERIOD"]]
    selected["source_value"] = selected[columns["OBS_VALUE"]]
    selected["source_status"] = selected[columns["OBS_STATUS"]]
    selected["unit"] = "percentage points of GDP"
    selected["source_series_key"] = (
        "Q."
        + selected["country_alpha2"].astype("string")
        + ".P.A.C"
    )
    structure = _dimension_columns(frame).get("STRUCTURE_ID")
    selected["source_structure_id"] = (
        selected[structure] if structure else f"BIS:{spec.dataset_id}({spec.version})"
    )
    return _finalise(selected, dataset_name="credit_gap", metadata=metadata)


def normalise_bis_debt_service(
    frame: pd.DataFrame,
    *,
    metadata: BisDownload | Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Select total-private and sector debt-service ratios from ``WS_DSR``."""

    spec = BIS_BULK_SPECS["debt_service"]
    columns = _require_dimensions(frame, spec)
    freq = _code(frame[columns["FREQ"]])
    country = _code(frame[columns["BORROWERS_CTY"]])
    borrower = _code(frame[columns["DSR_BORROWERS"]])
    selected_mask = freq.eq("Q") & borrower.isin(["P", "H", "N"])
    selected = frame.loc[selected_mask].copy()
    if selected.empty:
        return _empty_observations()

    selected_borrower = borrower.loc[selected.index]
    selected["feature"] = selected_borrower.map(
        {
            "P": "bis_private_debt_service_ratio",
            "H": "bis_household_debt_service_ratio",
            "N": "bis_corporate_debt_service_ratio",
        }
    )
    selected["country_alpha2"] = country.loc[selected.index]
    selected["country_label"] = _label(
        selected[columns["BORROWERS_CTY"]]
    )
    selected["source_period"] = selected[columns["TIME_PERIOD"]]
    selected["source_value"] = selected[columns["OBS_VALUE"]]
    selected["source_status"] = selected[columns["OBS_STATUS"]]
    selected["unit"] = "% income"
    selected["source_series_key"] = (
        "Q."
        + selected["country_alpha2"].astype("string")
        + "."
        + selected_borrower.astype("string")
    )
    structure = _dimension_columns(frame).get("STRUCTURE_ID")
    selected["source_structure_id"] = (
        selected[structure] if structure else f"BIS:{spec.dataset_id}({spec.version})"
    )
    return _finalise(selected, dataset_name="debt_service", metadata=metadata)


def normalise_bis_selected_property_prices(
    frame: pd.DataFrame,
    *,
    metadata: BisDownload | Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Select the four comparable BIS selected residential-price measures."""

    spec = BIS_BULK_SPECS["selected_property_prices"]
    columns = _require_dimensions(frame, spec)
    freq = _code(frame[columns["FREQ"]])
    country = _code(frame[columns["REF_AREA"]])
    value_type = _code(frame[columns["VALUE"]])
    unit_code = _code(frame[columns["UNIT_MEASURE"]])
    unit_label = _label(frame[columns["UNIT_MEASURE"]]).str.lower()
    measure = pd.Series(pd.NA, index=frame.index, dtype="string")
    measure.loc[unit_code.eq("771") | unit_label.str.contains("year-on-year", na=False)] = (
        "growth"
    )
    measure.loc[unit_label.str.contains("index", na=False)] = "index"
    selected_mask = (
        freq.eq("Q")
        & value_type.isin(["N", "R"])
        & measure.isin(["growth", "index"])
    )
    selected = frame.loc[selected_mask].copy()
    if selected.empty:
        return _empty_observations()

    keys = pd.DataFrame(
        {
            "value_type": value_type.loc[selected.index],
            "measure": measure.loc[selected.index],
        },
        index=selected.index,
    )
    selected["feature"] = pd.Series(
        list(zip(keys["value_type"], keys["measure"])), index=selected.index
    ).map(
        {
            ("R", "growth"): "bis_real_house_price_growth_yoy",
            ("N", "growth"): "bis_nominal_house_price_growth_yoy",
            ("R", "index"): "bis_real_house_price_index",
            ("N", "index"): "bis_nominal_house_price_index",
        }
    )
    selected["country_alpha2"] = country.loc[selected.index]
    selected["country_label"] = _label(selected[columns["REF_AREA"]])
    selected["source_period"] = selected[columns["TIME_PERIOD"]]
    selected["source_value"] = selected[columns["OBS_VALUE"]]
    selected["source_status"] = selected[columns["OBS_STATUS"]]
    selected["unit"] = _label(selected[columns["UNIT_MEASURE"]])
    selected["source_series_key"] = (
        "Q."
        + selected["country_alpha2"].astype("string")
        + "."
        + value_type.loc[selected.index].astype("string")
        + "."
        + unit_code.loc[selected.index].astype("string")
    )
    structure = _dimension_columns(frame).get("STRUCTURE_ID")
    selected["source_structure_id"] = (
        selected[structure] if structure else f"BIS:{spec.dataset_id}({spec.version})"
    )
    return _finalise(
        selected,
        dataset_name="selected_property_prices",
        metadata=metadata,
    )


NORMALISERS = {
    "total_credit": normalise_bis_total_credit,
    "credit_gap": normalise_bis_credit_gap,
    "debt_service": normalise_bis_debt_service,
    "selected_property_prices": normalise_bis_selected_property_prices,
}


def read_bis_bulk_csv(path: str | Path, dataset_name: str) -> pd.DataFrame:
    """Read a BIS flat CSV directly or from its official ZIP archive."""

    if dataset_name not in BIS_BULK_SPECS:
        raise ValueError(f"Unknown BIS dataset: {dataset_name}")
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"BIS bulk input is missing or empty: {path}")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, low_memory=False)
    elif path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as archive:
                members = [
                    name
                    for name in archive.namelist()
                    if name.lower().endswith(".csv") and not name.endswith("/")
                ]
                if not members:
                    raise ValueError(f"BIS archive contains no CSV file: {path}")
                preferred = [name for name in members if "csv_flat" in name.lower()]
                member = sorted(preferred or members)[0]
                with archive.open(member) as source:
                    frame = pd.read_csv(source, low_memory=False)
        except zipfile.BadZipFile as error:
            raise ValueError(f"Invalid BIS ZIP archive: {path}") from error
    else:
        raise ValueError(f"Unsupported BIS bulk input format: {path.suffix}")

    _require_dimensions(frame, BIS_BULK_SPECS[dataset_name])
    return frame


def download_bis_bulk_dataset(
    dataset_name: str,
    destination_dir: str | Path,
    *,
    session: requests.Session | None = None,
    timeout: int = 180,
    attempts: int = 3,
) -> BisDownload:
    """Download one official BIS bulk archive atomically with retry."""

    if dataset_name not in BIS_BULK_SPECS:
        raise ValueError(f"Unknown BIS dataset: {dataset_name}")
    if attempts < 1:
        raise ValueError("attempts must be at least one")
    spec = BIS_BULK_SPECS[dataset_name]
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / spec.archive_name
    own_session = session is None
    session = session or requests.Session()
    session.headers.update({"User-Agent": "BankEnv/2.0 (BIS history build)"})
    last_error: Exception | None = None

    try:
        for attempt in range(attempts):
            temporary_path: Path | None = None
            try:
                with session.get(spec.url, stream=True, timeout=timeout) as response:
                    response.raise_for_status()
                    retrieved_at = _utc_now()
                    with tempfile.NamedTemporaryFile(
                        dir=destination_dir,
                        prefix=f".{spec.archive_name}.",
                        suffix=".tmp",
                        delete=False,
                    ) as temporary:
                        temporary_path = Path(temporary.name)
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                temporary.write(chunk)
                    if not zipfile.is_zipfile(temporary_path):
                        raise ValueError(
                            f"BIS {spec.dataset_id} response is not a valid ZIP archive"
                        )
                    with zipfile.ZipFile(temporary_path) as archive:
                        if not any(
                            name.lower().endswith(".csv")
                            for name in archive.namelist()
                        ):
                            raise ValueError(
                                f"BIS {spec.dataset_id} archive contains no CSV"
                            )
                    temporary_path.replace(destination)
                    temporary_path = None
                    return BisDownload(
                        dataset=dataset_name,
                        dataset_id=spec.dataset_id,
                        path=str(destination.resolve()),
                        source_url=spec.url,
                        retrieved_at=retrieved_at,
                        source_vintage=response.headers.get("Last-Modified"),
                        etag=response.headers.get("ETag"),
                        bytes=destination.stat().st_size,
                        sha256=_sha256(destination),
                    )
            except (requests.RequestException, OSError, ValueError) as error:
                last_error = error
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                if attempt + 1 < attempts:
                    time.sleep(2**attempt)
    finally:
        if own_session:
            session.close()
    raise RuntimeError(
        f"BIS {spec.dataset_id} download failed after {attempts} attempts"
    ) from last_error


def local_bis_download_record(dataset_name: str, path: str | Path) -> BisDownload:
    """Create auditable metadata for a pre-downloaded official bulk file."""

    if dataset_name not in BIS_BULK_SPECS:
        raise ValueError(f"Unknown BIS dataset: {dataset_name}")
    spec = BIS_BULK_SPECS[dataset_name]
    path = Path(path)
    # Validate structure before returning a record used by a workflow manifest.
    read_bis_bulk_csv(path, dataset_name)
    return BisDownload(
        dataset=dataset_name,
        dataset_id=spec.dataset_id,
        path=str(path.resolve()),
        source_url=spec.url,
        retrieved_at=_utc_now(),
        source_vintage=datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        etag=None,
        bytes=path.stat().st_size,
        sha256=_sha256(path),
        retrieval_method="local_bulk_file",
    )


def build_bis_financial_history(
    inputs: Mapping[str, pd.DataFrame | str | Path],
    *,
    metadata: Mapping[str, BisDownload | Mapping[str, object]] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Normalise selected official BIS inputs into one long-form history."""

    unknown = sorted(set(inputs).difference(BIS_BULK_SPECS))
    if unknown:
        raise ValueError(f"Unknown BIS datasets: {unknown}")
    frames: list[pd.DataFrame] = []
    for dataset_name, source in inputs.items():
        frame = (
            source.copy()
            if isinstance(source, pd.DataFrame)
            else read_bis_bulk_csv(source, dataset_name)
        )
        normalised = NORMALISERS[dataset_name](
            frame,
            metadata=(metadata or {}).get(dataset_name),
        )
        frames.append(normalised)
    if not frames:
        return _empty_observations()
    result = pd.concat(frames, ignore_index=True)
    if start_year is not None:
        result = result[result["year"].ge(int(start_year))]
    if end_year is not None:
        result = result[result["year"].le(int(end_year))]
    return result.sort_values(
        ["country_code", "indicator_code", "period"]
    ).reset_index(drop=True)


def bis_coverage_report(observations: pd.DataFrame) -> dict[str, object]:
    """Return compact source and feature coverage for a workflow manifest."""

    if observations.empty:
        return {
            "rows": 0,
            "countries": 0,
            "features": 0,
            "first_period": None,
            "last_period": None,
            "by_feature": {},
        }
    by_feature: dict[str, dict[str, int | str]] = {}
    for feature, rows in observations.groupby("indicator_code", sort=True):
        by_feature[str(feature)] = {
            "rows": int(len(rows)),
            "countries": int(rows["country_code"].nunique()),
            "first_period": rows["period"].min().date().isoformat(),
            "last_period": rows["period"].max().date().isoformat(),
        }
    return {
        "rows": int(len(observations)),
        "countries": int(observations["country_code"].nunique()),
        "features": int(observations["indicator_code"].nunique()),
        "first_period": observations["period"].min().date().isoformat(),
        "last_period": observations["period"].max().date().isoformat(),
        "by_feature": by_feature,
    }


def write_bis_financial_history(
    observations: pd.DataFrame,
    *,
    output_path: str | Path = DEFAULT_CACHE_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    downloads: Iterable[BisDownload] = (),
) -> tuple[Path, Path]:
    """Write the normalized history and build manifest atomically."""

    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    output_tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    observations.to_parquet(output_tmp, index=False)
    output_tmp.replace(output_path)

    manifest = {
        "schema_version": 1,
        "built_at": _utc_now(),
        "output_path": str(output_path.resolve()),
        "coverage": bis_coverage_report(observations),
        "downloads": [download.to_dict() for download in downloads],
        "provenance_note": (
            "bis_private_credit_to_gdp_gap is sourced only from official "
            "BIS WS_CREDIT_GAP CG_DTYPE=C observations"
        ),
    }
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    shutil.move(str(manifest_tmp), manifest_path)
    return output_path, manifest_path


__all__ = [
    "BIS_BULK_SPECS",
    "DEFAULT_CACHE_PATH",
    "DEFAULT_MANIFEST_PATH",
    "FEATURE_METADATA",
    "BisBulkSpec",
    "BisDownload",
    "bis_coverage_report",
    "build_bis_financial_history",
    "download_bis_bulk_dataset",
    "local_bis_download_record",
    "normalise_bis_credit_gap",
    "normalise_bis_debt_service",
    "normalise_bis_selected_property_prices",
    "normalise_bis_total_credit",
    "read_bis_bulk_csv",
    "write_bis_financial_history",
]
