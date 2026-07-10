"""Verify the in-code crisis-label dictionary against the published dataset.

The training labels in ``src/crisis_labels.py`` were hand-transcribed from the
Laeven-Valencia database (IMF WP/26/94, May 2026). IMF web hosts return 403 to
non-browser clients, so the published workbook must be downloaded manually
(one time) and passed to this script. The script then:

1. Records the file's SHA-256 checksum in
   ``data/reference/crisis_label_source.json`` so future runs verify the same
   vintage is being used.
2. Parses the episode table (Excel or CSV) into country -> [(start, end)]
   periods.
3. Reconciles the parsed episodes against ``CrisisLabels.SYSTEMIC_CRISES`` and
   writes ``artifacts/crisis_label_reconciliation.json``.
4. Exits non-zero when the transcription and the dataset disagree.

Usage:
    python -m src.scripts.verify_crisis_labels --dataset path/to/wp26094.xlsx
    python -m src.scripts.verify_crisis_labels --dataset file.csv \
        --country-col ISO --start-col Start --end-col End
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR
from src.crisis_labels import CrisisLabels


REGISTRY_PATH = Path(BASE_DIR) / "data" / "reference" / "crisis_label_source.json"
REPORT_PATH = Path(BASE_DIR) / "artifacts" / "crisis_label_reconciliation.json"

# Column-name candidates seen across Laeven-Valencia releases.
COUNTRY_COLUMN_CANDIDATES = ["iso", "iso3", "country_code", "code", "wbcode"]
START_COLUMN_CANDIDATES = ["start", "start_year", "begin", "crisis_start", "year_start"]
END_COLUMN_CANDIDATES = ["end", "end_year", "crisis_end", "year_end"]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _match_column(columns, requested, candidates, role):
    if requested:
        for column in columns:
            if column.strip().lower() == requested.strip().lower():
                return column
        raise ValueError(f"Requested {role} column {requested!r} not in {list(columns)}")
    normalized = {column.strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        f"Could not find a {role} column in {list(columns)}; "
        f"pass it explicitly (candidates tried: {candidates})"
    )


def _read_table(path: Path, sheet=None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet or 0)
    return pd.read_csv(path)


def _parse_year(value):
    """Extract a four-digit year from ints, floats, or strings like '2019m8'."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        year = int(value)
        return year if 1900 <= year <= 2100 else None
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group(0)) if match else None


def load_dataset_episodes(
    path,
    country_col=None,
    start_col=None,
    end_col=None,
    sheet=None,
) -> dict:
    """Parse the published episode table into country -> [(start, end)]."""
    path = Path(path)
    frame = _read_table(path, sheet)
    frame.columns = [str(column) for column in frame.columns]
    country = _match_column(frame.columns, country_col, COUNTRY_COLUMN_CANDIDATES, "country")
    start = _match_column(frame.columns, start_col, START_COLUMN_CANDIDATES, "start-year")
    end = _match_column(frame.columns, end_col, END_COLUMN_CANDIDATES, "end-year")

    episodes = {}
    for _, row in frame.iterrows():
        code = str(row[country]).strip().upper()
        start_year = _parse_year(row[start])
        if not re.fullmatch(r"[A-Z]{3}", code) or start_year is None:
            continue
        end_year = _parse_year(row[end]) or start_year
        episodes.setdefault(code, []).append((start_year, end_year))
    for code in episodes:
        episodes[code] = sorted(set(episodes[code]))
    if not episodes:
        raise ValueError(f"No parsable crisis episodes found in {path}")
    return episodes


def reconcile(dataset_episodes: dict, transcribed: dict) -> dict:
    """Compare dataset episodes with the in-code dictionary."""
    dataset_countries = set(dataset_episodes)
    transcribed_countries = set(transcribed)
    matches = []
    mismatches = []
    for code in sorted(dataset_countries & transcribed_countries):
        expected = sorted(set(map(tuple, dataset_episodes[code])))
        actual = sorted(set(map(tuple, transcribed[code])))
        record = {"country": code, "dataset": expected, "transcribed": actual}
        (matches if expected == actual else mismatches).append(record)
    return {
        "countries_compared": len(dataset_countries & transcribed_countries),
        "matching_countries": len(matches),
        "mismatched_countries": [record["country"] for record in mismatches],
        "mismatches": mismatches,
        "only_in_dataset": sorted(dataset_countries - transcribed_countries),
        "only_in_transcription": sorted(transcribed_countries - dataset_countries),
    }


def verify(path, country_col=None, start_col=None, end_col=None, sheet=None,
           expected_sha256=None) -> dict:
    path = Path(path)
    checksum = sha256_file(path)

    registry = {}
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    pinned = expected_sha256 or registry.get("sha256")
    if pinned and pinned != checksum:
        raise ValueError(
            f"Dataset checksum mismatch: expected {pinned}, got {checksum}. "
            "If this is a new vintage, delete the registry entry deliberately."
        )

    episodes = load_dataset_episodes(path, country_col, start_col, end_col, sheet)
    labels = CrisisLabels()
    report = reconcile(episodes, labels.SYSTEMIC_CRISES)
    report["dataset_file"] = path.name
    report["dataset_sha256"] = checksum
    report["source_version"] = labels.SOURCE_VERSION

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(
            {
                "source_title": labels.SOURCE_TITLE,
                "source_version": labels.SOURCE_VERSION,
                "source_url": labels.SOURCE_URL,
                "file_name": path.name,
                "sha256": checksum,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Published L-V episode table (.xlsx/.csv)")
    parser.add_argument("--country-col", default=None)
    parser.add_argument("--start-col", default=None)
    parser.add_argument("--end-col", default=None)
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--sha256", default=None, help="Expected checksum of the dataset file")
    args = parser.parse_args()

    report = verify(
        args.dataset,
        country_col=args.country_col,
        start_col=args.start_col,
        end_col=args.end_col,
        sheet=args.sheet,
        expected_sha256=args.sha256,
    )
    print(json.dumps({k: report[k] for k in (
        "countries_compared", "matching_countries", "mismatched_countries",
        "only_in_dataset", "only_in_transcription", "dataset_sha256",
    )}, indent=2))
    clean = not report["mismatched_countries"] and not report["only_in_dataset"]
    print(f"Reconciliation report: {REPORT_PATH}")
    sys.exit(0 if clean else 1)


if __name__ == "__main__":
    main()
