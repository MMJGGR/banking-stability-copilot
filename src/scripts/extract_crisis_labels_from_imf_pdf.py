"""Reproduce the pinned Laeven-Valencia crisis-label artifact from Table A1.

Usage::

    python -m src.scripts.extract_crisis_labels_from_imf_pdf --pdf WP2694.pdf

The parser reads only Appendix I, Table A1 (PDF pages 27-31 / zero-based
indices 26-30), validates the published 164-row total, separates the three
borderline episodes named by the authors, and writes a source-traceable CSV.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re

import pandas as pd
import pdfplumber
import pycountry

TABLE_PAGE_INDICES = range(26, 31)
EPISODE_DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "reference"
    / "systemic_banking_crises_1970_2025.csv"
)
EXPECTED_TOTAL_EPISODES = 164
EXPECTED_SYSTEMIC_EPISODES = 161
EXPECTED_BORDERLINE_EPISODES = 3
SOURCE_TABLE = "Appendix I, Table A1"
SOURCE_URL = (
    "https://www.imf.org/en/publications/wp/issues/2026/05/14/"
    "systemic-banking-crises-database-1970-2025-576036"
)
EXPECTED_PDF_SHA256 = "266074fede721a5254b60b2c2800e68c996cf11305fcb42e8ed005845b1bebb1"
BORDERLINE_KEYS = {("NIC", 2018), ("LKA", 2023), ("VNM", 2022)}
COUNTRY_CODE_OVERRIDES = {
    "Cape Verde": "CPV",
    "Cote d'Ivoire": "CIV",
    "Democratic Republic of Congo": "COD",
    "Korea": "KOR",
    "Kyrgyz Republic": "KGZ",
    "Republic of Congo": "COG",
    "Russia": "RUS",
    "São Tomé and Príncipe": "STP",
    "Türkiye": "TUR",
}
WRAPPED_COUNTRY_PREFIXES = {
    "Bosnia and",
    "Central African",
    "Democratic",
    "Dominican",
    "São Tomé and",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _country_code(country_name: str) -> str:
    if country_name in COUNTRY_CODE_OVERRIDES:
        return COUNTRY_CODE_OVERRIDES[country_name]
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except LookupError as exc:
        raise ValueError(f"No ISO alpha-3 mapping for Table A1 country {country_name!r}") from exc


def _table_lines(pdf_path: Path) -> list[str]:
    with pdfplumber.open(pdf_path) as document:
        if len(document.pages) <= max(TABLE_PAGE_INDICES):
            raise ValueError("PDF is too short to contain Appendix I, Table A1")
        text = "\n".join(
            document.pages[index].extract_text(x_tolerance=1, y_tolerance=3) or ""
            for index in TABLE_PAGE_INDICES
        )
    return [line.strip() for line in text.splitlines() if line.strip()]


def extract_table_a1(pdf_path: Path, *, enforce_sha256: bool = True) -> pd.DataFrame:
    pdf_path = Path(pdf_path)
    pdf_hash = _sha256(pdf_path)
    if enforce_sha256 and pdf_hash != EXPECTED_PDF_SHA256:
        raise ValueError(
            "Unexpected IMF PDF checksum: "
            f"expected {EXPECTED_PDF_SHA256}, found {pdf_hash}"
        )

    row_pattern = re.compile(
        r"^(?P<country>.+?)\s+(?:[456]/\s+)?"
        r"(?P<start>(?:19|20)\d{2})\s+(?P<end>\S+)(?P<tail>.*)$"
    )
    rows: list[dict] = []
    pending_country: str | None = None
    for line in _table_lines(pdf_path):
        if pending_country:
            line = f"{pending_country} {line}"
            pending_country = None
        if line in WRAPPED_COUNTRY_PREFIXES:
            pending_country = line
            continue
        match = row_pattern.match(line)
        if match is None:
            continue
        country_name = match.group("country").strip()
        start_year = int(match.group("start"))
        raw_end = match.group("end").strip()
        end_year = int(raw_end) if re.fullmatch(r"(?:19|20)\d{2}", raw_end) else None
        country_code = _country_code(country_name)
        classification = (
            "borderline"
            if (country_code, start_year) in BORDERLINE_KEYS
            else "systemic"
        )
        source_end_text = raw_end
        tail_tokens = match.group("tail").strip().split()
        if tail_tokens and tail_tokens[0] in {"4/", "5/", "6/"}:
            source_end_text = f"{source_end_text} {tail_tokens[0]}"
        rows.append(
            {
                "country_code": country_code,
                "country_name": country_name,
                "start_year": start_year,
                "end_year": end_year,
                "label_end_year": end_year if end_year is not None else start_year,
                "classification": classification,
                "source_table_row": len(rows) + 1,
                "source_country_text": country_name,
                "source_start_text": str(start_year),
                "source_end_text": source_end_text,
                "source_table": SOURCE_TABLE,
                "source_url": SOURCE_URL,
                "source_pdf_sha256": pdf_hash,
            }
        )

    result = pd.DataFrame(rows)
    counts = result["classification"].value_counts().to_dict()
    expected = {
        "systemic": EXPECTED_SYSTEMIC_EPISODES,
        "borderline": EXPECTED_BORDERLINE_EPISODES,
    }
    if len(result) != EXPECTED_TOTAL_EPISODES or counts != expected:
        raise ValueError(
            f"Table A1 extraction mismatch: rows={len(result)}, classes={counts}, "
            f"expected_rows={EXPECTED_TOTAL_EPISODES}, expected_classes={expected}"
        )
    if result.duplicated(
        ["country_code", "start_year", "label_end_year", "classification"]
    ).any():
        raise ValueError("Duplicate Table A1 episodes after extraction")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=EPISODE_DATA_PATH)
    parser.add_argument(
        "--allow-different-pdf-checksum",
        action="store_true",
        help="Permit a re-issued PDF; the resulting checksum remains stored per row.",
    )
    args = parser.parse_args()
    result = extract_table_a1(
        args.pdf,
        enforce_sha256=not args.allow_different_pdf_checksum,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(
        f"Wrote {len(result)} episodes to {args.output} "
        f"({(result.classification == 'systemic').sum()} systemic, "
        f"{(result.classification == 'borderline').sum()} borderline)"
    )


if __name__ == "__main__":
    main()
