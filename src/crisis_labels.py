"""Official Laeven-Valencia systemic-banking-crisis labels.

The episode artifact is extracted from Appendix I, Table A1 of Laeven and
Valencia (2026), *Systemic Banking Crises Database: 1970-2025*, IMF Working
Paper WP/26/94.  It contains all 164 published episodes: 161 systemic crises
and the three stress episodes that the authors explicitly classify as
borderline.  Borderline rows remain excluded from model targets by default.

Table A1 is the dating authority used here.  ``end_year`` preserves the
published cell exactly (including the blank end cells for the 2022 Vietnam and
2023 Sri Lanka borderline rows); ``label_end_year`` is the operational end
used for year-level labels and equals the start year when Table A1 is blank.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


SOURCE_TITLE = "Systemic Banking Crises Database: 1970-2025"
SOURCE_VERSION = "IMF Working Paper WP/26/94 (May 2026)"
SOURCE_DOI = "10.5089/9798229045971.001"
SOURCE_URL = (
    "https://www.imf.org/en/publications/wp/issues/2026/05/14/"
    "systemic-banking-crises-database-1970-2025-576036"
)
SOURCE_TABLE_URL = (
    "https://www.elibrary.imf.org/view/journals/001/2026/094/"
    "article-A001-en.xml"
)
SOURCE_TABLE = "Appendix I, Table A1"
SOURCE_COVERAGE_END_YEAR = 2025
EPISODE_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "reference"
    / "systemic_banking_crises_1970_2025.csv"
)

EXPECTED_TOTAL_EPISODES = 164
EXPECTED_SYSTEMIC_EPISODES = 161
EXPECTED_BORDERLINE_EPISODES = 3
_REQUIRED_COLUMNS = {
    "country_code",
    "country_name",
    "start_year",
    "end_year",
    "label_end_year",
    "classification",
    "source_table_row",
    "source_country_text",
    "source_start_text",
    "source_end_text",
    "source_table",
    "source_url",
    "source_pdf_sha256",
}


def _load_episode_table(path: Path = EPISODE_DATA_PATH) -> pd.DataFrame:
    """Load and validate the pinned official episode artifact."""
    if not path.exists():
        raise FileNotFoundError(
            f"Official crisis-label artifact is missing: {path}"
        )

    frame = pd.read_csv(
        path,
        dtype={
            "country_code": "string",
            "country_name": "string",
            "classification": "string",
            "source_table_row": "string",
            "source_country_text": "string",
            "source_start_text": "string",
            "source_end_text": "string",
        },
        keep_default_na=True,
    )
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Crisis-label artifact is missing columns: {sorted(missing)}")

    frame["country_code"] = frame["country_code"].str.strip().str.upper()
    frame["classification"] = frame["classification"].str.strip().str.lower()
    frame["start_year"] = pd.to_numeric(frame["start_year"], errors="raise").astype(int)
    frame["end_year"] = pd.to_numeric(frame["end_year"], errors="coerce").astype("Int64")
    frame["label_end_year"] = pd.to_numeric(
        frame["label_end_year"], errors="raise"
    ).astype(int)

    if len(frame) != EXPECTED_TOTAL_EPISODES:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_EPISODES} official episodes, found {len(frame)}"
        )
    counts = frame["classification"].value_counts().to_dict()
    expected_counts = {
        "systemic": EXPECTED_SYSTEMIC_EPISODES,
        "borderline": EXPECTED_BORDERLINE_EPISODES,
    }
    if counts != expected_counts:
        raise ValueError(
            f"Unexpected crisis classifications: expected {expected_counts}, found {counts}"
        )
    if not frame["country_code"].str.fullmatch(r"[A-Z]{3}").all():
        bad = frame.loc[
            ~frame["country_code"].str.fullmatch(r"[A-Z]{3}"), "country_code"
        ].tolist()
        raise ValueError(f"Invalid ISO alpha-3 crisis codes: {bad}")
    if (frame["label_end_year"] < frame["start_year"]).any():
        raise ValueError("Crisis label_end_year cannot precede start_year")
    published_end = frame["end_year"].notna()
    if not (
        frame.loc[published_end, "end_year"].astype(int)
        == frame.loc[published_end, "label_end_year"]
    ).all():
        raise ValueError("Operational end years diverge from published Table A1 ends")
    if frame.duplicated(
        ["country_code", "start_year", "label_end_year", "classification"]
    ).any():
        raise ValueError("Duplicate crisis episodes in official label artifact")

    return frame


def _period_dict(frame: pd.DataFrame, classification: str) -> Dict[str, List[Tuple[int, int]]]:
    subset = frame.loc[frame["classification"] == classification]
    periods: Dict[str, List[Tuple[int, int]]] = {}
    for row in subset.itertuples(index=False):
        periods.setdefault(str(row.country_code), []).append(
            (int(row.start_year), int(row.label_end_year))
        )
    return {code: sorted(values) for code, values in periods.items()}


_OFFICIAL_EPISODES = _load_episode_table()
_SYSTEMIC_CRISES = _period_dict(_OFFICIAL_EPISODES, "systemic")
_BORDERLINE_CRISES = _period_dict(_OFFICIAL_EPISODES, "borderline")


class CrisisLabels:
    """Query the official 1970-2025 crisis episodes and build model targets.

    ``include_borderline=False`` is deliberate: the paper states that the
    Nicaragua (2018), Vietnam (2022), and Sri Lanka (2023) stress events did
    not meet its systemic-crisis definition.
    """

    SOURCE_TITLE = SOURCE_TITLE
    SOURCE_VERSION = SOURCE_VERSION
    SOURCE_DOI = SOURCE_DOI
    SOURCE_URL = SOURCE_URL
    SOURCE_TABLE_URL = SOURCE_TABLE_URL
    SOURCE_TABLE = SOURCE_TABLE
    SOURCE_COVERAGE_END_YEAR = SOURCE_COVERAGE_END_YEAR
    EPISODE_DATA_PATH = EPISODE_DATA_PATH
    EXPECTED_TOTAL_EPISODES = EXPECTED_TOTAL_EPISODES
    EXPECTED_SYSTEMIC_EPISODES = EXPECTED_SYSTEMIC_EPISODES
    EXPECTED_BORDERLINE_EPISODES = EXPECTED_BORDERLINE_EPISODES

    SYSTEMIC_CRISES = _SYSTEMIC_CRISES
    BORDERLINE_CRISES = _BORDERLINE_CRISES

    def __init__(self, include_borderline: bool = False):
        self.include_borderline = include_borderline
        classifications = ["systemic"]
        if include_borderline:
            classifications.append("borderline")
        self.episode_df = _OFFICIAL_EPISODES.loc[
            _OFFICIAL_EPISODES["classification"].isin(classifications)
        ].copy()

        self.crises = {
            country: list(periods)
            for country, periods in self.SYSTEMIC_CRISES.items()
        }
        if include_borderline:
            for country, periods in self.BORDERLINE_CRISES.items():
                self.crises.setdefault(country, []).extend(periods)
                self.crises[country] = sorted(self.crises[country])
        self._build_crisis_df()

    def _build_crisis_df(self) -> None:
        """Expand episode dates to one record per crisis-country-year."""
        records = []
        for episode in self.episode_df.itertuples(index=False):
            for year in range(int(episode.start_year), int(episode.label_end_year) + 1):
                records.append(
                    {
                        "country_code": str(episode.country_code),
                        "year": year,
                        "crisis": 1,
                        "crisis_start": int(episode.start_year),
                        "crisis_end": int(episode.label_end_year),
                        "classification": str(episode.classification),
                    }
                )
        self.crisis_df = pd.DataFrame.from_records(records)

    def get_episode_table(self, preserve_source_order: bool = True) -> pd.DataFrame:
        """Return the selected episode rows with source-level provenance."""
        frame = self.episode_df.copy()
        if preserve_source_order:
            return frame.reset_index(drop=True)
        return frame.sort_values(
            ["country_code", "start_year", "label_end_year"]
        ).reset_index(drop=True)

    def is_crisis_year(self, country_code: str, year: int) -> bool:
        """Return whether ``year`` is inside a selected crisis episode."""
        code = str(country_code).strip().upper()
        return any(start <= year <= end for start, end in self.crises.get(code, []))

    def get_crisis_target(self, country_code: str, year: int, horizon: int = 3) -> int:
        """Return 1 when a crisis occurs in ``[year + 1, year + horizon]``."""
        if horizon < 1:
            raise ValueError("horizon must be at least one year")
        return int(
            any(
                self.is_crisis_year(country_code, future_year)
                for future_year in range(int(year) + 1, int(year) + horizon + 1)
            )
        )

    def create_labeled_dataset(
        self,
        features_df: pd.DataFrame,
        year_col: str = "year",
        horizon: int = 3,
    ) -> pd.DataFrame:
        """Add ``crisis_target`` to a country-year feature table."""
        if "country_code" not in features_df.columns:
            raise ValueError("features_df must have 'country_code' column")

        labeled = features_df.copy()
        if year_col in labeled.columns:
            labeled["crisis_target"] = labeled.apply(
                lambda row: self.get_crisis_target(
                    row["country_code"], int(row[year_col]), horizon
                ),
                axis=1,
            )
        else:
            # Retained for backward compatibility; model-training panels should
            # always supply an explicit historical forecast-origin year.
            current_year = 2024
            labeled["crisis_target"] = labeled["country_code"].apply(
                lambda code: self.get_crisis_target(code, current_year, horizon)
            )
        return labeled

    def get_crisis_countries(
        self, year_range: Tuple[int, int] | None = None
    ) -> List[str]:
        """Return countries with a selected episode overlapping ``year_range``."""
        if year_range is None:
            return list(self.crises)
        range_start, range_end = year_range
        return [
            country
            for country, periods in self.crises.items()
            if any(start <= range_end and end >= range_start for start, end in periods)
        ]

    def get_crisis_summary(self) -> pd.DataFrame:
        """Return country-level episode counts and date coverage."""
        records = []
        for country, periods in self.crises.items():
            records.append(
                {
                    "country_code": country,
                    "n_crises": len(periods),
                    "total_crisis_years": sum(end - start + 1 for start, end in periods),
                    "latest_crisis": max(end for _, end in periods),
                    "first_crisis": min(start for start, _ in periods),
                }
            )
        return pd.DataFrame(records).sort_values(
            "latest_crisis", ascending=False
        ).reset_index(drop=True)
