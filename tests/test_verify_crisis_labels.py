import json
from pathlib import Path

import pandas as pd
import pytest

from src.crisis_labels import CrisisLabels
from src.scripts.verify_crisis_labels import (
    load_dataset_episodes,
    reconcile,
    sha256_file,
)


def _fixture_csv(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "Country": ["United States", "Kenya", "Kenya", "Freedonia"],
            "ISO": ["USA", "KEN", "KEN", "FRD"],
            "Start": [2007, 1992, "1985m4", 2003],
            "End": [2009, 1995, 1986, None],
        }
    )
    path = tmp_path / "episodes.csv"
    frame.to_csv(path, index=False)
    return path


def test_parser_reads_iso_and_period_columns(tmp_path):
    episodes = load_dataset_episodes(_fixture_csv(tmp_path))
    assert episodes["USA"] == [(2007, 2009)]
    assert episodes["KEN"] == [(1985, 1986), (1992, 1995)]
    # A missing end year falls back to the start year.
    assert episodes["FRD"] == [(2003, 2003)]


def test_parser_supports_explicit_columns_and_checksum(tmp_path):
    path = _fixture_csv(tmp_path)
    episodes = load_dataset_episodes(
        path, country_col="ISO", start_col="Start", end_col="End"
    )
    assert "USA" in episodes
    digest = sha256_file(path)
    assert len(digest) == 64
    assert digest == sha256_file(path)


def test_reconcile_flags_transcription_differences():
    dataset = {"USA": [(2007, 2009)], "KEN": [(1992, 1995)], "FRD": [(2003, 2003)]}
    transcribed = {"USA": [(2007, 2009)], "KEN": [(1992, 1994)], "GHA": [(2017, 2021)]}
    report = reconcile(dataset, transcribed)
    assert report["matching_countries"] == 1
    assert report["mismatched_countries"] == ["KEN"]
    assert report["only_in_dataset"] == ["FRD"]
    assert report["only_in_transcription"] == ["GHA"]


def test_full_verification_against_pinned_dataset_when_present():
    """End-to-end provenance check; runs once the published file is dropped in.

    The IMF web host rejects non-browser downloads, so the workbook must be
    fetched manually to data/reference/. Until then this test is skipped and
    the transcription remains flagged as unverified by the governance docs.
    """
    registry_path = Path("data/reference/crisis_label_source.json")
    if not registry_path.exists():
        pytest.skip("published Laeven-Valencia dataset not yet downloaded")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    dataset_path = Path("data/reference") / registry["file_name"]
    if not dataset_path.exists():
        pytest.skip("registered dataset file missing locally")
    assert sha256_file(dataset_path) == registry["sha256"]
    episodes = load_dataset_episodes(dataset_path)
    report = reconcile(episodes, CrisisLabels().SYSTEMIC_CRISES)
    assert not report["mismatched_countries"]
    assert not report["only_in_dataset"]
