import hashlib

import pandas as pd

from src.snapshot_manifest import (
    sha256_file,
    summarize_fsibsis_cache,
    summarize_wgi_cache,
    write_snapshot_manifest,
)


def test_sha256_file(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"banking-copilot")

    assert sha256_file(source) == hashlib.sha256(
        b"banking-copilot"
    ).hexdigest()


def test_write_snapshot_manifest(tmp_path):
    output = tmp_path / "nested" / "manifest.json"
    result = write_snapshot_manifest(
        {"snapshot_id": "2025-12-31"},
        output,
    )

    assert result == output
    assert '"snapshot_id": "2025-12-31"' in output.read_text(
        encoding="utf-8"
    )


def test_fsibsis_manifest_counts_indicator_labels(tmp_path):
    cache = tmp_path / "FSIBSIS_cache.parquet"
    pd.DataFrame(
        {
            "country_code": ["KEN", "KEN", "MOZ"],
            "COUNTRY": ["KEN", "KEN", "MOZ"],
            "SECTOR": ["Deposit takers"] * 3,
            "INDICATOR": ["Capital", "Loans", "Capital"],
            "2025-M12": [1.0, 2.0, None],
            "2026-M04": [3.0, None, 4.0],
            "2026-M07": [5.0, 6.0, 7.0],
        }
    ).to_parquet(cache, index=False)

    summary = summarize_fsibsis_cache(cache, pd.Timestamp("2026-06-30"))

    assert summary["rows"] == 3
    assert summary["countries"] == 2
    assert summary["indicators"] == 2
    assert summary["latest_period_label"] == "2026-M04"


def test_wgi_manifest_counts_governance_columns(tmp_path):
    cache = tmp_path / "WGI_cache.parquet"
    pd.DataFrame(
        {
            "country_code": ["KEN", "KEN", "MOZ"],
            "year": [2024, 2025, 2024],
            "voice_accountability": [50.0, 51.0, 40.0],
            "political_stability": [42.0, 43.0, 35.0],
            "empty_indicator": [None, None, None],
        }
    ).to_parquet(cache, index=False)

    summary = summarize_wgi_cache(cache, pd.Timestamp("2024-12-31"))

    assert summary["rows"] == 2
    assert summary["countries"] == 2
    assert summary["indicators"] == 2
    assert summary["latest_observation"] == "2024-12-31"
