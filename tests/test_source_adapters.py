import pandas as pd
import pytest

from src.sources.base import SourceAdapter, SourceUnavailableError


def build_adapter():
    return SourceAdapter(
        name="TEST",
        local_patterns=("*TEST*.csv",),
        required_columns=("SERIES_CODE", "COUNTRY", "INDICATOR"),
        api_url_env="UNSET_TEST_API_URL",
        bulk_url_env="UNSET_TEST_BULK_URL",
    )


def test_source_adapter_uses_valid_local_fallback(tmp_path):
    source = tmp_path / "latest_TEST.csv"
    pd.DataFrame(
        [
            {
                "SERIES_CODE": "TST.VALUE.A",
                "COUNTRY": "Testland",
                "INDICATOR": "Value",
                "2025": 1.0,
            }
        ]
    ).to_csv(source, index=False)

    result = build_adapter().fetch(tmp_path / "downloads", tmp_path)

    assert result.retrieval_method == "local_fallback"
    assert result.sha256
    assert result.bytes == source.stat().st_size


def test_source_adapter_rejects_invalid_local_fallback(tmp_path):
    pd.DataFrame([{"COUNTRY": "Testland"}]).to_csv(
        tmp_path / "bad_TEST.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="missing required columns"):
        build_adapter().fetch(tmp_path / "downloads", tmp_path)


def test_source_adapter_fails_without_any_source(tmp_path):
    with pytest.raises(SourceUnavailableError):
        build_adapter().fetch(tmp_path / "downloads", tmp_path)
