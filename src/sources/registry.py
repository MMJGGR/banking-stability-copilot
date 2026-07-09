"""Configured source-adapter registry."""

from src.sources.base import SourceAdapter


def build_source_adapters() -> dict[str, SourceAdapter]:
    return {
        "WEO": SourceAdapter(
            name="WEO",
            local_patterns=("*WEO*.csv", "data/**/*WEO*.csv"),
            required_columns=("SERIES_CODE", "COUNTRY", "INDICATOR"),
            api_url_env="WEO_API_EXPORT_URL",
            bulk_url_env="WEO_BULK_DOWNLOAD_URL",
        ),
        "FSIC": SourceAdapter(
            name="FSIC",
            local_patterns=("*FSIC*.csv", "data/**/*FSIC*.csv"),
            required_columns=("SERIES_CODE", "COUNTRY", "INDICATOR"),
            api_url_env="FSIC_API_EXPORT_URL",
            bulk_url_env="FSIC_BULK_DOWNLOAD_URL",
        ),
        "MFS": SourceAdapter(
            name="MFS",
            local_patterns=("*MFS*.csv", "data/**/*MFS*.csv"),
            required_columns=("SERIES_CODE", "COUNTRY", "INDICATOR"),
            api_url_env="MFS_API_EXPORT_URL",
            bulk_url_env="MFS_BULK_DOWNLOAD_URL",
        ),
        "FSIBSIS": SourceAdapter(
            name="FSIBSIS",
            local_patterns=("*FSIBSIS*.csv", "data/**/*FSIBSIS*.csv"),
            required_columns=("SERIES_CODE", "COUNTRY", "SECTOR", "INDICATOR"),
            api_url_env="FSIBSIS_API_EXPORT_URL",
            bulk_url_env="FSIBSIS_BULK_DOWNLOAD_URL",
        ),
        "WGI": SourceAdapter(
            name="WGI",
            local_patterns=("*wgi*.xlsx", "*WGI*.xlsx", "data/**/*wgi*.xlsx"),
            required_columns=(),
            api_url_env="WGI_API_EXPORT_URL",
            bulk_url_env="WGI_BULK_DOWNLOAD_URL",
        ),
    }
