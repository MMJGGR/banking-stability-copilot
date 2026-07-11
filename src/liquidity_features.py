"""Assemble the promoted liquidity features for the production training path.

This is the single source of truth for which staged liquidity features are
fed into the model. The pillar pipeline consumes any of these columns that are
present (each has a declared entry in ``FEATURE_RISK_DIRECTIONS``); this module
builds them from the cached WEO general-government series and the packaged
external-liquidity reference file so every training entry point
(``refresh_data``, ``build_local_snapshot``, retraining) includes them
consistently.

Assembly is best-effort: if a source is unavailable the affected columns are
simply omitted rather than failing the training run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import BASE_DIR


# Curated, non-duplicative additions. Keep in sync with the challenger feature
# lists in ``pillar_pipeline.ECONOMIC_FEATURES`` / ``FEATURE_RISK_DIRECTIONS``.
GOVERNMENT_LIQUIDITY_FEATURES = ["govt_interest_to_revenue", "govt_debt_to_revenue"]
EXTERNAL_LIQUIDITY_FEATURES = [
    "net_iip_gdp",
    "external_liabilities_gdp",
    "reserves_to_goods_services_imports",
    "gross_external_financing_need_proxy_gdp",
    "investment_income_debits_to_cxr",
]

EXTERNAL_REFERENCE_PATH = (
    Path(BASE_DIR) / "data" / "reference" / "external_liquidity_features.parquet"
)


def _government_features(as_of_date, model_countries) -> pd.DataFrame:
    from src.government_liquidity import (
        build_government_liquidity_features,
        load_weo_fiscal_observations,
    )

    observations = load_weo_fiscal_observations(
        as_of_date=as_of_date, model_countries=model_countries
    )
    features, _ = build_government_liquidity_features(observations)
    columns = ["country_code"] + [
        c for c in GOVERNMENT_LIQUIDITY_FEATURES if c in features.columns
    ]
    return features[columns].copy()


def _external_features() -> pd.DataFrame:
    if not EXTERNAL_REFERENCE_PATH.exists():
        return pd.DataFrame(columns=["country_code"])
    frame = pd.read_parquet(EXTERNAL_REFERENCE_PATH)
    columns = ["country_code"] + [
        c for c in EXTERNAL_LIQUIDITY_FEATURES if c in frame.columns
    ]
    external = frame[columns].copy()
    external["country_code"] = external["country_code"].astype(str).str.upper()
    return external


def assemble_liquidity_features(
    as_of_date=None,
    model_countries: list[str] | None = None,
) -> pd.DataFrame:
    """Return the merged government + external liquidity feature matrix.

    Best-effort: any source that raises or is missing is skipped so the
    production training run is never broken by an unavailable staged source.
    """
    frames = []
    try:
        gov = _government_features(as_of_date, model_countries)
        if not gov.empty:
            frames.append(gov)
    except Exception as exc:  # noqa: BLE001 - never break training on staged data
        print(f"  WARN government-liquidity features unavailable: {exc}")
    try:
        ext = _external_features()
        if not ext.empty and len(ext.columns) > 1:
            frames.append(ext)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN external-liquidity features unavailable: {exc}")

    if not frames:
        return pd.DataFrame(columns=["country_code"])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="country_code", how="outer")
    merged["country_code"] = merged["country_code"].astype(str).str.upper()
    return merged.drop_duplicates("country_code").reset_index(drop=True)
