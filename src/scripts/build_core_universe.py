"""Rebuild data/reference/core_universe.csv: UN LDCs plus World Bank IDA/blend.

The core universe is where BankEnv is judged: frontier and low-income
economies. IDA and blend lending status is the machine-readable counterpart
of IMF PRGT eligibility (same income cut-off); the World Bank revises it
each 1 July. Run: python -m src.scripts.build_core_universe
"""
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/reference/core_universe.csv")
# UN DESA list of least developed countries, updated December 2024.
UN_LDC = (
    "AFG AGO BGD BEN BFA BDI KHM CAF TCD COM COD DJI ERI ETH GMB GIN GNB HTI "
    "KIR LAO LSO LBR MDG MWI MLI MRT MOZ MMR NPL NER RWA SEN SLE SLB SOM SSD "
    "SDN TLS TGO TUV UGA TZA YEM ZMB"
).split()
WB_TO_IMF = {"XKX": "KOS"}  # World Bank Kosovo code vs the IMF code the app uses


def build() -> pd.DataFrame:
    assert len(set(UN_LDC)) == 44
    response = requests.get(
        "https://api.worldbank.org/v2/country",
        params={"format": "json", "per_page": 400},
        timeout=60,
    )
    response.raise_for_status()
    wb = {
        WB_TO_IMF.get(row["id"], row["id"]): row
        for row in response.json()[1]
        if row["lendingType"]["value"] != "Aggregates"
    }
    ida = {code for code, row in wb.items() if row["lendingType"]["id"] in ("IDX", "IDB")}
    if len(ida) < 60:  # a truncated API reply must not shrink the universe
        raise RuntimeError(f"World Bank API returned only {len(ida)} IDA/blend countries")
    codes = sorted(set(UN_LDC) | ida)
    return pd.DataFrame({
        "country_code": codes,
        "country_name": [wb.get(c, {}).get("name", "") for c in codes],
        "un_ldc": [c in UN_LDC for c in codes],
        "wb_lending": [wb.get(c, {}).get("lendingType", {}).get("value", "") for c in codes],
        "wb_income": [wb.get(c, {}).get("incomeLevel", {}).get("value", "") for c in codes],
        "basis": [
            "both" if c in UN_LDC and c in ida else "un_ldc" if c in UN_LDC else "wb_ida"
            for c in codes
        ],
    })


if __name__ == "__main__":
    frame = build()
    frame.to_csv(OUT, index=False)
    print(f"{len(frame)} countries: {frame.basis.value_counts().to_dict()}")
