"""Display-name resolution for IMF alpha-3 country codes.

The official IMF SDMX feeds identify countries only by alpha-3 code, so
normalized caches and serving artifacts built from them can carry blank
``country_name`` values. This module recovers display names from the code
via pycountry, with overrides for IMF codes that have no ISO 3166-1 entry.
"""

import pycountry

# IMF codes with no ISO 3166-1 entry.
IMF_CODE_OVERRIDES = {
    "KOS": "Kosovo",
    "UVK": "Kosovo",
    "WBG": "West Bank and Gaza",
    "EMU": "Euro Area",
}


def country_name_from_code(code, fallback_to_code: bool = False) -> str:
    """Best-effort display name for an alpha-3 country code.

    Returns '' for unknown codes, or the code itself when
    ``fallback_to_code`` is set (useful for UI display where a blank
    label is worse than showing the raw code).
    """
    if code is None:
        return ""
    normalized = str(code).strip().upper()
    if not normalized:
        return ""
    if normalized in IMF_CODE_OVERRIDES:
        return IMF_CODE_OVERRIDES[normalized]
    entry = pycountry.countries.get(alpha_3=normalized)
    if entry is not None:
        return getattr(entry, "common_name", None) or entry.name
    return normalized if fallback_to_code else ""


def fill_missing_country_names(
    df,
    code_col: str = "country_code",
    name_col: str = "country_name",
    fallback_to_code: bool = False,
):
    """Fill blank/NaN ``name_col`` values from ``code_col`` in place.

    Rows that already have a non-blank name are left untouched, so names
    provided by a data source always win over the derived ones.
    """
    if code_col not in df.columns:
        return df
    if name_col not in df.columns:
        df[name_col] = ""
    names = df[name_col].fillna("").astype(str).str.strip()
    missing = names == ""
    if missing.any():
        codes = df.loc[missing, code_col]
        mapping = {
            c: country_name_from_code(c, fallback_to_code=fallback_to_code)
            for c in codes.unique()
        }
        df.loc[missing, name_col] = codes.map(mapping)
    return df
