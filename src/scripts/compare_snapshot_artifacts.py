"""Compare two Banking Stability Copilot candidate snapshot bundles by natural observation keys."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.data_loader import parse_period_label


def _normalise(df: pd.DataFrame, keys: list[str], value_col: str, metadata: list[str]):
    use = df[keys + [value_col] + [c for c in metadata if c in df.columns]].copy()
    use[value_col] = pd.to_numeric(use[value_col], errors="coerce")
    use = use.dropna(subset=keys)
    agg = {value_col: "mean"}
    agg.update({c: "first" for c in metadata if c in use.columns})
    return use.groupby(keys, dropna=False, as_index=False).agg(agg)


def _imf(path: Path):
    df = pd.read_parquet(path)
    keys = ["country_code", "indicator_code", "frequency", "unit", "period"]
    metadata = ["observation_status", "latest_actual_year"]
    return _normalise(df, keys, "value", metadata), keys, metadata


def _fsibsis(path: Path):
    df = pd.read_parquet(path)
    id_cols = ["country_code", "SECTOR", "INDICATOR"]
    period_cols = [c for c in df.columns if pd.notna(parse_period_label(c))]
    if not period_cols:
        raise ValueError(f"No parseable FSIBSIS period columns in {path}; columns={list(df.columns)}")
    long = df.melt(id_vars=id_cols, value_vars=period_cols, var_name="period", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    keys = id_cols + ["period"]
    return _normalise(long, keys, "value", []), keys, []


def _wgi(path: Path):
    df = pd.read_parquet(path)
    value_cols = [c for c in df.columns if c not in {"country_code", "year"}]
    long = df.melt(id_vars=["country_code", "year"], value_vars=value_cols, var_name="indicator", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    keys = ["country_code", "year", "indicator"]
    return _normalise(long, keys, "value", []), keys, []


def compare(old, new, keys, metadata, value_col="value"):
    # Convert Arrow-backed key columns to pandas strings before the join; this
    # also handles columns that are entirely empty in one vintage.
    for frame in (old, new):
        for key in keys:
            if frame[key].dtype == "object" or str(frame[key].dtype).startswith("string"):
                frame[key] = frame[key].astype("string")
    merged = old.merge(new, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True)
    added = merged[merged["_merge"] == "right_only"]
    removed = merged[merged["_merge"] == "left_only"]
    both = merged[merged["_merge"] == "both"].copy()
    oldv, newv = both[f"{value_col}_old"], both[f"{value_col}_new"]
    revised_mask = ~np.isclose(oldv.fillna(np.nan), newv.fillna(np.nan), rtol=1e-10, atol=1e-12, equal_nan=True)
    status_changes = {}
    for col in metadata:
        old_col, new_col = f"{col}_old", f"{col}_new"
        if old_col in both.columns and new_col in both.columns:
            changed = ~(both[old_col].astype("string").fillna("<NA>").eq(both[new_col].astype("string").fillna("<NA>")))
            status_changes[col] = int(changed.sum())
            revised_mask |= changed
    revised = both[revised_mask]
    def samples(frame, side):
        out=[]
        for _, row in frame.head(12).iterrows():
            item={k: str(row[k]) for k in keys}
            if side == "both":
                item.update({"old_value": None if pd.isna(row[f"{value_col}_old"]) else float(row[f"{value_col}_old"]),
                             "new_value": None if pd.isna(row[f"{value_col}_new"]) else float(row[f"{value_col}_new"])})
            else:
                col=f"{value_col}_{'new' if side=='added' else 'old'}"
                item["value"]=None if pd.isna(row[col]) else float(row[col])
            out.append(item)
        return out
    return {
        "old_observations": int(len(old)), "new_observations": int(len(new)),
        "added": int(len(added)), "removed": int(len(removed)),
        "revised_or_status_changed": int(len(revised)),
        "metadata_change_counts": status_changes,
        "examples_added": samples(added, "added"),
        "examples_removed": samples(removed, "removed"),
        "examples_revised": samples(revised, "both"),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--old", required=True, type=Path)
    p.add_argument("--new", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a=p.parse_args()
    report={"comparison":"2026-11-12 baseline candidate vs 2026-11-21 candidate","sources":{}}
    for source in ["FSIC","MFS","WEO"]:
        old,keys,meta=_imf(a.old/f"cache/{source}_cache.parquet")
        new,_,_=_imf(a.new/f"cache/{source}_cache.parquet")
        report["sources"][source]=compare(old,new,keys,meta)
    old,keys,meta=_fsibsis(a.old/"cache/FSIBSIS_cache.parquet")
    new,_,_=_fsibsis(a.new/"cache/FSIBSIS_cache.parquet")
    report["sources"]["FSIBSIS"]=compare(old,new,keys,meta)
    old,keys,meta=_wgi(a.old/"cache/WGI_cache.parquet")
    new,_,_=_wgi(a.new/"cache/WGI_cache.parquet")
    report["sources"]["WGI"]=compare(old,new,keys,meta)
    report["summary"]={
        "added":sum(v["added"] for v in report["sources"].values()),
        "removed":sum(v["removed"] for v in report["sources"].values()),
        "revised_or_status_changed":sum(v["revised_or_status_changed"] for v in report["sources"].values()),
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,default=str)+"\\n")
    print(json.dumps(report["summary"],indent=2))
    for name, values in report["sources"].items():
        print(f"{name}: +{values['added']} -{values['removed']} revised/status {values['revised_or_status_changed']}")


if __name__ == "__main__":
    main()
