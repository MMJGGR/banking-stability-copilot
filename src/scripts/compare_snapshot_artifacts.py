"""Compare two Banking Stability Copilot candidate snapshot bundles by natural observation keys."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def _is_period_column(value):
    s = str(value).strip()
    if len(s) == 4 and s.isdigit():
        return 1900 <= int(s) <= 2100
    if len(s) == 7 and s[:4].isdigit():
        suffix = s[4:]
        return suffix in {f"-Q{i}" for i in range(1,5)} | {f"-M{i:02d}" for i in range(1,13)}
    return False

def _normalise(df, keys, value_col, metadata):
    use=df[keys+[value_col]+[c for c in metadata if c in df.columns]].copy()
    use[value_col]=pd.to_numeric(use[value_col],errors="coerce")
    use=use.dropna(subset=keys)
    agg={value_col:"mean"}; agg.update({c:"first" for c in metadata if c in use.columns})
    return use.groupby(keys,dropna=False,as_index=False).agg(agg)

def _imf(path):
    df=pd.read_parquet(path); keys=["country_code","indicator_code","frequency","unit","period"]
    meta=["observation_status","latest_actual_year"]
    return _normalise(df,keys,"value",meta),keys,meta

def _fsibsis(path):
    df=pd.read_parquet(path); ids=["country_code","SECTOR","INDICATOR"]
    periods=[c for c in df.columns if _is_period_column(c)]
    if not periods: raise ValueError(f"No period columns in {path}")
    long=df.melt(id_vars=ids,value_vars=periods,var_name="period",value_name="value")
    long["value"]=pd.to_numeric(long["value"],errors="coerce"); long=long.dropna(subset=["value"])
    keys=ids+["period"]
    return _normalise(long,keys,"value",[]),keys,[]

def _wgi(path):
    df=pd.read_parquet(path); vals=[c for c in df.columns if c not in {"country_code","year"}]
    long=df.melt(id_vars=["country_code","year"],value_vars=vals,var_name="indicator",value_name="value")
    long["value"]=pd.to_numeric(long["value"],errors="coerce"); long=long.dropna(subset=["value"])
    keys=["country_code","year","indicator"]
    return _normalise(long,keys,"value",[]),keys,[]

def compare(old,new,keys,metadata):
    for frame in (old,new):
        for key in keys:
            if frame[key].dtype=="object" or str(frame[key].dtype).startswith("string"):
                frame[key]=frame[key].astype("string")
    merged=old.merge(new,on=keys,how="outer",suffixes=("_old","_new"),indicator=True)
    added=merged[merged["_merge"]=="right_only"]; removed=merged[merged["_merge"]=="left_only"]
    both=merged[merged["_merge"]=="both"].copy()
    revised=~np.isclose(both["value_old"].fillna(np.nan),both["value_new"].fillna(np.nan),rtol=1e-10,atol=1e-12,equal_nan=True)
    changes={}
    for col in metadata:
        a,b=f"{col}_old",f"{col}_new"
        if a in both and b in both:
            mask=~both[a].astype("string").fillna("<NA>").eq(both[b].astype("string").fillna("<NA>"))
            changes[col]=int(mask.sum()); revised |= mask
    def examples(frame,side):
        out=[]
        for _,r in frame.head(10).iterrows():
            x={k:str(r[k]) for k in keys}
            if side=="revised":
                x.update(old_value=None if pd.isna(r.value_old) else float(r.value_old),new_value=None if pd.isna(r.value_new) else float(r.value_new))
            else:
                col="value_new" if side=="added" else "value_old"
                x["value"]=None if pd.isna(r[col]) else float(r[col])
            out.append(x)
        return out
    rev=both[revised]
    return {"old_observations":len(old),"new_observations":len(new),"added":len(added),"removed":len(removed),
            "revised_or_status_changed":len(rev),"metadata_change_counts":changes,
            "examples_added":examples(added,"added"),"examples_removed":examples(removed,"removed"),
            "examples_revised":examples(rev,"revised")}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--old",type=Path,required=True); p.add_argument("--new",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    report={"comparison":"2026-11-12 baseline candidate vs 2026-11-21 candidate","sources":{}}
    for source in ["FSIC","MFS","WEO"]:
        o,k,m=_imf(a.old/f"cache/{source}_cache.parquet"); n,_,_=_imf(a.new/f"cache/{source}_cache.parquet"); report["sources"][source]=compare(o,n,k,m)
    o,k,m=_fsibsis(a.old/"cache/FSIBSIS_cache.parquet"); n,_,_=_fsibsis(a.new/"cache/FSIBSIS_cache.parquet"); report["sources"]["FSIBSIS"]=compare(o,n,k,m)
    o,k,m=_wgi(a.old/"cache/WGI_cache.parquet"); n,_,_=_wgi(a.new/"cache/WGI_cache.parquet"); report["sources"]["WGI"]=compare(o,n,k,m)
    report["summary"]={key:sum(v[key] for v in report["sources"].values()) for key in ["added","removed","revised_or_status_changed"]}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2,default=str)+"\n")
    print(json.dumps(report["summary"],indent=2))
    for name,v in report["sources"].items(): print(f"{name}: +{v['added']} -{v['removed']} revised/status {v['revised_or_status_changed']}")
if __name__=="__main__": main()
