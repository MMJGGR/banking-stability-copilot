"""Uncapped residual-to-persistence development models (PRD v0.4).

The library stays broad. Weighting changes regularization influence, not
feature admission: every identity remains in the ledger. Results are
retrospective development evidence only.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh

from .discovery import WideRankTransform
from .inventory import ForecastDataError
from .discovery_execution import ALPHAS, WINDOWS, TARGETS, prepare_matrix, metrics, verify_snapshot, write_json

class WeightedKernelSpace:
    def __init__(self, mode="broad", variance_fraction=.9):
        if mode not in {"broad","profile","source"}:
            raise ForecastDataError("unknown weighting mode")
        self.mode=mode; self.variance_fraction=variance_fraction

    def fit(self, X: pd.DataFrame, ledger: pd.DataFrame):
        self.transformer_=WideRankTransform().fit(X)
        z=self.transformer_.transform(X)
        ids=self.transformer_.active_features_
        meta=ledger.set_index("predictor_id").reindex(ids)
        if meta.source.isna().any():
            raise ForecastDataError("missing source metadata")
        weights=np.ones(len(ids),dtype=float)
        groups=pd.DataFrame({"predictor_id":ids,"source":meta.source.to_numpy()})
        if self.mode=="source":
            sizes=groups.groupby("source").predictor_id.transform("size").to_numpy()
            weights=1/np.sqrt(sizes)
            groups["balance_group"]=groups.source
            groups["group_size"]=sizes
        elif self.mode=="profile":
            mask=X[ids].notna().to_numpy()
            digests=[]
            for j,name in enumerate(ids):
                digests.append(hashlib.sha256(np.ascontiguousarray(z[:,j]).tobytes()+mask[:,j].tobytes()).hexdigest())
            groups["profile_sha256"]=digests
            sizes=groups.groupby("profile_sha256").predictor_id.transform("size").to_numpy()
            weights=1/np.sqrt(sizes)
            groups["balance_group"]=groups.profile_sha256
            groups["group_size"]=sizes
        else:
            groups["balance_group"]="all"
            groups["group_size"]=1
        groups["feature_weight"]=weights
        self.weight_ledger_=groups
        self.weights_=weights
        self.reference_=z*weights
        K=self.reference_ @ self.reference_.T
        eigenvalues,u=eigh(K,check_finite=True,driver="evd")
        order=np.argsort(eigenvalues)[::-1];eigenvalues=eigenvalues[order];u=u[:,order]
        tol=max(float(eigenvalues[0]),1.)*np.finfo(float).eps*max(self.reference_.shape)
        keep=eigenvalues>tol
        self.eigenvalues_=eigenvalues[keep];self.u_=u[:,keep]
        if not len(self.eigenvalues_): raise ForecastDataError("zero rank")
        ratios=self.eigenvalues_/self.eigenvalues_.sum()
        self.n_components_=min(len(ratios),int(np.searchsorted(np.cumsum(ratios),self.variance_fraction)+1))
        self.reference_index_=X.index.copy()
        return self

    def predict(self, X: pd.DataFrame, y: pd.Series, *, alpha=100., residual_multiplier=1.):
        if not isinstance(y,pd.Series) or not y.index.equals(self.reference_index_):
            raise ForecastDataError("target alignment mismatch")
        values=y.to_numpy(dtype=float)
        if not np.isfinite(values).all(): raise ForecastDataError("nonfinite target")
        z=self.transformer_.transform(X)*self.weights_
        center=values.mean()
        penalty=np.full(len(self.eigenvalues_),alpha*residual_multiplier,dtype=float)
        penalty[:self.n_components_]=alpha
        coef=self.u_ @ ((self.u_.T @ (values-center))/(self.eigenvalues_+penalty))
        return center + (z @ self.reference_.T) @ coef


def execute(snapshot: Path, output: Path):
    snapshot=Path(snapshot).resolve(); output=Path(output)
    _,m2,verified=verify_snapshot(snapshot)
    if output.exists(): raise ForecastDataError("output must be new")
    output.mkdir(parents=True)
    registry=pd.read_csv(snapshot/"complete-registry.csv",low_memory=False)
    predictors=pd.read_parquet(snapshot/"panel/predictors.parquet",
        columns=["entity_code","predictor_id","feature_id","forecast_origin_year","observation_year","value","break_in_year","representation"])
    X,ledger,prep=prepare_matrix(predictors,registry)
    targets=pd.read_csv(snapshot/"panel/target-pairs.csv.gz")
    level=predictors.loc[predictors.representation.eq("level"),
        ["entity_code","forecast_origin_year","feature_id","value"]].rename(columns={"feature_id":"target_feature_id","value":"persistence"})
    t=targets.merge(level,on=["entity_code","forecast_origin_year","target_feature_id"],how="left",validate="many_to_one")
    t=t.loc[~t.target_break_in_year.fillna(False) & t.persistence.notna()].copy()
    t["available_year"]=pd.to_datetime(t.assumed_target_available_at).dt.year
    t["change_target"]=t.target_value-t.persistence
    t=t.set_index(["entity_code","forecast_origin_year"])
    t=t[t.index.isin(X.index[X.notna().any(axis=1)])]
    specs=[
      ("broad_residual_ridge","broad",1.),
      ("broad_residual_common_plus_direct","broad",10.),
      ("profile_balanced_residual_ridge","profile",1.),
      ("source_balanced_residual_ridge","source",1.),
    ]
    predictions=[]; reports=[]; skipped=[]
    for target,code in TARGETS.items():
      for horizon in [1,2]:
        a=t[t.target_name.eq(target)&t.horizon_years.eq(horizon)].sort_index()
        years=a.index.get_level_values("forecast_origin_year")
        for start,end in WINDOWS:
          train=a[(years<start)&a.available_year.lt(start)]
          test=a[(years>=start)&(years<=end)]
          inner_start=start-4
          inner_train=train[(train.index.get_level_values(1)<inner_start)&train.available_year.lt(inner_start)]
          inner_test=train[train.index.get_level_values(1)>=inner_start]
          ident={"target":target,"horizon":horizon,"outer_start":start,"outer_end":end,
                 "train_rows":len(train),"test_rows":len(test),"inner_train":len(inner_train),"inner_test":len(inner_test)}
          if min(len(train),len(inner_train))<20 or min(len(test),len(inner_test))<1:
              skipped.append({**ident,"reason":"insufficient_rows"});continue
          chosen={}
          for model,mode,mult in specs:
            sp=WeightedKernelSpace(mode).fit(X.loc[inner_train.index],ledger)
            trials=[]
            for alpha in ALPHAS:
              delta=sp.predict(X.loc[inner_test.index],inner_train.change_target,alpha=alpha,residual_multiplier=mult)
              level_pred=inner_test.persistence.to_numpy()+delta
              trials.append((metrics(inner_test.target_value,level_pred)["mae"],alpha))
            chosen[model]=min(trials,key=lambda x:(x[0],-x[1]))[1]
          spaces={}
          for mode in {"broad","profile","source"}:
            spaces[mode]=WeightedKernelSpace(mode).fit(X.loc[train.index],ledger)
            spaces[mode].weight_ledger_.to_csv(output/f"{target}-{horizon}y-{start}-{mode}-weights.csv",index=False)
          for model,mode,mult in specs:
            sp=spaces[mode]; alpha=chosen[model]
            delta=sp.predict(X.loc[test.index],train.change_target,alpha=alpha,residual_multiplier=mult)
            level_pred=test.persistence.to_numpy()+delta
            err=np.abs(level_pred-test.target_value.to_numpy())
            baseerr=np.abs(test.persistence.to_numpy()-test.target_value.to_numpy())
            frame=test[["target_value","persistence","target_year","target_status","change_target"]].copy()
            frame["predicted_change"]=delta;frame["prediction"]=level_pred;frame["model"]=model
            frame["target"]=target;frame["horizon"]=horizon;frame["outer_start"]=start;frame["alpha"]=alpha
            predictions.append(frame.reset_index())
            reports.append({**ident,"model":model,"alpha":alpha,"mode":mode,
                "learnable_features":len(sp.transformer_.active_features_),"components_90pct":sp.n_components_,
                "fraction_rows_beating_persistence":float((err<baseerr).mean()),
                "change_rmse":float(np.sqrt(np.mean((delta-test.change_target.to_numpy())**2))),
                **metrics(test.target_value,level_pred)})
          base=test[["target_value","persistence","target_year","target_status","change_target"]].copy()
          base["predicted_change"]=0.;base["prediction"]=base.persistence;base["model"]="persistence"
          base["target"]=target;base["horizon"]=horizon;base["outer_start"]=start
          predictions.append(base.reset_index())
          reports.append({**ident,"model":"persistence","fraction_rows_beating_persistence":0.,
                          "change_rmse":float(np.sqrt(np.mean(test.change_target.to_numpy()**2))),
                          **metrics(test.target_value,test.persistence)})
          write_json(output/f"{target}-{horizon}y-{start}-tuning.json",{"fold":ident,"chosen_alpha":chosen})
    if not predictions: raise ForecastDataError("no experiment ran")
    pred=pd.concat(predictions,ignore_index=True)
    pred.to_csv(output/"out-of-fold-predictions.csv.gz",index=False)
    folds=pd.DataFrame(reports);folds.to_csv(output/"fold-metrics.csv",index=False)
    aggregate=[]
    for (target,horizon,model),d in pred.groupby(["target","horizon","model"]):
        e=d.prediction-d.target_value
        de=d.predicted_change-d.change_target
        be=(d.persistence-d.target_value).abs()
        aggregate.append({"target":target,"horizon":int(horizon),"model":model,"rows":len(d),
                          "mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e**2))),"bias":float(e.mean()),
                          "change_rmse":float(np.sqrt(np.mean(de**2))),
                          "fraction_rows_beating_persistence":float((e.abs()<be).mean()) if model!="persistence" else 0.})
    pd.DataFrame(aggregate).to_csv(output/"aggregate-metrics.csv",index=False)
    report={"status":"completed_v0_4_residual_development_not_production_validation",
            "source_cutoff":m2["cutoff"],"inputs_verified":verified,"feature_cap":None,
            "models":[x[0] for x in specs]+["persistence"],"skipped":skipped,
            "final_confirmation_evaluated":False,"production_modified":False,
            "crisis_classifier_retrained":False,"preprocessing":prep,"aggregate_metrics":aggregate}
    write_json(output/"run.json",report)
    return report
