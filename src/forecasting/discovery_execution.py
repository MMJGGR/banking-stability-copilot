"""Registered broad discovery + matched DEVELOPMENT forecasts, isolated from serving.

python -m src.forecasting.discovery_execution --snapshot <M2 live directory> \
    --output <new directory>

Reads the pinned research snapshot; no network access, production writes,
classifier training, model promotion or final confirmation evaluation.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .discovery import BroadSpectralSpace
from .inventory import ForecastDataError
from .panel import TARGETS
from src.scripts.audit_forecasting_sources import sha256, output_destination

M2_ZIP_SHA = 'f670f2e7caff532201c1a198130d0f8b34cb04784cb643dae2c5466af9359c9b'
# Input verification uses the immutable artifact's own ledger, plus exact
# content digests of the three inputs printed into every execution report.
CURRENCY_UNITS = frozenset({'XDC', 'USD', 'EUR', 'XDR'})
ALPHAS = (10., 100., 1000.)
WINDOWS = ((2016, 2018), (2019, 2021))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n')


def verify_snapshot(root):
    root = Path(root).resolve()
    checks = json.loads((root/'output-checksums.json').read_text())
    verified = 0
    for relative, digest in checks.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
            raise ForecastDataError(f'M2 checksum/identity mismatch: {relative}')
        verified += 1
    report = json.loads((root/'run.json').read_text())
    if report.get('status') != 'completed_M2_data_execution_not_model_validation':
        raise ForecastDataError('M2 did not complete')
    return checks, report, verified


def prepare_matrix(predictors, registry):
    """Admit units explicitly; amounts normalized against their own past only.

    Own-history scale is mean absolute value of preceding observations, reset
    on a reported break. Two prior observations and a positive scale are
    required for that CELL, not for global admission of the predictor.
    """
    required = {'entity_code','predictor_id','feature_id','forecast_origin_year',
                'value','break_in_year','observation_year'}
    if not required <= set(predictors):
        raise ForecastDataError('Incomplete M2 predictors')
    if not registry.feature_id.is_unique:
        raise ForecastDataError('Nonunique feature registry')
    d = predictors[list(required)].copy()
    if d.duplicated(['entity_code','forecast_origin_year','predictor_id']).any():
        raise ForecastDataError('Duplicate predictor cell')
    if not (d.observation_year < d.forecast_origin_year).all():
        raise ForecastDataError('Input not before forecast origin')
    if not np.isfinite(d.value).all():
        raise ForecastDataError('Staged predictors contain nonfinite numeric values')
    r = registry.set_index('feature_id')
    if not set(d.feature_id) <= set(r.index):
        raise ForecastDataError('Unknown predictor identity')
    meta = d[['predictor_id','feature_id']].drop_duplicates().set_index('predictor_id')
    meta = meta.join(r[['source','INDICATOR','indicator_label','UNIT','SCALE']], on='feature_id')
    meta['unit_policy'] = np.where(meta.UNIT.isna(), 'quarantined_unresolved_unit',
        np.where(meta.UNIT.isin(CURRENCY_UNITS), 'causal_own_history_amount', 'observed_native_unit_rank'))
    all_columns = sorted(meta.index)
    row_index = pd.MultiIndex.from_frame(d[['entity_code','forecast_origin_year']].drop_duplicates()).sort_values()
    d['model_value'] = d.value
    d.loc[d.feature_id.map(r.UNIT).isna(), 'model_value'] = np.nan
    amount = d.feature_id.map(r.UNIT).isin(CURRENCY_UNITS)
    a = d.loc[amount, ['entity_code','predictor_id','forecast_origin_year','value','break_in_year']].copy()
    a = a.sort_values(['entity_code','predictor_id','forecast_origin_year'])
    for name in ['entity_code','predictor_id']:
        a[name] = a[name].astype('category')
    keys = ['entity_code','predictor_id']
    a['_segment'] = a.groupby(keys, observed=True, sort=False).break_in_year.cumsum()
    group_keys = [a[k] for k in keys+['_segment']]
    previous_count = a.groupby(keys+['_segment'], observed=True, sort=False).cumcount()
    absval = a.value.abs()
    previous_sum = absval.groupby(group_keys, observed=True, sort=False).cumsum()-absval
    denominator = previous_sum / previous_count.replace(0, np.nan)
    good = previous_count.ge(2) & denominator.gt(0)
    normalized = pd.Series(np.nan, index=a.index, dtype=float)
    normalized.loc[good] = np.arcsinh(a.loc[good,'value']/denominator.loc[good])
    d.loc[a.index,'model_value'] = normalized
    stats = d.groupby('predictor_id', observed=True).agg(
        source_cells=('value','size'), admitted_cells=('model_value','count'))
    ledger = meta.join(stats).reset_index()
    x = d.pivot(index=['entity_code','forecast_origin_year'],columns='predictor_id',values='model_value')
    x = x.reindex(index=row_index, columns=all_columns).astype(np.float32)
    summary = {'supplied_predictor_representations': len(all_columns), 'feature_count_cap': None,
        'unit_quarantined_representations': int(meta.unit_policy.eq('quarantined_unresolved_unit').sum()),
        'amount_representations': int(meta.unit_policy.eq('causal_own_history_amount').sum()),
        'amount_cells_without_supported_prior_scale': int((~good).sum()),
        'native_rank_representations': int(meta.unit_policy.eq('observed_native_unit_rank').sum()),
        'rows':len(x), 'entities':int(x.index.get_level_values(0).nunique()),
        'admitted_observed_cells':int(d.model_value.notna().sum()),
        'missingness_policy':'not an economic PCA input; median fill and explicit coverage diagnostics',
        'amount_policy':'asinh(value / preceding mean abs value), >=2 prior observations, reset at breaks'}
    return x, ledger, summary


def _corr(x,y):
    good=np.isfinite(x)&np.isfinite(y)
    if good.sum()<3 or np.std(x[good])==0 or np.std(y[good])==0:return None
    return float(np.corrcoef(x[good],y[good])[0,1])


def discovery(x,ledger,output):
    output.mkdir()
    year=int(x.index.get_level_values('forecast_origin_year').max())
    current=x.xs(year,level='forecast_origin_year')
    previous=x.xs(year-1,level='forecast_origin_year')
    space=BroadSpectralSpace(.9).fit(current)
    admission=space.transformer_.admission_.merge(ledger,on='predictor_id',validate='one_to_one')
    admission.to_csv(output/'all-feature-admission.csv',index=False)
    spectrum=space.spectrum();spectrum.to_csv(output/'spectrum.csv',index=False)
    loadings=space.loadings()
    pd.DataFrame(loadings,index=pd.Index(space.transformer_.active_features_,name='predictor_id'),
                 columns=[f'PC{i+1}' for i in range(loadings.shape[1])]).to_parquet(output/'all-component-loadings.parquet')
    for label, matrix in [('current',current),('previous_same_reference',previous)]:
        diag,scores,residual=space.describe(matrix)
        diag.to_csv(output/f'{label}-distinctness.csv')
        pd.DataFrame(scores,index=matrix.index).to_csv(output/f'{label}-coordinates.csv')
        pd.DataFrame(residual,index=matrix.index,columns=space.transformer_.active_features_).to_parquet(output/f'{label}-observed-residual-contributions.parquet')
    diag,scores,_=space.describe(current)
    k=space.n_components_
    contribution=(loadings[:,:k]**2) @ space.variance_ratios_[:k]
    f=pd.DataFrame({'predictor_id':space.transformer_.active_features_,
                    'common_variance_share':contribution}).merge(ledger,on='predictor_id',validate='one_to_one')
    f.to_csv(output/'all-feature-common-contributions.csv',index=False)
    f.groupby('source').agg(predictors=('predictor_id','size'),common_variance_share=('common_variance_share','sum')).to_csv(output/'source-contributions.csv')
    groups=space.exact_profile_groups(current);groups.to_csv(output/'exact-profile-groups.csv',index=False)
    representatives=groups.drop_duplicates('profile_sha256').predictor_id.tolist()
    distinct=BroadSpectralSpace(.9).fit(current[representatives])
    distinct.spectrum().to_csv(output/'exact-profile-collapse-spectrum.csv',index=False)
    missing_correlations=[_corr(scores[:,j],diag.observed_share.to_numpy()) for j in range(k)]
    result={'forecast_origin':year,'latest_observation_year_under_assumed_lag':year-1,
        'rows':len(current),'target_independent':True,'targets_used':0,
        'supplied_representations':len(current.columns),'learnable_representations':len(space.transformer_.active_features_),
        'numerical_rank':len(space.eigenvalues_),
        'components_for_variance':{str(p):space.components_for(p) for p in [.8,.9,.95]},
        'first_component_variance':float(space.variance_ratios_[0]),
        'unique_exact_profiles':len(representatives),
        'duplicate_profile_extra_columns':len(groups)-len(representatives),
        'deduplicated_components_for_90pct':distinct.n_components_,
        'pc_coverage_correlations':missing_correlations,
        'top_common_contributors':f.nlargest(12,'common_variance_share')[['source','INDICATOR','indicator_label','predictor_id','common_variance_share']].to_dict('records'),
        'limitations':['Descriptive retrospective representation; no risk ranking or causal claim.',
          'Median filling and sparse histories affect geometry; inspect coverage.',
          'Reference fitted at latest panel year; earlier coordinates are retrospective alignment, not prior forecasts.',
          'Exact-profile collapse is a diagnostic sensitivity, not economic equivalence or removal from the library.']}
    write_json(output/'summary.json',result)
    return result


def metrics(actual,predicted):
    e=np.asarray(predicted)-np.asarray(actual)
    if not np.isfinite(e).all():raise ForecastDataError('Nonfinite model output')
    return {'rows':len(e),'mae':float(np.abs(e).mean()),'rmse':float(np.sqrt(np.mean(e*e))), 'bias':float(e.mean())}


def development(x,targets,ledger,predictors,output):
    output.mkdir()
    # Persistence uses a genuinely observed value of the SAME target series at
    # the declared origin. No substitute or cross-frequency last-row fallback.
    level=predictors.loc[predictors.representation.eq('level'),
          ['entity_code','forecast_origin_year','feature_id','value']].rename(columns={'feature_id':'target_feature_id','value':'persistence'})
    t=targets.merge(level,on=['entity_code','forecast_origin_year','target_feature_id'],how='left',validate='many_to_one')
    t['cohort_exclusion']=np.where(t.target_break_in_year.fillna(False),'explicit_target_break',
                                 np.where(t.persistence.isna(),'no_same_series_persistence',''))
    t=t.set_index(['entity_code','forecast_origin_year'])
    available=x.notna().any(axis=1)
    t.loc[~t.index.isin(available.index[available]),'cohort_exclusion']='no_admitted_predictors'
    t[t.cohort_exclusion.ne('')].to_csv(output/'cohort-exclusions.csv')
    t=t[t.cohort_exclusion.eq('')].copy()
    t['available_year']=pd.to_datetime(t.assumed_target_available_at).dt.year
    predictions=[]; reports=[]; skipped=[]
    for target,code in TARGETS.items():
        compact=ledger.loc[ledger.source.eq('FSIC') & ledger.INDICATOR.eq(code),'predictor_id'].tolist()
        for horizon in [1,2]:
            a=t[t.target_name.eq(target)&t.horizon_years.eq(horizon)].sort_index()
            if not a.index.is_unique:raise ForecastDataError('Nonunique target origin')
            yy=a.index.get_level_values('forecast_origin_year')
            for start,end in WINDOWS:
                train=a[(yy<start)&a.available_year.lt(start)]
                test=a[(yy>=start)&(yy<=end)]
                inner_start=start-4
                inner_train=train[(train.index.get_level_values(1)<inner_start)&train.available_year.lt(inner_start)]
                inner_test=train[train.index.get_level_values(1)>=inner_start]
                ident={'target':target,'horizon':horizon,'outer_start':start,'outer_end':end,
                       'train_rows':len(train),'test_rows':len(test),'inner_train':len(inner_train),'inner_test':len(inner_test)}
                if min(len(train),len(inner_train))<20 or min(len(test),len(inner_test))<1:
                    skipped.append({**ident,'reason':'insufficient_declared_training_or_development_rows'});continue
                chosen={}; tuning=[]
                for name,cols in [('compact',compact),('broad',list(x.columns))]:
                    sp=BroadSpectralSpace(.9).fit(x.loc[inner_train.index,cols])
                    for mult,model in ([(1.,'compact_ridge')] if name=='compact' else [(1.,'broad_ridge'),(10.,'broad_common_plus_residual')]):
                        trials=[]
                        for alpha in ALPHAS:
                            pr=sp.ridge_predict(x.loc[inner_test.index,cols],inner_train.target_value,alpha=alpha,residual_multiplier=mult)
                            m=metrics(inner_test.target_value,pr)
                            trials.append((m['mae'],alpha))
                            tuning.append({'model':model,'alpha':alpha,**m})
                        chosen[model]=min(trials,key=lambda v:(v[0],-v[1]))[1]
                    del sp;gc.collect()
                frames=[]
                for name,cols in [('compact',compact),('broad',list(x.columns))]:
                    sp=BroadSpectralSpace(.9).fit(x.loc[train.index,cols])
                    audit=sp.transformer_.admission_
                    audit.to_csv(output/f'{target}-{horizon}y-{start}-{name}-admission.csv',index=False)
                    for mult,model in ([(1.,'compact_ridge')] if name=='compact' else [(1.,'broad_ridge'),(10.,'broad_common_plus_residual')]):
                        pr=sp.ridge_predict(x.loc[test.index,cols],train.target_value,alpha=chosen[model],residual_multiplier=mult)
                        frame=test[['target_value','persistence','target_year','target_status']].copy()
                        frame['prediction']=pr;frame['model']=model;frame['target']=target
                        frame['horizon']=horizon;frame['outer_start']=start;frame['alpha']=chosen[model]
                        frame['learnable_features']=len(sp.transformer_.active_features_);frame['components_90pct']=sp.n_components_
                        frames.append(frame.reset_index())
                        reports.append({**ident,'model':model,'alpha':chosen[model],
                            'supplied_features':len(cols),'learnable_features':len(sp.transformer_.active_features_),
                            'components_90pct':sp.n_components_,**metrics(test.target_value,pr)})
                    del sp;gc.collect()
                frame=test[['target_value','persistence','target_year','target_status']].copy()
                frame['prediction']=test.persistence;frame['model']='persistence';frame['target']=target
                frame['horizon']=horizon;frame['outer_start']=start;frames.append(frame.reset_index())
                reports.append({**ident,'model':'persistence',**metrics(test.target_value,test.persistence)})
                predictions+=frames
                write_json(output/f'{target}-{horizon}y-{start}-inner-tuning.json',{'fold':ident,'trials':tuning,'chosen_alpha':chosen})
                print('DEVELOPMENT_FOLD',target,horizon,start,len(train),len(test),chosen,flush=True)
    if not predictions:raise ForecastDataError('No declared experiment could run')
    pred=pd.concat(predictions,ignore_index=True)
    pred.to_csv(output/'out-of-fold-predictions.csv.gz',index=False)
    results=pd.DataFrame(reports);results.to_csv(output/'fold-metrics.csv',index=False)
    aggregate=[]
    for (target,horizon,model),d in pred.groupby(['target','horizon','model']):
        aggregate.append({'target':target,'horizon':int(horizon),'model':model,
                          **metrics(d.target_value,d.prediction)})
    agg=pd.DataFrame(aggregate);agg.to_csv(output/'aggregate-metrics.csv',index=False)
    subgroup=[]
    for (target,horizon,model,entity),d in pred.groupby(['target','horizon','model','entity_code']):
        subgroup.append({'target':target,'horizon':int(horizon),'model':model,'entity_code':entity,**metrics(d.target_value,d.prediction)})
    pd.DataFrame(subgroup).to_csv(output/'entity-metrics.csv',index=False)
    summary={'status':'retrospective_development_not_production_validation','outer_windows':WINDOWS,
        'penalty_grid':ALPHAS,'matched_target_origin_rows':int(len(pred[pred.model.eq('persistence')])),
        'model_folds_executed':len(results),'skipped':skipped,
        'models':['persistence','compact_ridge','broad_ridge','broad_common_plus_residual'],
        'final_confirmation_evaluated':False,'crisis_classifier_retrained':False,
        'aggregate_metrics':aggregate}
    write_json(output/'summary.json',summary)
    return summary


def run(snapshot, output, *, discovery_only=False):
    snapshot=Path(snapshot).resolve()
    output=output_destination(snapshot,Path(output))
    checks,m2,verified=verify_snapshot(snapshot)
    output.mkdir(parents=True)
    root=Path(__file__).resolve().parents[2]
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):commit='local_copy_recorded_code_hashes'
    record={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
        'code_commit':commit,'source_artifact':10463106988,'source_archive_sha256':M2_ZIP_SHA,
        'source_cutoff':m2['cutoff'],'inputs_verified':verified,
        'new_source_retrieval_performed':False,'production_modified':False,
        'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}
    try:
        registry=pd.read_csv(snapshot/'complete-registry.csv',low_memory=False)
        columns=['entity_code','predictor_id','feature_id','forecast_origin_year','observation_year','value','break_in_year','representation']
        p=pd.read_parquet(snapshot/'panel/predictors.parquet',columns=columns)
        x,ledger,prep=prepare_matrix(p,registry)
        ledger.to_csv(output/'all-predictor-representation-ledger.csv',index=False)
        write_json(output/'preprocessing.json',prep)
        with threadpool_limits(limits=2):
            record['discovery']=discovery(x,ledger,output/'discovery')
            if not discovery_only:
                targets=pd.read_csv(snapshot/'panel/target-pairs.csv.gz')
                record['development']=development(x,targets,ledger,p,output/'development')
        verify_snapshot(snapshot)
        record['status']='completed_discovery_and_development' if not discovery_only else 'completed_discovery_only'
        record['inputs_unchanged']=True
        record['completed_at']=datetime.now(timezone.utc).isoformat()
        record['source_series_in_library']=len(registry)
        record['preprocessing']=prep
        record['code_sha256']={str(f.relative_to(root)):sha256(f) for f in [Path(__file__),Path(__file__).with_name('discovery.py')]}
        write_json(output/'run.json',record)
        write_json(output/'output-checksums.json',{str(f.relative_to(output)):sha256(f) for f in sorted(output.rglob('*')) if f.is_file()})
        print('BROAD_EXECUTION_COMPLETE',record['status'],flush=True)
        return record
    except Exception as exc:
        record['status']='failed_incomplete_execution';record['error']=str(exc)
        write_json(output/'FAILED.json',record)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--discovery-only',action='store_true')
    a=parser.parse_args();run(a.snapshot,a.output,discovery_only=a.discovery_only)

if __name__=='__main__':main()
