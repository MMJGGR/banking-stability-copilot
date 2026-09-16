"""Resume the verified M2 source snapshot; no model training or production writes.

Completed IMF canonical stores retain their original checksummed raw inputs.
WGI entity recovery preserves the original values. Broad feature membership,
the published-score comparator and outcomes lacking inputs are explicit.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import pandas as pd
import requests
from .inventory import ForecastDataError
from .panel import annual_endpoints, build_retrospective_panel
from .wgi import MEASURES, records_from_pages, recover_wgi, source_country_ids
from src.scripts.audit_forecasting_sources import sha256, output_destination

ORIGINAL_CODE='b55adf71ac83d9c1cfa4186fe19b4954c53b875c'
ORIGINAL_ARTIFACT=10461733710
ORIGINAL_ARCHIVE_SHA='6c8122d3fff4d9a0468b9bb76229720916c29cf1e90cdffbd92994062625c30d'
WGI_RAW_SHA='f69df5978649612597245434ecde6331fda0face4c96090b696aaa4d4585f6e4'
SCORED_COHORT_PATH='docs/releases/2026-09-15/country-review.csv'
SCORED_COHORT_SHA='b8d0480a7d1fcc638e1cc4f59c0dc879d598fc7044beddb98c9f6a429711bbdb'


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')


def recover_wgi_download(raw_path: Path, destination: Path, *, retrieved_at: str, cutoff: str):
    """Keep original values; recover IDs from the official source dimension."""
    raw_hash=sha256(raw_path)
    pages_dir=destination/'identity-recovery-pages';pages_dir.mkdir(exist_ok=False)
    pages=[];receipts=[]
    catalog_url='https://api.worldbank.org/v2/sources/3/country/data?format=json&per_page=1000'
    catalog_response=requests.get(catalog_url,timeout=120)
    catalog_response.raise_for_status()
    catalog_path=destination/'source-country-dimension.json'
    catalog_path.write_bytes(catalog_response.content)
    catalog=source_country_ids(catalog_response.json())
    receipts.append({'url':catalog_url,'checked_at':datetime.now(timezone.utc).isoformat(),
                     'http_status':catalog_response.status_code,'bytes':len(catalog_response.content),
                     'file':catalog_path.name,'sha256':sha256(catalog_path)})
    for measure in MEASURES:
        page=1
        while True:
            url=f'https://api.worldbank.org/v2/country/all/indicator/{measure}'
            params={'source':3,'format':'json','per_page':20000,'page':page}
            response=requests.get(url,params=params,timeout=120)
            response.raise_for_status()
            payload=response.json()
            if not isinstance(payload,list) or len(payload)!=2 or not isinstance(payload[0],dict):
                raise ForecastDataError('Unrecognized World Bank pagination response')
            count=int(payload[0].get('pages',0))
            if int(payload[0].get('page',0))!=page or count<1 or count>100:
                raise ForecastDataError('Unreconciled World Bank pagination')
            path=pages_dir/f'{measure}-page-{page}.json';path.write_bytes(response.content)
            pages.append(payload)
            receipts.append({'url':response.url,'checked_at':datetime.now(timezone.utc).isoformat(),
                             'http_status':response.status_code,'bytes':len(response.content),
                             'file':str(path.relative_to(destination)),'sha256':sha256(path)})
            if page>=count: break
            page+=1
    write_json(destination/'identity-recovery-receipts.json',receipts)
    raw=pd.read_csv(raw_path,dtype={'country_code':str,'year':int},float_precision='round_trip')
    canonical,registry,entities,summary=recover_wgi(raw,records_from_pages(pages,catalog),retrieved_at=retrieved_at)
    future=canonical.observation_period>pd.Timestamp(cutoff)
    summary['after_cutoff_rows']=int(future.sum())
    canonical=canonical.loc[~future].copy()
    summary['canonical_cells']=len(canonical)
    canonical.to_parquet(destination/'canonical.parquet',index=False)
    registry.to_csv(destination/'registry.csv',index=False)
    entities.to_csv(destination/'entities.csv',index=False)
    endpoints=annual_endpoints(canonical)
    endpoints.to_parquet(destination/'annual-endpoints.parquet',index=False)
    summary.update(source_response_sha256=raw_hash,original_retrieved_at=retrieved_at,
                   metadata_recovered_at=receipts[-1]['checked_at'],original_values_preserved=True)
    write_json(destination/'summary.json',summary)
    write_json(destination/'retrieval.json',{
        'source':'WGI','original_raw_file':raw_path.name,'sha256':raw_hash,
        'bytes':raw_path.stat().st_size,'original_export_timestamp':retrieved_at,
        'identity_recovery':'Separate official country-dimension and observation API pages; every original value reconciled before IDs attached.',
        'historical_publication_dates':'not recovered; retrieval dates are not first-publication dates'})
    if sha256(raw_path)!=raw_hash:raise ForecastDataError('WGI original raw download changed')
    return registry,endpoints,summary


def verify_completed_imf(source_dir: Path):
    receipt=json.loads((source_dir/'retrieval.json').read_text())
    if sha256(source_dir/'structure.json')!=receipt['structure_sha256']:
        raise ForecastDataError('Archived DSD checksum mismatch')
    raw=source_dir/receipt['archived_raw_file']
    if sha256(raw)!=receipt['archived_sha256']:raise ForecastDataError('Compressed raw source checksum mismatch')
    digest=hashlib.sha256();size=0
    with gzip.open(raw,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block);size+=len(block)
    if digest.hexdigest()!=receipt['sha256'] or size!=receipt['bytes']:
        raise ForecastDataError('Raw source-content checksum mismatch')
    summary=json.loads((source_dir/'summary.json').read_text())
    if summary['source_response_sha256']!=receipt['sha256']:
        raise ForecastDataError('Source summary lineage mismatch')
    if summary['dated_rows']!=sum(summary[k] for k in ['rejected_rows','canonical_cells','conflicting_rows','equal_duplicate_rows']):
        raise ForecastDataError('Source ledger does not reconcile')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    previous=args.previous.resolve();out=output_destination(repo,args.output)
    prior=json.loads((previous/'run.json').read_text())
    if prior['code_commit']!=ORIGINAL_CODE or prior['status']!='failed_incomplete' or prior['error']!='ForecastDataError: Nonunique WGI observation':
        raise ForecastDataError('Input is not the expected partial M2 source build')
    # Only the completed IMF normalizer must be unchanged for source reuse.
    # The annual panel is rebuilt below and separately checked by output hashes.
    subprocess.run(['git','diff','--exit-code',ORIGINAL_CODE,'HEAD','--',
                    'src/forecasting/canonical.py'],cwd=repo,check=True)
    for name,digest in prior['serving_sha256_before'].items():
        if sha256(repo/name)!=digest:raise ForecastDataError('Serving comparison baseline changed')
    cohort_path=repo/SCORED_COHORT_PATH
    if sha256(cohort_path)!=SCORED_COHORT_SHA:raise ForecastDataError('Scored comparator cohort checksum mismatch')
    scored_list=pd.read_csv(cohort_path,usecols=['country_code']).country_code
    if scored_list.isna().any() or not scored_list.is_unique:raise ForecastDataError('Invalid scored cohort')
    scored=set(scored_list)
    summaries={s:verify_completed_imf(previous/s) for s in ['FSIC','FSIBSIS','MFS','WEO']}
    raw_files=list((previous/'WGI').glob('worldbank_WGI_*.csv'))
    if len(raw_files)!=1 or sha256(raw_files[0])!=WGI_RAW_SHA:
        raise ForecastDataError('Unexpected original WGI file')
    out.mkdir(parents=True,exist_ok=False)
    report={'status':'incomplete','source_artifact':ORIGINAL_ARTIFACT,
            'source_archive_sha256':ORIGINAL_ARCHIVE_SHA,'source_build_commit':ORIGINAL_CODE,
            'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
            'cutoff':prior['cutoff'],'started_at':datetime.now(timezone.utc).isoformat(),
            'sources':summaries,'model_training_performed':False,
            'serving_sha256_before':prior['serving_sha256_before'],
            'scored_cohort_reference':{'path':SCORED_COHORT_PATH,'sha256':SCORED_COHORT_SHA,'entities':len(scored)},
            'source_reuse':'All four complete IMF canonical stores reused from checksum-pinned artifact. No IMF redownload.'}
    write_json(out/'run.json',report)
    try:
        registries=[];all_endpoints=[]
        for source in ['FSIC','FSIBSIS','MFS','WEO']:
            shutil.copytree(previous/source,out/source)
            r=pd.read_csv(out/source/'registry.csv',dtype=str,keep_default_na=False)
            e=pd.read_parquet(out/source/'annual-endpoints.parquet')
            if not r.feature_id.is_unique:raise ForecastDataError('Nonunique source registry')
            registries.append(r);all_endpoints.append(e)
        wgi_dir=out/'WGI';wgi_dir.mkdir()
        raw=wgi_dir/raw_files[0].name;shutil.copy2(raw_files[0],raw)
        stamp=datetime.strptime(raw.stem.replace('worldbank_WGI_',''),'%Y-%m-%dT%H%M%SZ').replace(tzinfo=timezone.utc).isoformat()
        r,e,s=recover_wgi_download(raw,wgi_dir,retrieved_at=stamp,cutoff=prior['cutoff'])
        registries.append(r);all_endpoints.append(e);report['sources']['WGI']=s
        registry=pd.concat(registries,ignore_index=True)
        endpoint=pd.concat(all_endpoints,ignore_index=True)
        if not registry.feature_id.is_unique:raise ForecastDataError('Cross-source registry collision')
        registry.to_csv(out/'complete-registry.csv',index=False)
        countries=set(pd.read_parquet(repo/'cache/crisis_features.parquet',columns=['country_code']).country_code)
        if not scored<=countries:raise ForecastDataError('A published scored country is missing from the broad feature universe')
        panel=build_retrospective_panel(endpoint,registry,countries,start_origin=2003,end_origin=2023,information_lag_years=1)
        folder=out/'panel';folder.mkdir()
        panel.predictors.to_parquet(folder/'predictors.parquet',index=False)
        panel.context.to_parquet(folder/'context-candidates.parquet',index=False)
        panel.targets.to_csv(folder/'target-pairs.csv.gz',index=False)
        panel.target_exclusions.to_csv(folder/'target-exclusions.csv.gz',index=False)
        coverage=panel.predictors.groupby(['entity_code','forecast_origin_year']).agg(
            source_features=('feature_id','nunique'),predictor_variants=('predictor_id','nunique')).reset_index()
        coverage.to_csv(folder/'coverage.csv',index=False)
        panel.predictors[panel.predictors.entity_code.eq('KEN')&panel.predictors.forecast_origin_year.eq(2023)].to_csv(folder/'kenya-2023-input-example.csv',index=False)
        audited=panel.targets.merge(coverage[['entity_code','forecast_origin_year']],on=['entity_code','forecast_origin_year'],how='left',validate='many_to_one',indicator=True)
        matched=audited[audited._merge.eq('both')].drop(columns='_merge')
        without=audited[audited._merge.ne('both')].drop(columns='_merge')
        matched.to_csv(folder/'target-pairs-with-inputs.csv.gz',index=False)
        without.assign(exclusion_reason='no_predictor_observed_at_origin').to_csv(folder/'targets-without-inputs.csv',index=False)
        membership=pd.DataFrame({'entity_code':sorted(countries)})
        membership['in_published_scored_cohort']=membership.entity_code.isin(scored)
        membership.to_csv(folder/'cohort-membership.csv',index=False)
        shutil.copy2(cohort_path,folder/'scored-cohort-reference.csv')
        panel.summary.update(country_universe_source='cache/crisis_features.parquet (raw feature universe)',
                             published_scored_cohort_entities=len(scored),
                             extra_unscored_entities=sorted(countries-scored),
                             published_scored_cohort_origins=int(coverage.entity_code.isin(scored).sum()),
                             target_pairs_with_at_least_one_predictor=len(matched),
                             targets_without_any_predictor_at_origin=len(without))
        panel.summary['target_pair_country_coverage']=matched.groupby(['target_name','horizon_years']).entity_code.nunique().rename('countries').reset_index().to_dict('records')
        write_json(folder/'summary.json',panel.summary)
        for name,digest in prior['serving_sha256_before'].items():
            if sha256(repo/name)!=digest:raise ForecastDataError('Serving inputs changed during research')
        # Cohort labelling/export changes must not change existing panel values.
        expected={'predictors.parquet':'b861afeeb3db49e06281fae95eb5892227fb05a7d50f5e882a49473a0f01c77a',
                  'context-candidates.parquet':'fc2ce844604724037e00bd6af9354136427e92e2e9546c72dde084128b82ba3a',
                  'coverage.csv':'7528982ad95c44f5ce7114ab39eb787f8826ec8c6cfe5bc8f1590bb77a9af9d6'}
        if any(sha256(folder/name)!=digest for name,digest in expected.items()):
            raise ForecastDataError('Cohort clarification unexpectedly changed panel data')
        with gzip.open(folder/'target-pairs.csv.gz','rb') as f:
            if hashlib.sha256(f.read()).hexdigest()!='616cbc89a1f6d3ebde3130a49cf1f320e42da6c374b6ffb519cd016ac5b9ec5f':
                raise ForecastDataError('Cohort clarification unexpectedly changed target values')
        report.update(status='completed_M2_data_execution_not_model_validation',panel=panel.summary,
                      candidate_source_series=len(registry),inputs_unchanged=True,
                      panel_values_unchanged_from_initial_successful_build=True,
                      completed_at=datetime.now(timezone.utc).isoformat())
        write_json(out/'run.json',report)
        write_json(out/'prior-failed-run.json',prior)
        write_json(out/'output-checksums.json',{str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()})
        print('M2_DATA_EXECUTION_COMPLETE',json.dumps(report),flush=True)
    except Exception as error:
        report.update(status='failed_incomplete',error=f'{type(error).__name__}: {error}')
        write_json(out/'run.json',report)
        raise


if __name__=='__main__':main()
