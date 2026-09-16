"""Resume the verified M2 source snapshot; no model training or production writes.

The four IMF sources have already completed normalization. Their raw archives,
DSDs and source hashes are reverified here. A separately recorded World Bank
metadata retrieval recovers source IDs; the original WGI values must reconcile
unchanged. The enclosing workflow pins the entire input ZIP checksum.
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
from .wgi import MEASURES, records_from_pages, recover_wgi
from src.scripts.audit_forecasting_sources import sha256, output_destination

ORIGINAL_CODE='b55adf71ac83d9c1cfa4186fe19b4954c53b875c'
ORIGINAL_ARTIFACT=10461733710
ORIGINAL_ARCHIVE_SHA='6c8122d3fff4d9a0468b9bb76229720916c29cf1e90cdffbd92994062625c30d'
WGI_RAW_SHA='f69df5978649612597245434ecde6331fda0face4c96090b696aaa4d4585f6e4'


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')


def recover_wgi_download(raw_path: Path, destination: Path, *, retrieved_at: str, cutoff: str):
    """Keep original values; recover IDs from complete, archived API pages."""
    raw_hash=sha256(raw_path)
    pages_dir=destination/'identity-recovery-pages';pages_dir.mkdir(exist_ok=False)
    pages=[];receipts=[]
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
    canonical,registry,entities,summary=recover_wgi(raw,records_from_pages(pages),retrieved_at=retrieved_at)
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
                   metadata_recovered_at=receipts[-1]['checked_at'],
                   original_values_preserved=True)
    write_json(destination/'summary.json',summary)
    write_json(destination/'retrieval.json',{
        'source':'WGI','original_raw_file':raw_path.name,'sha256':raw_hash,
        'bytes':raw_path.stat().st_size,'original_export_timestamp':retrieved_at,
        'identity_recovery':'Separate official API pages; every original value reconciled before IDs attached.',
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
    # The IMF normalizer and annual panel definitions must match the completed
    # source run. New WGI recovery utilities are separate and explicitly tested.
    subprocess.run(['git','diff','--exit-code',ORIGINAL_CODE,'HEAD','--',
                    'src/forecasting/canonical.py','src/forecasting/panel.py'],cwd=repo,check=True)
    for name,digest in prior['serving_sha256_before'].items():
        if sha256(repo/name)!=digest:raise ForecastDataError('Serving comparison baseline changed')
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
        matched=panel.targets.merge(coverage[['entity_code','forecast_origin_year']],on=['entity_code','forecast_origin_year'],how='inner',validate='many_to_one')
        panel.summary['target_pairs_with_at_least_one_predictor']=len(matched)
        panel.summary['target_pair_country_coverage']=matched.groupby(['target_name','horizon_years']).entity_code.nunique().rename('countries').reset_index().to_dict('records')
        write_json(folder/'summary.json',panel.summary)
        for name,digest in prior['serving_sha256_before'].items():
            if sha256(repo/name)!=digest:raise ForecastDataError('Serving inputs changed during research')
        report.update(status='completed_M2_data_execution_not_model_validation',panel=panel.summary,
                      candidate_source_series=len(registry),inputs_unchanged=True,
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
