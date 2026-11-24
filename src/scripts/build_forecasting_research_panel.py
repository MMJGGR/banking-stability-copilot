"""Explicit live research retrieval and M2 panel; never trains or promotes.

Downloads complete official responses to a NEW research directory. Partitions
by full entity code to bound memory, while preserving global attached metadata.
The raw response and DSD are archived for every resulting canonical cell.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import pandas as pd
import requests

from src.forecasting.canonical import normalize_sdmx, _codes, CODELIST
from src.forecasting.inventory import ForecastDataError
from src.forecasting.panel import annual_endpoints, build_retrospective_panel
from src.scripts.audit_forecasting_sources import output_destination, sha256
from src.sources.sdmx import build_sdmx_sources


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=str,allow_nan=False)+'\n')


def partition_raw(path, destination):
    destination.mkdir()
    metadata=[];total=0
    for chunk in pd.read_csv(path,chunksize=150000,dtype=str):
        total+=len(chunk)
        dated=chunk.COUNTRY.notna() & chunk.TIME_PERIOD.notna()
        metadata.append(chunk.loc[~dated])
        for entity,group in chunk.loc[dated].groupby('COUNTRY',sort=False):
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',entity):
                raise ForecastDataError(f'Unsafe/unrecognized full entity code {entity!r}')
            p=destination/(entity+'.csv.gz')
            group.to_csv(p,index=False,mode='a',header=not p.exists(),compression='gzip')
    return pd.concat(metadata,ignore_index=True),total


def process_imf(source, client, output, cutoff):
    dest=output/source;dest.mkdir()
    url=client.structure_url()+'?references=all'
    response=requests.get(url,headers={'Accept':'application/vnd.sdmx.structure+json'},timeout=180)
    response.raise_for_status()
    (dest/'structure.json').write_bytes(response.content)
    structure=response.json()
    result=client.fetch(dest,timeout=600,retries=2)
    receipt=asdict(result)
    raw=Path(result.path)
    receipt['structure_url']=url
    receipt['structure_sha256']=sha256(dest/'structure.json')
    receipt['data_url']=client.data_url()
    write_json(dest/'retrieval.json',receipt)
    metadata,total=partition_raw(raw,dest/'partitions')
    metadata.to_csv(dest/'attribute-only-rows.csv.gz',index=False)
    regs=[];endpoints=[];counts=Counter();statuses=Counter();entity_rows=[]
    output_cells=dest/'canonical';output_cells.mkdir()
    for index,p in enumerate(sorted((dest/'partitions').glob('*.csv.gz'))):
        entity=p.name[:-7]
        frame=pd.concat([pd.read_csv(p,dtype=str),metadata],ignore_index=True)
        versions=frame.STRUCTURE_ID.dropna().unique()
        flows=[f for f in structure['data'].get('dataflows',[]) if f['id']==client.dataflow_id]
        expected={f'{client.agency}:{client.dataflow_id}({f["version"]})' for f in flows}
        if len(versions)!=1 or versions[0] not in expected:
            raise ForecastDataError(f'Data/structure edition mismatch: {source} {versions} {expected}')
        normalized=normalize_sdmx(frame,source,structure,retrieved_at=result.retrieved_at,cutoff=cutoff)
        normalized.observations.to_parquet(output_cells/(entity+'.parquet'),index=False)
        if not normalized.rejected.empty:normalized.rejected.to_csv(dest/(entity+'-rejected.csv.gz'),index=False)
        if not normalized.conflicts.empty:normalized.conflicts.to_csv(dest/(entity+'-conflicts.csv.gz'),index=False)
        regs.append(normalized.registry)
        endpoints.append(annual_endpoints(normalized.observations))
        for k,v in normalized.summary.items():
            if isinstance(v,int) and k not in ['raw_rows','attribute_only_rows','features','entities','unit_unresolved_features']:
                counts[k]+=v
        statuses.update(normalized.summary['status_counts'])
        entity_rows.append({'entity_code':entity,'canonical_cells':len(normalized.observations),
                            'features':len(normalized.registry),'conflicts':len(normalized.conflicts)})
        if index%40==0:print(source,'entities completed',index+1,flush=True)
        del frame,normalized
        gc.collect()
    registry=pd.concat(regs,ignore_index=True).drop_duplicates().reset_index(drop=True)
    if not registry.feature_id.is_unique:raise ForecastDataError('Cross-entity registry metadata conflict')
    registry.to_csv(dest/'registry.csv',index=False)
    endpoints=pd.concat(endpoints,ignore_index=True)
    endpoints.to_parquet(dest/'annual-endpoints.parquet',index=False)
    entity_frame=pd.DataFrame(entity_rows)
    country_lists=[c for c in structure['data']['codelists'] if c['id']=='CL_COUNTRY']
    names={v['id']:v.get('name',v['id']) for v in country_lists[0]['codes']} if len(country_lists)==1 else {}
    entity_frame['source_label']=entity_frame.entity_code.map(names).fillna('unmapped_in_general_country_codelist')
    entity_frame.to_csv(dest/'entities.csv',index=False)
    declared=_codes(structure,CODELIST[source])
    pd.DataFrame([{'indicator_code':k,'label':v,'has_pre_cutoff_observations':k in set(registry.INDICATOR)} for k,v in declared.items()]).to_csv(dest/'declared-measures.csv',index=False)
    summary={'source':source,'raw_rows':total,'attribute_only_rows':len(metadata),**dict(counts),
             'features':len(registry),'entities':len(entity_frame),'unresolved_unit_features':int(registry.UNIT.isna().sum()),
             'status_counts':dict(statuses),'declared_measures':len(declared),
             'source_response_sha256':result.sha256,'retrieved_at':result.retrieved_at}
    assert summary['dated_rows']+len(metadata)==total
    assert summary['dated_rows']==summary['rejected_rows']+summary['canonical_cells']+summary['conflicting_rows']+summary['equal_duplicate_rows']
    write_json(dest/'summary.json',summary)
    packed=dest/'raw-response.csv.gz'
    with raw.open('rb') as incoming,gzip.open(packed,'wb',compresslevel=5) as outgoing:shutil.copyfileobj(incoming,outgoing)
    assert sha256(raw)==result.sha256
    receipt['archived_raw_file']='raw-response.csv.gz';receipt['archived_sha256']=sha256(packed)
    write_json(dest/'retrieval.json',receipt)
    raw.unlink();shutil.rmtree(dest/'partitions')
    print('SOURCE_COMPLETE',json.dumps(summary),flush=True)
    return registry,endpoints,summary


def process_wgi(client,output,cutoff):
    from src.forecasting.resume_m2 import recover_wgi_download
    dest=output/'WGI';dest.mkdir()
    result=client.fetch(dest,timeout=120,retries=2)
    raw=Path(result.path)
    # Preserve the original receipt as well as the separate metadata-recovery
    # ledger. The values stay in their original download vintage.
    write_json(dest/'original-download-receipt.json',asdict(result))
    return recover_wgi_download(raw,dest,retrieved_at=result.retrieved_at,cutoff=cutoff)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cutoff',required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    out=output_destination(repo,args.output)
    manifest=repo/'artifacts/data_manifest.json';data_manifest=json.loads(manifest.read_text())
    protected={manifest:sha256(manifest)}
    for source in ['FSIC','FSIBSIS','MFS','WEO','WGI']:
        p=repo/'cache'/f'{source}_cache.parquet';h=sha256(p)
        if h!=data_manifest['artifacts'][str(p.relative_to(repo))]['sha256']:raise ForecastDataError('Serving source checksum mismatch')
        protected[p]=h
    members_path=repo/'cache/crisis_features.parquet';protected[members_path]=sha256(members_path)
    if protected[members_path]!=data_manifest['artifacts']['cache/crisis_features.parquet']['sha256']:raise ForecastDataError('Serving member-list checksum mismatch')
    countries=set(pd.read_parquet(members_path,columns=['country_code']).country_code)
    out.mkdir(parents=True)
    report={'status':'incomplete','started_at':datetime.now(timezone.utc).isoformat(),
            'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=repo).strip(),
            'cutoff':args.cutoff,'sources':{},'model_training_performed':False,
            'serving_sha256_before':{str(k.relative_to(repo)):v for k,v in protected.items()}}
    write_json(out/'run.json',report)
    try:
        clients=build_sdmx_sources();registries=[];endpoints=[]
        for source in ['FSIC','FSIBSIS','MFS','WEO','WGI']:
            r,e,s=(process_wgi(clients[source],out,args.cutoff) if source=='WGI' else process_imf(source,clients[source],out,args.cutoff))
            registries.append(r);endpoints.append(e);report['sources'][source]=s
            write_json(out/'run.json',report)
        registry=pd.concat(registries,ignore_index=True)
        endpoint=pd.concat(endpoints,ignore_index=True)
        registry.to_csv(out/'complete-registry.csv',index=False)
        panel=build_retrospective_panel(endpoint,registry,countries,start_origin=2003,end_origin=2023,information_lag_years=1)
        folder=out/'panel';folder.mkdir()
        panel.predictors.to_parquet(folder/'predictors.parquet',index=False)
        panel.context.to_parquet(folder/'context-candidates.parquet',index=False)
        panel.targets.to_csv(folder/'target-pairs.csv.gz',index=False)
        panel.target_exclusions.to_csv(folder/'target-exclusions.csv.gz',index=False)
        panel.predictors.groupby(['entity_code','forecast_origin_year']).agg(source_features=('feature_id','nunique'),predictor_variants=('predictor_id','nunique')).reset_index().to_csv(folder/'coverage.csv',index=False)
        panel.predictors[panel.predictors.entity_code.eq('KEN')&panel.predictors.forecast_origin_year.eq(2023)].to_csv(folder/'kenya-2023-input-example.csv',index=False)
        write_json(folder/'summary.json',panel.summary)
        for p,h in protected.items():
            if sha256(p)!=h:raise ForecastDataError('Production input mutated during research')
        report.update(status='completed_M2_data_execution_not_model_validation',panel=panel.summary,
                      inputs_unchanged=True,completed_at=datetime.now(timezone.utc).isoformat())
        write_json(out/'run.json',report)
        write_json(out/'output-checksums.json',{str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()})
        print('RESEARCH_COMPLETE',json.dumps(report),flush=True)
    except Exception as error:
        report.update(status='failed_incomplete',error=f'{type(error).__name__}: {error}')
        write_json(out/'run.json',report)
        raise


if __name__=='__main__':main()
