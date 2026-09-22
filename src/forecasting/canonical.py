"""Research-only SDMX normalizer preserving published identities and metadata.

No serving loader imports this module. OBS_VALUE is kept unchanged: SCALE is
recorded, never blindly applied a second time. Public-release/vintage times
remain unknown unless separately established. See PRD data contract v0.2.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import re
import numpy as np
import pandas as pd
from .inventory import ForecastDataError

DSD = {'FSIC': 'DSD_FSIC', 'FSIBSIS': 'DSD_FSIBSIS',
       'MFS': 'DSD_MFS_DCS', 'WEO': 'DSD_WEO'}
CODELIST = {'FSIC': 'CL_FSIC_INDICATOR', 'FSIBSIS': 'CL_FSIBSIS_INDICATOR',
            'MFS': 'CL_MFS_DCS_INDICATOR', 'WEO': 'CL_WEO_INDICATOR'}
TRANSFORM_UNIT = {'XDC':'XDC','USD':'USD','EUR':'EUR','XDR':'XDR',
                  'SA_XDC':'XDC','SA_USD':'USD','SA_EUR':'EUR','PT':'PT',
                  'EOP_PT':'PT','PCH_CP_A_PT':'PT','RMBBMPT_A_PT':'PT',
                  'PT_A_PT':'PT','PT_A':'PT','IX':'IX','SAF_IX':'IX',
                  'EOP_IX':'IX','PA_IX':'IX'}
REGIME = ('UNIT','SCALE','CONSOLIDATION_BASIS','INTRAGROUP_ADJUSTMENTS',
          'ACCOUNTING_STANDARDS','DEFINITION')

@dataclass
class CanonicalResult:
    observations: pd.DataFrame
    registry: pd.DataFrame
    rejected: pd.DataFrame
    conflicts: pd.DataFrame
    summary: dict


def _clean(value):
    if pd.isna(value): return None
    if isinstance(value, (np.integer, np.floating)): value=value.item()
    if isinstance(value, float) and value.is_integer(): return str(int(value))
    return str(value).strip() or None


def _codes(payload: dict, identifier: str) -> dict:
    found = [x for x in payload['data'].get('codelists',[]) if x['id']==identifier]
    if len(found)!=1: raise ForecastDataError(f'Expected one {identifier} codelist, found {len(found)}')
    return {x['id']:x.get('names',{}).get('en',x.get('name',x['id'])) for x in found[0]['codes']}


def _parse_period(label):
    label=str(label).strip()
    if re.fullmatch(r'\d{4}\.0',label): label=label[:-2]
    if re.fullmatch(r'\d{4}',label): return pd.Timestamp(label+'-12-31'),'A'
    if re.fullmatch(r'\d{4}-Q[1-4]',label): return pd.Period(label,freq='Q').end_time.normalize(),'Q'
    if re.fullmatch(r'\d{4}-M(0[1-9]|1[0-2])',label):
        return pd.Period(label.replace('-M','-'),freq='M').end_time.normalize(),'M'
    return pd.NaT,None


def normalize_sdmx(frame: pd.DataFrame, source: str, structure: dict,
                   *, retrieved_at: str, cutoff: str) -> CanonicalResult:
    """Normalize a COMPLETE raw response, including its attribute-only rows.

    A complete response is required because attributes may occur at its end.
    Missing dated observations are retained as explicit unavailable cells.
    Conflicting fully specified cells are quarantined, not averaged. Equal
    values with contradictory status are also conflicts, not free precedence.
    """
    if source not in DSD: raise ForecastDataError('Unsupported source')
    structures=[s for s in structure['data']['dataStructures'] if s['id']==DSD[source]]
    if len(structures)!=1: raise ForecastDataError('Ambiguous/missing DSD')
    comps=structures[0]['dataStructureComponents']
    dims=[x['id'] for x in sorted(comps['dimensionList']['dimensions'],key=lambda x:x['position'])]
    required=set(dims+['TIME_PERIOD','OBS_VALUE'])
    if not frame.columns.is_unique or not required<=set(frame):
        raise ForecastDataError(f'Missing/duplicate raw columns: {required-set(frame)}')
    # All source attributes remain in the immutable raw response. Drop only
    # columns empty throughout this response from this working representation.
    if pd.isna(pd.Timestamp(retrieved_at)) or pd.isna(pd.Timestamp(cutoff)):
        raise ForecastDataError('Missing retrieval/cutoff date')
    if not {'COUNTRY','INDICATOR','FREQUENCY'} <= set(dims):
        raise ForecastDataError('Unsupported source dimension contract')
    data=frame.dropna(axis=1, how="all").copy()
    for c in required:
        if c not in data: data[c] = None
    # Numeric values are never used to infer identities.
    for c in data.columns.difference(['OBS_VALUE']): data[c]=data[c].map(_clean)
    obs_mask=data[dims+['TIME_PERIOD']].notna().all(axis=1)
    metadata=data.loc[~obs_mask].copy()
    if metadata.OBS_VALUE.notna().any():
        raise ForecastDataError('Numeric observation with incomplete source key')
    obs=data.loc[obs_mask].copy()
    labels=_codes(structure,CODELIST[source])
    unknown=set(obs.INDICATOR)-set(labels)
    if unknown: raise ForecastDataError(f'Unregistered indicator codes: {sorted(unknown)}')
    inherited=0
    # SDMX attributes carry attachment rules; no cross-row ffill or guessed join.
    for attr in comps['attributeList'].get('attributes',[]):
        c=attr['id']; keys=attr.get('attributeRelationship',{}).get('dimensions')
        if c not in data or not keys or not set(keys)<=set(metadata): continue
        rows=metadata.loc[metadata[c].notna() & metadata[keys].notna().all(axis=1),keys+[c]].drop_duplicates()
        if rows.empty: continue
        if rows.duplicated(keys).any(): raise ForecastDataError(f'Conflicting attached metadata: {c}')
        joined=obs[keys].merge(rows,on=keys,how='left',validate='many_to_one')[c]
        joined.index=obs.index
        disagree=obs[c].notna() & joined.notna() & obs[c].ne(joined)
        if disagree.any(): raise ForecastDataError(f'Conflicting observation/attached metadata: {c}')
        inherited+=int((obs[c].isna() & joined.notna()).sum())
        obs[c]=obs[c].where(obs[c].notna(), joined)
    for c in REGIME:
        if c not in obs: obs[c]=None
    obs['unit_resolution']='source_attribute'
    transforms={}
    if source=='MFS':
        transforms=_codes(structure,'CL_MFS_TYPE_OF_TRANSFORMATION')
        if set(obs.TYPE_OF_TRANSFORMATION)-set(transforms): raise ForecastDataError('Unknown MFS transformation')
        inferred=obs.TYPE_OF_TRANSFORMATION.map(TRANSFORM_UNIT)
        conflict=obs.UNIT.notna() & inferred.notna() & obs.UNIT.ne(inferred)
        if conflict.any(): raise ForecastDataError('Transformation/unit disagreement')
        resolved=obs.UNIT.isna() & inferred.notna()
        obs.loc[resolved,'UNIT']=inferred[resolved]
        obs.loc[resolved,'unit_resolution']='registered_transformation_codelist'
    obs.loc[obs.UNIT.isna(),'unit_resolution']='unresolved'
    mapping={v:_parse_period(v) for v in obs.TIME_PERIOD.unique()}
    obs['observation_period']=pd.to_datetime(obs.TIME_PERIOD.map(lambda x:mapping[x][0]))
    inferred_freq=obs.TIME_PERIOD.map(lambda x:mapping[x][1])
    numeric=pd.to_numeric(obs.OBS_VALUE,errors='coerce')
    reason=pd.Series('',index=obs.index)
    reason.loc[obs.observation_period.isna()]='invalid_period'
    reason.loc[inferred_freq.ne(obs.FREQUENCY)]='frequency_period_mismatch'
    reason.loc[obs.OBS_VALUE.notna() & ~np.isfinite(numeric)]='invalid_numeric'
    obs['value']=numeric.astype(float)
    obs['status_code']=obs.STATUS.fillna('UNKNOWN') if 'STATUS' in obs else 'UNKNOWN'
    status_map=_codes(structure,'CL_OBS_STATUS')
    obs['status_label']=obs.status_code.map(status_map).fillna('unknown/unrecognized source status')
    # Do not guess that a blank/unknown status means actual. A number may still
    # be a retrospective research observation, but not a verified realization.
    obs['historical_status']='source_status_not_real_time_verified'
    future=obs.observation_period>pd.Timestamp(cutoff)
    reason.loc[future & reason.eq('')]='after_cutoff'
    obs['exclusion_reason']=reason
    rejected=obs.loc[reason.ne('')].copy()
    obs=obs.loc[reason.eq('')].copy()
    identity=[c for c in dims if c!='COUNTRY']+list(REGIME)
    identity=list(dict.fromkeys(identity))
    unique=obs[identity].drop_duplicates().copy()
    unique['identity_json']=unique.apply(lambda row:json.dumps({'source':source,**{k:_clean(v) for k,v in row.items()}},sort_keys=True,separators=(',',':')),axis=1)
    unique['feature_id']=unique.identity_json.map(lambda x:source+':'+hashlib.sha256(x.encode()).hexdigest())
    obs=obs.merge(unique,on=identity,how='left',validate='many_to_one')
    keys=['COUNTRY','feature_id','observation_period']
    grouped=obs.groupby(keys,dropna=False,sort=True)
    counts=grouped.size().rename('rows').to_frame()
    counts['distinct_values']=grouped.value.nunique(dropna=False)
    counts['distinct_status']=grouped.status_code.nunique(dropna=False)
    counts=counts.reset_index()
    conflicts=counts.loc[(counts.distinct_values>1)|(counts.distinct_status>1)].copy()
    obs=obs.merge(conflicts[keys].assign(_conflict=True),on=keys,how='left',validate='many_to_one')
    bad=obs._conflict.eq(True)
    clean=obs.loc[~bad].copy()
    # Column-level provenance remains in the archived raw response. Select
    # only deterministic canonical fields when identical observations coalesce.
    clean=clean.rename(columns={'COUNTRY':'entity_code'})
    clean['retrieved_at']=pd.Timestamp(retrieved_at)
    clean['available_at']=pd.NaT;clean['vintage_at']=pd.NaT
    fields=['entity_code','feature_id','observation_period','value','status_code',
            'status_label','historical_status','retrieved_at','available_at','vintage_at']
    fields+=[c for c in ['DERIVATION_TYPE','COUNTRY_UPDATE_DATE'] if c in clean]
    # Extra informational metadata is retained as a sorted set per cell, not
    # used to break a tie between conflicting measurements.
    extra=[c for c in fields if c in ['DERIVATION_TYPE','COUNTRY_UPDATE_DATE']]
    for c in extra:
        clean[c]=clean[c].fillna('UNKNOWN')
    if extra:
        cellkeys=['entity_code','feature_id','observation_period']
        unique_rows=clean[fields].drop_duplicates()
        multiple=unique_rows.duplicated(cellkeys,keep=False)
        singles=unique_rows.loc[~multiple]
        multiples=unique_rows.loc[multiple].copy()
        for c in extra:
            if not multiples.empty:
                multiples[c]=multiples.groupby(cellkeys)[c].transform(lambda x:'|'.join(sorted(set(x))))
        clean=pd.concat([singles,multiples],ignore_index=True)
    clean=clean[fields].drop_duplicates().sort_values(['entity_code','feature_id','observation_period']).reset_index(drop=True)
    if clean.duplicated(['entity_code','feature_id','observation_period']).any():raise ForecastDataError('Unresolved canonical key')
    registry=unique.copy()
    registry['source']=source;registry['indicator_label']=registry.INDICATOR.map(labels)
    registry['transformation_label']=registry.TYPE_OF_TRANSFORMATION.map(transforms) if 'TYPE_OF_TRANSFORMATION' in registry else ''
    registry['unit_resolution']=registry.feature_id.map(obs.groupby('feature_id').unit_resolution.first())
    registry['library_state']='canonical_retrospective_not_model_validated'
    if not registry.feature_id.is_unique:raise ForecastDataError('Nonunique feature ID')
    summary={'source':source,'raw_rows':len(data),'attribute_only_rows':len(metadata),'dated_rows':int(obs_mask.sum()),
             'rejected_rows':len(rejected),'canonical_cells':len(clean),'conflicting_cells':len(conflicts),
             'conflicting_rows':int(conflicts.rows.sum()),'equal_duplicate_rows':len(obs)-int(bad.sum())-len(clean),
             'missing_canonical_values':int(clean.value.isna().sum()),'features':len(registry),
             'entities':int(clean.entity_code.nunique()),'inherited_attribute_values':inherited,
             'unit_unresolved_features':int(registry.UNIT.isna().sum()),
             'raw_value_policy':'unchanged_no_scale_multiplication',
             'vintage_mode':'retrospective_latest_vintage','source_dimensions':dims,
             'status_counts':{str(k):int(v) for k,v in clean.status_code.value_counts().items()}}
    assert summary['dated_rows']==summary['rejected_rows']+summary['canonical_cells']+summary['conflicting_rows']+summary['equal_duplicate_rows']
    return CanonicalResult(clean,registry.sort_values('feature_id').reset_index(drop=True),rejected,conflicts,summary)
