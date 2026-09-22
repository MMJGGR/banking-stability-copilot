"""Recover World Bank source entity IDs without changing the saved WGI values.

Some territories have both main-endpoint ID fields empty. Recover their
published code from the WGI source-specific Country dimension; never combine
blank IDs, guess mappings, or drop records. Archive the catalogue separately.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .inventory import ForecastDataError

MEASURES = {
    'GOV_WGI_VA.SC':'voice_accountability', 'GOV_WGI_PV.SC':'political_stability',
    'GOV_WGI_GE.SC':'govt_effectiveness', 'GOV_WGI_RQ.SC':'regulatory_quality',
    'GOV_WGI_RL.SC':'rule_of_law', 'GOV_WGI_CC.SC':'control_corruption',
}


def source_country_ids(payload: dict) -> dict[str, str]:
    """Read the complete official WGI source-3 Country dimension."""
    if not isinstance(payload, dict) or int(payload.get('pages', 0)) != 1 or int(payload.get('page', 0)) != 1:
        raise ForecastDataError('Incomplete source-country catalogue')
    sources=[x for x in payload.get('source',[]) if str(x.get('id'))=='3']
    if len(sources)!=1: raise ForecastDataError('Not the WGI source-country catalogue')
    concepts=[c for c in sources[0].get('concept',[]) if str(c.get('id','')).lower()=='country']
    if len(concepts)!=1: raise ForecastDataError('Missing country dimension')
    rows=concepts[0].get('variable',[])
    if not rows or len(rows)!=int(payload.get('total',-1)):
        raise ForecastDataError('Truncated country dimension')
    mapping={};codes=set()
    for row in rows:
        code=str(row.get('id') or '').strip();name=str(row.get('value') or '').strip()
        if not code or not name or code in codes or name in mapping:
            raise ForecastDataError('Ambiguous source-country catalogue')
        mapping[name]=code;codes.add(code)
    return mapping


def records_from_pages(pages: list, country_catalog: dict[str,str] | None = None) -> pd.DataFrame:
    rows=[]
    for payload in pages:
        if not isinstance(payload,list) or len(payload)!=2 or not isinstance(payload[1],list):
            raise ForecastDataError('Unrecognized World Bank page')
        for record in payload[1]:
            if record.get('value') is None: continue
            country=record.get('country') or {}
            source_id=str(country.get('id') or '').strip()
            name=str(country.get('value') or '').strip()
            basis='main_endpoint_country_id'
            if country_catalog is not None:
                source_id=country_catalog.get(name,'')
                basis='source_country_dimension'
            measure=(record.get('indicator') or {}).get('id')
            date=str(record.get('date') or '')
            if not source_id or not name or measure not in MEASURES or not date.isdigit():
                raise ForecastDataError('Missing/unrecognized World Bank entity or measure')
            rows.append({'source_entity_id':source_id,'source_id_basis':basis,'country_name':name,
                         'source_iso3':str(record.get('countryiso3code') or '').strip(),
                         'indicator_code':measure,'year':int(date),'verification_value':record['value']})
    if not rows: raise ForecastDataError('No populated WGI observations')
    data=pd.DataFrame(rows).drop_duplicates()
    keys=['source_entity_id','indicator_code','year']
    if data.duplicated(keys).any(): raise ForecastDataError('Conflicting World Bank source identity')
    return data


def recover_wgi(raw: pd.DataFrame, metadata: pd.DataFrame, *, retrieved_at: str):
    """Enrich a fixed raw download using separately retrieved verified IDs.

    Name/measure/year is used only as a checked one-to-one reconciliation key,
    never as the final series identity. Every original row must reconcile to
    a live source ID with unchanged numerical value. Original values survive
    unchanged, including minor JSON/CSV floating-point serialization effects.
    """
    required={'indicator_code','feature_name','country_code','country_name','year','value'}
    if not required<=set(raw) or not set(metadata.columns)>= {'source_entity_id','country_name','source_iso3','indicator_code','year','verification_value'}:
        raise ForecastDataError('Incomplete WGI recovery inputs')
    original=raw.copy(); keys=['country_name','indicator_code','year']
    if original.duplicated(keys).any() or metadata.duplicated(keys).any():
        raise ForecastDataError('Country-name reconciliation is not one-to-one')
    if not original.indicator_code.isin(MEASURES).all(): raise ForecastDataError('Unregistered WGI measure')
    if not original.feature_name.eq(original.indicator_code.map(MEASURES)).all():
        raise ForecastDataError('WGI measure/label mismatch')
    joined=original.merge(metadata,on=keys,how='left',validate='one_to_one')
    if joined.source_entity_id.isna().any(): raise ForecastDataError('Unreconciled WGI source record')
    original_iso=joined.country_code.fillna('').str.strip()
    if not original_iso.eq(joined.source_iso3).all():
        raise ForecastDataError('Source ISO metadata changed: explicit review required')
    values=pd.to_numeric(joined.value,errors='raise').astype(float)
    verification=pd.to_numeric(joined.verification_value,errors='raise').astype(float)
    if not np.isfinite(values).all() or not values.between(0,100).all():
        raise ForecastDataError('Invalid 0-100 governance-score values')
    difference=(values-verification).abs()
    if not np.isfinite(verification).all() or (difference>1e-12).any():
        raise ForecastDataError('WGI values changed; cannot silently mix retrieval vintages')
    basis=joined.source_id_basis if 'source_id_basis' in joined else pd.Series('main_endpoint_country_id',index=joined.index)
    published=basis.eq('source_country_dimension')
    if (published & original_iso.ne('') & original_iso.ne(joined.source_entity_id)).any():
        raise ForecastDataError('Source-country code disagrees with reported ISO3')
    joined['entity_code']=original_iso.where(original_iso.ne(''),'WB:'+joined.source_entity_id)
    joined.loc[published,'entity_code']=joined.loc[published,'source_entity_id']
    joined['source_id_basis']=basis
    identity_map=joined[['source_entity_id','source_id_basis','entity_code','country_name','source_iso3']].drop_duplicates()
    if identity_map.source_entity_id.duplicated().any() or identity_map.entity_code.duplicated().any():
        raise ForecastDataError('Ambiguous source entity crosswalk')
    identity_map['mapping_status']=np.where(identity_map.source_iso3.eq(''),
        'source_id_preserved_iso_mapping_unresolved','source_iso3')
    recovered=identity_map.source_id_basis.eq('source_country_dimension')
    identity_map.loc[recovered,'mapping_status']='verified_source_country_dimension'
    joined['feature_id']='WGI:'+joined.indicator_code
    joined['observation_period']=pd.to_datetime(joined.year.astype(str)+'-12-31')
    joined['status_code']='UNKNOWN'
    joined['available_at']=pd.NaT; joined['vintage_at']=pd.NaT
    joined['retrieved_at']=retrieved_at
    joined['historical_status']='current_vintage_historical_availability_unverified'
    joined['value']=values
    fields=['entity_code','source_entity_id','country_name','feature_id','observation_period',
            'value','status_code','available_at','vintage_at','retrieved_at','historical_status']
    data=joined[fields].sort_values(['entity_code','feature_id','observation_period']).reset_index(drop=True)
    if data.duplicated(['entity_code','feature_id','observation_period']).any():
        raise ForecastDataError('Nonunique repaired WGI observation')
    registry=pd.DataFrame([{'feature_id':'WGI:'+code,'INDICATOR':code,'indicator_label':name,
        'source':'WGI','FREQUENCY':'A','UNIT':'SCORE_0_100','SCALE':'0',
        'unit_resolution':'official_SC_measure_definition',
        'library_state':'canonical_retrospective_not_model_validated'} for code,name in MEASURES.items()])
    summary={'source':'WGI','raw_rows':len(raw),'canonical_cells':len(data),
             'features':len(registry),'entities':len(identity_map),'conflicting_cells':0,
             'unresolved_unit_features':0,'rows_with_missing_iso3_retained':int(original_iso.eq('').sum()),
             'entities_with_missing_iso3_retained':int(identity_map.source_iso3.eq('').sum()),
             'values_preserved':True,'maximum_verification_roundoff':float(difference.max()),
             'historical_release_dates_available':False}
    return data,registry,identity_map.sort_values('entity_code').reset_index(drop=True),summary
