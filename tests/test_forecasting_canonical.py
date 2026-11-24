"""Source-grounded contract tests, not forecast validation."""
import json
import numpy as np
import pandas as pd
import pytest
from src.forecasting.canonical import normalize_sdmx
from src.forecasting.inventory import ForecastDataError


def structure(source='FSIC'):
    ds={'FSIC':'DSD_FSIC','FSIBSIS':'DSD_FSIBSIS','MFS':'DSD_MFS_DCS','WEO':'DSD_WEO'}
    cl={'FSIC':'CL_FSIC_INDICATOR','FSIBSIS':'CL_FSIBSIS_INDICATOR','MFS':'CL_MFS_DCS_INDICATOR','WEO':'CL_WEO_INDICATOR'}
    dims=['COUNTRY','INDICATOR','FREQUENCY']
    if source in ['FSIC','FSIBSIS']:dims.insert(1,'SECTOR')
    if source=='MFS':dims.insert(2,'TYPE_OF_TRANSFORMATION')
    return {'data':{'dataStructures':[{'id':ds[source],'dataStructureComponents':{
        'dimensionList':{'dimensions':[{'id':c,'position':i} for i,c in enumerate(dims)]},
        'attributeList':{'attributes':[{'id':'UNIT','attributeRelationship':{'dimensions':['INDICATOR']}}]}}}],
        'codelists':[{'id':cl[source],'codes':[{'id':'TEST','name':'A real measure'}]},
                     {'id':'CL_OBS_STATUS','codes':[{'id':'B','name':'Time series break'},{'id':'F','name':'Forecast value'}]},
                     {'id':'CL_MFS_TYPE_OF_TRANSFORMATION','codes':[{'id':'XDC','name':'Domestic currency'},{'id':'SA_XDC','name':'Seasonally adjusted domestic currency'},{'id':'PCH_CP_A_PT','name':'Percent change'}]}]}}


def row(**kw):
    return dict({'COUNTRY':'KEN','SECTOR':'S12C','INDICATOR':'TEST','FREQUENCY':'A','TIME_PERIOD':'2020',
                 'OBS_VALUE':12.,'SCALE':0,'UNIT':'PT','STATUS':None},**kw)


def run(rows,source='FSIC'):
    return normalize_sdmx(pd.DataFrame(rows),source,structure(source),retrieved_at='2026-09-16T00:00:00Z',cutoff='2026-09-16')


def test_group_codes_do_not_truncate_or_become_conflicts():
    x=run([row(COUNTRY='G110'),row(COUNTRY='G119',OBS_VALUE=15)],'WEO')
    assert set(x.observations.entity_code)=={'G110','G119'}
    assert x.summary['conflicting_cells']==0


def test_mfs_adjusted_unadjusted_and_growth_are_distinct():
    x=run([row(TYPE_OF_TRANSFORMATION=t,UNIT=None,OBS_VALUE=v) for t,v in [('XDC',100),('SA_XDC',99),('PCH_CP_A_PT',2)]],'MFS')
    assert x.summary['features']==3 and x.summary['conflicting_cells']==0
    assert set(x.registry.UNIT)=={'XDC','PT'}
    assert x.registry.unit_resolution.eq('registered_transformation_codelist').all()


def test_source_metadata_attachment_is_not_a_measurement():
    x=run([row(UNIT=None),row(COUNTRY=None,SECTOR=None,FREQUENCY=None,TIME_PERIOD=None,OBS_VALUE=None)])
    assert len(x.observations)==1 and x.registry.UNIT.tolist()==['PT']
    assert x.summary['inherited_attribute_values']==1 and x.summary['attribute_only_rows']==1


def test_conflicting_attached_unit_cannot_be_guessed():
    rows=[row(UNIT=None)]+[row(COUNTRY=None,SECTOR=None,FREQUENCY=None,TIME_PERIOD=None,OBS_VALUE=None,UNIT=u) for u in ['PT','USD']]
    with pytest.raises(ForecastDataError,match='Conflicting attached'):run(rows)


def test_attached_unit_cannot_contradict_observation():
    with pytest.raises(ForecastDataError,match='observation/attached'):
        run([row(UNIT='USD'),row(COUNTRY=None,SECTOR=None,FREQUENCY=None,TIME_PERIOD=None,OBS_VALUE=None)])


@pytest.mark.parametrize('field,value',[('SECTOR','S12O'),('SCALE',6),('UNIT','USD'),('CONSOLIDATION_BASIS','X')])
def test_semantic_dimensions_change_identity(field,value):
    x=run([row(),row(**{field:value})]);assert len(x.registry)==2


def test_large_raw_amount_is_not_multiplied_by_scale():
    x=run([row(OBS_VALUE=2e12,SCALE=6,UNIT='XDC')])
    assert x.observations.value.iloc[0]==2e12
    assert x.registry.SCALE.iloc[0]=='6'


def test_missing_value_and_negative_zero_are_not_fabricated():
    x=run([row(COUNTRY=c,OBS_VALUE=v) for c,v in [('AAA',None),('BBB',-2),('CCC',0)]])
    assert len(x.observations)==3 and x.summary['missing_canonical_values']==1
    assert x.observations.set_index('entity_code').value.loc['BBB']==-2


def test_status_break_is_decoded_not_hidden():
    x=run([row(STATUS='B')]);assert x.observations.status_label.iloc[0]=='Time series break'
    assert x.observations.available_at.isna().all() and x.observations.vintage_at.isna().all()


@pytest.mark.parametrize('seed',[0,42,2026])
def test_conflicts_duplicates_and_input_order(seed):
    rows=[row(),row(),row(OBS_VALUE=13),row(COUNTRY='BBB',STATUS='B')]
    a=run(rows);b=run(pd.DataFrame(rows).sample(frac=1,random_state=seed).to_dict('records'))
    assert a.summary==b.summary and a.summary['conflicting_cells']==1
    pd.testing.assert_frame_equal(a.observations,b.observations)
    pd.testing.assert_frame_equal(a.registry,b.registry)


def test_status_disagreement_quarantines_identical_values():
    assert run([row(STATUS='B'),row(STATUS='F')]).summary['conflicting_cells']==1


@pytest.mark.parametrize('kwargs,reason',[
    ({'FREQUENCY':'Q'},'frequency_period_mismatch'),
    ({'OBS_VALUE':'#VALUE!'},'invalid_numeric'),
    ({'TIME_PERIOD':'2027'},'after_cutoff'),
    ({'OBS_VALUE':np.inf},'invalid_numeric'),
])
def test_invalid_rows_reconciled(kwargs,reason):
    x=run([row(),row(COUNTRY='BBB',**kwargs)])
    assert x.summary['canonical_cells']==1 and x.summary['rejected_rows']==1
    assert x.rejected.exclusion_reason.iloc[0]==reason


def test_unknown_indicator_and_missing_numeric_identity_fail():
    with pytest.raises(ForecastDataError,match='Unregistered'):run([row(INDICATOR='NEW')])
    with pytest.raises(ForecastDataError,match='incomplete'):run([row(COUNTRY=None)])


def test_full_calendar_parsing_and_source_immutability():
    frame=pd.DataFrame([row(TIME_PERIOD='2020.0'),row(TIME_PERIOD='2021-Q4',FREQUENCY='Q'),row(TIME_PERIOD='2022-M12',FREQUENCY='M')])
    before=frame.copy(deep=True)
    x=normalize_sdmx(frame,'FSIC',structure(),retrieved_at='2026-09-16',cutoff='2026-09-16')
    assert set(x.observations.observation_period.dt.year)=={2020,2021,2022}
    pd.testing.assert_frame_equal(frame,before)


from src.forecasting.panel import annual_endpoints, build_retrospective_panel


def panel_fixture():
    r=pd.DataFrame([dict(feature_id='f',source='FSIC',INDICATOR='AQ12_CFSI_PT',SECTOR='S12CFSI',UNIT='PT',SCALE='0',FREQUENCY='A')])
    d=pd.DataFrame([dict(entity_code=c,feature_id='f',observation_period=f'{y}-12-31',value=v,status_code='UNKNOWN') for c in ['KEN','G110'] for y,v in [(2017,1.),(2018,2.),(2020,4.),(2021,5.)]])
    return annual_endpoints(d),r


def test_panel_exact_calendar_lags_and_targets():
    e,r=panel_fixture();x=build_retrospective_panel(e,r,{'KEN'},start_origin=2019,end_origin=2020)
    p=x.predictors
    assert not ((p.forecast_origin_year==2020)&p.representation.eq('level')).any()
    assert not ((p.forecast_origin_year==2021)&p.representation.eq('change_1y')).any()
    row=p[p.forecast_origin_year.eq(2019)&p.representation.eq('change_1y')]
    assert row.value.tolist()==[1.]
    assert len(x.context)>0 and set(x.context.entity_code)=={'G110'}
    assert set(x.targets.entity_code)=={'KEN'} and not x.targets.target_publication_verified.any()
    assert x.summary['models_fitted']==0


def test_projection_and_non_endpoint_rows_cannot_become_annual_labels():
    d=pd.DataFrame([dict(entity_code='KEN',feature_id='f',observation_period=p,value=v,status_code=s) for p,v,s in [('2020-06-30',1,'UNKNOWN'),('2020-12-31',10,'F'),('2021-12-31',np.nan,'UNKNOWN')]])
    a=annual_endpoints(d)
    assert len(a)==2 and not a.retrospective_eligible.any()


def test_break_in_any_month_blocks_corresponding_change():
    e,r=panel_fixture()
    rows=e[['entity_code','feature_id','observation_period','value','status_code']].copy()
    rows=pd.concat([rows,pd.DataFrame([dict(entity_code='KEN',feature_id='f',observation_period='2018-03-31',value=1.5,status_code='B')])],ignore_index=True)
    x=build_retrospective_panel(annual_endpoints(rows),r,{'KEN'},start_origin=2019,end_origin=2020)
    assert not x.predictors.representation.eq('change_1y').any()


def test_target_definitions_never_collapse_distinct_regimes():
    e,r=panel_fixture();other=e.copy();other.feature_id='f2';other.value+=5
    reg=pd.concat([r,r.assign(feature_id='f2')],ignore_index=True)
    x=build_retrospective_panel(pd.concat([e,other]),reg,{'KEN'},start_origin=2019,end_origin=2020)
    assert x.targets.empty
    assert 'ambiguous_target_definition' in set(x.target_exclusions.exclusion_reason)


def test_panel_reordering_and_future_values_do_not_change_past_predictors():
    e,r=panel_fixture();a=build_retrospective_panel(e,r,{'KEN'},start_origin=2019,end_origin=2020)
    later=e.copy();later.loc[later.observation_year>=2020,'value']=999
    b=build_retrospective_panel(later.iloc[::-1],r,{'KEN'},start_origin=2019,end_origin=2020)
    pd.testing.assert_frame_equal(a.predictors,b.predictors)


def test_panel_does_not_accept_unknown_features_or_duplicate_cells():
    e,r=panel_fixture()
    with pytest.raises(ForecastDataError,match='Unregistered'):
        build_retrospective_panel(e.assign(feature_id='unknown'),r,{'KEN'})
    with pytest.raises(ForecastDataError,match='Nonunique'):
        build_retrospective_panel(pd.concat([e,e]),r,{'KEN'})
