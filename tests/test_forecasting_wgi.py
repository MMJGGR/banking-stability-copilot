import numpy as np
import pandas as pd
import pytest
from src.forecasting.wgi import records_from_pages,recover_wgi
from src.forecasting.inventory import ForecastDataError


def fixture():
    raw=pd.DataFrame([{'indicator_code':'GOV_WGI_GE.SC','feature_name':'govt_effectiveness',
                      'country_code':iso,'country_name':name,'year':2020,'value':v}
                     for iso,name,v in [('KEN','Kenya',50.),('', 'Territory A',60.),('', 'Territory B',70.)]])
    records=[{'country':{'id':id_,'value':r.country_name},'countryiso3code':r.country_code,
              'date':str(r.year),'indicator':{'id':r.indicator_code},'value':r.value}
             for id_,r in zip(['KE','TA','TB'],raw.itertuples())]
    return raw,records


def test_missing_iso_entities_survive_without_guessing_or_merging():
    raw,records=fixture();before=raw.copy(deep=True)
    obs,reg,ids,s=recover_wgi(raw,records_from_pages([[{},records]]),retrieved_at='2026-09-16')
    assert set(obs.entity_code)=={'KEN','WB:TA','WB:TB'}
    assert s['rows_with_missing_iso3_retained']==2 and s['values_preserved']
    assert len(obs)==len(raw) and len(reg)==6
    assert ids.mapping_status.eq('source_id_preserved_iso_mapping_unresolved').sum()==2
    pd.testing.assert_frame_equal(raw,before)


@pytest.mark.parametrize('kind',['numeric_revision','missing_id','name_collision','iso_change','conflicting_record'])
def test_reconciliation_failures_are_not_silently_deduplicated(kind):
    raw,records=fixture()
    if kind=='numeric_revision':records[0]['value']=51.
    if kind=='missing_id':records[0]['country']['id']=''
    if kind=='name_collision':records[2]['country']['value']=records[1]['country']['value']
    if kind=='iso_change':records[0]['countryiso3code']='DIFFERENT'
    if kind=='conflicting_record':records.append({**records[0],'value':51.})
    with pytest.raises(ForecastDataError):
        recover_wgi(raw,records_from_pages([[{},records]]),retrieved_at='2026-09-16')


def test_equal_pagination_duplicates_are_idempotent_and_row_order_invariant():
    raw,records=fixture()
    a=recover_wgi(raw,records_from_pages([[{},records]]),retrieved_at='2026-09-16')
    b=recover_wgi(raw.iloc[::-1],records_from_pages([[{},records[::-1]+records]]),retrieved_at='2026-09-16')
    for i in range(3):pd.testing.assert_frame_equal(a[i],b[i])
    assert a[3]==b[3]


def test_empty_and_invalid_pages_fail():
    for pages in [[],[[{},None]],[[{},[]]]]:
        with pytest.raises(ForecastDataError):records_from_pages(pages)


def test_incompatible_measure_scale_is_not_silently_accepted():
    raw,records=fixture();raw.loc[0,'value']=-2.5;records[0]['value']=-2.5
    with pytest.raises(ForecastDataError,match='0-100'):
        recover_wgi(raw,records_from_pages([[{},records]]),retrieved_at='2026-09-16')
