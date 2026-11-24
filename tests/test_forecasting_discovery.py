"""Behaviour tests; synthetic data is not predictive-validation evidence."""
import pickle
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from src.forecasting.discovery import WideRankTransform, BroadSpectralSpace
from src.forecasting.discovery_execution import prepare_matrix
from src.forecasting.inventory import ForecastDataError


def matrix(n=35,p=70):
    rng=np.random.default_rng(172)
    x=pd.DataFrame(rng.normal(size=(n,p)),columns=[f'feature{i}' for i in range(p)])
    x.loc[::3,'feature0']=np.nan
    x['constant']=1.;x['unknown']=np.nan
    return x


def test_no_feature_cap_more_features_than_rows():
    x=matrix();s=BroadSpectralSpace().fit(x)
    assert len(s.transformer_.active_features_)==70
    assert len(s.transformer_.admission_)==72
    assert set(s.transformer_.admission_.state)=={'learnable','constant_observed_in_reference','all_missing_in_reference'}


@pytest.mark.parametrize('seed',[0,7,42])
def test_row_and_column_order_do_not_change_spectrum_or_predictions(seed):
    x=matrix();y=x.feature1-x.feature2
    a=BroadSpectralSpace().fit(x)
    reordered=x.sample(frac=1,random_state=seed)[x.columns[::-1]]
    b=BroadSpectralSpace().fit(reordered)
    np.testing.assert_allclose(a.eigenvalues_,b.eigenvalues_,rtol=1e-11,atol=1e-10)
    np.testing.assert_allclose(a.ridge_predict(x,y),b.ridge_predict(x,y.reindex(reordered.index)),rtol=1e-10,atol=1e-10)


def test_prediction_and_future_only_features_cannot_change_fitted_preprocessing():
    x=matrix();s=BroadSpectralSpace().fit(x)
    before=pickle.dumps(s.transformer_)
    revised=x.copy();revised['feature1']=1e12;revised['unknown']=10
    s.transformer_.transform(revised)
    assert pickle.dumps(s.transformer_)==before
    assert 'unknown' not in s.transformer_.active_features_


def test_low_variance_feature_is_retained_and_can_predict():
    rng=np.random.default_rng(21)
    x=pd.DataFrame({'a':rng.normal(size=100),'b':rng.normal(size=100)*1e-9})
    s=BroadSpectralSpace(.5).fit(x)
    assert set(s.transformer_.active_features_)=={'a','b'}
    y=x.b.rank()/100
    prediction=s.ridge_predict(x,y,alpha=.01)
    assert np.corrcoef(prediction,y)[0,1]>.99


def test_full_spectral_ridge_matches_reference_solver():
    x=matrix();y=pd.Series(np.sin(np.arange(len(x))),index=x.index)
    s=BroadSpectralSpace().fit(x)
    expected=Ridge(alpha=100.,solver='svd').fit(s.reference_,y).predict(s.reference_)
    np.testing.assert_allclose(expected,s.ridge_predict(x,y,alpha=100.),atol=1e-10)


def test_small_feature_space_and_pickle_supported():
    x=matrix(n=70,p=3);s=BroadSpectralSpace().fit(x)
    restored=pickle.loads(pickle.dumps(s))
    np.testing.assert_allclose(s.loadings(),restored.loadings())
    assert s.components_for(.95)<=3


def test_all_missing_query_is_unsupported_not_safe():
    x=matrix();s=BroadSpectralSpace().fit(x)
    q=x.iloc[:1].copy();q[:]=np.nan
    diag,_,_=s.describe(q)
    assert diag.status.iloc[0]=='insufficient_observed_inputs'
    assert np.isnan(diag.common_component_distance.iloc[0])
    assert np.isnan(diag.observed_residual_mse.iloc[0])


def test_loadings_reconstruct_all_training_variance_without_sign_constraints():
    x=matrix();s=BroadSpectralSpace().fit(x);v=s.loadings()
    np.testing.assert_allclose(s.reference_,s.reference_@v@v.T,atol=1e-9)
    assert (v<0).any() and (v>0).any()
    assert s.spectrum().cumulative_variance_ratio.iloc[-1]==pytest.approx(1.)


def test_exact_profile_duplicates_auditable_not_erased():
    x=matrix();x['duplicate']=x.feature1
    s=BroadSpectralSpace().fit(x);g=s.exact_profile_groups(x).set_index('predictor_id')
    assert g.loc['duplicate','profile_sha256']==g.loc['feature1','profile_sha256']
    assert g.loc['duplicate','profile_size']==2
    assert 'duplicate' in s.transformer_.feature_names_in_


@pytest.mark.parametrize('field',['target','country_code','forecast_origin_year'])
def test_metadata_never_becomes_economic_coordinate(field):
    x=matrix();x[field]=1.
    with pytest.raises(ForecastDataError):WideRankTransform().fit(x)


def panel():
    rows=[]
    for entity in ['A','B']:
        for year in range(2003,2012):
            for feature,unit in [('money','XDC'),('ratio','PT'),('unknown','')]:
                rows.append({'entity_code':entity,'forecast_origin_year':year,
                    'observation_year':year-1,'predictor_id':feature+'::level','feature_id':feature,
                    'value':float((year-2001)**2)*(10 if entity=='B' and feature=='money' else 1),
                    'break_in_year':year==2009})
    registry=pd.DataFrame({'feature_id':['money','ratio','unknown'], 'source':['MFS','FSIC','WEO'],
        'INDICATOR':['money','ratio','u'], 'indicator_label':['money','ratio','u'],
        'UNIT':['XDC','PT',np.nan], 'SCALE':[6,0,0]})
    return pd.DataFrame(rows),registry


def test_causal_amount_normalization_ignores_future_values_and_currency_units():
    d,r=panel();x,ledger,_=prepare_matrix(d,r)
    updated=d.copy();updated.loc[updated.forecast_origin_year>=2008,'value']*=100
    y,_,_=prepare_matrix(updated,r)
    pd.testing.assert_frame_equal(x.loc[pd.IndexSlice[:,2003:2007],:],y.loc[pd.IndexSlice[:,2003:2007],:])
    np.testing.assert_allclose(x.loc['A','money::level'],x.loc['B','money::level'],equal_nan=True)
    assert x['unknown::level'].isna().all()
    assert len(ledger)==3


def test_amount_break_resets_history_without_erasing_library_feature():
    d,r=panel();x,ledger,_=prepare_matrix(d,r)
    assert np.isnan(x.loc[('A',2009),'money::level'])
    assert np.isnan(x.loc[('A',2010),'money::level'])
    assert np.isfinite(x.loc[('A',2011),'money::level'])
    assert np.isfinite(x.loc[('A',2009),'ratio::level'])


def test_duplicate_or_future_predictor_cells_fail():
    d,r=panel()
    with pytest.raises(ForecastDataError):prepare_matrix(pd.concat([d,d.iloc[[0]]]),r)
    d.loc[0,'observation_year']=3000
    with pytest.raises(ForecastDataError):prepare_matrix(d,r)


def test_no_target_driven_feature_selection_y_ignored():
    x=matrix();a=WideRankTransform().fit(x,y=np.arange(len(x)))
    b=WideRankTransform().fit(x,y=-np.arange(len(x)))
    np.testing.assert_array_equal(a.transform(x),b.transform(x))


def test_invalid_penalty_or_alignment_rejected():
    x=matrix();s=BroadSpectralSpace().fit(x);y=pd.Series(np.arange(len(x)),index=x.index)
    with pytest.raises(ForecastDataError):s.ridge_predict(x,y,alpha=0)
    with pytest.raises(ForecastDataError):s.ridge_predict(x,y.iloc[::-1])
