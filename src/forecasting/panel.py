"""Exact-calendar, ragged retrospective panel. No fitting or score changes.

The initial information lag is an explicit one-calendar-year research
assumption, NOT proof of historical publication. Annual, Q4 and December
values stay separate by feature ID. No ratio/stock/flow aggregation occurs.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .inventory import ForecastDataError

RESEARCH_STATUSES = frozenset({'UNKNOWN', 'A', 'B', 'E', 'P'})
TARGETS = {
    'npl_ratio': 'AQ12_CFSI_PT',
    'capital_adequacy': 'FSI688_CFSI_PT',
    'liquidity_st_liabilities': 'FSI765_CFSI_PT',
}


def annual_endpoints(observations: pd.DataFrame) -> pd.DataFrame:
    """Keep exact year ends and report any break in that feature's year."""
    required={'entity_code','feature_id','observation_period','value','status_code'}
    if not required<=set(observations):raise ForecastDataError('Incomplete canonical observations')
    d=observations.copy()
    d['observation_period']=pd.to_datetime(d.observation_period,errors='raise')
    if d.observation_period.isna().any():raise ForecastDataError('Missing observation period')
    keys=['entity_code','feature_id','observation_period']
    if d.duplicated(keys).any():raise ForecastDataError('Canonical observations are not unique')
    d['observation_year']=d.observation_period.dt.year
    d['_break']=d.status_code.eq('B')
    breaks=d.groupby(['entity_code','feature_id','observation_year'])._break.any()
    d=d[d.observation_period.dt.month.eq(12)&d.observation_period.dt.day.eq(31)].copy()
    d=d.join(breaks.rename('break_in_year'),on=['entity_code','feature_id','observation_year'])
    d['retrospective_eligible']=d.status_code.isin(RESEARCH_STATUSES)&np.isfinite(d.value)
    return d.drop(columns='_break').sort_values(keys).reset_index(drop=True)


@dataclass
class PanelResult:
    predictors: pd.DataFrame
    context: pd.DataFrame
    targets: pd.DataFrame
    target_exclusions: pd.DataFrame
    summary: dict


def build_retrospective_panel(endpoints: pd.DataFrame, registry: pd.DataFrame,
                              country_codes: set[str], *,
                              start_origin: int=2003, end_origin: int=2023,
                              information_lag_years: int=1) -> PanelResult:
    """Build all observed feature cells and exact target pairs, no imputation.

    Unknown units and unclassified entities remain staged and visible. This
    constructor does NOT declare them model-admitted. Context is not broadcast
    until its entity role and downstream experiment are explicitly registered.
    No past predictive outcomes are examined to select features or parameters.
    """
    if not isinstance(information_lag_years,int) or isinstance(information_lag_years,bool) or information_lag_years<1:
        raise ForecastDataError('Declare a positive calendar-year information lag')
    if start_origin>end_origin or not country_codes:raise ForecastDataError('Invalid panel configuration')
    if not registry.feature_id.is_unique:raise ForecastDataError('Registry identities must be unique')
    e=endpoints.copy()
    if not set(e.feature_id)<=set(registry.feature_id):raise ForecastDataError('Unregistered feature')
    if e.duplicated(['entity_code','feature_id','observation_year']).any():raise ForecastDataError('Nonunique annual endpoint')
    if not pd.to_datetime(e.observation_period).eq(pd.to_datetime(e.observation_year.astype(str)+'-12-31')).all():
        raise ForecastDataError('Only exact year-end source endpoints accepted')
    valid=e.loc[e.retrospective_eligible].copy()
    fields=['entity_code','feature_id','observation_year','observation_period','value','status_code','break_in_year']
    level=valid[fields].copy()
    level['forecast_origin_year']=level.observation_year+information_lag_years
    level['representation']='level'
    lag=valid[fields].copy()
    lag['forecast_origin_year']=lag.observation_year+information_lag_years+1
    lag['representation']='lag_1y'
    older=valid[['entity_code','feature_id','observation_year','value','break_in_year']].copy()
    older['observation_year']+=1
    joined=valid.merge(older,on=['entity_code','feature_id','observation_year'],suffixes=('','_prior'),validate='one_to_one')
    delta=joined.loc[~joined.break_in_year & ~joined.break_in_year_prior,fields+['value_prior']].copy()
    delta['value']=delta.value-delta.pop('value_prior')
    delta['forecast_origin_year']=delta.observation_year+information_lag_years
    delta['representation']='change_1y'
    values=pd.concat([level,lag,delta],ignore_index=True)
    values=values[values.forecast_origin_year.between(start_origin,end_origin)].copy()
    values['predictor_id']=values.feature_id+'::'+values.representation
    values['information_age_years']=values.forecast_origin_year-values.observation_year
    assert (values.information_age_years>=information_lag_years).all()
    values['experiment_mode']='retrospective_latest_vintage'
    values['admission_state']='staged_not_model_admitted'
    values=values.sort_values(['entity_code','forecast_origin_year','predictor_id']).reset_index(drop=True)
    if values.duplicated(['entity_code','forecast_origin_year','predictor_id']).any():raise ForecastDataError('Duplicate predictor cell')
    pred=values[values.entity_code.isin(country_codes)].copy()
    context=values[~values.entity_code.isin(country_codes)].copy()
    r=registry.copy()
    for c in ['SECTOR','UNIT','FREQUENCY','SCALE']:
        if c not in r:r[c]=None
    target_registry=r[r.source.eq('FSIC') & r.INDICATOR.isin(TARGETS.values()) &
                      r.SECTOR.eq('S12CFSI') & r.UNIT.eq('PT') &
                      r.FREQUENCY.eq('A') & r.SCALE.astype(str).isin(['0','0.0'])]
    code_to_name={v:k for k,v in TARGETS.items()}
    mapping=target_registry.set_index('feature_id').INDICATOR.map(code_to_name)
    truth=valid[valid.entity_code.isin(country_codes)&valid.feature_id.isin(mapping.index)].copy()
    truth['target_name']=truth.feature_id.map(mapping)
    pairs=[];exclusions=[]
    origins=pd.MultiIndex.from_product([sorted(country_codes),range(start_origin,end_origin+1)],names=['entity_code','forecast_origin_year']).to_frame(index=False)
    for target in TARGETS:
        selected=truth[truth.target_name.eq(target)].copy()
        for horizon in [1,2]:
            grid=origins.copy();grid['target_name']=target;grid['horizon_years']=horizon
            grid['target_year']=grid.forecast_origin_year+horizon
            dup=selected.duplicated(['entity_code','observation_year'],keep=False)
            ambiguous=set(zip(selected.loc[dup,'entity_code'],selected.loc[dup,'observation_year']))
            exact=selected.loc[~dup,['entity_code','observation_year','value','feature_id','status_code','break_in_year']].rename(columns={'observation_year':'target_year','value':'target_value','feature_id':'target_feature_id','status_code':'target_status','break_in_year':'target_break_in_year'})
            grid=grid.merge(exact,on=['entity_code','target_year'],how='left',validate='many_to_one')
            grid['exclusion_reason']=['ambiguous_target_definition' if (cc,y) in ambiguous else 'unavailable_exact_observed_target' if pd.isna(v) else '' for cc,y,v in zip(grid.entity_code,grid.target_year,grid.target_value)]
            grid['target_end']=pd.to_datetime(grid.target_year.astype(str)+'-12-31')
            grid['assumed_target_available_at']=pd.to_datetime((grid.target_year+information_lag_years).astype(str)+'-12-31')
            grid['target_publication_verified']=False
            grid['experiment_mode']='retrospective_latest_vintage'
            pairs.append(grid[grid.exclusion_reason.eq('')]);exclusions.append(grid[grid.exclusion_reason.ne('')])
    targets=pd.concat(pairs,ignore_index=True)
    rejected=pd.concat(exclusions,ignore_index=True)
    coverage=pred.groupby(['entity_code','forecast_origin_year']).predictor_id.nunique()
    summary={'status':'M2_retrospective_panel_not_predictive_validation',
             'forecast_origin_start':start_origin,'forecast_origin_end':end_origin,
             'information_lag_years_assumed':information_lag_years,
             'country_universe':'explicit_current_serving_membership_for_research_comparability',
             'country_members':len(country_codes),'country_origins_with_predictors':len(coverage),
             'countries_with_predictors':int(pred.entity_code.nunique()),
             'raw_source_features_in_country_panel':int(pred.feature_id.nunique()),
             'predictor_representations':int(pred.predictor_id.nunique()),
             'observed_predictor_cells':len(pred),'context_candidate_cells':len(context),
             'context_entities':sorted(context.entity_code.unique().tolist()),
             'target_pairs':len(targets),'unavailable_or_ambiguous_target_pairs':len(rejected),
             'target_counts':targets.groupby(['target_name','horizon_years']).size().rename('pairs').reset_index().to_dict('records'),
             'missing_values_imputed':0,'models_fitted':0,'final_holdout_evaluated':False,
             'limitations':['No verified historical public-release/vintage ledger.',
                            'Unknown units and domestic-currency scale normalization still require model-admission review.',
                            'Source families and global/regional context are retained, not silently counted as countries.',
                            'Annual endpoints do not imply that stocks, flows and ratios have identical semantics.']}
    return PanelResult(pred,context,targets,rejected,summary)
