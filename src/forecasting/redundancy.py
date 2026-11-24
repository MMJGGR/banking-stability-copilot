"""Target-independent reference sensitivities; no feature deletion or serving use.

See PRD v0.4. Source balancing is a diagnostic assumption, not an economically
preferred weighting. Every weight is fitted only on the supplied reference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from .inventory import ForecastDataError


class WeightedReferenceGeometry:
    """PCA of a preprocessed matrix with positive, frozen column weights."""
    def __init__(self, mode='unweighted', variance_fraction=.9):
        if mode not in {'unweighted', 'exact_profile', 'source_energy'}:
            raise ForecastDataError('Unknown weighting mode')
        if not 0 < variance_fraction <= 1:
            raise ForecastDataError('Invalid variance fraction')
        self.mode, self.variance_fraction = mode, variance_fraction

    @staticmethod
    def _validate(z, observed):
        if not isinstance(z, pd.DataFrame) or z.empty or not z.index.is_unique or not z.columns.is_unique:
            raise ForecastDataError('Unique nonempty reference matrix required')
        if not isinstance(observed, pd.DataFrame) or not observed.index.equals(z.index) or not observed.columns.equals(z.columns):
            raise ForecastDataError('Observation masks must exactly align')
        if observed.isna().any().any() or any(t != bool for t in observed.dtypes):
            raise ForecastDataError('Boolean observation masks required')
        if not np.isfinite(z.to_numpy(dtype=float)).all():
            raise ForecastDataError('Preprocessed inputs must be finite')
        return z.to_numpy(dtype=float), observed.to_numpy()

    def fit(self, z, observed, sources):
        a, mask = self._validate(z, observed)
        if len(z) < 2 or not isinstance(sources, pd.Series) or not sources.index.equals(z.columns):
            raise ForecastDataError('At least two rows and aligned source labels required')
        if sources.isna().any() or sources.astype(str).str.strip().eq('').any():
            raise ForecastDataError('Missing source identity')
        self.columns_ = z.columns.copy()
        self.sources_ = sources.copy()
        self.mean_ = a.mean(axis=0)
        centered = a - self.mean_
        energy = (centered**2).sum(axis=0)
        if not np.all(energy > 0) or not mask.any(axis=0).all():
            raise ForecastDataError('Only varying observed inputs enter this kernel; retain others in admission ledger')
        # Canonical reference row order prevents hashing from depending on row order.
        order = z.index.argsort()
        hashes = []
        for j in range(a.shape[1]):
            values = np.ascontiguousarray(a[order, j]).copy()
            values[values == 0] = 0.0  # canonicalize signed zero
            hashes.append(hashlib.sha256(values.tobytes()+mask[order, j].tobytes()).hexdigest())
        groups = pd.Series(hashes, index=z.columns)
        counts = groups.map(groups.value_counts()).astype(float)
        weights = pd.Series(1., index=z.columns)
        if self.mode == 'exact_profile':
            weights = 1 / np.sqrt(counts)
        elif self.mode == 'source_energy':
            totals = pd.Series(energy, index=z.columns).groupby(sources, sort=True).sum()
            weights = np.sqrt((totals.sum()/len(totals)) / sources.map(totals))
        if not np.isfinite(weights).all() or not weights.gt(0).all():
            raise ForecastDataError('Invalid fitted weights')
        self.weights_ = weights
        self.weight_ledger_ = pd.DataFrame({'source': sources, 'profile_sha256': groups,
            'profile_size': counts.astype(int), 'weight': weights,
            'unweighted_reference_energy': energy,
            'weighted_reference_energy': energy*weights.to_numpy()**2})
        self.weight_ledger_.index.name = 'predictor_id'
        b = centered * weights.to_numpy()
        n, p = b.shape
        if n <= p:
            vals, u = eigh(b @ b.T, driver='evd')
            idx = np.argsort(vals)[::-1]; vals, u = vals[idx], u[:, idx]
            tol = max(float(vals[0]), 1.) * np.finfo(float).eps * max(n,p)
            keep = vals > tol
            self.eigenvalues_ = vals[keep]
            self.loadings_ = (b.T @ u[:,keep]) / np.sqrt(self.eigenvalues_)
        else:
            vals, v = eigh(b.T @ b, driver='evd')
            idx = np.argsort(vals)[::-1]; vals, v = vals[idx], v[:,idx]
            tol = max(float(vals[0]), 1.) * np.finfo(float).eps * max(n,p)
            keep = vals > tol
            self.eigenvalues_, self.loadings_ = vals[keep], v[:,keep]
        if not len(self.eigenvalues_):
            raise ForecastDataError('Zero numerical rank')
        self.reference_rows_ = n
        self.variance_ratios_ = self.eigenvalues_/self.eigenvalues_.sum()
        self.n_components_ = self.components_for(self.variance_fraction)
        self.features_retained_ = p
        return self

    def components_for(self, fraction):
        if not 0 < fraction <= 1:
            raise ForecastDataError('Invalid requested variance fraction')
        return min(len(self.eigenvalues_), int(np.searchsorted(np.cumsum(self.variance_ratios_),fraction))+1)

    def spectrum(self):
        return pd.DataFrame({'component':np.arange(1,len(self.eigenvalues_)+1),
            'explained_variance_ratio':self.variance_ratios_,
            'cumulative_variance_ratio':np.cumsum(self.variance_ratios_)})

    def contributions(self):
        k = self.n_components_
        result = self.weight_ledger_.copy()
        result['common_variance_share'] = self.loadings_[:,:k]**2 @ self.variance_ratios_[:k]
        return result

    def describe(self, z, observed):
        a, mask = self._validate(z, observed)
        if not z.columns.equals(self.columns_):
            raise ForecastDataError('Frozen reference schema mismatch')
        b = (a-self.mean_)*self.weights_.to_numpy()
        k = self.n_components_
        scores = b @ self.loadings_[:,:k]
        residual = b - scores @ self.loadings_[:,:k].T
        observed_mass = mask @ (self.weights_.to_numpy()**2)
        common = np.sum(scores**2/(self.eigenvalues_[:k]/(self.reference_rows_-1)),axis=1)/k
        residual_mse = np.sum(np.where(mask,residual**2,0),axis=1)/np.maximum(observed_mass,np.finfo(float).tiny)
        common[observed_mass==0] = np.nan
        residual_mse[observed_mass==0] = np.nan
        return pd.DataFrame({'observed_share':mask.mean(axis=1),
            'common_component_distance':common,'observed_weighted_residual_mse':residual_mse,
            'status':np.where(observed_mass>0,'descriptive_not_risk_or_forecast','insufficient_observed_inputs')}, index=z.index)


def run(snapshot, output):
    """Execute fixed v0.4 diagnostics without loading any target file."""
    from .discovery import BroadSpectralSpace
    from .discovery_execution import prepare_matrix, verify_snapshot, write_json
    from src.scripts.audit_forecasting_sources import output_destination, sha256
    from threadpoolctl import threadpool_limits
    snapshot = Path(snapshot).resolve()
    output = output_destination(snapshot, Path(output))
    _, prior, verified = verify_snapshot(snapshot)
    # Paths are obtained from the immutable M2 inventory rather than a target file.
    predictors_path = snapshot/'panel'/'predictors.parquet'
    registry_path = snapshot/'feature-registry.csv'
    if not predictors_path.is_file() or not registry_path.is_file():
        raise ForecastDataError('Expected registered M2 predictor/registry paths unavailable')
    predictors = pd.read_parquet(predictors_path)
    registry = pd.read_csv(registry_path)
    x, ledger, preprocessing = prepare_matrix(predictors, registry)
    del predictors
    year = int(x.index.get_level_values('forecast_origin_year').max())
    if year != 2023:
        raise ForecastDataError('Reference origin differs from preregistered 2023')
    current = x.xs(year,level='forecast_origin_year')
    previous = x.xs(year-1,level='forecast_origin_year')
    output.mkdir(parents=True)
    records = []; geometries = {}
    with threadpool_limits(limits=2):
        original = BroadSpectralSpace(.9).fit(current)
        tr = original.transformer_
        cols = pd.Index(tr.active_features_, name='predictor_id')
        z = pd.DataFrame(tr.transform(current),index=current.index,columns=cols)
        mask = current[cols].notna()
        oldz = pd.DataFrame(tr.transform(previous),index=previous.index,columns=cols)
        oldmask = previous[cols].notna()
        source = ledger.set_index('predictor_id').source.reindex(cols)
        tr.admission_.merge(ledger,on='predictor_id',validate='one_to_one').to_csv(output/'all-feature-admission.csv',index=False)
        for mode in ['unweighted','exact_profile','source_energy']:
            g = WeightedReferenceGeometry(mode).fit(z,mask,source)
            g.spectrum().to_csv(output/f'{mode}-spectrum.csv',index=False)
            c = g.contributions().join(ledger.set_index('predictor_id')[['INDICATOR','indicator_label','UNIT','unit_policy']])
            c.to_csv(output/f'{mode}-all-feature-contributions.csv')
            by_source = c.groupby('source').agg(predictors=('weight','size'),weighted_energy=('weighted_reference_energy','sum'),common_variance_share=('common_variance_share','sum'))
            by_source.to_csv(output/f'{mode}-source-contributions.csv')
            d = g.describe(z,mask); d.to_csv(output/f'{mode}-reference-distinctness.csv')
            g.describe(oldz,oldmask).to_csv(output/f'{mode}-previous-frozen-reference.csv')
            geometries[mode] = d
            records.append({'mode':mode,'reference_inputs':len(cols),'features_dropped_by_weighting':0,
                'components_80':g.components_for(.8),'components_90':g.components_for(.9),'components_95':g.components_for(.95),
                'first_component_share':float(g.variance_ratios_[0]),'unique_profiles':int(c.profile_sha256.nunique()),
                'spectrum_sum':float(g.variance_ratios_.sum())})
            if mode == 'unweighted':
                np.testing.assert_allclose(g.variance_ratios_,original.variance_ratios_,rtol=1e-9,atol=1e-12)
            if mode == 'exact_profile':
                reps = g.weight_ledger_.reset_index().drop_duplicates('profile_sha256').predictor_id.tolist()
                collapsed = BroadSpectralSpace(.9).fit(current[reps])
                np.testing.assert_allclose(g.variance_ratios_,collapsed.variance_ratios_,rtol=1e-8,atol=1e-11)
            if mode == 'source_energy':
                np.testing.assert_allclose(by_source.weighted_energy.to_numpy(),np.repeat(by_source.weighted_energy.iloc[0],len(by_source)),rtol=1e-10)
        correlations = []
        for mode in ['exact_profile','source_energy']:
            for field in ['common_component_distance','observed_weighted_residual_mse']:
                correlations.append({'mode':mode,'field':field,'spearman_vs_unweighted':float(geometries['unweighted'][field].corr(geometries[mode][field],method='spearman'))})
    summary = {'status':'completed_descriptive_sensitivity_not_predictive_validation',
        'source_cutoff':prior['cutoff'],'reference_origin':year,'latest_observation_year_under_lag':year-1,
        'reference_entities':len(current),'all_representations_in_ledger':len(current.columns),
        'varying_inputs_retained_in_every_sensitivity':len(cols),'feature_cap':None,
        'targets_loaded':0,'forecast_models_fitted':0,'production_classifier_retrained':False,
        'verified_M2_files':verified,'preprocessing':preprocessing,'specifications':records,
        'geometry_sensitivity':correlations,'baseline_reproduced':True,'duplicate_collapse_equivalence_passed':True,
        'equal_source_energy_passed':True,'limitations':[
            'Source-energy balance is a diagnostic assumption, not a selected model or economic weighting.',
            'Reference origin 2023 is retrospective and is not a current-2026 feature snapshot.',
            'Distinctness is not risk, causality, significance or forecast skill.',
            'Historical publication/vintage dates remain unverified.']}
    write_json(output/'summary.json',summary)
    checks = {str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output/'output-checksums.json',checks)
    print(json.dumps(summary,indent=2,allow_nan=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    run(args.snapshot,args.output)
