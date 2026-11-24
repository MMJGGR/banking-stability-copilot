"""Uncapped, target-independent structure and broad spectral regression.

Variance and distinctness are descriptive, not risk or predictive validity.
All transforms are fitted on the reference/training rows only. Full library
identities remain auditable even when a fold cannot learn from a column.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from .baseline import _validate
from .inventory import ForecastDataError


class WideRankTransform(TransformerMixin, BaseEstimator):
    """Observed empirical midranks; missing values map to the reference median.

    No target, feature-count budget, sign restriction or coverage threshold.
    Rank transforms address units/outliers, not cross-country currency identity:
    amounts must first pass the independently declared economic-unit policy.
    """
    def fit(self, X: pd.DataFrame, y=None):
        d = _validate(X).sort_index(axis=1)
        self.feature_names_in_ = np.asarray(d.columns, dtype=object)
        self.n_features_in_ = len(d.columns)
        self.sorted_values_ = {}
        rows = []
        for name in d:
            values = np.sort(d[name].dropna().to_numpy())
            reason = 'learnable'
            if len(values) == 0:
                reason = 'all_missing_in_reference'
            elif values[0] == values[-1]:
                reason = 'constant_observed_in_reference'
            else:
                self.sorted_values_[name] = values
            rows.append({'predictor_id': name, 'reference_observations': len(values),
                         'reference_rows': len(d), 'state': reason})
        if not self.sorted_values_:
            raise ForecastDataError('No varying observed reference predictors')
        self.active_features_ = list(self.sorted_values_)
        self.admission_ = pd.DataFrame(rows)
        a = self._ranks(d)
        # Each observed column has zero mean under empirical midranks. Missing
        # median fills do not create an economic missingness coordinate.
        count = d[self.active_features_].notna().sum().to_numpy()
        self.scale_ = np.sqrt(np.sum(a*a, axis=0) / count)
        self.mean_ = (a / self.scale_).mean(axis=0)
        return self

    def _ranks(self, X):
        out = np.zeros((len(X), len(self.active_features_)), dtype=np.float64)
        for j, name in enumerate(self.active_features_):
            values = X[name].to_numpy(dtype=float)
            mask = np.isfinite(values)
            ref = self.sorted_values_[name]
            lo = np.searchsorted(ref, values[mask], side='left')
            hi = np.searchsorted(ref, values[mask], side='right')
            out[mask, j] = (lo + hi) / len(ref) - 1.0
        return out

    def transform(self, X):
        check_is_fitted(self, 'scale_')
        d = _validate(X)
        if set(d.columns) != set(self.feature_names_in_):
            raise ForecastDataError('Reference/prediction feature schema mismatch')
        return self._ranks(d) / self.scale_ - self.mean_


class BroadSpectralSpace:
    """Complete PCA spectrum in the smaller of row and feature spaces.

    The 90% (or supplied) variance fraction sets a data-dependent component
    count, not a predictor subset. The ridge/residual path keeps every
    numerically identifiable direction, including low-variance directions.
    """
    def __init__(self, variance_fraction=0.9):
        if not 0 < variance_fraction <= 1:
            raise ForecastDataError('Variance fraction must lie in (0,1]')
        self.variance_fraction = variance_fraction

    def fit(self, X):
        if len(X) < 2:
            raise ForecastDataError('At least two reference observations required')
        self.transformer_ = WideRankTransform().fit(X)
        self.reference_ = self.transformer_.transform(X)
        n, p = self.reference_.shape
        if n <= p:
            eigenvalues, u = eigh(self.reference_ @ self.reference_.T,
                                  check_finite=True, driver='evd')
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues, u = eigenvalues[order], u[:, order]
            tol = max(float(eigenvalues[0]), 1.) * np.finfo(float).eps * max(n, p)
            keep = eigenvalues > tol
            self.eigenvalues_, self.u_ = eigenvalues[keep], u[:, keep]
            self.loadings_ = None
        else:
            eigenvalues, v = eigh(self.reference_.T @ self.reference_, driver='evd')
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues, v = eigenvalues[order], v[:, order]
            tol = max(float(eigenvalues[0]), 1.) * np.finfo(float).eps * max(n, p)
            keep = eigenvalues > tol
            self.eigenvalues_ = eigenvalues[keep]
            self.loadings_ = v[:, keep]
            self.u_ = (self.reference_ @ self.loadings_) / np.sqrt(self.eigenvalues_)
        if not len(self.eigenvalues_):
            raise ForecastDataError('Zero numerical rank')
        self.variance_ratios_ = self.eigenvalues_ / self.eigenvalues_.sum()
        self.n_components_ = self.components_for(self.variance_fraction)
        self.reference_index_ = X.index.copy()
        return self

    def components_for(self, fraction):
        return min(len(self.eigenvalues_),
                   int(np.searchsorted(np.cumsum(self.variance_ratios_), fraction) + 1))

    def loadings(self):
        if self.loadings_ is None:
            self.loadings_ = (self.reference_.T @ self.u_) / np.sqrt(self.eigenvalues_)
        return self.loadings_

    def kernel(self, X):
        return self.transformer_.transform(X) @ self.reference_.T

    def ridge_predict(self, X, y, *, alpha=100., residual_multiplier=1.):
        if not isinstance(y, pd.Series) or not y.index.equals(self.reference_index_):
            raise ForecastDataError('Targets must exactly align with reference rows')
        values = y.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ForecastDataError('Observed finite training outcomes required')
        if not np.isfinite([alpha, residual_multiplier]).all() or min(alpha, residual_multiplier) <= 0:
            raise ForecastDataError('Strictly positive finite penalties required')
        center = values.mean()
        penalty = np.full(len(self.eigenvalues_), alpha * residual_multiplier)
        penalty[:self.n_components_] = alpha
        coefficient = self.u_ @ ((self.u_.T @ (values-center)) / (self.eigenvalues_ + penalty))
        return center + self.kernel(X) @ coefficient

    def describe(self, X):
        """Distinctness and residual contributions, only on observed cells.

        No country risk tiers, causal labels or calibrated confidence is inferred.
        An all-missing fitted row is marked unsupported rather than median-safe.
        """
        z = self.transformer_.transform(X)
        k = self.n_components_
        v = self.loadings()[:, :k]
        scores = z @ v
        observed = X[self.transformer_.active_features_].notna().to_numpy()
        counts = observed.sum(axis=1)
        residual = z - scores @ v.T
        residual_contrib = np.where(observed, residual**2, np.nan)
        residual_mse = np.nansum(residual_contrib, axis=1) / np.maximum(counts, 1)
        pc_dist = np.sum(scores**2 / (self.eigenvalues_[:k]/max(len(self.reference_)-1,1)), axis=1)/k
        pc_dist[counts == 0] = np.nan
        residual_mse[counts == 0] = np.nan
        diag = pd.DataFrame({'observed_fitted_features': counts,
            'fitted_features': observed.shape[1], 'observed_share': counts/observed.shape[1],
            'common_component_distance': pc_dist, 'observed_residual_mse': residual_mse,
            'status': np.where(counts > 0, 'descriptive_not_a_risk_score', 'insufficient_observed_inputs')}, index=X.index)
        return diag, scores, residual_contrib

    def spectrum(self):
        return pd.DataFrame({'component': np.arange(1, len(self.eigenvalues_)+1),
            'variance': self.eigenvalues_ / max(len(self.reference_)-1,1),
            'explained_variance_ratio': self.variance_ratios_,
            'cumulative_variance_ratio': np.cumsum(self.variance_ratios_)})

    def exact_profile_groups(self, X):
        """Diagnostic duplicates; never delete identities from the library."""
        z = self.transformer_.transform(X)
        mask = X[self.transformer_.active_features_].notna().to_numpy()
        ids = []
        for j, name in enumerate(self.transformer_.active_features_):
            digest = hashlib.sha256(np.ascontiguousarray(z[:, j]).tobytes()+mask[:, j].tobytes()).hexdigest()
            ids.append({'predictor_id': name, 'profile_sha256': digest})
        out = pd.DataFrame(ids)
        out['profile_size'] = out.groupby('profile_sha256').predictor_id.transform('size')
        return out
