import math
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances
from threadpoolctl import threadpool_limits

from src.stable_distance import stable_nan_euclidean
from src.pillar_pipeline import PillarInferencePipeline


def test_large_common_coordinates_do_not_erase_small_distances():
    # The 1e15 common coordinate cancels exactly when subtracted first.
    x = np.array([1e15, 3., 4., np.nan])
    y = np.array([1e15, 0., 0., 90.])
    assert stable_nan_euclidean(x, y) == 5 * math.sqrt(4 / 3)
    assert stable_nan_euclidean(x, x) == 0


def test_metric_matches_existing_mathematical_definition():
    rng = np.random.default_rng(47)
    for _ in range(100):
        x, y = rng.normal(size=(2, 30))
        x[rng.random(30) < .2] = np.nan
        y[rng.random(30) < .2] = np.nan
        expected = nan_euclidean_distances(x[None, :], y[None, :])[0, 0]
        assert stable_nan_euclidean(x, y) == pytest.approx(expected, rel=2e-14)
        assert stable_nan_euclidean(x, y) == stable_nan_euclidean(y, x)


def test_no_shared_coordinates_and_explicit_missing_sentinel():
    assert np.isnan(stable_nan_euclidean([1, np.nan], [np.nan, 2]))
    assert stable_nan_euclidean([1, -99], [4, 10], missing_values=-99) == 3 * math.sqrt(2)


@pytest.mark.parametrize('x,y', [([1, 2], [1]), ([[1, 2]], [[1, 2]]), ([np.inf], [1])])
def test_invalid_distance_inputs_fail(x, y):
    with pytest.raises(ValueError):
        stable_nan_euclidean(x, y)


def matrix():
    rng = np.random.default_rng(171)
    n = 40
    frame = pd.DataFrame({
        'country_code': [f'C{i:02d}' for i in range(n)],
        'gdp_growth': rng.normal(3, 1, n),
        'inflation': rng.normal(4, 1, n),
        'gdp_per_capita': rng.uniform(2000, 70000, n),
        'capital_adequacy': rng.uniform(10, 25, n),
        'npl_ratio': rng.uniform(1, 15, n),
        'loan_concentration': rng.uniform(10, 50, n),
        'nominal_gdp': np.repeat(1e15, n),
    })
    frame.loc[::3, 'npl_ratio'] = np.nan
    frame.loc[::4, 'loan_concentration'] = np.nan
    frame.loc[::5, 'inflation'] = np.nan
    return frame


def test_pipeline_uses_stable_metric_and_is_exact_across_threads_and_batches():
    raw = matrix()
    pipeline = PillarInferencePipeline().fit(raw)
    assert pipeline.imputer_.metric is stable_nan_euclidean
    restored = pickle.loads(pickle.dumps(pipeline))
    expected = pipeline.impute(raw).sort_index()
    expected_scores = pipeline.transform(raw).set_index('country_code').sort_index()
    for threads in [1, 2, 4]:
        with threadpool_limits(limits=threads):
            for data in [raw, raw.iloc[::-1], raw.sample(frac=1, random_state=42)]:
                pd.testing.assert_frame_equal(expected, restored.impute(data).sort_index(), check_exact=True)
                actual = restored.transform(data).set_index('country_code').sort_index()
                pd.testing.assert_series_equal(expected_scores.risk_score, actual.risk_score, check_exact=True)
            pd.testing.assert_frame_equal(expected.loc[['C00']], restored.impute(raw.iloc[[0]]), check_exact=True)
    # Changed inputs are recomputed, not served from a country-only cache.
    revised = raw.iloc[[0]].copy()
    revised['capital_adequacy'] = 2.0
    used = restored.impute(revised)
    assert used.loc['C00', 'capital_adequacy'] == 2.0


def test_legacy_estimator_metric_is_not_replaced_on_deserialization():
    legacy = KNNImputer(n_neighbors=2, metric='nan_euclidean').fit([[1., np.nan], [2., 3.], [4., 5.]])
    restored = pickle.loads(pickle.dumps(legacy))
    assert restored.metric == 'nan_euclidean'


def test_knn_preserves_observed_data():
    frame = matrix().set_index('country_code')
    imputer = KNNImputer(n_neighbors=5, weights='distance', metric=stable_nan_euclidean)
    imputed = pd.DataFrame(imputer.fit_transform(frame), index=frame.index, columns=frame.columns)
    assert ((frame == imputed) | frame.isna()).all().all()
