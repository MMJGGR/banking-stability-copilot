"""Numerically stable implementation of the existing missing-aware distance.

The banking matrix contains both ratios and nominal monetary amounts. Computing
squared distance as x.x + y.y - 2*x.y can cancel the small differences between
large, similar observations. Calculate their differences first instead. The
metric, missing-data weighting and KNN neighbor/weight policies are unchanged.

This top-level callable is pickleable. Previously persisted estimators keep
their original metric; only explicitly rebuilt pipelines use this correction.
"""
from __future__ import annotations

import math
import numpy as np


def stable_nan_euclidean(x, y, *, missing_values=np.nan):
    """Return sqrt(n_features / n_observed) * Euclidean observed distance.

    Compatible with sklearn.impute.KNNImputer's callable metric contract.
    No common observed coordinates produce NaN, as in nan_euclidean. Python's
    hypot avoids BLAS-dependent dot-product subtraction and square overflow.
    """
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    if left.ndim != 1 or left.shape != right.shape:
        raise ValueError("Distance inputs must be equal-length 1-D vectors")
    if np.isinf(left).any() or np.isinf(right).any():
        raise ValueError("Infinite values are not valid distance inputs")
    if np.isnan(missing_values):
        observed = ~(np.isnan(left) | np.isnan(right))
    else:
        observed = ~(np.isnan(left) | np.isnan(right) |
                     (left == missing_values) | (right == missing_values))
    count = int(observed.sum())
    if count == 0:
        return np.nan
    differences = left[observed] - right[observed]
    return math.hypot(*differences.tolist()) * math.sqrt(left.size / count)
