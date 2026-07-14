import numpy as np
from sklearn.base import clone

from src.crisis_estimators import SignConstrainedLogisticRegression


def test_sign_constrained_estimator_is_cloneable_and_nonnegative():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 3))
    # The second variable has a deliberately negative sample relationship; the
    # governance bound must set it to zero rather than reverse its direction.
    y = (1.5 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(size=300) > 0.7).astype(int)
    estimator = clone(SignConstrainedLogisticRegression(C=2.0))
    estimator.fit(X, y)
    assert (estimator.coef_ >= -1e-12).all()
    assert estimator.coef_[0, 0] > 0


def test_probability_is_monotonic_in_every_oriented_feature():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ]
        * 30
    )
    y = ((X[:, 0] + X[:, 1]) >= 2).astype(int)
    model = SignConstrainedLogisticRegression(C=10.0).fit(X, y)
    base = np.array([[0.5, 0.5]])
    for column in range(2):
        higher = base.copy()
        higher[0, column] += 1.0
        assert model.predict_proba(higher)[0, 1] >= model.predict_proba(base)[0, 1]


def test_sample_weights_change_the_fitted_probability():
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    equal = SignConstrainedLogisticRegression(class_weight=None).fit(X, y)
    weighted = SignConstrainedLogisticRegression(class_weight=None).fit(
        X, y, sample_weight=np.array([1.0, 10.0, 1.0, 10.0])
    )
    assert weighted.predict_proba([[0.5]])[0, 1] > equal.predict_proba([[0.5]])[0, 1]


def test_regularization_scale_preserves_a_clear_signal():
    rng = np.random.default_rng(19)
    signal = rng.normal(size=500)
    X = signal.reshape(-1, 1)
    y = (signal > 1.0).astype(int)

    model = SignConstrainedLogisticRegression(C=0.25, class_weight=None).fit(X, y)
    probability = model.predict_proba(X)[:, 1]

    assert model.coef_[0, 0] > 1.0
    assert probability[y == 1].mean() > probability[y == 0].mean() + 0.5
