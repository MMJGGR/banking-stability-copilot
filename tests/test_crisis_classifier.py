import numpy as np
import pandas as pd

import src.crisis_classifier as classifier_module
from src.crisis_classifier import CrisisClassifier


def test_classifier_fit_produces_cross_validation_scores(monkeypatch, tmp_path):
    """Regression test for the ROC AUC validation path."""
    monkeypatch.setattr(classifier_module, "HAS_XGBOOST", False)
    monkeypatch.setattr(classifier_module, "HAS_SHAP", False)

    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(size=(60, 4)),
        columns=["gdp_growth", "inflation", "npl_ratio", "capital_adequacy"],
    )
    y = pd.Series(([0, 1] * 30), name="crisis_target")

    classifier = CrisisClassifier(n_estimators=10, max_depth=2)
    classifier.output_dir = str(tmp_path)
    classifier.fit(X, y, cv=3)

    assert classifier.fitted_
    assert len(classifier.cv_scores_) == 3
    assert np.isfinite(classifier.cv_scores_).all()
    assert (tmp_path / "cv_roc_curve.png").exists()


def test_classifier_supports_country_grouped_cross_validation(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(classifier_module, "HAS_XGBOOST", False)
    monkeypatch.setattr(classifier_module, "HAS_SHAP", False)

    rng = np.random.default_rng(7)
    groups = np.repeat([f"C{i:02d}" for i in range(20)], 2)
    y = pd.Series(np.tile([0, 1], 20), name="crisis_target")
    X = pd.DataFrame(
        rng.normal(size=(40, 3)),
        columns=["gdp_growth", "inflation", "npl_ratio"],
    )

    classifier = CrisisClassifier(n_estimators=5, max_depth=2)
    classifier.output_dir = str(tmp_path)
    classifier.fit(X, y, cv=2, groups=groups)

    assert classifier.fitted_
    assert len(classifier.cv_scores_) == 2
