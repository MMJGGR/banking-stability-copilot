import numpy as np
import pandas as pd
import pytest

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


def test_classifier_predictions_use_training_medians_and_survive_round_trip(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(classifier_module, "HAS_XGBOOST", False)
    monkeypatch.setattr(classifier_module, "HAS_SHAP", False)

    rng = np.random.default_rng(11)
    X = pd.DataFrame(
        {
            "gdp_growth": rng.normal(size=60),
            "inflation": rng.normal(size=60),
            "npl_ratio": rng.normal(size=60),
        }
    )
    X.loc[::5, "inflation"] = np.nan
    y = pd.Series([0, 1] * 30, name="crisis_target")

    classifier = CrisisClassifier(n_estimators=5, max_depth=2)
    classifier.output_dir = str(tmp_path)
    classifier.fit(X, y, cv=3)

    target = pd.DataFrame(
        [{"gdp_growth": 0.1, "inflation": np.nan, "npl_ratio": -0.2}]
    )
    mixed_batch = pd.concat(
        [
            target,
            pd.DataFrame(
                [
                    {
                        "gdp_growth": 1000.0,
                        "inflation": 1000.0,
                        "npl_ratio": 1000.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    expected = classifier.predict_proba(target)[0]
    assert classifier.predict_proba(mixed_batch)[0] == pytest.approx(expected)

    artifact_path = tmp_path / "classifier.pkl"
    classifier.save(str(artifact_path))
    loaded = CrisisClassifier.load(str(artifact_path))

    assert loaded.feature_fill_values_.equals(classifier.feature_fill_values_)
    assert loaded.calibrated_model is not None
    assert loaded.predict_proba(target)[0] == pytest.approx(expected)


def test_classifier_rejects_missing_prediction_features(monkeypatch, tmp_path):
    monkeypatch.setattr(classifier_module, "HAS_XGBOOST", False)
    monkeypatch.setattr(classifier_module, "HAS_SHAP", False)

    X = pd.DataFrame(
        {
            "gdp_growth": np.arange(30, dtype=float),
            "inflation": np.arange(30, dtype=float) / 2,
        }
    )
    y = pd.Series([0, 1] * 15, name="crisis_target")
    classifier = CrisisClassifier(n_estimators=5, max_depth=2)
    classifier.output_dir = str(tmp_path)
    classifier.fit(X, y, cv=3)

    with pytest.raises(ValueError, match="missing trained features"):
        classifier.predict_proba(pd.DataFrame({"gdp_growth": [1.0]}))


def test_review_threshold_policy_preserves_recall_floor():
    y = np.array([0, 0, 0, 0, 1, 1])
    proba = np.array([0.05, 0.20, 0.30, 0.90, 0.40, 0.80])

    summary = classifier_module._threshold_policy_summary(y, proba)

    assert summary["review"]["recall"] >= 0.60
    assert summary["high_recall"]["recall"] >= 0.70
    assert summary["balanced"]["f1"] >= summary["review"]["f1"] or summary[
        "review"
    ]["recall"] >= 0.60
    assert summary["review"]["policy"] == "review"
