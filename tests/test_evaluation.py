"""Metric conventions: positive class, ROC-AUC from scores, original units."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src import config
from src.evaluation import (
    ResultsRecorder,
    before_after_table,
    classification_metrics,
    get_scores,
    majority_baseline,
    regression_metrics,
)


def _toy_binary(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = np.where(X[:, 0] + 0.8 * rng.normal(size=n) > 0.6, "negative", "positive")
    return X, y


# ---------------------------------------------------------------- positive class
def test_precision_recall_use_negative_as_positive_class():
    y_true = np.array(["negative", "negative", "positive", "positive", "positive", "positive"])
    y_pred = np.array(["negative", "positive", "positive", "positive", "positive", "negative"])
    m = classification_metrics(y_true, y_pred, y_score=np.array([.9, .4, .1, .2, .3, .6]))
    # For "negative": TP=1, FN=1, FP=1  -> precision 0.5, recall 0.5
    assert m["Precision (negative)"] == pytest.approx(0.5)
    assert m["Recall (negative)"] == pytest.approx(0.5)
    # For "positive" recall would be 3/4 - make sure we did not compute that
    assert m["Recall (negative)"] != pytest.approx(0.75)
    assert config.POS_LABEL == "negative"


# ---------------------------------------------------------------- ROC-AUC
def test_roc_auc_uses_scores_not_hard_labels():
    X, y = _toy_binary()
    model = LogisticRegression(random_state=config.RANDOM_STATE).fit(X, y)
    scores, source = get_scores(model, X)
    y_pred = model.predict(X)
    from_scores = classification_metrics(y, y_pred, scores)["ROC-AUC"]
    from_labels = classification_metrics(y, y_pred, (y_pred == "negative").astype(float))["ROC-AUC"]
    assert source == "predict_proba"
    assert from_scores != pytest.approx(from_labels)


def test_get_scores_never_calls_predict():
    X, y = _toy_binary()

    class NoPredict(LogisticRegression):
        def predict(self, X):
            raise AssertionError("get_scores must not call predict()")

    model = NoPredict(random_state=config.RANDOM_STATE).fit(X, y)
    get_scores(model, X)   # would raise if predict() were used


def test_decision_function_sign_matches_probability_for_negative_class():
    X, y = _toy_binary()
    # SVC without probability calibration (the `probability` argument is deprecated
    # in scikit-learn 1.9) exposes only decision_function - the path used in Phase 7.
    svc = SVC(kernel="linear", random_state=config.RANDOM_STATE).fit(X, y)
    scores, source = get_scores(svc, X)
    assert source == "decision_function"
    # classes_ are sorted: ["negative", "positive"]; higher score must mean "more negative"
    assert list(svc.classes_) == ["negative", "positive"]
    auc = classification_metrics(y, svc.predict(X), scores)["ROC-AUC"]
    assert auc > 0.8   # a flipped sign would give roughly 1 - auc < 0.2


def test_recorder_classification_stores_score_source_and_confusion():
    X, y = _toy_binary()
    model = LogisticRegression(random_state=config.RANDOM_STATE).fit(X, y)
    rec = ResultsRecorder("classification")
    row = rec.record("LR", model, X, y, verbose=False)
    assert row["Score source"] == "predict_proba"
    assert rec.confusion["LR"].to_numpy().sum() == len(y)
    assert list(rec.table().columns[:2]) == ["Rank", "Accuracy"]


def test_majority_baseline():
    y_train = np.array(["positive"] * 7 + ["negative"] * 3)
    base = majority_baseline(y_train, np.array(["positive", "negative"]))
    assert base["majority_class"] == "positive"
    assert base["Accuracy"] == 0.5 and base["Recall (negative)"] == 0.0
    assert base["ROC-AUC"] == 0.5


# ---------------------------------------------------------------- regression
def test_regression_metrics_are_in_original_units():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    pred = y + np.array([1.0, -1.0, 2.0, -2.0])
    m1, m10 = regression_metrics(y, pred), regression_metrics(10 * y, 10 * pred)
    assert m1["MAE"] == pytest.approx(1.5)
    assert m10["RMSE"] == pytest.approx(10 * m1["RMSE"])      # scales with the units
    assert m10["R2"] == pytest.approx(m1["R2"])                # R2 is unit-free


def test_before_after_reports_no_improvement_honestly():
    t = before_after_table([
        {"model": "A", "metric": "R2", "before": 0.9, "after": 0.8, "higher_is_better": True},
        {"model": "B", "metric": "RMSE", "before": 2.0, "after": 1.5, "higher_is_better": False},
    ])
    assert t["Improved?"].tolist() == ["no", "yes"]
