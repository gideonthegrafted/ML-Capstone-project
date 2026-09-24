"""Leakage-sensitive preprocessing behaviour: splits, labels, scalers, dense guard."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from src import config
from src.data_loading import load_raw
from src.preprocessing import (
    DenseFloat32,
    build_regression_preprocessor,
    build_tfidf,
    derive_sentiment_label,
    drop_exact_duplicates,
    make_stratify_bins,
    split_classification,
    split_regression,
)


# ---------------------------------------------------------------- labels
def test_label_rule_on_synthetic_values():
    ratings = pd.Series(["1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5", "Like", None])
    labels, _ = derive_sentiment_label(ratings)
    assert labels.tolist()[:3] == ["negative"] * 3
    assert labels.iloc[3:6].isna().all()          # ambiguous middle dropped
    assert labels.tolist()[6:9] == ["positive"] * 3
    assert labels.iloc[9:].isna().all()           # "Like" and missing dropped


def test_label_counts_on_real_data():
    labels, report = derive_sentiment_label(load_raw(config.REVIEWS)["Rating"])
    assert (labels == "positive").sum() == 6_274
    assert (labels == "negative").sum() == 2_428
    assert labels.isna().sum() == 1_298
    assert report["count"].sum() == 10_000
    assert config.POS_LABEL == "negative"          # minority class is the positive label


# ---------------------------------------------------------------- duplicates
def test_drop_exact_duplicates_real_student():
    out, rep = drop_exact_duplicates(load_raw(config.STUDENT))
    assert rep["duplicates_removed"] == 127 and rep["rows_after"] == 9_873 == len(out)


# ---------------------------------------------------------------- splits
def _student_xy():
    df, _ = drop_exact_duplicates(load_raw(config.STUDENT))
    return df.drop(columns=config.STUDENT.target), df[config.STUDENT.target]


def test_regression_split_reproducible_and_target_unchanged():
    X, y = _student_xy()
    a = split_regression(X, y)
    b = split_regression(X, y)
    assert a[0].index.equals(b[0].index) and a[1].index.equals(b[1].index)
    X_train, X_test, y_train, y_test = a
    assert list(X_train.columns) == list(X.columns)          # bins never become a feature
    pd.testing.assert_series_equal(y_train, y.loc[y_train.index])   # target values untouched
    assert len(X_test) == round(config.TEST_SIZE * len(X))


def test_regression_stratification_is_configurable():
    X, y = _student_xy()
    stratified = split_regression(X, y, n_bins=10)[1].index
    plain = split_regression(X, y, n_bins=0)[1].index
    assert not stratified.equals(plain)
    bins = make_stratify_bins(y, 10)
    assert bins.nunique() == 10


def test_classification_split_is_stratified():
    y = pd.Series(["negative"] * 300 + ["positive"] * 700)
    X = pd.DataFrame({"x": range(1000)})
    _, _, y_tr, y_te = split_classification(X, y)
    assert (y_te == "negative").mean() == pytest.approx(0.30, abs=0.005)
    assert (y_tr == "negative").mean() == pytest.approx(0.30, abs=0.005)


# ---------------------------------------------------------------- leakage
def test_scaler_fitted_on_training_data_only():
    """Train and test have very different statistics; only train should end up at mean 0."""
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(0, 1, 500), "c": ["x"] * 500})
    test = pd.DataFrame({"a": rng.normal(100, 1, 200), "c": ["x"] * 200})
    pre = build_regression_preprocessor(["a"], ["c"]).fit(train)
    assert abs(pre.transform(train)[:, 0].mean()) < 1e-9
    assert pre.transform(test)[:, 0].mean() > 50       # would be ~0 if fitted on test/all data


class _FitSizeSpy(BaseEstimator, TransformerMixin):
    fit_sizes: list = []

    def fit(self, X, y=None):
        _FitSizeSpy.fit_sizes.append(len(X))
        return self

    def transform(self, X):
        return X


def test_pipeline_transformers_refit_per_cv_fold():
    _FitSizeSpy.fit_sizes = []
    X = pd.DataFrame({"a": np.arange(100.0)})
    y = np.arange(100.0)
    pipe = Pipeline([("spy", _FitSizeSpy()), ("model", LinearRegression())])
    cross_validate(pipe, X, y, cv=KFold(5, shuffle=True, random_state=config.RANDOM_STATE))
    assert _FitSizeSpy.fit_sizes == [80] * 5          # never fitted on all 100 rows


# ---------------------------------------------------------------- text / dense
def test_tfidf_keeps_negations():
    vec = build_tfidf(max_features=None, min_df=1).fit(["not good at all", "no taste", "good food"])
    vocab = set(vec.vocabulary_)
    assert {"not", "no"} <= vocab


def test_dense_conversion_float32_and_guard():
    X = sparse.random(10, 20, density=0.2, format="csr", random_state=0)
    out = DenseFloat32().fit_transform(X)
    assert out.dtype == np.float32 and out.shape == (10, 20)
    with pytest.raises(MemoryError):
        DenseFloat32(max_mb=0.0001).fit_transform(X)
