"""Splitting, deterministic row cleaning and (unfitted) preprocessing builders.

Two kinds of step live here:

1. Deterministic row-level steps that need no statistics from the data
   (dropping exact duplicate rows, turning a rating into a sentiment label).
   These may run before the train/test split because they learn nothing.

2. Builders that RETURN UNFITTED transformers (imputers, encoders, scalers,
   TF-IDF). They are placed inside sklearn Pipelines in the notebooks, so they
   are fitted on the training portion only - including inside every
   cross-validation fold. Nothing in this module fits on the full dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


# --------------------------------------------------------------------------
# Deterministic row-level cleaning
# --------------------------------------------------------------------------
def drop_exact_duplicates(
    df: pd.DataFrame, subset: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """Drop rows that repeat an earlier row (first occurrence is kept).

    ``subset=None`` means every column must match. Returns the de-duplicated
    frame and a before/after report so the decision is visible.
    """
    before = len(df)
    mask = df.duplicated(subset=subset, keep="first")
    out = df.loc[~mask].copy()
    report = {
        "key": "all columns" if subset is None else ", ".join(subset),
        "rows_before": before,
        "duplicates_removed": int(mask.sum()),
        "rows_after": len(out),
        "duplicates_remaining_under_key": int(out.duplicated(subset=subset).sum()),
    }
    return out, report


def derive_sentiment_label(ratings: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Map raw ratings to "positive" / "negative"; everything else becomes NaN.

    Rule (approved decision, docs/clarifications.md):
        rating >= RATING_POSITIVE_MIN (4)  -> "positive"
        rating <= RATING_NEGATIVE_MAX (2)  -> "negative"
        2 < rating < 4 (2.5, 3, 3.5)       -> dropped as ambiguous
        missing / non-numeric (e.g. "Like") -> dropped as unusable

    Returns the label series (same index) and a report table of how each raw
    rating value was handled, with counts.
    """
    numeric = pd.to_numeric(ratings, errors="coerce")
    labels = pd.Series(np.nan, index=ratings.index, dtype="object")
    labels[numeric >= config.RATING_POSITIVE_MIN] = config.LABEL_POSITIVE
    labels[numeric <= config.RATING_NEGATIVE_MAX] = config.LABEL_NEGATIVE

    def outcome(raw, num):
        if pd.isna(raw):
            return "dropped: missing rating"
        if pd.isna(num):
            return "dropped: non-numeric rating"
        if num >= config.RATING_POSITIVE_MIN:
            return config.LABEL_POSITIVE
        if num <= config.RATING_NEGATIVE_MAX:
            return config.LABEL_NEGATIVE
        return "dropped: ambiguous middle rating"

    raw_as_text = ratings.astype("object").where(ratings.notna(), None)
    report = (
        pd.DataFrame({"raw_rating": raw_as_text, "numeric": numeric})
        .assign(outcome=lambda d: [outcome(r, n) for r, n in zip(d.raw_rating, d.numeric)])
        .fillna({"raw_rating": "<missing>"})
        .groupby(["raw_rating", "outcome"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["outcome", "raw_rating"])
        .reset_index(drop=True)
    )
    return labels, report


# --------------------------------------------------------------------------
# Train/test splitting
# --------------------------------------------------------------------------
def make_stratify_bins(y: pd.Series, n_bins: int) -> pd.Series:
    """Quantile bins of a continuous target, used ONLY to stratify the split.

    The returned codes are never added to X and the target is not modified.
    ``duplicates="drop"`` merges bins whose edges coincide (possible with
    heavily repeated target values).
    """
    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop").rename(f"{y.name}_strat_bin")


def split_regression(X: pd.DataFrame, y: pd.Series, n_bins: int | None = None):
    """80:20 split for regression, optionally stratified on target quantile bins.

    ``n_bins`` defaults to ``config.REGRESSION_STRATIFY_BINS``; pass 0 or set
    that config value to None to use a plain random split.
    Returns X_train, X_test, y_train, y_test.
    """
    if n_bins is None:
        n_bins = config.REGRESSION_STRATIFY_BINS
    strata = make_stratify_bins(y, n_bins) if n_bins else None
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=strata,
    )


def split_classification(X, y):
    """Stratified 80:20 split for classification (class proportions preserved)."""
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def row_overlap_report(X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    """Count test rows whose feature values also appear in the training set.

    This is reported, not "fixed": with low-cardinality features some overlap
    happens by chance and is not leakage in itself.
    """
    train_keys = set(map(tuple, X_train.astype(str).to_numpy()))
    in_train = [tuple(row) in train_keys for row in X_test.astype(str).to_numpy()]
    n = int(np.sum(in_train))
    return {
        "test_rows": len(X_test),
        "test_rows_seen_in_train": n,
        "pct_test_rows_seen_in_train": round(100 * n / max(len(X_test), 1), 2),
    }


# --------------------------------------------------------------------------
# Unfitted transformer builders (always used inside a Pipeline)
# --------------------------------------------------------------------------
def build_regression_preprocessor(
    numeric: list[str] | tuple[str, ...],
    categorical: list[str] | tuple[str, ...],
    scale: bool = True,
) -> ColumnTransformer:
    """Imputation + encoding (+ optional scaling) for tabular regression data.

    numeric     -> median imputation -> StandardScaler (if ``scale``)
    categorical -> most-frequent imputation -> one-hot (binary columns become one 0/1 column)
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    categorical_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), list(numeric)),
            ("cat", Pipeline(categorical_steps), list(categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_tfidf(max_features: int | None, **overrides) -> TfidfVectorizer:
    """TF-IDF vectoriser with project defaults; vocabulary is learned at fit time.

    No stock stop-word list is used: sklearn's English list removes negations
    such as "not" and "no", which carry sentiment. ``max_features`` must be
    chosen explicitly (Phase 6 measures vocabulary size before fixing it).
    """
    params = dict(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=max_features,
        dtype=np.float32,
    )
    params.update(overrides)
    return TfidfVectorizer(**params)


class DenseFloat32(BaseEstimator, TransformerMixin):
    """Convert a sparse matrix to a dense float32 array, with a memory guard.

    GaussianNB cannot take sparse input. This makes the conversion explicit
    and refuses to allocate more than ``max_mb`` megabytes.
    """

    def __init__(self, max_mb: float = config.MAX_DENSE_MB):
        self.max_mb = max_mb

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_rows, n_cols = X.shape
        needed_mb = n_rows * n_cols * 4 / 1024**2
        if needed_mb > self.max_mb:
            raise MemoryError(
                f"Dense conversion of {n_rows}x{n_cols} needs {needed_mb:.0f} MB "
                f"> limit {self.max_mb} MB. Reduce max_features."
            )
        if sparse.issparse(X):
            return X.astype(np.float32).toarray()
        return np.asarray(X, dtype=np.float32)
