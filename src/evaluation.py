"""Metrics, the result recorder and comparison-table helpers.

Metric conventions are defined ONCE here and applied everywhere:

Regression
    R2, RMSE and MAE on the ORIGINAL target units (the target is never scaled).

Classification (binary sentiment)
    * pos_label = config.POS_LABEL = "negative" (the minority class). Binary
      precision, recall and F1 are reported for that class and named as such.
    * Weighted precision/recall/F1 average the per-class scores weighted by
      class support (the rubric asks for weighted F1).
    * ROC-AUC is computed from continuous scores - predict_proba() or
      decision_function() - NEVER from hard predict() labels. ``get_scores``
      is the only way scores are obtained and it never calls predict().
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from . import config


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    """R2, RMSE and MAE in the original units of the target."""
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def get_scores(model, X, pos_label=config.POS_LABEL) -> tuple[np.ndarray, str]:
    """Continuous scores for ``pos_label`` and the method that produced them.

    Prefers predict_proba; falls back to decision_function. For a binary
    decision_function sklearn scores ``classes_[1]``, so the sign is flipped
    when ``pos_label`` is ``classes_[0]``. Never uses predict().
    """
    classes = list(model.classes_)
    if pos_label not in classes:
        raise ValueError(f"pos_label {pos_label!r} not in model classes {classes}")
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            return proba[:, classes.index(pos_label)], "predict_proba"
        except AttributeError:
            # e.g. SVC(probability=False) exposes the attribute but cannot use it
            pass
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        if scores.ndim != 1:
            raise ValueError("decision_function returned multi-column scores; binary task expected")
        return (scores if classes.index(pos_label) == 1 else -scores), "decision_function"
    raise TypeError(f"{type(model).__name__} exposes neither predict_proba nor decision_function")


def classification_metrics(y_true, y_pred, y_score, pos_label=config.POS_LABEL) -> dict:
    """All six required classification metrics (confusion matrix returned separately)."""
    y_true = np.asarray(y_true)
    y_pos = (y_true == pos_label).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        f"Precision ({pos_label})": float(precision_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)),
        f"Recall ({pos_label})": float(recall_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)),
        f"F1 ({pos_label})": float(f1_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)),
        "Precision (weighted)": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall (weighted)": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1 (weighted)": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_pos, y_score)),
    }


def labelled_confusion_matrix(y_true, y_pred, labels=config.CLASS_LABELS) -> pd.DataFrame:
    """Confusion matrix as a DataFrame with explicit actual/predicted labels."""
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return pd.DataFrame(
        cm,
        index=[f"actual: {l}" for l in labels],
        columns=[f"predicted: {l}" for l in labels],
    )


def majority_baseline(y_train, y_test, pos_label=config.POS_LABEL) -> dict:
    """Metrics of always predicting the training majority class (a reference, not a model)."""
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
    X_dummy = np.zeros((len(y_test), 1))
    y_pred = dummy.predict(X_dummy)
    y_score, _ = get_scores(dummy, X_dummy, pos_label)
    return {"majority_class": dummy.classes_[np.argmax(dummy.class_prior_)],
            **classification_metrics(y_test, y_pred, y_score, pos_label)}


# --------------------------------------------------------------------------
# Result recorder: one evaluation per fitted model, one source for the table
# --------------------------------------------------------------------------
@dataclass
class ResultsRecorder:
    """Evaluates fitted pipelines on the held-out test set and stores the results.

    The comparison table is assembled from the stored results, so the numbers
    in each notebook section and in the final table can never disagree.
    """

    task: str                                  # "regression" | "classification"
    results: dict = field(default_factory=dict)
    confusion: dict = field(default_factory=dict)
    predictions: dict = field(default_factory=dict)

    def record(self, name: str, model, X_test, y_test, fit_seconds: float | None = None,
               X_train=None, y_train=None, verbose: bool = True) -> dict:
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_seconds = time.perf_counter() - t0
        if self.task == "regression":
            row = regression_metrics(y_test, y_pred)
            if X_train is not None and y_train is not None:
                row["Train R2"] = float(r2_score(y_train, model.predict(X_train)))
        elif self.task == "classification":
            y_score, source = get_scores(model, X_test)
            row = classification_metrics(y_test, y_pred, y_score)
            row["Score source"] = source
            self.confusion[name] = labelled_confusion_matrix(y_test, y_pred)
            self.predictions[name] = {"y_pred": y_pred, "y_score": y_score}
        else:
            raise ValueError(f"unknown task {self.task!r}")
        if fit_seconds is not None:
            row["Fit time (s)"] = round(fit_seconds, 3)
        row["Predict time (s)"] = round(predict_seconds, 3)
        self.results[name] = row
        if verbose:
            self._print(name, row)
        return row

    def _print(self, name: str, row: dict) -> None:
        print(f"== {name} ==  (held-out test set, n={self._n(name)})")
        for key, value in row.items():
            print(f"  {key:<22} {value:.4f}" if isinstance(value, float) else f"  {key:<22} {value}")
        if name in self.confusion:
            print(self.confusion[name].to_string())

    def _n(self, name):
        cm = self.confusion.get(name)
        return int(cm.to_numpy().sum()) if cm is not None else "see split"

    def table(self, sort_by: str | None = None) -> pd.DataFrame:
        """Comparison table, ranked (descending) by ``sort_by`` with a Rank column."""
        sort_by = sort_by or ("R2" if self.task == "regression" else "F1 (weighted)")
        df = pd.DataFrame.from_dict(self.results, orient="index")
        df.index.name = "Model"
        df = df.sort_values(sort_by, ascending=False)
        df.insert(0, "Rank", range(1, len(df) + 1))
        return df


# --------------------------------------------------------------------------
# Tables on disk
# --------------------------------------------------------------------------
def save_table(df: pd.DataFrame, name: str, floatfmt: str = ".4f") -> tuple:
    """Save a table as CSV and Markdown under results/tables (dev runs get a suffix)."""
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{name}{config.OUTPUT_SUFFIX}"
    csv_path = config.TABLES_DIR / f"{stem}.csv"
    md_path = config.TABLES_DIR / f"{stem}.md"
    df.to_csv(csv_path)
    md_path.write_text(df.to_markdown(floatfmt=floatfmt) + "\n", encoding="utf-8")
    return csv_path, md_path


def before_after_table(rows: list[dict]) -> pd.DataFrame:
    """Tuning report: one row per (model, metric) with before, after and change.

    Each dict needs: model, metric, before, after, higher_is_better.
    The "Improved?" column states the direction honestly, including "no".
    """
    df = pd.DataFrame(rows)
    df["Change"] = df["after"] - df["before"]
    better = np.where(df["higher_is_better"], df["Change"] > 0, df["Change"] < 0)
    df["Improved?"] = np.where(df["Change"] == 0, "no change", np.where(better, "yes", "no"))
    return df.drop(columns="higher_is_better").rename(columns={"before": "Before", "after": "After"})
