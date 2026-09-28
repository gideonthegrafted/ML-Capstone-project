"""Classification Part A: label/cleaning contract, TF-IDF leakage, scores, saved pipelines."""
import hashlib
import json
import sys

import joblib
import numpy as np
import pytest

from src import config
from src.data_loading import load_raw
from src.evaluation import classification_metrics, get_scores
from src.preprocessing import build_tfidf, prepare_reviews

sys.path.insert(0, str(config.PROJECT_ROOT / "scripts"))
import validate_project  # noqa: E402

MANIFEST = config.MODELS_DIR / "classification_manifest.json"
needs_run = pytest.mark.skipif(not MANIFEST.exists(), reason="run notebooks/classification.ipynb first")


def test_prepare_reviews_matches_verified_counts():
    df, steps = prepare_reviews(load_raw(config.REVIEWS))
    assert [n for _, n in steps] == [10_000, 9_955, 8_696, 8_691]
    assert df["label"].value_counts().to_dict() == {"positive": 6_263, "negative": 2_428}
    assert "7514" not in df.columns


def test_split_is_stratified_and_text_only():
    X_train, X_test, y_train, y_test = validate_project.rebuild_classification_split()
    assert list(X_train.columns) == ["Review"]
    assert (len(X_train), len(X_test)) == (6_952, 1_739)
    assert abs((y_train == "negative").mean() - (y_test == "negative").mean()) < 0.001


def test_tfidf_learns_vocabulary_from_training_rows_only():
    X_train, X_test, _, _ = validate_project.rebuild_classification_split()
    vec = build_tfidf(config.TFIDF_MAX_FEATURES).fit(X_train["Review"])
    train_only = build_tfidf(None, min_df=1).fit(X_train["Review"]).vocabulary_
    assert set(vec.vocabulary_) <= set(train_only)          # nothing from test-only text
    assert len(vec.vocabulary_) == config.TFIDF_MAX_FEATURES == 5000


def test_classification_notebook_selects_models_on_training_data_only():
    import nbformat
    nb = nbformat.read(config.NOTEBOOKS_DIR / "classification.ipynb", as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert validate_project.calls_using_test_data(code) == []


@needs_run
def test_saved_classifiers_reproduce_metrics_with_score_based_auc():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    X_train, X_test, _, y_test = validate_project.rebuild_classification_split()
    fp = hashlib.sha256(",".join(map(str, sorted(X_test.index))).encode()).hexdigest()
    assert fp == manifest["test_index_sha256"]
    assert sum(m["variant"] == "baseline" for m in manifest["models"]) == 5
    for e in manifest["models"]:
        pipe = joblib.load(config.MODELS_DIR / e["file"])
        s, src = get_scores(pipe, X_test)
        assert src in ("predict_proba", "decision_function")
        m = classification_metrics(y_test, pipe.predict(X_test), s)
        for k in ("Accuracy", "Recall (negative)", "F1 (weighted)", "ROC-AUC"):
            assert m[k] == pytest.approx(e[k], abs=1e-9), (e["file"], k)


@needs_run
def test_weighted_f1_is_really_weighted():
    from sklearn.metrics import f1_score
    X_train, X_test, _, y_test = validate_project.rebuild_classification_split()
    e = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"][0]
    pred = joblib.load(config.MODELS_DIR / e["file"]).predict(X_test)
    per_class = f1_score(y_test, pred, average=None, labels=["negative", "positive"])
    support = np.array([(y_test == c).sum() for c in ["negative", "positive"]])
    assert e["F1 (weighted)"] == pytest.approx(np.average(per_class, weights=support), abs=1e-12)
