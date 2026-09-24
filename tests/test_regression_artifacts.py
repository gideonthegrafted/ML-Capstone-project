"""Regression track: training-only model selection, saved pipelines and the common split.

The artefact tests need the notebook to have been executed (they skip otherwise);
the static-scan tests always run.
"""
import hashlib
import json
import sys

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src import config
from src.evaluation import regression_metrics

sys.path.insert(0, str(config.PROJECT_ROOT / "scripts"))
import validate_project  # noqa: E402

MANIFEST = config.MODELS_DIR / "regression_manifest.json"
needs_run = pytest.mark.skipif(not MANIFEST.exists(), reason="run notebooks/regression.ipynb first")


# ---------------------------------------------------------------- static scan
def test_scan_flags_model_selection_on_test_data():
    bad = 'GridSearchCV(pipe, grid, cv=CV).fit(X_test, y_test)\ncross_validate(pipe, X_test, y_test, cv=CV)'
    assert len(validate_project.calls_using_test_data(bad)) >= 2


def test_scan_accepts_training_only_code():
    good = ('cross_validate(pipe, X_train, y_train, cv=CV)\n'
            'search = GridSearchCV(pipe, grid, cv=CV)\nsearch.fit(X_train, y_train)\n'
            'pred = search.best_estimator_.predict(X_test)')
    assert validate_project.calls_using_test_data(good) == []


def test_regression_notebook_selects_models_on_training_data_only():
    import nbformat
    nb = nbformat.read(config.NOTEBOOKS_DIR / "regression.ipynb", as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert validate_project.calls_using_test_data(code) == []


# ---------------------------------------------------------------- saved artefacts
@needs_run
def test_saved_models_share_the_independently_rebuilt_split():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    _, X_test, _, _ = validate_project.rebuild_regression_split()
    fp = hashlib.sha256(",".join(map(str, sorted(X_test.index))).encode()).hexdigest()
    assert fp == manifest["test_index_sha256"]


@needs_run
def test_saved_pipelines_round_trip_to_recorded_metrics():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    X_train, X_test, _, y_test = validate_project.rebuild_regression_split()
    assert sum(m["variant"] == "baseline" for m in manifest["models"]) == 10
    for entry in manifest["models"]:
        pipe = joblib.load(config.MODELS_DIR / entry["file"])
        assert isinstance(pipe, Pipeline)
        first, second = pipe.predict(X_test), pipe.predict(X_test)
        assert np.array_equal(first, second), f"{entry['file']}: predictions not deterministic"
        m = regression_metrics(y_test, first)
        for k in ("R2", "RMSE", "MAE"):
            assert m[k] == pytest.approx(entry[k], abs=1e-9), (entry["file"], k)


@needs_run
def test_saved_scalers_were_fitted_on_training_rows_only():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    X_train, X_test, _, _ = validate_project.rebuild_regression_split()
    numeric = list(config.STUDENT.numeric_features)
    train_mean = X_train[numeric].mean().to_numpy()
    full_mean = np.concatenate([X_train[numeric], X_test[numeric]]).mean(axis=0)   # what a leaky scaler would hold
    assert not np.allclose(train_mean, full_mean)                                   # the check can tell them apart
    for entry in manifest["models"]:
        scaler = joblib.load(config.MODELS_DIR / entry["file"]).named_steps["preprocess"] \
            .named_transformers_["num"].named_steps["scale"]
        assert np.allclose(scaler.mean_[:len(numeric)], train_mean), entry["file"]


@needs_run
def test_model_files_are_small_enough_to_commit():
    for f in config.MODELS_DIR.glob("regression_*.joblib"):
        assert f.stat().st_size / 1e6 <= config.MAX_MODEL_FILE_MB, f.name
