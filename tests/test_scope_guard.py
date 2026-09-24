"""Review 1 scope: exactly 10 regressors, exactly 5 Part A classifiers, no Review 2 code.

This file is exempt from the scope scan because it has to name the forbidden estimators.
"""
import importlib
import re
import sys

import pytest

from src import classification_models, config
from src.classification_models import CLASSIFICATION_PART_A
from src.regression_models import REGRESSION_ALGORITHMS

sys.path.insert(0, str(config.PROJECT_ROOT / "scripts"))
import validate_project  # noqa: E402


def test_algorithm_counts_and_variants():
    assert len(REGRESSION_ALGORITHMS) == 10
    assert [a.estimator for a in REGRESSION_ALGORITHMS] == [
        "LinearRegression", "Ridge", "Lasso", "ElasticNet",
        "PolynomialFeatures+LinearRegression", "DecisionTreeRegressor",
        "RandomForestRegressor", "GradientBoostingRegressor", "SVR", "KNeighborsRegressor",
    ]
    assert [a.estimator for a in CLASSIFICATION_PART_A] == [
        "LogisticRegression", "KNeighborsClassifier", "GaussianNB", "DecisionTreeClassifier", "SVC",
    ]


def test_no_part_b_builder_or_stub():
    names = dir(classification_models)
    assert not [n for n in names if re.search(r"part_?b", n, re.I)]
    forbidden = {"RandomForestClassifier", "AdaBoostClassifier", "GradientBoostingClassifier",
                 "BaggingClassifier", "MLPClassifier"}
    assert not forbidden & {a.estimator for a in CLASSIFICATION_PART_A}


def test_no_clustering_module_or_notebook():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.clustering")
    assert not (config.NOTEBOOKS_DIR / "clustering.ipynb").exists()


def test_scope_scan_of_all_code_and_notebooks_is_clean():
    hits = []
    for label, text in validate_project.code_sources():
        if label in validate_project.SCOPE_GUARD_EXEMPT:
            continue
        hits += [f"{label}: {d}" for d, p in validate_project.FORBIDDEN_PATTERNS.items() if re.search(p, text)]
    assert hits == []


def test_scope_patterns_catch_part_b_but_not_in_scope_regressors():
    pats = validate_project.FORBIDDEN_PATTERNS.values()
    assert any(re.search(p, "RandomForestClassifier(") for p in pats)
    assert any(re.search(p, "from sklearn.cluster import KMeans") for p in pats)
    assert not any(re.search(p, "RandomForestRegressor(n_estimators=100)") for p in pats)
    assert not any(re.search(p, "GradientBoostingRegressor()") for p in pats)
