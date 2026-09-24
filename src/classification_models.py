"""Catalogue of the five Classification PART A algorithms (Review 1 only).

This module is a CHECKLIST, not an implementation. The classifiers are built,
trained and evaluated visibly in notebooks/classification.ipynb.

Scope: Part A only. Part B belongs to Review 2 and deliberately has no stub,
placeholder or commented-out code here - the scope guard in
scripts/validate_project.py and tests/ fails if Part B estimators appear.

Source: 23CSE301 Capstone Guidelines, section 3.2 (algorithm list and notes).
"""
from __future__ import annotations

from .regression_models import AlgorithmSpec

CLASSIFICATION_PART_A: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec("A1", "Logistic Regression", "LogisticRegression", True,
                  "Baseline classifier; interpret coefficients/odds", ("odds-ratio table",)),
    AlgorithmSpec("A2", "K-Nearest Neighbors", "KNeighborsClassifier", True,
                  "Tune k; discuss distance metrics", ("k tuning", "distance-metric comparison")),
    AlgorithmSpec("A3", "Gaussian Naive Bayes", "GaussianNB", True,
                  "Discuss conditional independence assumption", ("independence discussion",)),
    AlgorithmSpec("A4", "Decision Tree Classifier", "DecisionTreeClassifier", False,
                  "Tune max_depth; visualise the tree", ("max_depth tuning", "tree plot")),
    AlgorithmSpec("A5", "Support Vector Machine (SVC)", "SVC", True,
                  "Tune C and kernel; scale features", ("C x kernel tuning",)),
)
# "scaled" for text: TF-IDF rows are L2-normalised, which is the scaling step
# for every model. The Decision Tree ignores scale entirely.

CLASSIFICATION_ALGORITHM_NAMES = tuple(a.name for a in CLASSIFICATION_PART_A)
assert len(CLASSIFICATION_PART_A) == 5
