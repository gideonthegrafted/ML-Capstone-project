"""Catalogue of the ten required regression algorithms (Review 1).

This module is a CHECKLIST, not an implementation. The estimators are built,
trained and evaluated visibly in notebooks/regression.ipynb. The catalogue
exists so that tests and scripts/validate_project.py can verify that exactly
these ten algorithms were produced, using the sklearn class named here.

Source: 23CSE301 Capstone Guidelines, section 3.1 (algorithm list and notes).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmSpec:
    number: str
    name: str                 # display name used in notebooks and tables
    estimator: str            # sklearn class that must appear in the fitted pipeline
    scaled: bool              # receives standardised numeric features
    guideline_note: str       # verbatim intent of the course guideline note
    required_outputs: tuple[str, ...]


REGRESSION_ALGORITHMS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec("1", "Linear Regression", "LinearRegression", True,
                  "Baseline; interpret coefficients", ("coefficient table",)),
    AlgorithmSpec("2", "Ridge Regression", "Ridge", True,
                  "L2 regularisation; tune alpha", ("alpha tuning",)),
    AlgorithmSpec("3", "Lasso Regression", "Lasso", True,
                  "L1 regularisation; observe feature sparsity", ("zero-coefficient count",)),
    AlgorithmSpec("4", "ElasticNet Regression", "ElasticNet", True,
                  "Combined L1+L2; tune l1_ratio", ("alpha x l1_ratio tuning",)),
    AlgorithmSpec("5", "Polynomial Regression", "PolynomialFeatures+LinearRegression", True,
                  "Apply PolynomialFeatures then Linear Regression; compare degrees",
                  ("degree 1/2/3 comparison",)),
    AlgorithmSpec("6", "Decision Tree Regressor", "DecisionTreeRegressor", True,
                  "Tune max_depth; show feature importance", ("max_depth tuning", "feature importance")),
    AlgorithmSpec("7", "Random Forest Regressor", "RandomForestRegressor", True,
                  "Ensemble baseline; tune n_estimators", ("n_estimators tuning", "feature importance")),
    AlgorithmSpec("8", "Gradient Boosting Regressor", "GradientBoostingRegressor", True,
                  "sklearn GBM or XGBoost; tune learning_rate", ("learning_rate tuning",)),
    AlgorithmSpec("9", "Support Vector Regressor", "SVR", True,
                  "Scale features first; tune C and kernel", ("C x kernel tuning",)),
    AlgorithmSpec("10", "K-Nearest Neighbors Regressor", "KNeighborsRegressor", True,
                  "Tune k; discuss impact of scaling", ("k tuning", "scaled vs unscaled")),
)
# Note on "scaled": approved decision (docs/clarifications.md C2) - all ten
# models share ONE preprocessor, so trees also receive standardised inputs.
# Tree splits are invariant to monotonic rescaling, so this changes nothing
# for them while satisfying "same preprocessed dataset" literally.

REGRESSION_ALGORITHM_NAMES = tuple(a.name for a in REGRESSION_ALGORITHMS)
assert len(REGRESSION_ALGORITHMS) == 10
