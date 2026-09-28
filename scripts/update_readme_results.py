"""Regenerate the README "Results" section from results/tables/*.csv (no hand-typed numbers).

    python scripts/update_readme_results.py      # run after scripts/run_all.py --track all
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
readme = (ROOT / "README.md").read_text(encoding="utf-8")


def md(df, fmt=".4f"):
    return df.to_markdown(floatfmt=fmt)


reg = pd.read_csv(T / "regression_comparison.csv", index_col=0)
reg_cv = pd.read_csv(T / "regression_cv_all_models.csv", index_col=0)
reg_t = pd.read_csv(T / "regression_tuned_results.csv", index_col=0)
clf = pd.read_csv(T / "classification_comparison.csv", index_col=0)
clf_cv = pd.read_csv(T / "classification_cv_all_models.csv", index_col=0)
clf_t = pd.read_csv(T / "classification_tuned_results.csv", index_col=0)
base = pd.read_csv(T / "classification_majority_baseline.csv", index_col=0)

reg_table = reg[["R²", "RMSE", "MAE"]].join(reg_cv[["CV R² (mean)", "CV R² (std)"]])
clf_table = clf[["Accuracy", "Precision (negative)", "Recall (negative)", "Weighted F1", "ROC-AUC"]].join(
    clf_cv[["CV F1w (mean)", "CV F1w (std)"]])
clf_base = base[["Accuracy", "Precision (negative)", "Recall (negative)", "F1 (weighted)", "ROC-AUC"]].rename(
    columns={"F1 (weighted)": "Weighted F1"})

results = f"""## Results

*Generated from `results/tables/*.csv` of the full run (`python scripts/run_all.py --track all`); do not
edit by hand. Test = held-out test set; CV = 5-fold cross-validation on the training set only.*

### Regression — Student Performance (test set: 1,975 rows)
Baseline (default-hyperparameter) models, ranked by test R²:

{md(reg_table)}

Tuned models (GridSearchCV on the two training-CV nominees; see `regression.ipynb` §21):

{md(reg_t[["Tuned configuration", "Tuned CV R² (train-side)", "Tuned test R²", "Tuned test RMSE", "Tuned test MAE"]], ".6f")}

### Classification Part A — Restaurant Reviews (test set: 1,739 reviews; positive class = `negative`)
Baseline models, ranked by weighted F1 (ROC-AUC from `predict_proba` / `decision_function`):

{md(clf_table)}

Majority-class baseline (reference, not a model):

{md(clf_base)}

Tuned models (GridSearchCV on the two training-CV leaders; see `classification.ipynb` §15):

{md(clf_t, ".4f")}

These tables are results, not conclusions: model selection and interpretation are the team's
(✍️ cells in the notebooks).
"""
readme = re.sub(r"## Results\n.*?(?=\n## AI-assistance disclosure)", results, readme, flags=re.S)
(ROOT / "README.md").write_text(readme, encoding="utf-8")
print("README results section regenerated")
