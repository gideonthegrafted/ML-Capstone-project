# Predictive Analytics Across Domains: Student Performance, Sentiment Classification and Customer Segmentation

**23CSE301 Machine Learning — Capstone Project (B.Tech. CSE, III Year, 2026-27)**

> ## Scope: Review 1
> This repository contains the **Review 1** deliverable only:
> * the **complete regression track** — 10 algorithms (`notebooks/regression.ipynb`);
> * **Classification Part A** — 5 algorithms (`notebooks/classification.ipynb`).
>
> Classification Part B and the clustering track belong to Review 2 and are **deliberately absent**.
> A scope guard in `scripts/validate_project.py` fails if they appear.

> **Build status:** all Review 1 machine-learning work is implemented, executed and validated
> (`python scripts/validate_project.py`: 0 FAIL). The project is **not yet submission-ready**: the team
> must still write the ✍️ TEAM ANALYSIS REQUIRED cells and register one engineered feature per track
> (see *Outstanding team work*).

## Team

| Member | Primary track ownership |
|---|---|
| _Student 1 — name_ | _to be filled in by the team_ |
| _Student 2 — name_ | _to be filled in by the team_ |
| _Student 3 — name_ | _to be filled in by the team_ |

## Datasets

| Track | Dataset | Shape | Target | Notes |
|---|---|---|---|---|
| Regression (R1) | [Student Performance (Multiple Linear Regression)](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression) | 10,000 × 6 | `Performance Index` (10–100) | ⚠️ **Synthetic**: the author states it was "created for illustrative purposes" |
| Classification (R1) | [10000 Restaurant Reviews](https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews) | 10,000 × 8 | Sentiment derived from `Rating`: ≥ 4 positive, ≤ 2 negative, middle ratings dropped | Text classification with TF-IDF; `pos_label = "negative"` (minority class) |
| Clustering (R2) | [Cafe Sales](https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training) (supplied as a cleaned derivative) | 10,000 × 8 | — | Inspected and documented only; nothing is modelled in Review 1 |

Details — checksums, licences, excluded columns and the reason for each — are in
[`docs/dataset_sources.md`](docs/dataset_sources.md). Decisions on ambiguous points are in
[`docs/clarifications.md`](docs/clarifications.md).
Instructor confirmation of the dataset choice is an outstanding administrative item (C7).

## Setup

Requires **Python ≥ 3.11** (tested with 3.13.2). Create the virtual environment **outside** any
OneDrive, Dropbox or Google Drive folder, so that the sync client does not lock thousands of files.

**Windows (PowerShell)**
```powershell
py -3.13 -m venv "$env:USERPROFILE\.venvs\ml-capstone"
& "$env:USERPROFILE\.venvs\ml-capstone\Scripts\python.exe" -m pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv ~/.venvs/ml-capstone
~/.venvs/ml-capstone/bin/python -m pip install -r requirements.txt
```

In VS Code, select this interpreter with *Python: Select Interpreter*, then choose it as the notebook
kernel.

### Get the data (not committed — see `docs/dataset_sources.md`)
Download the three CSV files from the Kaggle pages above (or copy them from the team's shared folder),
then run:
```bash
python scripts/setup_data.py --source "<folder containing the CSVs>"
```
The script verifies each SHA-256 checksum, copies the files into `data/raw/` read-only, and refuses to
overwrite anything already there.

## How to run

```bash
python scripts/run_all.py --track all          # execute both notebooks top to bottom, then validate
python scripts/run_all.py --track regression --mode dev   # quick pass; outputs get a _dev suffix
python -m pytest -q                            # test suite
python scripts/validate_project.py             # rubric, integrity and scope checks
python scripts/update_readme_results.py        # refresh the README results tables from results/tables
```
You can also open the notebooks and use *Restart Kernel and Run All*.

## Repository structure

```
README.md  requirements.txt  .gitignore
data/raw/            raw CSVs (git-ignored, read-only, checksum-verified)
data/processed/      regenerable artefacts (provenance.json)
notebooks/           regression.ipynb, classification.ipynb  <- the primary deliverable
src/                 shared support code (paths, contracts, loading/audit, preprocessing builders,
                     metric conventions, plotting, feature-engineering hook, algorithm catalogues)
scripts/             run_all.py, validate_project.py, setup_data.py
results/             tables/, figures/, tuning/
models/              saved pipelines (compressed joblib)
docs/                rubric_checklist.md, dataset_sources.md, clarifications.md,
                     regression_audit.md, final_review1_audit.md, viva_guide.md
tests/               pytest suite (leakage, metric conventions, contracts, scope guard)
```

## Methodology (fixed conventions)

* `random_state = 42` everywhere; one 80:20 split per track, shared by every model in that track.
* Regression split stratified on 10 target-quantile bins (split only — see C1).
* Classification split stratified on the label.
* All imputers, encoders, scalers and TF-IDF sit inside scikit-learn `Pipeline`s, so they are fitted on
  training data only, including within every CV fold.
* Model selection and tuning use 5-fold cross-validation on the training set; the test set is scored once.
* Regression metrics are R², RMSE and MAE, in the original units.
* Classification metrics are accuracy, precision, recall, weighted F1, ROC-AUC (from scores, never
  hard labels) and the confusion matrix, with a majority-class baseline for reference.

## Results

*Generated from `results/tables/*.csv` of the full run (`python scripts/run_all.py --track all`); do not
edit by hand. Test = held-out test set; CV = 5-fold cross-validation on the training set only.*

### Regression — Student Performance (test set: 1,975 rows)
Baseline (default-hyperparameter) models, ranked by test R²:

| Model                         |     R² |   RMSE |    MAE |   CV R² (mean) |   CV R² (std) |
|:------------------------------|-------:|-------:|-------:|---------------:|--------------:|
| Ridge Regression              | 0.9886 | 2.0524 | 1.6415 |         0.9887 |        0.0005 |
| Linear Regression             | 0.9886 | 2.0525 | 1.6416 |         0.9887 |        0.0005 |
| Polynomial Regression         | 0.9886 | 2.0525 | 1.6416 |         0.9887 |        0.0005 |
| Gradient Boosting Regressor   | 0.9880 | 2.1075 | 1.6839 |         0.9877 |        0.0007 |
| Support Vector Regressor      | 0.9860 | 2.2755 | 1.7953 |         0.9849 |        0.0001 |
| Random Forest Regressor       | 0.9847 | 2.3737 | 1.9153 |         0.9849 |        0.0007 |
| Lasso Regression              | 0.9806 | 2.6760 | 2.1457 |         0.9804 |        0.0007 |
| K-Nearest Neighbors Regressor | 0.9760 | 2.9736 | 2.3687 |         0.9744 |        0.0010 |
| Decision Tree Regressor       | 0.9748 | 3.0500 | 2.4536 |         0.9746 |        0.0008 |
| ElasticNet Regression         | 0.8620 | 7.1337 | 5.9257 |         0.8602 |        0.0016 |

Tuned models (GridSearchCV on the two training-CV nominees; see `regression.ipynb` §21):

| Model (tuned)               | Tuned configuration                                             |   Tuned CV R² (train-side) |   Tuned test R² |   Tuned test RMSE |   Tuned test MAE |
|:----------------------------|:----------------------------------------------------------------|---------------------------:|----------------:|------------------:|-----------------:|
| Ridge Regression            | alpha=0.5623                                                    |                   0.988676 |        0.988579 |          2.052441 |         1.641532 |
| Gradient Boosting Regressor | learning_rate=0.1, max_depth=2, n_estimators=400, subsample=0.8 |                   0.988088 |        0.988113 |          2.093957 |         1.677911 |

### Classification Part A — Restaurant Reviews (test set: 1,739 reviews; positive class = `negative`)
Baseline models, ranked by weighted F1 (ROC-AUC from `predict_proba` / `decision_function`):

| Model                        |   Accuracy |   Precision (negative) |   Recall (negative) |   Weighted F1 |   ROC-AUC |   CV F1w (mean) |   CV F1w (std) |
|:-----------------------------|-----------:|-----------------------:|--------------------:|--------------:|----------:|----------------:|---------------:|
| Support Vector Machine (SVC) |     0.9609 |                 0.9428 |              0.9156 |        0.9607 |    0.9876 |          0.9520 |         0.0074 |
| Logistic Regression          |     0.9517 |                 0.9427 |              0.8807 |        0.9512 |    0.9895 |          0.9412 |         0.0047 |
| Decision Tree Classifier     |     0.8689 |                 0.7590 |              0.7778 |        0.8694 |    0.8420 |          0.8616 |         0.0076 |
| Gaussian Naive Bayes         |     0.8355 |                 0.6567 |              0.8621 |        0.8413 |    0.8438 |          0.8193 |         0.0106 |
| K-Nearest Neighbors          |     0.7320 |                 0.7778 |              0.0576 |        0.6369 |    0.5525 |          0.6418 |         0.0061 |

Majority-class baseline (reference, not a model):

| Model                               |   Accuracy |   Precision (negative) |   Recall (negative) |   Weighted F1 |   ROC-AUC |
|:------------------------------------|-----------:|-----------------------:|--------------------:|--------------:|----------:|
| Majority-class baseline (reference) |     0.7205 |                 0.0000 |              0.0000 |        0.6035 |    0.5000 |

Tuned models (GridSearchCV on the two training-CV leaders; see `classification.ipynb` §15):

| Model (tuned)                |   Accuracy |   Precision (negative) |   Recall (negative) |   F1 (weighted) |   ROC-AUC | Score source (ROC-AUC)   |
|:-----------------------------|-----------:|-----------------------:|--------------------:|----------------:|----------:|:-------------------------|
| Support Vector Machine (SVC) |     0.9586 |                 0.9157 |              0.9383 |          0.9587 |    0.9865 | decision_function        |
| Logistic Regression          |     0.9528 |                 0.8870 |              0.9527 |          0.9533 |    0.9898 | predict_proba            |

These tables are results, not conclusions: model selection and interpretation are the team's
(✍️ cells in the notebooks).

## AI-assistance disclosure (guideline 7.5)

An AI coding assistant (Claude, by Anthropic) was used for **code scaffolding**:
* the `src/` support modules, scripts and tests;
* the notebook structure;
* the factual explanations of algorithms and ML concepts;
* the data-audit facts, which were computed from the files;
* the documentation in `docs/`.

It did **not** write, and must not write:
* EDA observations;
* the feature-engineering choice and justification;
* model-selection reasoning;
* the interpretation of results or the conclusions.

These are the cells marked **✍️ TEAM ANALYSIS REQUIRED** in the notebooks, and the feature
registration in `src/feature_engineering.py`. All git commits are made by the team members themselves.

## Outstanding team work

`python scripts/validate_project.py` lists these items. It reports *NOT SUBMISSION-READY* until all
of them are done:
* every ✍️ TEAM ANALYSIS REQUIRED cell written (regression 23, classification 19);
* at least one engineered feature registered with a justification, per track (rubric B3);
* track ownership filled in above;
* instructor confirmation of the datasets.

## Limitations

* The regression data is synthetic, so conclusions describe the generating process, not real students.
* Sentiment labels are derived from star ratings, which only approximate the sentiment of the text.
* Known execution noise: on Windows, joblib may print `KeyError` tracebacks from its resource tracker to
  the console during notebook runs; they do not reach the notebooks or affect any result.
