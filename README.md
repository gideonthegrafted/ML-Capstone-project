# Predictive Analytics Across Domains: Student Performance, Sentiment Classification and Customer Segmentation

**23CSE301 Machine Learning — Capstone Project (B.Tech. CSE, III Year, 2026-27)**

> ## Scope: Review 1
> This repository contains the **Review 1** deliverable only:
> * the **complete regression track** — 10 algorithms (`notebooks/regression.ipynb`);
> * **Classification Part A** — 5 algorithms (`notebooks/classification.ipynb`).
>
> Classification Part B and the clustering track belong to Review 2 and are **deliberately absent**.
> A scope guard in `scripts/validate_project.py` fails if they appear.

> **Build status:** Phase 2 of 11 — project foundation. The notebooks are skeletons with explained
> sections; the algorithms are not implemented yet. **No results exist yet**, and none are quoted here.

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
docs/                rubric_checklist.md, dataset_sources.md, clarifications.md
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

*No results yet.* Tables will be filled in from `results/tables/` after the full run (Phases 5 and 8).
Numbers are never copied from anywhere else.

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
* every ✍️ TEAM ANALYSIS REQUIRED cell written;
* at least one engineered feature registered with a justification, per track (rubric B3);
* track ownership filled in above;
* instructor confirmation of the datasets.

## Limitations

* The regression data is synthetic, so conclusions describe the generating process, not real students.
* Sentiment labels are derived from star ratings, which only approximate the sentiment of the text.
