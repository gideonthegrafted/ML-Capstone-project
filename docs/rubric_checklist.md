# Review 1 rubric checklist

Every Review 1 criterion is mapped to the artefact that satisfies it, with an **honest status**.
Status values:
* **Not started**
* **Foundation ready** — the infrastructure exists, but the notebook section is not built yet
* **Done** — built, executed and verified
* **✍️ Team** — needs student-written content

`python scripts/validate_project.py` checks the mechanical items automatically.

*Last updated: Phase 3 (regression audit, EDA, split, preprocessing). In the table below, "regression: done"
means the regression half is built and executed; the classification half follows in Phase 6.*

| # | Criterion (marks) | Full-marks descriptor (abridged) | Artefact | Status |
|---|---|---|---|---|
| A1 | Dataset loading & audit (1) | Shape, dtypes, missing counts, target distribution | `regression.ipynb` §3–10, `classification.ipynb` §2–3; `src/data_loading.py` | Regression: done · Classification: Phase 6 |
| A2 | EDA visualisations (2) | Distribution per feature, correlation heatmap, target distribution, ≥ 2 feature–target scatter plots | `regression.ipynb` §11–15 (target, 5 feature distributions, heatmap, 2 scatter plots, box plots), `classification.ipynb` §6 | Regression: done · Classification: Phase 6 |
| A3 | Insight commentary (1) | Written observation after each major plot | ✍️ cells `REG-A3-TARGET`, `-FEATURES`, `-CORRELATION`, `-RELATIONSHIPS` | ✍️ Team |
| B1 | Data cleaning (1) | Justified missing-value strategy; duplicates and outliers checked and treated | regression §8 (missing), §9 (duplicates), §13 (outliers); classification §4; ✍️ `REG-B1-DUPLICATES`, `REG-B1-OUTLIERS`, `CLF-B1` | Regression: done (checks) + ✍️ Team |
| B2 | Encoding, scaling & splitting (1) | Suitable encoding; scaler fitted on train only; stratified split | regression §17 (10-bin stratified split) and §18 (pipeline, train-only fit demonstrated); classification §5 and §7 | Regression: done · Classification: Phase 6 |
| B3 | Feature engineering (1) | ≥ 1 engineered feature with written justification | `src/feature_engineering.py` (ships empty); pipeline hook wired in regression §19; ✍️ `REG-B3`, `CLF-B3` | ✍️ Team |
| C1 | Regression implementation (4) | All 10 algorithms trained, no errors | `regression.ipynb` §20 (ten algorithm subsections) | Not started (Phase 4) |
| C2 | Comparative evaluation (2) | One table: R², RMSE, MAE for all 10, ranked by R² | `regression.ipynb` §20; `results/tables/regression_comparison.*` | Not started (Phase 5) |
| C3 | Hyperparameter tuning (2) | Grid/random search on ≥ 2 models, best parameters and improvement | `regression.ipynb` §20; `results/tuning/` | Not started (Phase 5) |
| C4 | Visualisation (1) | Residual and predicted-vs-actual plots for the best model; tree feature importance | `regression.ipynb` §20; `results/figures/` | Not started (Phase 5) |
| D1 | Part A implementation (2) | All 5 Part A algorithms trained, no errors | `classification.ipynb` §10.1–10.5 | Not started (Phase 7) |
| D2 | Evaluation (1) | Accuracy, weighted F1, confusion matrix per algorithm; comparison table | `classification.ipynb` §10–11 and §14 (all six G 3.2 metrics) | Not started (Phases 7–8) |
| E1 | Presentation quality (1) | Clear narrative; every member can explain the code | Notebook structure (what / why / concept per section) | ✍️ Team |
| — | Viva (5) | — | Viva notes in each algorithm section | ✍️ Team |

## Guideline requirements outside the rubric table

| Requirement | Source | Artefact | Status |
|---|---|---|---|
| random_state = 42 everywhere | G 7.1 | `src/config.py` `RANDOM_STATE`; test | Done |
| Scalers and encoders fitted on training data only | G 7.1 | Pipelines; `tests/test_preprocessing.py` (train-only fit, per-fold refit) | Done (mechanism); applied in Phases 3–8 |
| Consistent 80:20 split per track | G 7.2 | `TEST_SIZE = 0.20`; `split_regression`, `split_classification` | Done (mechanism) |
| Cross-validation for the top 2 models per track | G 7.2 | regression §20, classification §12 | Not started |
| Results in summary tables | G 7.2 | `ResultsRecorder.table()`, `save_table()` | Foundation ready |
| Plot title, labels, legend; tight layout; colourblind palette | G 7.3 | `src/plotting.py` `finalize()` enforces these; test | Done (mechanism) |
| Notebooks run top to bottom with outputs, no errors | G 7.1, D1 | `scripts/run_all.py`; validation "Notebooks" group | Skeletons execute cleanly |
| GitHub repository with meaningful commit history | G 8, D2 | Commits made by the students themselves (C17) | ✍️ Team |
| `requirements.txt` with versions | G 8 | Pinned, tested in the venv | Done |
| Required directory structure | G 8 | README, requirements, data/, notebooks/, models/ | Done |
| AI assistance cited in README | G 7.5 | README "AI-assistance disclosure" | Done (keep it accurate) |
| External code cited | G 7.5 | Notebook "References" sections | Ongoing |
