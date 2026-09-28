# Final Review 1 audit (examiner view)

This audit was written by checking the actual artefacts against the 23CSE301 Guidelines PDF (Review 1
rubric) and the team's Review 1 build specification. The evidence comes from:
* the final clean-kernel runs of both notebooks;
* `python -m pytest`;
* `python scripts/validate_project.py`.

**Status values:** PASS · TEAM INPUT REQUIRED · FAIL.

## Section A — Dataset & EDA (4 marks)

| Rubric item | Required | Actual evidence | Location | Test / validation evidence | Status | Remaining team work |
|---|---|---|---|---|---|---|
| A1 Loading & audit | Shape, dtypes, missing values, target/class distribution | Contract-checked loads (10,000×6; 10,000×8); dtype/missing/unique audit tables; target statistics; rating→label table; class balance | reg §3–11; clf §3–8 | `test_data_contracts.py` (11 tests); validation "Raw data" | PASS | — |
| A2 EDA visualisations | Distribution per feature, heatmap, target distribution, ≥ 2 feature–target scatter plots | Reg: 5 feature distributions, target plot, heatmap, 2 scatter plots, box plots. Clf: rating→label, class balance, length, 4 attribute distributions, Spearman heatmap, 2 scatter plots, top terms | reg §11–15; clf §5–6, §10 | validation "required figures exist" (both tracks) | PASS | — |
| A3 Insight commentary | Written observation after each major plot | ✍️ placeholders after every major plot | reg `REG-A3-*`, `REG-LEAKAGE`; clf `CLF-A3-*`, `CLF-TARGET`, `CLF-LEAKAGE` | validation counts unwritten cells | TEAM INPUT REQUIRED | Write each observation |

## Section B — Preprocessing & feature engineering (3 marks)

| Rubric item | Required | Actual evidence | Location | Test / validation evidence | Status | Remaining team work |
|---|---|---|---|---|---|---|
| B1 Cleaning | Missing values (justified), duplicates, outliers checked and treated | Reg: 0 missing (imputers in the pipeline), 127 exact duplicates removed, IQR and z outlier table. Clf: 45 empty reviews dropped, 5 re-posts removed, duplicate keys investigated | reg §8, §9, §13; clf §7–8 | `test_drop_exact_duplicates_real_student`, `test_prepare_reviews_matches_verified_counts` | PASS (checks) | Written justification: `REG-B1-DUPLICATES`, `REG-B1-OUTLIERS`, `CLF-B1` |
| B2 Encoding, scaling, split | Encoding; scaler fitted on train only; stratified split | One-hot (binary); StandardScaler; TF-IDF (L2) fitted in pipelines on train only; stratified splits (regression: 10 target bins, C1) | reg §17–18; clf §11–12 | Scaler holds training means (all 12 saved regressors); TF-IDF vocabulary = training-only vocabulary (all 7 classifiers); per-fold refit test | PASS | — |
| B3 Feature engineering | ≥ 1 engineered feature with written justification | Hook, pipeline step and with/without comparison ready; **nothing registered** (by design, G 7.5) | reg §19, §20.12; clf §12.1; `src/feature_engineering.py` | validation: 0 registered → PENDING | TEAM INPUT REQUIRED | Register one feature per track and justify it (`REG-B3`, `CLF-B3`) |

## Section C — Regression (9 marks)

| Rubric item | Required | Actual evidence | Location | Test / validation evidence | Status | Remaining team work |
|---|---|---|---|---|---|---|
| C1 Implementation | All 10 algorithms, predictions, no errors | 10 baseline pipelines; 0 execution errors | reg §20.1–20.10 | 10 baseline models saved; estimator classes match the catalogue | PASS | Interpretation cells `REG-M1`…`REG-M10` |
| C2 Comparison | R², RMSE, MAE for all 10 on the same test split, ranked by R² | `regression_comparison` (+ CV columns in `regression_final_comparison`) | reg §20.11, §21.4 | Common split fingerprint; table = saved models (≤ 4.4e-16); refit reproduces (0.0) | PASS | — |
| C3 Tuning | Grid/random search on ≥ 2 models; best parameters and improvement | GridSearchCV on Ridge (25) and Gradient Boosting (72), nominated by training CV; before/after tables | reg §21.2–21.3 | "formal tuning ≥ 2" PASS; static scan: no test data in any CV/search | PASS | `REG-C3-TUNING` |
| C4 Visualisation | Residual + predicted-vs-actual for the best model; tree feature importance | Both tuned finalists: predicted vs actual, residuals and distribution, importance/coefficients; RF/DT importance | reg §21.5, §20.6–20.7 | required figures exist | PASS | `REG-C4-DIAGNOSTICS`, `REG-MODEL-SELECTION` |
| 5-fold CV for the two best models (G 3.1) | — | CV (mean, std, folds) for all 10 on the training set | reg §21.1 | static scan PASS | PASS | `REG-CV` |

## Section D — Classification Part A (3 marks)

| Rubric item | Required | Actual evidence | Location | Test / validation evidence | Status | Remaining team work |
|---|---|---|---|---|---|---|
| D1 Implementation | 5 Part A algorithms, predictions, no errors | LR, KNN, GaussianNB, DT, SVC; 0 errors; **no Part B** | clf §14.1–14.5 | Exactly 5 saved baselines; scope guard PASS | PASS | `CLF-M1`…`CLF-M5` |
| D2 Evaluation | Accuracy, weighted F1, confusion matrix per model; comparison table (G 3.2 adds precision, recall, ROC-AUC) | All six metrics per model; table ranked by weighted F1 plus majority baseline; 7 confusion matrices; ROC curves | clf §14.6, §15.3 | ROC-AUC source verified (`predict_proba` / `decision_function`); pos_label `negative`; weighted-F1 test; table = saved models | PASS | `CLF-DIAGNOSTICS` |
| Training-side CV + tuning of two leaders (G 7.2, spec §6.2) | — | Stratified 5-fold CV of all 5; GridSearchCV on the top 2 (SVC, LR) | clf §15.1–15.2 | static scan PASS; 2 tuned models saved | PASS | `CLF-CV`, `CLF-TUNING`, `CLF-MODEL-SELECTION` |

## Section E — Presentation, notebook quality, integrity, reproducibility

| Rubric item | Required | Actual evidence | Location | Test / validation evidence | Status | Remaining team work |
|---|---|---|---|---|---|---|
| Explanation alongside code (G 7.1) | Markdown before code blocks | 0 code cells directly after another code cell; no cell > 80 lines; Technical Result cells | both notebooks | validation "Notebook quality" | PASS | — |
| Top-to-bottom execution with outputs (G 7.1, D1) | No errors, outputs visible | Clean-kernel runs, 0 errors, 0 stderr outputs | both notebooks | validation "Notebooks" | PASS | — |
| Plot standards (G 7.3) | Title, axis labels, legend, tight layout, colourblind palette | Enforced by `plotting.finalize` (raises otherwise) | all figures | `test_finalize_rejects_unlabelled_plots_and_saves` | PASS | — |
| Academic integrity (G 7.5) | Analysis by the team; AI use cited | 42 ✍️ cells; no AI-written conclusions; README disclosure | notebooks, README | validation counts placeholders | TEAM INPUT REQUIRED | Write all ✍️ cells |
| Reproducibility | random_state=42, pinned dependencies, relative paths | `config.RANDOM_STATE`; `requirements.txt` pins verified; no absolute paths; bit-exact refits | `src/config.py` | `test_no_hardcoded_absolute_paths_in_code`; refit check 0.0 | PASS | — |
| Scope | No Part B, no clustering | Scope scan of all code and notebook cells | — | `test_scope_guard.py`; validation | PASS | — |
| GitHub history (G 8) | Meaningful commits | Per-student milestone commits | GitHub | — | TEAM INPUT REQUIRED | Push the final commits |
| E1 / Viva | Every member explains the code | `docs/viva_guide.md` (15 algorithms) and viva notes per section | docs, notebooks | — | TEAM INPUT REQUIRED | Revise; own one track each |

## A. Critical failures
None: every mechanical requirement passes validation.

## B. Medium-priority items
1. **Rubric B3 (1 mark) is incomplete until the team registers an engineered feature per track.**
   The hook, pipeline and comparison cells are ready.
2. **Default KNN classification is dominated by all-zero TF-IDF vectors (C22).** It is reported
   factually; how to discuss it is the team's decision.
3. **Instructor confirmation of the self-selected datasets is still outstanding (C7).**

## C. Cosmetic
1. joblib prints `KeyError` tracebacks from its resource tracker to the console on Windows. They do not
   reach the notebooks or affect any result.
2. The regression comparison bar chart starts at 0, deliberately, so the tiny differences look flat.

## D. TEAM ANALYSIS REQUIRED
* Regression: 23 cells.
* Classification: 19 cells.
* One engineered feature per track.
* Track ownership in the README.
* `validate_project.py` lists them.

## E. Final steps before submission
1. Register and justify one engineered feature per track in `src/feature_engineering.py`.
2. Write every ✍️ cell. Replace each with `### Team analysis — <ID>` and your text.
3. Re-run `python scripts/run_all.py --track all`; the numbers will change if a feature was added.
4. Run `python scripts/update_readme_results.py` and re-read the Technical Result cells.
5. Run `python scripts/validate_project.py --strict` until it prints `STATUS: SUBMISSION-READY`.
6. Commit and push, each member committing their own analysis.
