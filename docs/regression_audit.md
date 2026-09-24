# Regression Review 1 audit

*Produced at the end of Phase 5 from a clean-kernel run of `notebooks/regression.ipynb`
(85 code cells executed, 0 errors, 0 stderr outputs, 30 figures), `python -m pytest` (46 passed) and
`python scripts/validate_project.py` (regression groups: all PASS).*

**Status values**
* **PASS** — implemented, executed and verified by the evidence listed.
* **TEAM INPUT REQUIRED** — the machinery is done, but the rubric item needs student-written content.
* **FAIL** — not met. No item is currently FAIL.

| # | Requirement (source) | Implemented? | Executed? | Verified? | Notebook location | Evidence | Status |
|---|---|---|---|---|---|---|---|
| 1 | Shape, dtypes, missing counts, target distribution (R1 A1) | Yes | Yes | Yes | §3–§11 | Contract check (10,000 × 6, SHA-256); audit table; target stats; `test_data_contracts.py` | PASS |
| 2 | Distribution plot per feature, heatmap, target distribution, ≥ 2 scatter plots (R1 A2) | Yes | Yes | Yes | §11, §12, §14, §15 | Figures `regression_target_distribution`, `_feature_distributions` (5 features), `_correlation_heatmap`, `_scatter_feature_target` (2 scatter plots), `_boxplots_by_feature` | PASS |
| 3 | Written observation after each major plot (R1 A3) | Placeholders | — | Detected | §11–§16 | `REG-A3-TARGET`, `-FEATURES`, `-CORRELATION`, `-RELATIONSHIPS`, `REG-LEAKAGE` | TEAM INPUT REQUIRED |
| 4 | Missing values, duplicates, outliers checked and treated (R1 B1) | Yes | Yes | Yes | §8, §9, §13 | 0 missing; 127 exact duplicates removed (9,873 rows kept); 0 IQR/z outliers; imputers in the pipeline | PASS (checks) |
| 5 | Justification of the cleaning strategy (R1 B1) | Placeholders | — | Detected | §9, §13 | `REG-B1-DUPLICATES`, `REG-B1-OUTLIERS` | TEAM INPUT REQUIRED |
| 6 | Encoding; scaler fitted on train only; stratified split (R1 B2, G 7.1) | Yes | Yes | Yes | §17, §18 | 10-bin stratified split 7,898 / 1,975; scaler means = training means (§18 demo; validation check on all 12 saved scalers; `test_saved_scalers_were_fitted_on_training_rows_only`) | PASS |
| 7 | ≥ 1 engineered feature with written justification (R1 B3) | Hook only (by design) | Yes (pass-through) | Reported incomplete | §19, §20.12 | `feature_engineering.status()` → 0 registered; validation PENDING | TEAM INPUT REQUIRED |
| 8 | All 10 algorithms trained, no errors (R1 C1, G 3.1) | Yes | Yes | Yes | §20.1–§20.10 | 10 baseline pipelines saved; catalogue class check PASS | PASS |
| 9 | Algorithm-specific outputs (G 3.1 notes) | Yes | Yes | Yes | §20.1–§20.10 | Coefficients; ridge path; lasso zero count; enet α×l1 grid; degrees 1–3; depth/importance; n_estimators/importance; learning_rate; kernel×C, gamma, scaling; k, scaling, metrics | PASS |
| 10 | Same preprocessed data and same test set for all 10 (G 3.1, 7.2) | Yes | Yes | Yes | §17, §20 | Test-split fingerprint rebuilt independently = manifest; one shared `build_regression_pipeline` | PASS |
| 11 | Single table R², RMSE, MAE for all 10, ranked by R² (R1 C2, G 7.2) | Yes | Yes | Yes | §20.11, §21.4 | `regression_comparison.csv/.md`; recomputed from pipelines (difference 0.0); table vs saved models 4.4e-16 | PASS |
| 12 | 5-fold CV R² for the two best models (G 3.1, 7.2) | Yes (all 10) | Yes | Yes | §21.1 | `regression_cv_all_models`; mean, std, fold scores; training data only (static scan PASS) | PASS |
| 13 | Grid/random search on ≥ 2 models; best params and improvement (R1 C3) | Yes | Yes | Yes | §21.2–§21.3 | Ridge (25 candidates), Gradient Boosting (72); `regression_tuning_summary`, `_tuning_test_before_after`, `results/tuning/*_grid.csv`; test set not used for tuning | PASS |
| 14 | Residual + predicted-vs-actual plots for the best model (R1 C4) | Yes (both finalists) | Yes | Yes | §21.5 | `regression_diag_pred_vs_actual_*`, `regression_diag_residuals_*` | PASS |
| 15 | Feature importance for ≥ 1 tree model (R1 C4) | Yes | Yes | Yes | §20.6, §20.7, §21.5 | `regression_random_forest_feature_importance`, `_decision_tree_feature_importance`, tuned GB importance | PASS |
| 16 | Title, labelled axes, legend, tight layout, colourblind palette (G 7.3) | Yes | Yes | Yes | all figures | `plotting.finalize` rejects unlabelled axes; 30 figures inspected | PASS |
| 17 | random_state = 42 everywhere (G 7.1) | Yes | Yes | Yes | all | `config.RANDOM_STATE`; refit from scratch reproduces all 12 models exactly (difference 0.0) | PASS |
| 18 | Notebook runs top to bottom, outputs visible (G 7.1, D1) | Yes | Yes | Yes | whole notebook | `run_all.py` clean kernel: 85/85 cells, 0 errors | PASS |
| 19 | Markdown explaining each major code block (G 7.1) | Yes | — | Yes | whole notebook | 0 code cells directly after another code cell; 115 Markdown cells | PASS |
| 20 | Results in summary tables, not scattered prints (G 7.2) | Yes | Yes | Yes | §20.11, §21 | 13 tables (CSV + Markdown) | PASS |
| 21 | Model interpretation, model selection, conclusions (G 7.5; R1 E1, viva) | Placeholders | — | Detected | §20.x, §21, §22 | `REG-M1`…`REG-M10`, `REG-CV`, `REG-C3-TUNING`, `REG-C4-DIAGNOSTICS`, `REG-MODEL-SELECTION`, `REG-CONCLUSION` | TEAM INPUT REQUIRED |
| 22 | Saved pipelines reload and predict identically | Yes | Yes | Yes | §21.6 | 12 files, 9.9 MB total (largest 9.4 MB); reload + refit checks PASS | PASS |
| 23 | No Review 2 / Part B / clustering code | Yes | — | Yes | — | Scope-guard scan PASS | PASS |

## Known issues (not affecting results)
* During notebook execution on Windows, joblib's worker resource tracker prints `KeyError` tracebacks
  about its temporary folders to the **console**. They do not appear in the notebook and do not affect
  any result: all metrics reproduce exactly on reload and on refit. This could not be reproduced outside
  the Jupyter kernel.
* The Random Forest uses `n_jobs=1`. With several threads, the per-tree predictions are summed in a
  varying order, which changes results at about 1e-14 between calls.
