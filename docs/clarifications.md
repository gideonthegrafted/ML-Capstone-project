# Clarifications log

This log records ambiguities in the course documents, what we decided about each, and what is still
open with the instructor. **Status** is one of *Decided*, *Open — ask instructor* or *Review 2*.

Sources: **G** = 23CSE301 Capstone Guidelines PDF (section), **R1** = Review 1 rubric criterion.

---

### C1 — Stratified split for a continuous regression target · *Decided*
**Ambiguity.** R1 B2 asks for a "stratified train/test split". Stratification is defined for class
labels, but the regression target `Performance Index` is continuous (91 distinct values).

**Decision.** For the split only, the target is cut into **10 quantile bins** and those bins are
passed to `train_test_split(stratify=...)`. This means every part of the target range appears in
train and test in the same proportion.
* The bins are **never** a model feature, and the target values are **never** modified.
* The bins are not stored in `X`. A test checks this (`tests/test_preprocessing.py`).
* The setting is configurable: `REGRESSION_STRATIFY_BINS` in `src/config.py`. Set it to `None` for a
  plain random split.

### C2 — "Same preprocessed dataset" and scaling of tree models · *Decided*
**Ambiguity.** G 3.1 says all ten regressors must be trained on the *same preprocessed dataset*.
Tree-based models do not need scaling.

**Decision.** All ten regressors share **one** preprocessor: median imputation and standardisation for
the numeric columns, one-hot encoding for `Extracurricular Activities`. Standardising is a monotonic
rescaling, and tree splits are unaffected by it. Trees therefore behave identically, and the
requirement is met literally.

### C3 — Sentiment label derivation · *Decided (fixed for Review 2 as well)*
**Ambiguity.** The Restaurant Reviews data has no sentiment column. `Rating` is stored as text and
contains 1–5 in half steps, one non-numeric value `"Like"`, and 38 missing values.

**Decision (approved option A).** Rating ≥ 4 → `positive`; rating ≤ 2 → `negative`. Ratings strictly
between 2 and 4 (2.5, 3, 3.5) are dropped as ambiguous. Missing and non-numeric ratings are dropped.

Verified on the raw file (10,000 rows), **before** the Phase 6 text and duplicate cleaning:

| Outcome | Raw rating values | Rows |
|---|---|---|
| positive | 4, 4.5, 5 | 6,274 |
| negative | 1, 1.5, 2 | 2,428 |
| dropped — ambiguous middle | 2.5, 3, 3.5 | 1,259 |
| dropped — missing rating | (empty) | 38 |
| dropped — non-numeric | "Like" | 1 |
| **Total** | | **10,000** |

That leaves 8,702 labelled rows (72.1 % positive / 27.9 % negative). The final counts after dropping
rows without text and duplicates are reported in `classification.ipynb`, sections 3–4.

*Note:* the Kaggle description suggests "above 3 positive, below 3 negative". That rule would label
3.5 as positive and 2.5 as negative. We chose the stricter rule deliberately; a different rule gives
different class counts.

### C4 — Positive class for binary precision/recall · *Decided*
`pos_label = "negative"`, the minority class. Binary precision, recall and F1 are reported for
`negative` and labelled "(negative)" in every table. Weighted versions are reported alongside.
Tests check that the intended class is used (`tests/test_evaluation.py`).

### C5 — Classification metrics: rubric subset vs. guideline list · *Decided*
R1 D2 lists accuracy, weighted F1 and the confusion matrix. G 3.2 says accuracy, precision, recall,
weighted F1, the confusion matrix and ROC-AUC are **all mandatory**. We implement all six.

### C6 — SVC probability output deprecated in scikit-learn 1.9 · *Decided*
The build specification suggests `SVC(probability=True)` for ROC-AUC. In the pinned scikit-learn
(1.9.1), the `probability` parameter is **deprecated**: it emits a FutureWarning and will be removed in
1.11. SVC ROC-AUC is therefore computed from `decision_function()`, which is a valid continuous score.
`src/evaluation.get_scores()` never uses `predict()` for ROC-AUC, and it records which score was used.
Grid search for SVC scores on training cross-validation folds only.

### C7 — Datasets "assigned by instructor" · *Open — ask instructor*
G 1 and the overview table say datasets are *assigned by the instructor*. This team selected its
datasets. The technical build goes ahead with the selected datasets. **Instructor approval or
confirmation of the three datasets is an outstanding administrative item.**

### C8 — "Distribution plot for each feature" on text data · *Decided*
R1 A2 asks for a distribution plot for each feature. For the classification track the model input is
TF-IDF (thousands of word features), so a plot per TF-IDF feature is meaningless. The EDA instead
plots:
* the distributions of the numeric attributes (`Pictures`, reviewer counts parsed from `Metadata`);
* review length;
* class balance;
* the most frequent terms per class.

### C9 — Where EDA sits relative to the split · *Decided (revised in Phase 3)*
The build specification asks that no data-driven decision be made before the split. The Phase 3
notebook structure (requested by the team) places the EDA (regression sections 11–16) **before** the
train/test split (section 17). The regression EDA is therefore **descriptive, on the full de-duplicated
data (9,873 rows)**. It is kept leakage-safe as follows:
* Deterministic row cleaning (exact duplicates; for classification, label derivation and missing
  text) runs before the split because it learns nothing from the data.
* **No EDA statistic is used to fit anything.** Every fitted step (imputation, encoding, scaling,
  and later tuning) is inside a `Pipeline` fitted on the training rows only. Section 18 demonstrates this.
* The one residual risk is that the team's *choice* of engineered feature is informed by full-data
  plots. To limit this, the feature must be row-wise and its effect is measured with training-set
  cross-validation and the single test evaluation in section 20.
* The audit (A1) reports the full raw file (10,000 rows).

### C10 — Duplicate handling · *Decided (justification: team)*
* **Student Performance:** the 127 exact duplicate rows are removed before the split. The 792 rows
  that repeat only the *feature* values are kept, because 647 of those feature combinations have
  different targets. They are legitimate observations, and the number of repeats is close to what
  chance alone produces with only 64,800 possible feature combinations.
* **Restaurant Reviews:** the duplicate key is investigated on the actual columns in Phase 6. There
  are 36 exact duplicate rows and 42 duplicates on (Restaurant, Reviewer, Review). The chosen key is
  reported with before/after counts.
* The written justification is a ✍️ team cell (`REG-B1`, `CLF-B1`).

### C11 — README timing · *Decided*
G 4 (D5) requires the README only at Review 2. We maintain it from Review 1 onwards.

### C12 — GUI · *Review 2*
The GUI is optional (G 4, D8) and is a Review 2 bonus (G 6). It is deferred to Review 2 and nothing is
built for Review 1.

### C13 — Bonus arithmetic · *Open — ask instructor*
G 6 offers "up to +2 marks in Review 2" on a 25-mark review, but also says the displayed total "is
capped at 20 for official grading purposes unless otherwise communicated". As written, a 25-mark
review capped at 20 would make the bonus unreachable. We need to ask what the cap means.

### C14 — Cross-validation for clustering · *Review 2*
G 7.2 asks for cross-validation "in each track". Agglomerative clustering cannot assign unseen points
without refitting, and clustering has no held-out labels. We should raise this before Review 2.
Also for Review 2: agglomerative clustering on 10,000 rows needs an n × n distance matrix (about
0.8 GB in float64), and the Cafe Sales data has no ground-truth labels for G 3.3's after-the-fact
validation.

### C15 — Raw data in the repository · *Decided*
G 8 allows either the raw files or a script to obtain them. We use the script
(`scripts/setup_data.py`) with checksums; the reasons are in `docs/dataset_sources.md`.

### C16 — Deviations from the team's build specification · *Decided*
* The **notebooks are the implementation** and the execution engine. `scripts/run_all.py` executes
  them top to bottom. There is no separate script that re-implements the models, so the notebook and
  a script cannot drift apart.
* `src/regression_models.py` and `src/classification_models.py` are **catalogues** (the checklist of
  required algorithms that validation checks against), not model builders.
* No notebook generator is kept in the repository, so nothing can overwrite team-written cells.
* `scripts/setup_data.py` was added for data acquisition, because raw data is not committed.

### C17 — Commits · *Decided*
The three team members make all commits themselves (at least 7–8 each), under their own names. The
AI assistant does not commit and does not configure any git identity.
