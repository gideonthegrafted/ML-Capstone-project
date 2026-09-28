# Viva guide: the 15 Review 1 algorithms

This guide contains **general, factual** material about each algorithm, for every team member to revise.
It deliberately contains **no dataset-specific interpretation**: questions such as "why did model X win
on our data?" must be answered in the team's own words, from the notebooks' outputs.

For each algorithm:
* three likely questions, with factual answers;
* the important hyperparameters;
* one common misconception;
* what scaling does and doesn't do;
* the computational cost.

---

## Regression (10)

### 1. Linear Regression
* **Q: How are the coefficients found?** Ordinary least squares: minimise Σ(y − ŷ)². The solution is
  closed-form; there is no learning rate and no iteration.
* **Q: What does a coefficient mean?** The change in the prediction for a one-unit increase in that
  input, holding the other inputs fixed.
* **Q: What does R² measure?** The share of the target's variance explained. 1 is perfect; 0 is no better
  than predicting the mean; R² can be negative on test data.
* **Hyperparameters:** essentially none (`fit_intercept`).
* **Misconception:** "a large coefficient means an important feature." Coefficient size depends on the
  feature's units; compare standardised coefficients instead.
* **Scaling:** doesn't change the predictions; it only makes the coefficients comparable.
* **Cost:** very fast. Dominated by solving the least-squares system (roughly n·p²).

### 2. Ridge Regression
* **Q: What does the L2 penalty do?** Adds α·Σb² to the loss, which shrinks every coefficient towards
  zero, reduces variance and stabilises correlated inputs.
* **Q: What happens as α → 0 and as α → ∞?** As α → 0 it becomes ordinary Linear Regression. As α → ∞
  all coefficients shrink to 0, so it predicts the mean.
* **Q: Does Ridge remove features?** No. Coefficients shrink but essentially never become exactly 0.
* **Hyperparameters:** `alpha`.
* **Misconception:** "regularisation always improves test scores." If the model isn't over-fitting,
  shrinkage can only add bias.
* **Scaling:** **required.** The penalty treats all coefficients equally, so inputs must share a scale.
* **Cost:** as fast as Linear Regression.

### 3. Lasso Regression
* **Q: Why does Lasso give exact zeros?** The L1 penalty α·Σ|b| has a corner at 0, so the optimum often
  lies exactly on an axis.
* **Q: How is it solved?** By coordinate descent, updating one coefficient at a time.
* **Q: What happens with correlated inputs?** It tends to keep one of them and drop the others, somewhat arbitrarily.
* **Hyperparameters:** `alpha`.
* **Misconception:** "a zero coefficient proves a feature is useless." It is only uninformative *given
  the other features* at that α.
* **Scaling:** **required.**
* **Cost:** fast, iterative.

### 4. ElasticNet Regression
* **Q: What is `l1_ratio`?** The mix of L1 and L2 penalty: 1 is pure Lasso, and values near 0 behave like Ridge.
* **Q: Why combine the two?** To get sparsity from L1 with the stability of L2 on correlated features.
* **Q: How many hyperparameters must be tuned?** Two: `alpha` and `l1_ratio`.
* **Hyperparameters:** `alpha`, `l1_ratio`.
* **Misconception:** "the default α = 1 is a sensible choice." α is on the scale of the loss; the default
  can over-regularise badly.
* **Scaling:** **required.**
* **Cost:** fast, iterative.

### 5. Polynomial Regression (PolynomialFeatures + Linear Regression)
* **Q: Is it a linear model?** Yes, it's linear in its coefficients. The *inputs* are expanded with
  powers and interactions.
* **Q: How many terms does degree d produce?** C(p + d, d) − 1 terms for p inputs; the number grows very quickly.
* **Q: How do you choose the degree?** By cross-validation on the training set, never with the test set.
* **Hyperparameters:** `degree`, `interaction_only`.
* **Misconception:** "a higher degree is always better." It usually over-fits and extrapolates badly.
* **Scaling:** needed in practice, so that powers of large-range inputs don't become numerically extreme.
* **Cost:** grows with the number of expanded terms.

### 6. Decision Tree Regressor
* **Q: How is a split chosen?** The feature and threshold that most reduce the squared error of the child nodes.
* **Q: What does a leaf predict?** The mean target of the training rows that reach it.
* **Q: Why does an unrestricted tree over-fit?** It keeps splitting until it memorises the training data.
* **Hyperparameters:** `max_depth`, `min_samples_leaf`, `min_samples_split`.
* **Misconception:** "trees extrapolate." A tree can't predict outside the range of training targets.
* **Scaling:** **no effect.** Splits are thresholds on one feature at a time.
* **Cost:** fast to train (about n·p·log n); prediction is O(depth).

### 7. Random Forest Regressor
* **Q: Where does the randomness come from?** Bootstrap samples of rows for each tree (and optional random
  feature subsets at each split).
* **Q: Why does averaging help?** Averaging many de-correlated high-variance trees reduces variance.
* **Q: What is feature importance?** The mean decrease in impurity from splits on each feature. It is
  biased towards features with many distinct values.
* **Hyperparameters:** `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`.
* **Misconception:** "more trees cause over-fitting." Adding trees doesn't increase over-fitting; the
  performance levels off.
* **Scaling:** no effect.
* **Cost:** trees are independent, so it parallelises well; model files can be large.

### 8. Gradient Boosting Regressor
* **Q: How does boosting differ from a random forest?** Trees are built *sequentially*, each fitted to
  the residuals (the negative gradient) of the current model, and they are shallow.
* **Q: What does `learning_rate` do?** It scales each tree's contribution. A smaller rate needs more trees
  but often generalises better.
* **Q: Can it over-fit?** Yes, with too many rounds, deep trees or a large learning rate.
* **Hyperparameters:** `learning_rate`, `n_estimators`, `max_depth`, `subsample`.
* **Misconception:** "learning_rate and n_estimators are independent." They trade off against each other.
* **Scaling:** no effect.
* **Cost:** sequential, so less parallelism than a random forest.

### 9. Support Vector Regressor (SVR)
* **Q: What is the ε-insensitive loss?** Errors smaller than ε are ignored; larger ones are penalised linearly.
* **Q: What does C do?** It sets the penalty on points outside the ε-tube. A larger C fits the training data more tightly.
* **Q: What is gamma (RBF kernel)?** How far one point's influence reaches. A large gamma gives a very
  local, wiggly function.
* **Hyperparameters:** `kernel`, `C`, `epsilon`, `gamma`.
* **Misconception:** "SVR works well without scaling." Kernels depend on distances or dot products, so
  scaling matters in general. Measure it rather than assume.
* **Scaling:** generally required.
* **Cost:** training is roughly quadratic to cubic in the number of rows.

### 10. K-Nearest Neighbors Regressor
* **Q: What does "training" do?** It only stores the data; all the work happens at prediction time.
* **Q: How does k affect bias and variance?** A small k gives low bias and high variance; a large k gives a smoother, more biased model.
* **Q: Why is scaling discussed?** Distances add up per-feature differences, so the feature with the
  largest range dominates.
* **Hyperparameters:** `n_neighbors`, `weights`, `metric` / `p`.
* **Misconception:** "scaling always improves KNN." Scaling gives every feature equal weight in the
  distance. Whether that helps depends on how informative each feature is.
* **Scaling:** changes which neighbours are chosen.
* **Cost:** prediction is O(n) per query without an index; ties between equidistant neighbours can
  break differently on different machines.

---

## Classification Part A (5)

### A1. Logistic Regression
* **Q: Why "regression"?** It regresses the log-odds linearly on the inputs; classes come from thresholding the probability.
* **Q: What is an odds ratio?** e^b: the multiplicative change in the odds for a one-unit increase in the input.
* **Q: What does `class_weight="balanced"` do?** It weights errors inversely to class frequency, which
  usually raises minority-class recall.
* **Hyperparameters:** `C` (inverse regularisation), `penalty`, `class_weight`, `max_iter`.
* **Misconception:** "the 0.5 threshold is fixed." It can be moved to trade precision against recall.
* **Scaling:** needed for comparable coefficients and regularisation (L2-normalised TF-IDF provides it).
* **Cost:** fast on sparse data.

### A2. K-Nearest Neighbors Classifier
* **Q: How is a class predicted?** By majority vote among the k nearest training points; predict_proba
  is the vote share.
* **Q: Euclidean or cosine for text?** For non-empty unit vectors they rank neighbours identically, but
  they treat all-zero vectors differently.
* **Q: What is the curse of dimensionality?** In high dimensions, distances between points become similar,
  so "nearest" carries less information.
* **Hyperparameters:** `n_neighbors`, `weights`, `metric`.
* **Misconception:** "KNN has no training cost, so it is cheap." Prediction compares against every training row.
* **Scaling:** decides the distance.
* **Cost:** memory-heavy (it stores all training data); prediction is slow for large n.

### A3. Gaussian Naive Bayes
* **Q: What are the two assumptions?** (1) Features are *conditionally independent* given the class.
  (2) Each feature is *normally distributed* within each class.
* **Q: What is `var_smoothing`?** A small variance added to every feature for numerical stability.
* **Q: Why does it need dense input?** scikit-learn's `GaussianNB` computes per-feature means and variances on dense arrays.
* **Hyperparameters:** `var_smoothing`, `priors`.
* **Misconception:** "naive Bayes probabilities are well calibrated." Violated independence usually
  makes them extreme (close to 0 or 1).
* **Scaling:** doesn't change the decision.
* **Cost:** very fast (a single pass); memory is the dense matrix.

### A4. Decision Tree Classifier
* **Q: What is Gini impurity?** 1 − Σpₖ²: the probability that two random samples from a node belong to different classes.
* **Q: How are probabilities produced?** From the class shares in the leaf. For pure leaves they are 0 or 1.
* **Q: How do you control over-fitting?** With `max_depth`, `min_samples_leaf` and pruning (`ccp_alpha`).
* **Hyperparameters:** `max_depth`, `min_samples_leaf`, `criterion`, `class_weight`.
* **Misconception:** "feature importance shows the direction of the effect." It shows only the size of the impurity reduction.
* **Scaling:** no effect.
* **Cost:** fast; an unrestricted tree on sparse text can be very deep.

### A5. Support Vector Machine (SVC)
* **Q: What are support vectors?** The training points on or inside the margin; only they define the boundary.
* **Q: Linear vs RBF kernel?** Linear gives a straight hyperplane. RBF allows curved boundaries through
  similarity exp(−γ‖a − b‖²).
* **Q: How is ROC-AUC computed without probabilities?** From `decision_function` (the signed distance to
  the boundary). AUC needs only a ranking, not calibrated probabilities.
* **Hyperparameters:** `C`, `kernel`, `gamma`, `class_weight`.
* **Misconception:** "SVC(probability=True) is needed for ROC-AUC." It isn't, and it is deprecated in scikit-learn 1.9.
* **Scaling:** required (L2-normalised TF-IDF provides it).
* **Cost:** training is roughly quadratic or worse in n; it is the slowest of the five here.

---

## Cross-cutting questions (all members)
* **Why split before fitting anything?** So that no statistic learned from the test rows (scaling means,
  TF-IDF vocabulary, hyperparameters) influences the model. Otherwise the test score is optimistic.
* **Why use a Pipeline?** It refits every preprocessing step inside each CV fold and applies (never refits)
  it to test data.
* **Why rank classification by weighted F1 and not accuracy?** With 72 % positive reviews, always
  predicting "positive" already gives 72 % accuracy. Weighted F1 and minority-class recall reveal that.
* **Why is ROC-AUC computed from scores?** ROC sweeps every threshold over a continuous score. Hard 0/1
  labels give only one point on the curve.
* **Why random_state=42?** Reproducibility: the same split, bootstrap samples and tie-breaking on every run.
