| Model                        |   Rank (weighted F1) |   Accuracy |   Precision (negative) |   Recall (negative) |   Weighted F1 |   ROC-AUC | Score source (ROC-AUC)   |
|:-----------------------------|---------------------:|-----------:|-----------------------:|--------------------:|--------------:|----------:|:-------------------------|
| Support Vector Machine (SVC) |                    1 |     0.9609 |                 0.9428 |              0.9156 |        0.9607 |    0.9876 | decision_function        |
| Logistic Regression          |                    2 |     0.9517 |                 0.9427 |              0.8807 |        0.9512 |    0.9895 | predict_proba            |
| Decision Tree Classifier     |                    3 |     0.8689 |                 0.7590 |              0.7778 |        0.8694 |    0.8420 | predict_proba            |
| Gaussian Naive Bayes         |                    4 |     0.8355 |                 0.6567 |              0.8621 |        0.8413 |    0.8438 | predict_proba            |
| K-Nearest Neighbors          |                    5 |     0.7320 |                 0.7778 |              0.0576 |        0.6369 |    0.5525 | predict_proba            |
