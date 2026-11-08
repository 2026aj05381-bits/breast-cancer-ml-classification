# Breast Cancer Classification — Machine Learning Assignment 2

**BITS Pilani WILP| M.Tech (AI & ML) | Machine Learning**

## a. Problem statement

Compare five supervised classification algorithms on the same breast cancer dataset. Train the models on a training split, evaluate them using six classification metrics on a held-out test split, and provide an interactive Streamlit application that loads saved models to evaluate uploaded test data without retraining.

This project is for educational evaluation only, **not clinical diagnosis**.

## b. Dataset description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository
- **Source:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Records:** 569
- **Predictor features:** 30 numerical measurements of cell nuclei, including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension, with mean, standard error, and worst-value measurements.
- **Target:** Diagnosis: `M` (malignant) mapped to `1`, `B` (benign) mapped to `0`.
- **Preprocessing:** Assign column headers to the raw `wdbc.data` file, exclude the patient identifier, map the target to numeric values, check missing values, and use an 80%/20% stratified train/test split with `random_state=42`.
- **Split:** 455 training records, 114 testing records. Scaling is included in the saved pipelines for models that use it.

## c. GitHub repository link

**Repository:** https://github.com/2026aj05381-bits/breast-cancer-ml-classification.git

## d. Models used and comparison of evaluation metrics

The following results were obtained on the same held-out test dataset. For precision, recall, F1, and AUC, the positive class is malignant (`1`).

| ML model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| K-Nearest Neighbors (KNN) | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine (SVM) | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations on model performance

| Model | Observation |
|---|---|
| Logistic Regression | Strong results across metrics; the highest AUC (0.9960), indicating excellent ranking of malignant versus benign cases on this test set. |
| Decision Tree | Lowest accuracy, AUC, F1, and MCC among the five evaluated models on this split. |
| KNN | High accuracy (0.9561) and precision (0.9744), with recall of 0.9048. |
| Gaussian Naive Bayes | Precision of 1.0000 but the lowest recall (0.8333), meaning it missed more malignant cases than the other models. |
| SVM | Highest accuracy (0.9737), F1 (0.9630), and MCC (0.9442), with precision of 1.0000 and recall of 0.9286. |
| **Overall winner** | **SVM**, based on the best combined performance on accuracy, F1, and MCC for this held-out test set. Logistic Regression has the highest AUC. |

These observations describe this dataset and test split; they do not establish general clinical performance.

## Streamlit application

The app supports choosing one of five previously trained models, uploading a test CSV (or Excel file), viewing predictions, and inspecting evaluation metrics and a confusion matrix or classification report. A bundled sample `test_data.csv` is also provided.

**Live app:** https://breast-cancer-ml-classification-vv9hzzjitphkreb5yoggr9.streamlit.app/
### Run locally

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch the application from the directory containing `app.py`:

```bash
python -m streamlit run app.py
```

The uploaded CSV must contain all 30 feature columns and either `target` (0/1) or `Diagnosis` (B/M). The app loads the saved model files from `model/` and does not retrain on uploaded test data.

## Repository contents

```text
app.py
requirements.txt
README.md
test_data.csv
2026aj05381.ipynb
model/
    logistic_regression.joblib
    decision_tree.joblib
    knn.joblib
    naive_bayes.joblib
    svm.joblib

```

