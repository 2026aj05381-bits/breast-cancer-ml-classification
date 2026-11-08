
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# 1. Configure application
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

st.title("🔬 Breast Cancer Classification")
st.write("Evaluate five pretrained ML models without retraining.")
st.caption("Educational demonstration only. Not for clinical diagnosis.")

# 2. Select model
selected_model = st.selectbox(
    "Select Machine Learning Model",
    list(MODEL_FILES.keys()),
    key="model_selector"
)

# 3. Select dataset source
data_source = st.radio(
    "Choose Test Dataset Source",
    [
        "Use Sample Test Dataset",
        "Upload CSV or Excel File"
    ],
    key="dataset_source"
)

if data_source == "Use Sample Test Dataset":
    source = BASE_DIR / "test_data.csv"

    if not source.is_file():
        st.error("test_data.csv not found in project folder.")
        st.stop()

else:
    source = st.file_uploader(
        "Upload Test Dataset",
        type=["csv", "xlsx"],
        key="test_dataset_uploader"
    )

    if source is None:
        st.info("Please upload a CSV or Excel file.")
        st.stop()

# 4. Read dataset
try:
    filename = source.name.lower()

    if filename.endswith(".csv"):
        test_df = pd.read_csv(source)
    elif filename.endswith(".xlsx"):
        test_df = pd.read_excel(source)
    else:
        st.error("Only CSV and XLSX files are supported.")
        st.stop()

except Exception as error:
    st.error(f"Unable to read dataset: {error}")
    st.stop()

if test_df.empty:
    st.error("Dataset is empty.")
    st.stop()

test_df.columns = test_df.columns.astype(str).str.strip()

if test_df.columns.duplicated().any():
    st.error("Duplicate column names found.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(test_df.head(10), use_container_width=True)
st.write("Total Samples:", len(test_df))

# 5. Prepare target labels
if "target" in test_df.columns:
    labels = pd.to_numeric(
        test_df["target"],
        errors="coerce"
    )

elif "Diagnosis" in test_df.columns:
    labels = (
        test_df["Diagnosis"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"M": 1, "B": 0})
    )

else:
    st.error("Dataset must contain target or Diagnosis column.")
    st.stop()

if labels.isna().any() or not labels.isin([0, 1]).all():
    st.error("Target labels must be 0/1 or Diagnosis M/B.")
    st.stop()

y_test_app = labels.astype(int)

X_test_app = test_df.drop(
    columns=["target", "Diagnosis", "ID", "id"],
    errors="ignore"
)

# 6. Load saved model
model_path = BASE_DIR / "model" / MODEL_FILES[selected_model]

if not model_path.is_file():
    st.error(f"Model file missing: {model_path.name}")
    st.stop()

try:
    loaded_model = load_model(str(model_path))
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

# 7. Validate input features
if hasattr(loaded_model, "feature_names_in_"):
    expected_features = list(
        loaded_model.feature_names_in_
    )

    missing = [
        col for col in expected_features
        if col not in X_test_app.columns
    ]

    extra = [
        col for col in X_test_app.columns
        if col not in expected_features
    ]

    if missing or extra:
        st.error(
            f"Feature mismatch. Missing: {missing}; "
            f"Unexpected: {extra}"
        )
        st.stop()

    X_test_app = X_test_app[expected_features]

elif X_test_app.shape[1] != 30:
    st.error("Expected 30 input features.")
    st.stop()

X_test_app = X_test_app.apply(
    pd.to_numeric,
    errors="coerce"
)

if X_test_app.isna().any().any():
    st.error("Feature data contains missing or invalid values.")
    st.stop()

# 8. Predict using pretrained model
try:
    predictions = loaded_model.predict(X_test_app)

    class_probabilities = loaded_model.predict_proba(
        X_test_app
    )

    positive_index = list(
        loaded_model.classes_
    ).index(1)

    probabilities = class_probabilities[:, positive_index]

except Exception as error:
    st.error(f"Prediction failed: {error}")
    st.stop()

st.success(f"{selected_model} loaded successfully!")

# 9. Display predictions
prediction_df = pd.DataFrame({
    "Actual": y_test_app.to_numpy(),
    "Predicted": predictions,
    "Malignant Probability": probabilities
})

prediction_df["Result"] = [
    "Correct" if actual == predicted else "Incorrect"
    for actual, predicted in zip(y_test_app, predictions)
]

st.subheader("Prediction Results")
st.dataframe(prediction_df, use_container_width=True)

# 10. Calculate six metrics
accuracy = accuracy_score(y_test_app, predictions)

if y_test_app.nunique() == 2:
    auc = roc_auc_score(y_test_app, probabilities)
else:
    auc = float("nan")

precision = precision_score(
    y_test_app, predictions, zero_division=0
)

recall = recall_score(
    y_test_app, predictions, zero_division=0
)

f1 = f1_score(
    y_test_app, predictions, zero_division=0
)

mcc = matthews_corrcoef(y_test_app, predictions)

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy", "AUC", "Precision",
        "Recall", "F1 Score", "MCC"
    ],
    "Value": [
        accuracy, auc, precision,
        recall, f1, mcc
    ]
})

st.subheader("Model Evaluation Metrics")

st.dataframe(
    metrics_df.style.format(
        {"Value": "{:.4f}"},
        na_rep="N/A"
    ),
    use_container_width=True
)

if y_test_app.nunique() < 2:
    st.warning("AUC requires both target classes.")

# 11. Confusion matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_test_app,
    predictions,
    labels=[0, 1]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Benign", "Actual Malignant"],
    columns=["Predicted Benign", "Predicted Malignant"]
)

st.dataframe(cm_df, use_container_width=True)

# ---------------------------------------------
# 12. Classification Report Table
# ---------------------------------------------

st.subheader("Classification Report")

# Get classification report as a dictionary
report_dict = classification_report(
    y_test_app,
    predictions,
    labels=[0, 1],
    target_names=["Benign", "Malignant"],
    output_dict=True,
    zero_division=0
)

# Convert report into a DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Remove accuracy row because it has a
# different structure from class-level metrics
report_df = report_df.drop(
    index="accuracy",
    errors="ignore"
)

# Rename rows
report_df = report_df.rename(
    index={
        "macro avg": "Macro Average",
        "weighted avg": "Weighted Average"
    }
)

# Rename columns
report_df = report_df.rename(
    columns={
        "precision": "Precision",
        "recall": "Recall",
        "f1-score": "F1 Score",
        "support": "Support"
    }
)

# Show metrics in a clean table
st.dataframe(
    report_df.style
        .format({
            "Precision": "{:.2f}",
            "Recall": "{:.2f}",
            "F1 Score": "{:.2f}",
            "Support": "{:.0f}"
        })
        .background_gradient(
            cmap="Blues",
            subset=["Precision", "Recall", "F1 Score"]
        ),
    use_container_width=True,
    hide_index=False
)

# Show overall accuracy separately
st.metric(
    label="Overall Accuracy",
    value=f"{accuracy_score(y_test_app, predictions) * 100:.2f}%"
)


# 13. Download predictions
st.download_button(
    "Download Predictions as CSV",
    data=prediction_df.to_csv(index=False),
    file_name="prediction_results.csv",
    mime="text/csv",
    key="download_predictions"
)
