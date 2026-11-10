
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef
)

st.title("Adult Income Prediction")
st.write("Machine Learning Assignment 2")

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Gaussian Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl"
}

selected_model = st.selectbox(
    "Select Machine Learning Model",
    list(model_files.keys())
)

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Data")
    st.dataframe(data.head())

    if st.button("Predict"):
        model_path = os.path.join(
            "model", model_files[selected_model]
        )

        model = joblib.load(model_path)

        X_input = data.drop(columns=["income"], errors="ignore")
        predictions = model.predict(X_input)

        result = data.copy()
        result["Predicted Income"] = predictions
        if "income" in data.columns:
            y_actual = data["income"]

            st.subheader("Model Evaluation")

            st.write("Accuracy:", accuracy_score(y_actual, predictions))
            st.write("Precision:", precision_score(y_actual, predictions, zero_division=0))
            st.write("Recall:", recall_score(y_actual, predictions, zero_division=0))
            st.write("F1 Score:", f1_score(y_actual, predictions, zero_division=0))
            st.write("MCC:", matthews_corrcoef(y_actual, predictions))
if hasattr(model, "predict_proba"):
    y_scores = model.predict_proba(X_input)[:, 1]
elif hasattr(model, "decision_function"):
    y_scores = model.decision_function(X_input)
else:
    y_scores = None

if y_scores is not None and y_actual.nunique() == 2:
    st.write("AUC:", roc_auc_score(y_actual, y_scores))
else:
    st.write("AUC: Not available")
            st.write("Confusion Matrix")
            cm = confusion_matrix(y_actual, predictions, labels=[0, 1])

            st.dataframe(
                pd.DataFrame(
                    cm,
                    index=["Actual 0", "Actual 1"],
                    columns=["Predicted 0", "Predicted 1"]
                )
            )

        st.write("Prediction Results")
        st.dataframe(result)

        csv = result.to_csv(index=False)

        st.download_button(
            "Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )
