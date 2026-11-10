
import streamlit as st
import pandas as pd
import joblib
import os

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

        predictions = model.predict(data)

        result = data.copy()
        result["Predicted Income"] = predictions

        st.write("Prediction Results")
        st.dataframe(result)

        csv = result.to_csv(index=False)

        st.download_button(
            "Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )
