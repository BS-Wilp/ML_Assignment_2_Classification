
# Adult Income Prediction

BITS WILP - Machine Learning Assignment 2

## a. Problem Statement

The objective of this project is to predict whether an individual's annual income exceeds $50,000 using machine learning classification algorithms.

Five classification models are implemented and evaluated to compare their performance on the Adult Income dataset.

The project also includes a Streamlit web application that allows users to select a trained model, upload a CSV file, generate income predictions, and download the prediction results.

## b. Dataset Description

The project uses the Adult Income dataset, also known as the Census Income dataset.

The dataset contains demographic and employment-related information such as age, education, occupation, marital status, working hours, and other attributes.

The target variable is `income`, which represents two income categories:

- 0: Income less than or equal to $50,000
- 1: Income greater than $50,000

The dataset includes numerical and categorical features. Data preprocessing is performed to prepare the features for machine learning classification.

The application is tested using `test_data.csv`, containing 100 records with known income labels.

## c. GitHub Repository Link

GitHub Repository:

https://github.com/BS-Wilp/ML_Assignment_2_Classification

Streamlit Application:

https://bs-wilp-adult-income-prediction.streamlit.app/

## d. Models Used and Performance Comparison

The following five machine learning classification algorithms are implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Model Evaluation Metrics

The models are evaluated using:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9000 | 0.9088 | 0.8261 | 0.7600 | 0.7917 | 0.7271 |
| Decision Tree | 0.8600 | 0.9336 | 0.7391 | 0.6800 | 0.7083 | 0.6174 |
| KNN | 0.8800 | 0.8709 | 0.7407 | 0.8000 | 0.7692 | 0.6892 |
| Gaussian Naive Bayes | 0.4600 | 0.6944 | 0.3117 | 0.9600 | 0.4706 | 0.2607 |
| Support Vector Machine | 0.8800 | 0.8960 | 0.8095 | 0.6800 | 0.7391 | 0.6662 |

## e. Observations on Model Performance

### 1. Logistic Regression

Logistic Regression achieved the highest accuracy of 90% among the five models.

It also achieved the highest F1 Score of 0.7917 and MCC of 0.7271.

This indicates that Logistic Regression provides the best overall classification performance on the selected test dataset.

### 2. Decision Tree

Decision Tree achieved an accuracy of 86% and an AUC of 0.9336.

It obtained the highest AUC among all five models, indicating strong discrimination between the two income classes across different classification thresholds.

However, its recall of 0.68 was lower than that of Logistic Regression and KNN.

### 3. K-Nearest Neighbors (KNN)

KNN achieved an accuracy of 88%, recall of 0.80, and F1 Score of 0.7692.

Its recall was higher than that of Logistic Regression, Decision Tree, and SVM.

This suggests that KNN identifies a relatively high proportion of individuals belonging to the higher-income category.

### 4. Gaussian Naive Bayes

Gaussian Naive Bayes achieved an accuracy of 46%, the lowest among the five models.

However, it achieved the highest recall of 0.96.

Its low precision of 0.3117 indicates a large number of false positive predictions.

Therefore, Gaussian Naive Bayes is less suitable when overall classification accuracy and precision are important.

### 5. Support Vector Machine (SVM)

SVM achieved an accuracy of 88%, precision of 0.8095, and F1 Score of 0.7391.

Its precision was relatively high, indicating that most records predicted as belonging to the higher-income category were correctly classified.

However, its recall of 0.68 was lower than that of Logistic Regression and KNN.

### Overall Conclusion

Based on the evaluation results, Logistic Regression is the best overall performing model on the selected test dataset, achieving the highest accuracy, F1 Score, and MCC.

Decision Tree achieved the highest AUC, while Gaussian Naive Bayes achieved the highest recall.

The comparison demonstrates that different machine learning algorithms perform differently depending on the evaluation metric considered.

## Streamlit Application

The Streamlit application provides the following features:

- Selection of one of the five trained machine learning models
- Uploading CSV files for prediction
- Displaying uploaded data
- Generating income predictions
- Displaying Accuracy, AUC, Precision, Recall, F1 Score, and MCC when actual labels are available
- Displaying the confusion matrix
- Downloading prediction results as a CSV file

## How to Run

1. Install the required dependencies:

   `pip install -r requirements.txt`

2. Run the Streamlit application:

   `streamlit run app.py`

## Project Files

- `ML_Assignment_2_Classification.ipynb` - Machine learning training notebook
- `app.py` - Streamlit application
- `test_data.csv` - Test dataset
- `model/` - Five trained machine learning models
- `requirements.txt` - Required Python libraries
- `README.md` - Project documentation
