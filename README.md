
# Customer Churn Prediction

An end-to-end Machine Learning project that predicts customer churn using the IBM Telco Customer Churn dataset. It includes data preprocessing, exploratory analysis, model training, hyperparameter optimization, evaluation, feature importance, and an interactive Streamlit application.

## Project Overview

Customer churn occurs when customers stop using a company's services. Predicting churn helps businesses identify customers who may leave and take appropriate retention measures.

This project compares Logistic Regression and Random Forest classifiers and uses an optimized Random Forest model for the interactive prediction application.

## Objectives

- Analyze customer data and identify churn patterns.
- Build and compare classification models.
- Optimize model hyperparameters.
- Evaluate models using classification metrics.
- Select a classification threshold based on validation results.
- Identify important features influencing predictions.
- Provide an interactive customer churn prediction interface.

## Dataset

**Source:** IBM Telco Customer Churn Dataset

- Total records: 7,043
- Total columns: 21
- Target variable: `Churn`
- Target classes: Yes and No

The `customerID` column is excluded from model training because it is an identifier rather than a predictive customer attribute.

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Model Storage | Joblib |
| Web Application | Streamlit |
| Testing | Pytest |
| Version Control | Git, GitHub |

## Project Workflow

1. Data exploration and analysis
2. Data cleaning and preprocessing
3. Feature encoding and scaling
4. Logistic Regression and Random Forest training
5. Model evaluation and cross-validation
6. Hyperparameter optimization
7. Classification threshold analysis
8. Feature importance and explainability
9. Interactive Streamlit application
10. Automated testing

## Model Performance

The optimized Random Forest model was evaluated on a held-out test set using a classification threshold of 0.55.

| Metric | Test Result |
|---|---:|
| Accuracy | 77% |
| Precision (Churn) | 55% |
| Recall (Churn) | 71% |
| F1-score (Churn) | 62% |

### Confusion Matrix

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 817 | 218 |
| Actual Churn | 108 | 266 |

The model correctly identified 266 out of 374 actual churn customers.

The selected threshold of 0.55 achieved a validation F1-score of 0.6388 and recall of 0.7157.

## Feature Importance

The Random Forest model identified the following important features:

1. Tenure
2. Total Charges
3. Contract Type
4. Monthly Charges
5. Online Security
6. Tech Support
7. Internet Service
8. Payment Method

Feature importance indicates model associations, not causation.

## Application Features

- Customer information input form
- Churn prediction (Yes/No)
- Churn probability percentage
- Low, Medium, and High risk categories
- Global feature importance visualization

## Project Structure

```text
Customer-Churn-Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│
├── docs/
├── models/
│
├── reports/
│   └── figures/
│
├── src/
│   ├── explore_data.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── optimize.py
│   ├── compare_models.py
│   ├── threshold_analysis.py
│   ├── final_evaluation.py
│   └── feature_importance.py
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_prediction.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/AbhinavBZ/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the IBM Telco Customer Churn CSV dataset and place it at:

```text
data/raw/Telco-Customer-Churn.csv
```

### 5. Run the application

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

## Run Automated Tests

```bash
python -m pytest tests/ -v
```

The current test suite contains four passing tests covering preprocessing, prediction, and threshold configuration.

## Future Improvements

- Add SHAP-based individual prediction explanations.
- Improve recall while controlling false positives.
- Add customer-level risk reports.
- Deploy the application online.
- Introduce model monitoring and retraining.

## Author

**Abhinav Bhardwaj**

B.Tech – Computer Science and Engineering

GitHub: [AbhinavBZ](https://github.com/AbhinavBZ)

---

*This project was developed as an end-to-end machine learning implementation for customer churn prediction.*
