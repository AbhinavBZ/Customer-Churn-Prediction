
import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from preprocessing import load_and_clean_data, prepare_data


# Load dataset
df = load_and_clean_data("data/raw/Telco-Customer-Churn.csv")

X_train, X_test, y_train, y_test, _ = prepare_data(df)


# Load all four models
models = {
    "Logistic Regression (Baseline)":
        "models/logistic_regression.joblib",

    "Random Forest (Baseline)":
        "models/random_forest.joblib",

    "Logistic Regression (Optimized)":
        "models/logistic_regression_optimized.joblib",

    "Random Forest (Optimized)":
        "models/random_forest_optimized.joblib"
}


results = []

for name, path in models.items():

    print(f"\nEvaluating {name}...")

    model = joblib.load(path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob)
    }

    results.append(metrics)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# Create comparison table
results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

os.makedirs("reports", exist_ok=True)

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nComparison saved to reports/model_comparison.csv")
