import os
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from preprocessing import load_and_clean_data, prepare_data


DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
MODEL_PATH = "models/random_forest_optimized.joblib"
THRESHOLD = 0.55

# Load data and recreate the original train-test split
df = load_and_clean_data(DATA_PATH)
X_train, X_test, y_train, y_test, _ = prepare_data(df)

# Load the optimized model and fit it on the complete training set
model = joblib.load(MODEL_PATH)
model.fit(X_train, y_train)

# Predict probabilities on the untouched test set
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

# Calculate final metrics
results = {
    "Threshold": THRESHOLD,
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-Score": f1_score(y_test, y_pred),
    "ROC-AUC": roc_auc_score(y_test, y_prob),
    "PR-AUC": average_precision_score(y_test, y_prob)
}

print("\n========== FINAL MODEL EVALUATION ==========")

for metric, value in results.items():
    print(f"{metric:<12}: {value:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save final model and evaluation results
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(model, "models/final_model.joblib")

pd.DataFrame([results]).to_csv(
    "reports/final_evaluation.csv",
    index=False
)

with open("models/threshold.json", "w") as file:
    json.dump({"threshold": THRESHOLD}, file, indent=4)

print("\nFinal model and evaluation results saved successfully!")