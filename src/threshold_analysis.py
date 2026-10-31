
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split

from preprocessing import load_and_clean_data, prepare_data


# Load and split data
df = load_and_clean_data("data/raw/Telco-Customer-Churn.csv")

X_train, X_test, y_train, y_test, _ = prepare_data(df)

# Split training data into train and validation sets
X_subtrain, X_val, y_subtrain, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# Load optimized Random Forest
model = joblib.load("models/random_forest_optimized.joblib")

# Train on subtraining data only
model.fit(X_subtrain, y_subtrain)

# Get validation probabilities
y_prob = model.predict_proba(X_val)[:, 1]

results = []

for threshold in np.arange(0.20, 0.71, 0.05):

    y_pred = (y_prob >= threshold).astype(int)

    results.append({
        "Threshold": round(threshold, 2),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "F1-Score": f1_score(y_val, y_pred)
    })

results_df = pd.DataFrame(results)

print("\n========== THRESHOLD ANALYSIS ==========")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "reports/threshold_analysis.csv",
    index=False
)

print("\nThreshold analysis saved successfully!")
