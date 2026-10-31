
import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from preprocessing import load_and_clean_data, prepare_data


# 1. Load and prepare dataset
DATA_PATH = "data/raw/Telco-Customer-Churn.csv"

df = load_and_clean_data(DATA_PATH)

X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

# Create models directory
os.makedirs("models", exist_ok=True)


# 2. Define Logistic Regression
logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])


# 3. Define Random Forest
random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])


# 4. Train and evaluate models
models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    print(f"\n{name} Results")
    print("-" * 40)
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall   :", round(recall_score(y_test, y_pred), 4))
    print("F1-score :", round(f1_score(y_test, y_pred), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_test, y_prob), 4))

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".joblib"

    joblib.dump(model, f"models/{filename}")

    print(f"Model saved: models/{filename}")


print("\nAll models trained and saved successfully!")
