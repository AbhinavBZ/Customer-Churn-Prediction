
import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from preprocessing import load_and_clean_data, prepare_data


# 1. Load and prepare data
df = load_and_clean_data("data/raw/Telco-Customer-Churn.csv")

X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# 2. Define Logistic Regression pipeline
logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

logistic_params = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__class_weight": [None, "balanced"]
}


# 3. Define Random Forest pipeline
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ))
])

rf_params = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__class_weight": [None, "balanced"]
}


# 4. Define hyperparameter searches
searches = {
    "Logistic Regression": RandomizedSearchCV(
        logistic_pipeline,
        param_distributions=logistic_params,
        n_iter=10,
        scoring="roc_auc",
        cv=5,
        random_state=42,
        n_jobs=-1,
        refit=True
    ),

    "Random Forest": RandomizedSearchCV(
        rf_pipeline,
        param_distributions=rf_params,
        n_iter=10,
        scoring="roc_auc",
        cv=5,
        random_state=42,
        n_jobs=-1,
        refit=True,
        pre_dispatch="2*n_jobs"
    )
}


# 5. Train and optimize
results = []

for name, search in searches.items():

    print(f"\nOptimizing {name}...")
    search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(search.best_params_)

    print("Best CV ROC-AUC:", round(search.best_score_, 4))

    # Save optimized model
    filename = name.lower().replace(" ", "_") + "_optimized.joblib"

    joblib.dump(search.best_estimator_, f"models/{filename}")

    print(f"Saved: models/{filename}")

    results.append({
        "Model": name,
        "Best CV ROC-AUC": search.best_score_,
        "Best Parameters": str(search.best_params_)
    })


# 6. Save optimization results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "reports/optimization_results.csv",
    index=False
)

print("\n========== OPTIMIZATION SUMMARY ==========")
print(results_df.to_string(index=False))

print("\nOptimization completed successfully!")
