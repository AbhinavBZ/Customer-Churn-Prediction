
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import cross_validate

from preprocessing import load_and_clean_data, prepare_data


# 1. Load and prepare the dataset
df = load_and_clean_data("data/raw/Telco-Customer-Churn.csv")

X_train, X_test, y_train, y_test, _ = prepare_data(df)

os.makedirs("reports/figures", exist_ok=True)
sns.set_theme(style="whitegrid")


# 2. Load saved models
models = {
    "Logistic Regression": joblib.load(
        "models/logistic_regression.joblib"
    ),
    "Random Forest": joblib.load(
        "models/random_forest.joblib"
    )
}

results = []
plt.figure(figsize=(8, 6))


# 3. Evaluate each model
for name, model in models.items():

    print(f"\n========== {name} ==========")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"]
    )

    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(
        f"reports/figures/{name.lower().replace(' ', '_')}_confusion.png"
    )
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.savefig(
        f"reports/figures/{name.lower().replace(' ', '_')}_roc.png"
    )
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {name}")
    plt.savefig(
        f"reports/figures/{name.lower().replace(' ', '_')}_pr.png"
    )
    plt.close()

    ap = average_precision_score(y_test, y_prob)

    # Cross-validation on training data only
    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring=["roc_auc", "f1", "recall"],
        n_jobs=-1
    )

    print("\n5-Fold Cross-Validation:")
    print("Mean ROC-AUC:", round(cv_scores["test_roc_auc"].mean(), 4))
    print("Mean F1:", round(cv_scores["test_f1"].mean(), 4))
    print("Mean Recall:", round(cv_scores["test_recall"].mean(), 4))

    results.append({
        "Model": name,
        "Test ROC-AUC": auc,
        "Test PR-AUC": ap,
        "CV ROC-AUC": cv_scores["test_roc_auc"].mean(),
        "CV F1": cv_scores["test_f1"].mean(),
        "CV Recall": cv_scores["test_recall"].mean()
    })


# 4. Save evaluation summary
results_df = pd.DataFrame(results)

print("\n========== FINAL EVALUATION SUMMARY ==========")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "reports/model_evaluation.csv",
    index=False
)

print("\nEvaluation completed successfully!")
print("Charts and metrics saved in reports/")
