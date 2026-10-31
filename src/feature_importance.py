
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import load_and_clean_data, prepare_data


# Load dataset and recreate the original split
df = load_and_clean_data("data/raw/Telco-Customer-Churn.csv")
X_train, X_test, y_train, y_test, _ = prepare_data(df)

# Load trained models
rf_model = joblib.load("models/final_model.joblib")
lr_model = joblib.load("models/logistic_regression.joblib")

# Get transformed feature names
feature_names = (
    rf_model.named_steps["preprocessor"].get_feature_names_out()
)

# Create output directory
os.makedirs("reports/figures", exist_ok=True)


# ------------------------------------------------
# 1. Random Forest Feature Importance
# ------------------------------------------------

rf_importance = rf_model.named_steps[
    "classifier"
].feature_importances_

rf_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_importance
}).sort_values("Importance", ascending=False)

rf_df.to_csv(
    "reports/random_forest_feature_importance.csv",
    index=False
)

print("\n========== RANDOM FOREST TOP 15 FEATURES ==========")
print(rf_df.head(15).to_string(index=False))

# Plot top 15 features
top_rf = rf_df.head(15).sort_values("Importance")

plt.figure(figsize=(10, 7))
plt.barh(top_rf["Feature"], top_rf["Importance"])
plt.title("Top 15 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(
    "reports/figures/random_forest_feature_importance.png",
    dpi=300
)
plt.close()


# ------------------------------------------------
# 2. Logistic Regression Coefficients
# ------------------------------------------------

lr_feature_names = (
    lr_model.named_steps["preprocessor"].get_feature_names_out()
)

lr_coefficients = lr_model.named_steps[
    "classifier"
].coef_[0]

lr_df = pd.DataFrame({
    "Feature": lr_feature_names,
    "Coefficient": lr_coefficients,
    "Absolute_Importance": abs(lr_coefficients)
}).sort_values("Absolute_Importance", ascending=False)

lr_df.to_csv(
    "reports/logistic_regression_coefficients.csv",
    index=False
)

print("\n========== LOGISTIC REGRESSION TOP 15 FEATURES ==========")
print(lr_df.head(15).to_string(index=False))

# Plot top 15 coefficients by absolute magnitude
top_lr = lr_df.head(15).sort_values("Coefficient")

plt.figure(figsize=(10, 7))
plt.barh(
    top_lr["Feature"],
    top_lr["Coefficient"]
)
plt.axvline(x=0, linewidth=1)
plt.title("Top 15 Logistic Regression Coefficients")
plt.xlabel("Coefficient (positive = higher churn tendency)")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(
    "reports/figures/logistic_regression_coefficients.png",
    dpi=300
)
plt.close()


print("\nFeature importance analysis completed successfully!")
print("CSV files and visualizations saved in reports/")
