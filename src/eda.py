
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Create output directory
os.makedirs("reports/figures", exist_ok=True)

# Clean TotalCharges for analysis
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)

print("Blank/invalid TotalCharges:", df["TotalCharges"].isna().sum())

# These records have zero tenure, so fill their TotalCharges with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

sns.set_theme(style="whitegrid")

# 1. Churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.savefig("reports/figures/churn_distribution.png")
plt.show()

# 2. Monthly charges vs churn
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="MonthlyCharges",
    hue="Churn",
    bins=30,
    kde=True
)
plt.title("Monthly Charges vs Customer Churn")
plt.savefig("reports/figures/monthly_charges.png")
plt.show()

# 3. Churn by contract type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Customer Churn by Contract Type")
plt.savefig("reports/figures/churn_by_contract.png")
plt.show()

# 4. Churn rate by contract type
print("\nChurn rate by contract:")
print(
    df.groupby("Contract", observed=True)["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .round(2)
)

print("\nEDA completed. Charts saved in reports/figures/")
