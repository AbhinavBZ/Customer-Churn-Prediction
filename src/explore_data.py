
import pandas as pd

DATA_PATH = "data/raw/Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)

print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

print("\n========== COLUMN INFORMATION ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE RECORDS ==========")
print(df.duplicated().sum())

print("\n========== CHURN DISTRIBUTION ==========")
print(df["Churn"].value_counts())

print("\n========== CHURN PERCENTAGE ==========")
print(df["Churn"].value_counts(normalize=True) * 100)

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())
