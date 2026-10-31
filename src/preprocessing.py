
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )

    # Fill invalid values
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Remove customer identifier
    df=df.drop(columns=["customerID"])

    # Convert target into binary values
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def prepare_data(df):

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Identify feature types
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # Split before fitting preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":

    DATA_PATH = "data/raw/Telco-Customer-Churn.csv"

    df = load_and_clean_data(DATA_PATH)

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    print("Preprocessing completed successfully!")
    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)
    print("Training target distribution:")
    print(y_train.value_counts())
    print("\nTesting target distribution:")
    print(y_test.value_counts())

    print("\nNumerical and categorical preprocessing pipelines created.")
