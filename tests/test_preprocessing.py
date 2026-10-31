
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from preprocessing import load_and_clean_data, prepare_data


def test_load_and_clean_data():
    sample = pd.DataFrame({
        "customerID": ["C001", "C002", "C003", "C004"],
        "gender": ["Male", "Female", "Male", "Female"],
        "tenure": [1, 12, 24, 36],
        "TotalCharges": ["", "100", "500", "900"],
        "Churn": ["Yes", "No", "Yes", "No"]
    })

    path = Path("tests/sample_data.csv")
    sample.to_csv(path, index=False)

    try:
        df = load_and_clean_data(path)

        assert "customerID" not in df.columns
        assert df["Churn"].tolist() == [1, 0, 1, 0]
        assert df["TotalCharges"].tolist() == [0, 100, 500, 900]
    finally:
        path.unlink(missing_ok=True)


def test_prepare_data():
    sample = pd.DataFrame({
        "gender": ["Male", "Female"] * 10,
        "tenure": list(range(20)),
        "MonthlyCharges": [50.0 + i for i in range(20)],
        "Contract": ["Month-to-month", "Two year"] * 10,
        "Churn": [0, 1] * 10
    })

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(sample)

    assert len(X_train) + len(X_test) == 20
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert preprocessor is not None
