
import json
from pathlib import Path

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "final_model.joblib"
THRESHOLD_PATH = ROOT_DIR / "models" / "threshold.json"


def test_model_prediction():
    model = joblib.load(MODEL_PATH)

    customer = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.0,
        "TotalCharges": 180.0
    }])

    probabilities = model.predict_proba(customer)

    assert probabilities.shape == (1, 2)
    assert 0 <= probabilities[0][1] <= 1
    assert abs(probabilities.sum() - 1.0) < 1e-6


def test_threshold_configuration():
    with open(THRESHOLD_PATH, "r") as file:
        config = json.load(file)

    threshold = config["threshold"]

    assert isinstance(threshold, (int, float))
    assert 0 <= threshold <= 1
