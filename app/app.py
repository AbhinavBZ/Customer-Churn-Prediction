
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "final_model.joblib"
THRESHOLD_PATH = ROOT_DIR / "models" / "threshold.json"
IMPORTANCE_PATH = ROOT_DIR / "reports" / "random_forest_feature_importance.csv"


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# Load model and threshold
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, "r") as file:
        threshold = json.load(file)["threshold"]

    return model, threshold


model, THRESHOLD = load_model()


# Application heading
st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a telecom customer is likely to leave "
    "the service using a trained Machine Learning model."
)

st.divider()


# Customer input form
st.subheader("Enter Customer Information")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

        tenure = st.slider("Tenure (Months)", 0, 72, 12)

        phone = st.selectbox("Phone Service", ["Yes", "No"])

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with col2:
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=1000.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=100000.0,
            value=840.0
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True
    )


# Prediction
if submitted:

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    probability = model.predict_proba(customer)[0][1]

    prediction = int(probability >= THRESHOLD)

    st.divider()
    st.subheader("Prediction Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is unlikely to churn")

    with result_col2:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    # Risk category based on probability bands
    if probability < 0.30:
        risk = "Low"
    elif probability < 0.60:
        risk = "Medium"
    else:
        risk = "High"

    st.metric("Risk Category", risk)

    st.progress(float(probability))

    st.caption(
        f"Classification threshold: {THRESHOLD:.2f}. "
        "Risk categories are presentation bands, separate from "
        "the model's classification threshold."
    )


# Global feature importance
st.divider()
st.subheader("Important Churn Factors")

if IMPORTANCE_PATH.exists():

    importance = pd.read_csv(IMPORTANCE_PATH).head(10)

    st.bar_chart(
        importance.set_index("Feature")["Importance"],
        horizontal=True
    )

    st.caption(
        "These are global Random Forest feature importances, "
        "not explanations specific to the entered customer."
    )

else:
    st.info("Feature importance report is not available.")

st.divider()

st.caption(
    "Customer Churn Prediction | Machine Learning Project"
)
