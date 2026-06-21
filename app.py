"""
Credit Card Fraud Detection - Streamlit Web App
--------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_detection_model.pkl")
    amount_scaler = joblib.load("amount_scaler.pkl")
    time_scaler = joblib.load("time_scaler.pkl")
    return model, amount_scaler, time_scaler

try:
    model, amount_scaler, time_scaler = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

st.title("💳 Credit Card Fraud Detection")
st.write(
    "This app uses a trained Machine Learning model (Random Forest) to predict "
    "whether a credit card transaction is **fraudulent** or **normal**, based on "
    "anonymized transaction features."
)

if not model_loaded:
    st.error(
        "⚠️ Model files not found. Please make sure 'fraud_detection_model.pkl', "
        "'amount_scaler.pkl', and 'time_scaler.pkl' are in the same directory as this app."
    )
    st.stop()

st.divider()

mode = st.sidebar.radio(
    "Choose Input Mode",
    ["Quick Demo (Amount & Time only)", "Upload CSV (Batch Prediction)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About this project**\n\n"
    "Trained on the Kaggle Credit Card Fraud dataset (284,807 transactions, "
    "0.17% fraud rate). Class imbalance handled using SMOTE oversampling. "
    "Model: Random Forest Classifier."
)

if mode == "Quick Demo (Amount & Time only)":
    st.subheader("🔎 Quick Transaction Check")
    st.info(
        "Note: The real dataset features (V1-V28) are PCA-transformed and not "
        "human-readable. This demo lets you test with Amount/Time while using "
        "random/zero values for the PCA components, for demonstration purposes."
    )

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)
    with col2:
        time_val = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=50000.0, step=1000.0)

    use_random_pca = st.checkbox("Use random PCA feature values (simulate variation)", value=True)

    if st.button("🔍 Predict Transaction", type="primary"):
        amount_scaled = amount_scaler.transform([[amount]])[0][0]
        time_scaled = time_scaler.transform([[time_val]])[0][0]

        if use_random_pca:
            pca_features = np.random.normal(0, 1, 28)
        else:
            pca_features = np.zeros(28)

        
        input_features = list(pca_features) + [amount_scaled, time_scaled]

        prediction = model.predict([input_features])[0]
        probability = model.predict_proba([input_features])[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"🚨 **FRAUD DETECTED** — Fraud Probability: {probability:.2%}")
        else:
            st.success(f"✅ **NORMAL TRANSACTION** — Fraud Probability: {probability:.2%}")

        st.progress(min(float(probability), 1.0))


else:
    st.subheader("📂 Batch Prediction via CSV Upload")
    st.write(
        "Upload a CSV with the same columns as the training data "
        "(V1-V28, Amount, Time) — no 'Class' column needed."
    )

    uploaded_file = st.file_uploader("Upload transactions CSV", type=["csv"])

    if uploaded_file is not None:
        uploaded_file.seek(0)
        batch_df = pd.read_csv(uploaded_file)
        batch_df.columns = batch_df.columns.str.strip()
        st.write("Preview of uploaded data:")
        st.dataframe(batch_df.head())
        st.write("Detected columns:", batch_df.columns.tolist())

        if st.button("🔍 Run Batch Prediction", type="primary"):
            try:
                df_copy = batch_df.copy()
                df_copy.columns = df_copy.columns.str.strip()

                required_raw = ["Amount", "Time"]
                missing_raw = [c for c in required_raw if c not in df_copy.columns]
                if missing_raw:
                    st.error(f"Uploaded CSV is missing required columns: {missing_raw}")
                    st.stop()

                df_copy["Amount_scaled"] = amount_scaler.transform(df_copy[["Amount"]])
                df_copy["Time_scaled"] = time_scaler.transform(df_copy[["Time"]])
                df_copy.drop(columns=["Amount", "Time"], inplace=True)

                expected_cols = list(model.feature_names_in_)
                missing_model_cols = [c for c in expected_cols if c not in df_copy.columns]
                if missing_model_cols:
                    st.error(f"Uploaded CSV is missing expected feature columns: {missing_model_cols}")
                    st.stop()

                df_copy = df_copy[expected_cols]

                predictions = model.predict(df_copy)
                probabilities = model.predict_proba(df_copy)[:, 1]

                results = batch_df.copy()
                results["Prediction"] = np.where(predictions == 1, "Fraud", "Normal")
                results["Fraud_Probability"] = probabilities

                st.success(f"Processed {len(results)} transactions!")
                st.write(f"🚨 Fraud detected: {(predictions == 1).sum()} transactions")

                st.dataframe(results)

                csv_output = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Results CSV",
                    data=csv_output,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error processing file: {e}")

st.divider()
st.caption("Built with Python, Scikit-learn, and Streamlit | ML Project Demo | Muhammad Hassan - AI/ML Engineer")
