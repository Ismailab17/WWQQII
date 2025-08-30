# app.py
import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Water Quality Index Predictor", layout="centered")

st.title("🌊 Water Quality Index Predictor")
st.markdown("Predict **NSF WQI** and **CCME WQI** based on water quality parameters.")

# Load models and scalers
mlp_nsf = joblib.load("mlp_nsf_model.pkl")
scaler_nsf = joblib.load("scaler_nsf.pkl")
mlp_ccme = joblib.load("mlp_ccme_model.pkl")
scaler_ccme = joblib.load("scaler_ccme.pkl")

st.header("NSF WQI Input Parameters")
NO3 = st.number_input("Nitrate (NO₃⁻)", min_value=0.0, max_value=100.0, value=10.0)
COD = st.number_input("COD (mg/L)", min_value=0.0, max_value=500.0, value=30.0)
FC = st.number_input("Fecal Coliform (FC)", min_value=0.0, max_value=1000.0, value=50.0)
BOD = st.number_input("BOD (mg/L)", min_value=0.0, max_value=50.0, value=5.0)
TDS = st.number_input("TDS (mg/L)", min_value=0.0, max_value=2000.0, value=300.0)

st.header("CCME WQI Input Parameters")
PO4 = st.number_input("Phosphate (PO₄³⁻)", min_value=0.0, max_value=20.0, value=2.0)

if st.button("Calculate WQIs"):
    # Prepare inputs
    X_nsf_input = np.array([[NO3, COD, FC, BOD, TDS]])
    X_nsf_scaled = scaler_nsf.transform(X_nsf_input)
    nsf_pred = mlp_nsf.predict(X_nsf_scaled)[0]

    X_ccme_input = np.array([[COD, FC, BOD, PO4, NO3]])
    X_ccme_scaled = scaler_ccme.transform(X_ccme_input)
    ccme_pred = mlp_ccme.predict(X_ccme_scaled)[0]

    st.subheader("💧 Predicted Water Quality Indexes")
    st.write(f"**NSF WQI:** {nsf_pred:.2f}")
    st.write(f"**CCME WQI:** {ccme_pred:.2f}")

    # Interpretation functions
    def nsf_interpret(wqi):
        if wqi > 90:
            return "Excellent"
        elif wqi > 70:
            return "Good"
        elif wqi > 50:
            return "Fair"
        else:
            return "Poor"

    def ccme_interpret(wqi):
        if wqi > 80:
            return "Excellent"
        elif wqi > 60:
            return "Good"
        elif wqi > 45:
            return "Fair"
        else:
            return "Poor"

    st.write(f"**NSF WQI Interpretation:** {nsf_interpret(nsf_pred)}")
    st.write(f"**CCME WQI Interpretation:** {ccme_interpret(ccme_pred)}")
