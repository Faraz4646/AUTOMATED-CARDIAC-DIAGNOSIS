
import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Automated Cardiac Diagnosis (Demo)")

st.title("Automated Cardiac Diagnosis — Risk Prediction")
st.caption("Educational demo only — not a medical device or clinical advice.")

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        pack = pickle.load(f)
    return pack

st.sidebar.header("About")
st.sidebar.write("""
This demo predicts heart disease risk using a logistic regression model
trained on the UCI-style heart dataset.
""")

# Feature inputs (basic subset; you can expand)
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    sex = st.selectbox("Sex (1=Male, 0=Female)", options=[1,0], index=0)
    cp = st.selectbox("Chest Pain Type (0–3 or 1–4)", options=[0,1,2,3,4], index=1)
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=220, value=130)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=240)
with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0,1], index=0)
    restecg = st.selectbox("Resting ECG (0–2)", options=[0,1,2], index=0)
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (0/1)", options=[0,1], index=0)
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

slope = st.selectbox("Slope of ST segment (0–2 or 1–3)", options=[0,1,2,3], index=1)
ca = st.number_input("Number of major vessels (ca)", min_value=0, max_value=4, value=0, step=1)
thal = st.selectbox("Thal (0–3 or 3/6/7)", options=[0,1,2,3,6,7], index=4)

# Collect into a DataFrame matching training features if possible
def build_input_df(feature_names):
    raw = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    df = pd.DataFrame([raw])
    # Reindex to model feature order (if features differ, align intersection)
    cols = [c for c in feature_names if c in df.columns]
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        st.warning(f"Missing features not in UI: {missing}. They will be dropped.")
    return df.reindex(columns=cols, fill_value=0)

if st.button("Predict Risk"):
    try:
        pack = load_model()
        scaler, model, feature_names = pack["scaler"], pack["model"], pack["feature_names"]
        X = build_input_df(feature_names)
        Xs = scaler.transform(X)
        prob = float(model.predict_proba(Xs)[:,1][0])
        pred = "High Risk" if prob >= 0.5 else "Low Risk"
        st.subheader(f"Prediction: {pred}")
        st.write(f"Estimated probability of heart disease: **{prob:.2f}**")
        st.info("If you have symptoms like chest pain, shortness of breath, or fainting, seek medical care immediately.")
    except FileNotFoundError:
        st.error("model.pkl not found. Please run `python train.py` first.")
    except Exception as e:
        st.error(f"Error: {e}")
