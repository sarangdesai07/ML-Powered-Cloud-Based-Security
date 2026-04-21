import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CICIDS 2017 Multi-Class IDS",
    layout="wide"
)

# ---------------- DARK THEME ----------------
st.markdown("""
<style>
.stApp { background-color: #0f172a; }

h1 { color: #e2e8f0; text-align: center; }

.stNumberInput input {
    background-color: #1e293b !important;
    color: #f1f5f9 !important;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    height: 45px;
}

.result-card {
    padding: 20px;
    border-radius: 10px;
    margin-top: 25px;
}

.attack {
    background-color: #3f1d1d;
    border-left: 6px solid #ef4444;
    color: #f8fafc;
}

.benign {
    background-color: #1e3a2f;
    border-left: 6px solid #22c55e;
    color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("../models/final_model.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    label_encoder = joblib.load("../models/label_encoder.pkl")
    selected_features = joblib.load("../models/selected_features.pkl")
except:
    st.error("Model files not found. Train model first.")
    st.stop()

# ---------------- HEADER ----------------
st.title("CICIDS 2017 Multi-Class Intrusion Detection")
st.markdown("Detect BENIGN, DoS, DDoS, PortScan, Bot, BruteForce, Infiltration")

st.markdown("---")

# ---------------- INPUT MODE ----------------
input_mode = st.radio(
    "Select Input Mode",
    ["Manual Entry", "Upload CSV"]
)

input_df = None

# ==============================
# MANUAL ENTRY
# ==============================
if input_mode == "Manual Entry":

    col1, col2 = st.columns(2)
    input_data = []

    for i, feature in enumerate(selected_features):
        if i % 2 == 0:
            with col1:
                val = st.number_input(feature, value=0.0)
                input_data.append(val)
        else:
            with col2:
                val = st.number_input(feature, value=0.0)
                input_data.append(val)

    input_df = pd.DataFrame([input_data], columns=selected_features)

# ==============================
# CSV UPLOAD
# ==============================
else:
    uploaded = st.file_uploader(
        "Upload CSV with exactly ONE row",
        type=["csv"]
    )

    if uploaded:
        try:
            # Auto detect delimiter (comma or tab)
            df = pd.read_csv(uploaded, sep=None, engine="python")

            # Remove hidden spaces
            df.columns = df.columns.str.strip()

            if len(df) != 1:
                st.error("CSV must contain exactly one row.")

            elif not all(col in df.columns for col in selected_features):
                missing = [col for col in selected_features if col not in df.columns]
                st.error(f"Missing columns: {missing[:5]} ...")
            else:
                input_df = df[selected_features]
                st.success("Valid file uploaded.")

        except Exception as e:
            st.error(f"Error reading CSV: {e}")

st.markdown("---")

# ==============================
# PREDICTION
# ==============================
if st.button("Analyze Network Traffic", use_container_width=True):

    if input_df is None:
        st.error("Provide input first.")
    else:
        input_scaled = scaler.transform(input_df)
        probs = model.predict_proba(input_scaled)[0]

        # Get class names
        classes = label_encoder.classes_

        benign_index = list(classes).index("BENIGN")
        benign_prob = probs[benign_index]

        # Highest attack probability
        attack_probs = {
            cls: prob for cls, prob in zip(classes, probs)
            if cls != "BENIGN"
        }

        top_attack = max(attack_probs, key=attack_probs.get)
        top_attack_prob = attack_probs[top_attack]

        # -------- SECURITY-AWARE DECISION --------
        if top_attack_prob > 0.35:   # threshold (tuneable)
            final_label = top_attack
            confidence = top_attack_prob * 100
        else:
            final_label = "BENIGN"
            confidence = benign_prob * 100

        # -------- DISPLAY RESULT --------
        if final_label == "BENIGN":
            st.markdown(f"""
            <div class="result-card benign">
                <h3>Traffic Status: BENIGN</h3>
                <p><strong>Confidence:</strong> {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card attack">
                <h3>Attack Detected: {final_label}</h3>
                <p><strong>Confidence:</strong> {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

        # -------- PROBABILITY BREAKDOWN --------
        st.markdown("### Class Probability Breakdown")

        for cls, prob in sorted(zip(classes, probs), key=lambda x: -x[1]):
            st.write(f"{cls}: {prob*100:.2f}%")
            st.progress(float(prob))