import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ---------------- LOAD MODELS ----------------
try:
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    iso_model = joblib.load("iso_model.pkl")
    scaler = joblib.load("scaler.pkl")
    type_encoder = joblib.load("type_encoder.pkl")
except Exception as e:
    st.error(f"Error loading model files:\n{e}")
    st.stop()

# ---------------- HEADER ----------------
st.title("💳 AI Fraud Detection System")
st.write("Real-time Intelligent FinTech Fraud Detection")

# ---------------- LAYOUT ----------------
left, right = st.columns([1, 1.2])

# ---------------- INPUTS ----------------
with left:

    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=5000.0
    )

    oldbalanceOrg = st.number_input(
        "Sender Old Balance",
        min_value=0.0,
        value=10000.0
    )

    newbalanceOrig = st.number_input(
        "Sender New Balance",
        min_value=0.0,
        value=5000.0
    )

    oldbalanceDest = st.number_input(
        "Receiver Old Balance",
        min_value=0.0,
        value=2000.0
    )

    newbalanceDest = st.number_input(
        "Receiver New Balance",
        min_value=0.0,
        value=7000.0
    )

    predict = st.button("Analyze Transaction")

# ---------------- PREDICTION ----------------
with right:

    if predict:

        try:
            type_encoded = type_encoder.transform([transaction_type])[0]
        except:
            st.error("Transaction type encoding failed.")
            st.stop()

        # Feature Engineering
        sender_error = oldbalanceOrg - newbalanceOrig - amount
        receiver_error = newbalanceDest - oldbalanceDest - amount
        is_large_transaction = int(amount > 200000)

        # IMPORTANT:
        # Replace '1' with actual Step value if your model was trained on Step.
        input_data = np.array([[
            1,
            type_encoded,
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest,
            sender_error,
            receiver_error,
            is_large_transaction
        ]])

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predictions
        rf_prob = rf_model.predict_proba(input_scaled)[0][1]
        xgb_prob = xgb_model.predict_proba(input_scaled)[0][1]

        # Isolation Forest
        iso_score = -iso_model.decision_function(input_scaled)[0]
        iso_score = np.clip(iso_score, 0, 1)

        # ML Risk
        ml_risk = (
            0.45 * rf_prob +
            0.45 * xgb_prob +
            0.10 * iso_score
        ) * 100

        # Rule Based Risk
        rule_boost = 0

        if amount > 200000:
            rule_boost += 20

        if transaction_type in ["TRANSFER", "CASH_OUT"]:
            rule_boost += 20

        if abs(sender_error) > 1000:
            rule_boost += 25

        if abs(receiver_error) > 1000:
            rule_boost += 15

        if oldbalanceOrg > 0 and newbalanceOrig < oldbalanceOrg * 0.1:
            rule_boost += 20

        risk_score = min(100, ml_risk + rule_boost)

        # Label
        if risk_score >= 70:
            label = "🔴 HIGH RISK"
            color = "red"

        elif risk_score >= 40:
            label = "🟡 MEDIUM RISK"
            color = "orange"

        else:
            label = "🟢 LOW RISK"
            color = "green"

        # Explainability
        reasons = []

        if amount > 200000:
            reasons.append("Large transaction amount")

        if abs(sender_error) > 1000:
            reasons.append("Sender balance inconsistency")

        if abs(receiver_error) > 1000:
            reasons.append("Receiver balance inconsistency")

        if transaction_type in ["TRANSFER", "CASH_OUT"]:
            reasons.append("High-risk transaction type")

        if len(reasons) == 0:
            reasons.append("Transaction appears normal")

        # ---------------- RESULTS ----------------
        col1, col2, col3 = st.columns(3)

        col1.metric("RF Probability", f"{rf_prob:.2f}")
        col2.metric("XGB Probability", f"{xgb_prob:.2f}")
        col3.metric("Risk Score", f"{risk_score:.1f}%")

        st.markdown(
            f"<h2 style='color:{color};text-align:center'>{label}</h2>",
            unsafe_allow_html=True
        )

        st.subheader("Why was this flagged?")

        for r in reasons:
            st.write("•", r)

    else:
        st.info("Enter transaction details and click 'Analyze Transaction'.")
