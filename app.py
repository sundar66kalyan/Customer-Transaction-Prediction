import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Customer Transaction Prediction", layout="wide")

st.title("🏦 Customer Transaction Prediction")
st.markdown("### Predict customer transaction behavior")

st.info("📢 **Demo Mode Active** - Using intelligent predictions based on customer profiles")

# Sidebar for input
st.sidebar.header("📊 Customer Features")

annual_income = st.sidebar.number_input("Annual Income ($)", min_value=0, max_value=500000, value=50000, step=1000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
age = st.sidebar.slider("Age", 18, 100, 35)
previous_transactions = st.sidebar.number_input("Previous Transactions (6 months)", min_value=0, max_value=100, value=10)
avg_transaction_amount = st.sidebar.number_input("Average Transaction Amount ($)", min_value=0, max_value=10000, value=500)

# Prediction logic
def predict_transaction(income, credit, age, transactions, avg_amount):
    # Weighted score based on customer behavior patterns
    income_score = min(income / 100000, 1.0) * 0.25
    credit_score_norm = (credit - 300) / 550 * 0.25
    transaction_score = min(transactions / 50, 1.0) * 0.35
    amount_score = min(avg_amount / 2000, 1.0) * 0.15
    
    total_score = income_score + credit_score_norm + transaction_score + amount_score
    probability = min(max(total_score, 0), 1)
    
    # Age adjustment
    if 25 <= age <= 45:
        probability = min(probability * 1.1, 1.0)
    elif age > 60:
        probability = probability * 0.9
        
    return 1 if probability > 0.5 else 0, probability

# Prediction button
if st.button("🔮 Predict Transaction", type="primary", use_container_width=True):
    prediction, probability = predict_transaction(
        annual_income, credit_score, age, previous_transactions, avg_transaction_amount
    )
    
    st.markdown("---")
    st.subheader("📈 Prediction Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 1:
            st.success("### ✅ Will Transact")
            st.metric("Confidence", f"{probability:.1%}")
        else:
            st.error("### ❌ Will Not Transact")
            st.metric("Confidence", f"{(1-probability):.1%}")
    
    with col2:
        st.markdown("**Probability Score**")
        st.progress(probability)
        st.caption(f"{probability:.1%} chance of transaction")
    
    with col3:
        if probability > 0.7:
            st.markdown("**Recommendation:** 🎯 High Priority")
        elif probability > 0.4:
            st.markdown("**Recommendation:** 📌 Medium Priority")
        else:
            st.markdown("**Recommendation:** ⚠️ Low Priority")
    
    # Detailed insights
    st.markdown("---")
    st.subheader("📊 Key Insights")
    
    insights = []
    if annual_income > 75000:
        insights.append("✅ High income earner - good target for premium products")
    if credit_score > 700:
        insights.append("✅ Excellent credit score - low risk customer")
    if previous_transactions > 20:
        insights.append("✅ Active transaction history - engaged customer")
    if age < 30:
        insights.append("📱 Young customer - target with digital offers")
    elif age > 50:
        insights.append("💼 Mature customer - focus on wealth management")
    
    if insights:
        for insight in insights:
            st.write(insight)
    else:
        st.write("📌 Standard customer profile - standard engagement recommended")
    
    if probability > 0.7:
        st.balloons()

# Feature importance explanation
with st.expander("ℹ️ How predictions work"):
    st.markdown("""
    **Model Features & Weights:**
    - **Annual Income (25%)** - Higher income indicates better transaction capability
    - **Credit Score (25%)** - Good credit suggests financial responsibility
    - **Previous Transactions (35%)** - Past behavior best predictor of future action
    - **Average Transaction Amount (15%)** - Higher spend indicates engagement
    
    **Note:** This is a demo model. For production, use your trained LightGBM model.
    """)

st.markdown("---")
st.markdown("👨‍💻 Built with Streamlit | Customer Transaction Prediction Model")
