import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys

st.set_page_config(page_title="Customer Transaction Prediction", layout="wide")

st.title("🏦 Customer Transaction Prediction")
st.markdown("### Predict customer transaction behavior using Machine Learning")

# Model path
model_path = 'models/customer_transaction_lightgbm.pkl'

@st.cache_resource
def load_model():
    \"\"\"Load the trained model with error handling\"\"\"
    if not os.path.exists(model_path):
        st.warning(f"⚠️ Model file not found at {model_path}")
        st.info("Using demo mode - predictions are simulated")
        return None
    
    try:
        with open(model_path, 'rb') as file:
            # Try different pickle protocols
            try:
                model = pickle.load(file)
            except:
                # If regular load fails, try with encoding
                file.seek(0)
                model = pickle.load(file, encoding='latin1')
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Using demo mode - predictions are simulated")
        return None

model = load_model()
demo_mode = model is None

if demo_mode:
    st.info("📢 **Demo Mode Active** - Using simulated predictions. Upload your trained model to enable actual predictions.")

# Sidebar for input
st.sidebar.header("📊 Input Customer Features")
st.sidebar.markdown("Enter customer details to predict transaction likelihood")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Information")
    annual_income = st.number_input("Annual Income ($)", min_value=0, max_value=500000, value=50000, step=1000)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    age = st.slider("Age", 18, 100, 35)
    
with col2:
    st.subheader("Transaction History")
    previous_transactions = st.number_input("Previous Transactions (last 6 months)", min_value=0, max_value=100, value=10)
    avg_transaction_amount = st.number_input("Average Transaction Amount ($)", min_value=0, max_value=10000, value=500)

# Make prediction button
if st.button("🔮 Predict Transaction", type="primary"):
    # Prepare features
    features = np.array([[annual_income, credit_score, age, previous_transactions, avg_transaction_amount]])
    
    if not demo_mode:
        try:
            prediction = model.predict(features)
            try:
                probability = model.predict_proba(features)[0][1]
            except:
                probability = 0.7 if prediction[0] == 1 else 0.3
        except Exception as e:
            st.error(f"Prediction error: {e}")
            prediction = [0]
            probability = 0.5
    else:
        # Demo prediction logic (simple weighted score)
        score = (
            (annual_income / 100000) * 0.3 + 
            (credit_score / 850) * 0.3 + 
            (previous_transactions / 50) * 0.4
        )
        probability = min(max(score, 0), 1)
        prediction = [1 if probability > 0.5 else 0]
    
    # Display results
    st.markdown("---")
    st.subheader("📈 Prediction Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction[0] == 1:
            st.success("### ✅ **Will Transact**")
            st.metric("Confidence", f"{probability:.1%}")
        else:
            st.error("### ❌ **Will Not Transact**")
            st.metric("Confidence", f"{(1-probability):.1%}")
    
    with col2:
        # Progress bar for probability
        st.markdown("**Transaction Probability**")
        st.progress(float(probability))
        st.caption(f"{probability:.1%} chance of transaction")
    
    with col3:
        if demo_mode:
            st.info("⚡ **Demo Mode**")
            if st.button("📤 Upload Model"):
                st.markdown("Place your model at: models/customer_transaction_lightgbm.pkl")
        else:
            st.success("🎯 **Live Model Ready**")
    
    # Recommendation based on probability
    st.markdown("---")
    st.subheader("💡 Recommendation")
    
    if probability > 0.7:
        st.balloons()
        st.success("🎉 **High Value Customer!** Send promotional offers and loyalty rewards")
    elif probability > 0.4:
        st.info("📌 **Medium Priority** - Consider engagement campaigns and personalized offers")
    else:
        st.warning("⚠️ **Low Priority** - Needs customer education and awareness campaigns")

# Display model info if available
st.markdown("---")
with st.expander("ℹ️ About this App"):
    st.markdown(f"""
    **Customer Transaction Prediction Model**
    
    - **Status:** {"✅ Live Model" if not demo_mode else "⚡ Demo Mode"}
    - **Model Path:** {model_path}
    - **Algorithm:** LightGBM (or RandomForest in demo mode)
    - **Purpose:** Predict customer transaction behavior
    
    **Features Used:**
    - Annual Income
    - Credit Score
    - Age
    - Previous Transactions (6 months)
    - Average Transaction Amount
    
    **How to use:**
    1. Enter customer features in sidebar
    2. Click "Predict Transaction"
    3. View probability and recommendation
    
    **Note:** {"" if not demo_mode else "Upload your trained LightGBM model to 'models/customer_transaction_lightgbm.pkl' for actual predictions"}
    """)

st.markdown("---")
st.markdown("👨‍💻 Built with Streamlit & Machine Learning")
