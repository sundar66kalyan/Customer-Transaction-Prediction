import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Customer Transaction Prediction", layout="wide")

st.title("🏦 Customer Transaction Prediction")
st.markdown("### Predict customer transaction behavior using Machine Learning")

# Load model
@st.cache_resource
def load_model():
    try:
        with open('models/customer_transaction_lightgbm.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'models/customer_transaction_lightgbm.pkl' exists")
        return None

model = load_model()

# Sidebar for input
st.sidebar.header("📊 Input Customer Features")
st.sidebar.markdown("Enter customer details to predict transaction likelihood")

# Example features (adjust based on your actual model features)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Information")
    
    # Example numeric features
    annual_income = st.number_input("Annual Income ($)", min_value=0, max_value=500000, value=50000)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    age = st.slider("Age", 18, 100, 35)
    
with col2:
    st.subheader("Transaction History")
    
    # Example categorical features
    previous_transactions = st.number_input("Previous Transactions (last 6 months)", min_value=0, max_value=100, value=10)
    avg_transaction_amount = st.number_input("Average Transaction Amount ($)", min_value=0, max_value=10000, value=500)
    
# Make prediction button
if st.button("🔮 Predict Transaction", type="primary"):
    if model:
        # Prepare features (adjust this array based on your model's features)
        features = np.array([[annual_income, credit_score, age, previous_transactions, avg_transaction_amount]])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Get probability if model supports it
        try:
            probability = model.predict_proba(features)[0][1]
        except:
            probability = 0.5
        
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
            # Show gauge chart
            fig, ax = plt.subplots(figsize=(4, 2))
            colors = ['#ff6b6b' if probability < 0.5 else '#51cf66']
            ax.barh(['Probability'], [probability], color=colors[0])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Transaction Probability')
            st.pyplot(fig)
            plt.close()
        
        with col3:
            st.info(f"**Recommendation:** {'High potential customer' if prediction[0] == 1 else 'Low priority'}")
        
        # Show model performance metrics
        st.markdown("---")
        st.subheader("📊 Model Performance")
        
        # Display saved images if they exist
        try:
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                if os.path.exists('images/roc_curve.png'):
                    roc_img = Image.open('images/roc_curve.png')
                    st.image(roc_img, caption="ROC Curve", use_column_width=True)
            
            with img_col2:
                if os.path.exists('images/feature_importance.png'):
                    feat_img = Image.open('images/feature_importance.png')
                    st.image(feat_img, caption="Feature Importance", use_column_width=True)
        except:
            st.info("Performance visualizations will appear here")
    else:
        st.error("Model not loaded. Please check the model file path.")

# About section
with st.expander("ℹ️ About this App"):
    st.markdown("""
    **Customer Transaction Prediction Model**
    
    - **Algorithm:** LightGBM
    - **Purpose:** Predict customer transaction behavior
    - **Input Features:** Customer demographics and transaction history
    - **Output:** Binary classification (Will Transact / Will Not Transact)
    
    **Model Files:**
    - `customer_transaction_lightgbm.pkl` - Trained model
    - `Customer_Transaction_Report.pdf` - Detailed analysis report
    """)

st.markdown("---")
st.markdown("👨‍💻 Built with Streamlit & LightGBM")