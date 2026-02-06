"""
Customer Retention Analytics Dashboard
Interactive Streamlit app for churn prediction and insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Customer Retention Analytics",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-high {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-low {
        background: linear-gradient(90deg, #00c853 0%, #69f0ae 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E3A5F;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load trained model and artifacts."""
    model_path = "churn_model.pkl"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def predict_churn(artifacts, tenure, monthly_charges, total_charges, contract, payment_method):
    """Make churn prediction for a customer."""
    model = artifacts["model"]
    label_encoders = artifacts["label_encoders"]
    scaler = artifacts["scaler"]
    
    # Encode categorical variables
    contract_encoded = label_encoders["Contract"].transform([contract])[0]
    payment_encoded = label_encoders["PaymentMethod"].transform([payment_method])[0]
    
    # Create feature array
    features = pd.DataFrame({
        "Tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Contract": [contract_encoded],
        "PaymentMethod": [payment_encoded]
    })
    
    # Scale numerical features
    numerical_cols = ["Tenure", "MonthlyCharges", "TotalCharges"]
    features[numerical_cols] = scaler.transform(features[numerical_cols])
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return prediction, probability


def main():
    # Header
    st.markdown('<p class="main-header">📊 Customer Retention Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    artifacts = load_model()
    
    if artifacts is None:
        st.error("⚠️ Model not found! Please run the training pipeline first:")
        st.code("python generate_data.py\npython train_model.py", language="bash")
        return
    
    # Sidebar - Customer Input
    st.sidebar.header("🧑‍💼 Customer Details")
    st.sidebar.markdown("Enter customer information to predict churn risk.")
    
    tenure = st.sidebar.slider(
        "Months with Company",
        min_value=1,
        max_value=72,
        value=12,
        help="How long the customer has been with the company"
    )
    
    monthly_charges = st.sidebar.slider(
        "Monthly Charges (₹)",
        min_value=18.0,
        max_value=120.0,
        value=65.0,
        step=0.50,
        help="Customer's monthly bill amount"
    )
    
    total_charges = st.sidebar.number_input(
        "Total Charges (₹)",
        min_value=0.0,
        max_value=10000.0,
        value=float(tenure * monthly_charges),
        step=10.0,
        help="Total amount charged to date"
    )
    
    contract = st.sidebar.selectbox(
        "Contract Type",
        options=["Month-to-month", "One year", "Two year"],
        help="Customer's contract duration"
    )
    
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        options=["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        help="Customer's preferred payment method"
    )
    
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔮 Predict Churn Risk", use_container_width=True)
    
    # Main Panel
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Customer Profile Summary")
        
        profile_col1, profile_col2, profile_col3 = st.columns(3)
        with profile_col1:
            st.metric("Tenure", f"{tenure} months")
        with profile_col2:
            st.metric("Monthly Charges", f"${monthly_charges:.2f}")
        with profile_col3:
            st.metric("Total Charges", f"${total_charges:.2f}")
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>Contract:</strong> {contract} &nbsp;|&nbsp;
            <strong>Payment:</strong> {payment_method}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🎯 Prediction Result")
        
        if predict_button:
            prediction, probability = predict_churn(
                artifacts, tenure, monthly_charges, total_charges, contract, payment_method
            )
            
            churn_pct = probability * 100
            
            if probability >= 0.5:
                st.markdown(f'<div class="risk-high">⚠️ HIGH RISK<br>{churn_pct:.1f}% Churn Probability</div>', 
                           unsafe_allow_html=True)
                st.warning("**Recommendation:** Proactive retention intervention recommended.")
            else:
                st.markdown(f'<div class="risk-low">✅ LOW RISK<br>{churn_pct:.1f}% Churn Probability</div>', 
                           unsafe_allow_html=True)
                st.success("**Status:** Customer appears satisfied.")
            
            # Progress bar
            st.markdown("### Risk Level")
            st.progress(probability)
        else:
            st.info("👈 Enter customer details and click **Predict Churn Risk**")
    
    # Feature Importance Section
    st.markdown("---")
    st.subheader("📈 Feature Importance Analysis")
    st.caption("Which factors contribute most to churn predictions?")
    
    model = artifacts["model"]
    feature_names = artifacts["feature_names"]
    
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)
    
    # Horizontal bar chart
    st.bar_chart(
        importance_df.set_index("Feature"),
        horizontal=True,
        height=300
    )
    
    # Insights
    top_feature = importance_df.iloc[-1]["Feature"]
    st.info(f"💡 **Key Insight:** '{top_feature}' is the strongest predictor of customer churn in this model.")
    
    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit | Customer Churn Prediction System")


if __name__ == "__main__":
    main()

