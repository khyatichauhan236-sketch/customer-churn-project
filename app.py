import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .main-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        margin-bottom: 1.5rem;
    }
    .result-churn {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
    }
    .result-safe {
        background: linear-gradient(135deg, #55efc4, #00b894);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
    }
    .metric-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    h1 { color: #2d3436 !important; }
    .sidebar-title { color: #667eea; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Header ────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.markdown("<h1 style='font-size:3.5rem;margin:0;'>✈️</h1>", unsafe_allow_html=True)
with col_title:
    st.markdown("<h1 style='margin:0;padding-top:0.3rem;'>Customer Churn Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#636e72;font-size:1rem;margin:0;'>Powered by Random Forest | Travel Industry</p>", unsafe_allow_html=True)

st.markdown("---")

# ── About Section ─────────────────────────────────────────────────
with st.expander("ℹ️ About this App", expanded=False):
    st.markdown("""
    This app predicts whether a **travel industry customer** is likely to **churn** (leave) or **stay**.
    
    **Model:** Random Forest Classifier (100 trees)  
    **Dataset:** 954 customer records with 6 behavioral & demographic features  
    **Developed by:** Khyati Chauhan | KU2507U0406 | B.Tech Hons. CSE GenAI (A) IBM
    
    Fill in the customer details in the sidebar and click **Predict Churn** to get the prediction.
    """)

# ── Sidebar Inputs ────────────────────────────────────────────────
st.sidebar.markdown("## 🎛️ Customer Details")
st.sidebar.markdown("Fill in the customer information below:")
st.sidebar.markdown("---")

age = st.sidebar.slider(
    "👤 Customer Age",
    min_value=18, max_value=80, value=34, step=1,
    help="Age of the customer in years"
)

frequent_flyer = st.sidebar.selectbox(
    "✈️ Frequent Flyer?",
    options=["No", "Yes"],
    help="Is the customer a frequent flyer?"
)

annual_income = st.sidebar.selectbox(
    "💰 Annual Income Class",
    options=["Low Income", "Middle Income", "High Income"],
    help="Customer's annual income category"
)

services_opted = st.sidebar.slider(
    "🛎️ Services Opted",
    min_value=1, max_value=9, value=3, step=1,
    help="Number of additional services the customer has opted for"
)

account_synced = st.sidebar.selectbox(
    "📱 Account Synced to Social Media?",
    options=["No", "Yes"],
    help="Has the customer synced their account to social media?"
)

booked_hotel = st.sidebar.selectbox(
    "🏨 Booked Hotel or Not?",
    options=["No", "Yes"],
    help="Has the customer booked a hotel through our platform?"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.info("🌲 Random Forest\n\n📊 Accuracy: ~82%\n\n📈 AUC: ~0.85")

# ── Encode Inputs ─────────────────────────────────────────────────
income_map = {"Low Income": 0, "Middle Income": 1, "High Income": 2}
binary_map = {"No": 0, "Yes": 1}

input_data = pd.DataFrame({
    "Age": [int(age)],
    "FrequentFlyer": [binary_map[frequent_flyer]],
    "AnnualIncomeClass": [income_map[annual_income]],
    "ServicesOpted": [int(services_opted)],
    "AccountSyncedToSocialMedia": [binary_map[account_synced]],
    "BookedHotelOrNot": [binary_map[booked_hotel]]
})

# ── Main Content ──────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📋 Customer Profile Summary")
    profile_data = {
        "Feature": ["Age", "Frequent Flyer", "Annual Income", "Services Opted",
                    "Social Media Synced", "Hotel Booked"],
        "Value": [f"{age} years", frequent_flyer, annual_income,
                  f"{services_opted} service(s)", account_synced, booked_hotel]
    }
    profile_df = pd.DataFrame(profile_data)
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### 🎯 Prediction")
    predict_btn = st.button("🔮 Predict Churn", use_container_width=True)

    if predict_btn:
        if not model_loaded:
            st.error("⚠️ model.pkl not found! Please run the notebook first to generate the model.")
        else:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            churn_prob = probability[1] * 100
            stay_prob = probability[0] * 100

            if prediction == 1:
                st.markdown(f"""
                <div class="result-churn">
                    🚨 HIGH CHURN RISK<br>
                    <span style='font-size:1rem;font-weight:normal;'>
                    This customer is likely to churn.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    ✅ LOW CHURN RISK<br>
                    <span style='font-size:1rem;font-weight:normal;'>
                    This customer is likely to stay.
                    </span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability bars
            st.markdown("**📊 Prediction Confidence**")
            st.progress(int(stay_prob), text=f"Stay Probability: {stay_prob:.1f}%")
            st.progress(int(churn_prob), text=f"Churn Probability: {churn_prob:.1f}%")

            # Recommendation
            st.markdown("---")
            st.markdown("**💡 Recommendation**")
            if prediction == 1:
                st.warning("""
                **Action Required:** Consider the following retention strategies:
                - 🎁 Offer a personalized discount or loyalty reward
                - 📞 Schedule a customer success call
                - 🎯 Provide exclusive frequent flyer upgrade
                - 🏨 Offer complimentary hotel booking for next trip
                """)
            else:
                st.success("""
                **Customer is Retained!** Keep up the good work:
                - ⭐ Enroll in premium loyalty program
                - 📧 Send quarterly engagement newsletters
                - 🎉 Reward with milestone bonuses
                """)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown("**👩‍🎓 Student:** Khyati Chauhan")
with col_f2:
    st.markdown("**🆔 KU ID:** KU2507U0406")
with col_f3:
    st.markdown("**📧** ku2507u0406@karnavatiuniversity.com")
