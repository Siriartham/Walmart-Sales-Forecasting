import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="🛒",
    layout="wide"
)

# -----------------------------
# LOAD MODEL (FIXED PATH)
# -----------------------------
# IMPORTANT: We use a relative path because on the cloud, 
# the model file is in the same folder as app.py
try:
    model = joblib.load('walmart_model_rf.pkl') 
except FileNotFoundError:
    st.error("Model file not found! Please ensure 'walmart_model_rf.pkl' is in the root folder of your GitHub repo.")
    st.stop()

# -----------------------------
# HEADER
# -----------------------------
st.title("🛒 Walmart Weekly Sales Forecasting")
st.markdown("""
Predict Walmart weekly sales using a trained **Random Forest Regressor** 
based on store information, economic indicators, and historical sales trends.
""")
st.success("✅ Model Status: Active | Random Forest Regressor | R² Score: 0.98")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("📋 Store Information")

store = st.sidebar.selectbox("Store Number", options=list(range(1, 46)))
holiday = st.sidebar.selectbox("Holiday Week", options=[0, 1], help="0 = Non-Holiday, 1 = Holiday")

st.sidebar.header("📈 Economic Indicators")
temp = st.sidebar.slider("Temperature (°F)", -20, 110, 60)
fuel = st.sidebar.slider("Fuel Price ($)", 2.0, 5.0, 3.0)
cpi = st.sidebar.number_input("Consumer Price Index (CPI)", value=200.0)
unemp = st.sidebar.slider("Unemployment Rate (%)", 3.0, 15.0, 8.0)

st.sidebar.header("📅 Date Information")
month = st.sidebar.selectbox("Month", options=list(range(1, 13)))
year = st.sidebar.selectbox("Year", options=[2010, 2011, 2012])
day = st.sidebar.slider("Day of Month", 1, 31, 15)
week = st.sidebar.slider("Week Number", 1, 53, 26)

st.sidebar.header("📊 Historical Sales")
last_week_sales = st.sidebar.number_input("Last Week Sales ($)", min_value=0.0, value=1000000.0, step=10000.0)
rolling_mean = st.sidebar.number_input("4-Week Rolling Average Sales ($)", min_value=0.0, value=1000000.0, step=10000.0)

# -----------------------------
# INPUT DATAFRAME
# -----------------------------
# The columns MUST be in the exact order used during model.fit()
input_data = pd.DataFrame({
    "Store": [store],
    "Holiday_Flag": [holiday],
    "Temperature": [temp],
    "Fuel_Price": [fuel],
    "CPI": [cpi],
    "Unemployment": [unemp],
    "Day": [day],
    "Month": [month],
    "Year": [year],
    "Week_Number": [week],
    "last_week_sales": [last_week_sales],
    "rolling_mean_4": [rolling_mean]
})

# -----------------------------
# PREDICTION DISPLAY
# -----------------------------
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Summary")
    st.dataframe(input_data)

with col2:
    st.subheader("Prediction")
    if st.button("🚀 Predict Weekly Sales"):
        prediction = model.predict(input_data)
        
        # Making the output BIG and a centered
        st.markdown(f"""
            <div style="background-color:#d4edda; padding:20px; border-radius:10px; text-align:center; border: 2px solid #c3e6cb">
                <h3 style="color:#155724; margin:0;">Predicted Weekly Sales</h3>
                <h1 style="color:#155724; margin:0;">${prediction[0]:,.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.info("Adjust parameters in the sidebar and click the button to see the result.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Data Science Portfolio Project | Built with Random Forest & Streamlit")
