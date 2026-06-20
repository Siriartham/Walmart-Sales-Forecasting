import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the saved model pipeline
# Make sure this filename matches exactly what you saved in your notebook
model = joblib.load(r'C:\Users\DELL\AI_Walmart_Sales_Forecasting\dataset\walmart_model_rf.pkl')

st.set_page_config(page_title="Walmart Sales Predictor", page_icon="🛒")

st.title("Walmart Weekly Sales Forecasting")
st.write("This app uses a **Random Forest Regressor** to predict next week's sales based on economic factors and historical trends.")

# 2. Create input fields in the sidebar
st.sidebar.header("Input Store Details")

# Basic Store Info
store = st.sidebar.selectbox("Select Store", options=list(range(1, 46)))
holiday = st.sidebar.selectbox("Is it a Holiday Week?", options=[0, 1])

# Economic Factors
st.sidebar.subheader("Economic Indicators")
temp = st.sidebar.slider("Temperature", -20, 110, 60)
fuel = st.sidebar.slider("Fuel Price", 2.0, 5.0, 3.0)
cpi = st.sidebar.number_input("CPI", value=200.0)
unemp = st.sidebar.slider("Unemployment Rate", 3.0, 15.0, 8.0)
# Date Information
st.sidebar.subheader("Date Details")
month = st.sidebar.selectbox("Month", options=list(range(1, 13)))
year = st.sidebar.selectbox("Year", options=[2010, 2011, 2012])
day = st.sidebar.slider("Day of Month", 1, 31, 15)
week = st.sidebar.slider("Week Number", 1, 53, 26)

# --- IMPORTANT: TIME SERIES FEATURES ---
st.sidebar.subheader("Historical Trends")
last_week_sales = st.sidebar.number_input("Last Week's Sales ($)", min_value=0.0, value=1000000.0)
rolling_mean = st.sidebar.number_input("4-Week Rolling Average Sales ($)", min_value=0.0, value=1000000.0)

# 3. Create the input dataframe
# IMPORTANT: The order of these columns MUST be exactly the same as your X_train
input_dict = {
    'Store': [store],
    'Holiday_Flag': [holiday],
    'Temperature': [temp],
    'Fuel_Price': [fuel],
    'CPI': [cpi],
    'Unemployment': [unemp],
    'Day': [day],
    'Month': [month],
    'Year': [year],
    'Week_Number': [week],
    'last_week_sales': [last_week_sales],
    'rolling_mean_4': [rolling_mean]
}

input_data = pd.DataFrame(input_dict)

# 4. Prediction logic
if st.button("Predict Weekly Sales"):
    try:
        prediction = model.predict(input_data)
        
        # Display the result in a nice box
        st.balloons()
        st.markdown("---")
        st.subheader("Prediction Result:")
        st.success(f"### The Predicted Weekly Sales are: **${prediction[0]:,.2f}**")
        
        # Add a small disclaimer for the user
        st.info("Note: This prediction is based on a Random Forest model trained on historical Walmart data.")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Please ensure your model pipeline matches the input features exactly.")