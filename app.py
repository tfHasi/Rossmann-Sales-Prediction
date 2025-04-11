import streamlit as st
import pickle
import pandas as pd
import numpy as np
import datetime
import xgboost as xgb

# --- Step 1: Load Your Pre-trained Model ---
@st.cache_resource
def load_model():
    try:
        with open("./Model/xgb_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error("Error loading model: " + str(e))
        st.stop()

model = load_model()

# --- Step 2: Set Up the App Title and Description ---
st.title("Rossmann Sales Forecast App")
st.write("This app predicts sales for Rossmann stores for the 2015-2016 period.")
st.markdown("**Important Features Include:** Open, StoreType_b, Promo, CompetitionDistance, StateHoliday_0, DayOfWeek, Store, Assortment_c, Promo2, and StoreType_d.")

# --- Step 3: Create User Inputs ---
st.subheader("Input Store and Date Information")

# Basic numerical inputs
store = st.number_input("Store ID", min_value=1, max_value=1115, value=1)
day_of_week = st.number_input("Day of Week (1=Monday, 7=Sunday)", min_value=1, max_value=7, value=3)

# Date picker restricted to the 2015-2016 period
date_input = st.date_input("Date", value=datetime.date(2015, 1, 1), 
                           min_value=datetime.date(2015, 1, 1), max_value=datetime.date(2016, 12, 31))

open_input = st.selectbox("Is store open?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
promo = st.selectbox("Promo running?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
school_holiday = st.selectbox("School holiday?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
competition_distance = st.number_input("Competition Distance", min_value=0.0, value=500.0, step=10.0)

# Categorical inputs matching one-hot encoding steps
st.subheader("Categorical Variables")
state_holiday = st.selectbox("State Holiday", options=["0", "a", "b", "c"])
store_type = st.selectbox("Store Type", options=["a", "b", "c", "d"])
assortment = st.selectbox("Assortment", options=["a", "b", "c"])
promo_interval = st.selectbox("Promo Interval", 
                              options=["None", "Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec"])

# Input for Promo2 feature
promo2 = st.selectbox("Is Promo2 active?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# --- Step 4: Define a Function to Preprocess Inputs ---
def preprocess_input():
    # Date-based features
    month = date_input.month
    year = date_input.year
    
    # Build basic dataframe with initial features
    data = {
        "Store": [store],
        "DayOfWeek": [day_of_week],
        "Open": [open_input],
        "Promo": [promo],
        "SchoolHoliday": [school_holiday],
        "CompetitionDistance": [competition_distance],
        "Promo2": [promo2],
        "Month": [month],
        "Year": [year]
    }
    
    df = pd.DataFrame(data)
    
    # One-hot encode StateHoliday
    for col in ['0', 'a', 'b', 'c']:
        df[f"StateHoliday_{col}"] = 1 if state_holiday == col else 0

    # One-hot encode StoreType
    for col in ['a', 'b', 'c', 'd']:
        df[f"StoreType_{col}"] = 1 if store_type == col else 0

    # One-hot encode Assortment
    for col in ['a', 'b', 'c']:
        df[f"Assortment_{col}"] = 1 if assortment == col else 0

    # One-hot encode PromoInterval
    for interval in ['None', 'Jan,Apr,Jul,Oct', 'Feb,May,Aug,Nov', 'Mar,Jun,Sept,Dec']:
        df[f"PromoInterval_{interval}"] = 1 if promo_interval == interval else 0

    # Competition feature: For this simplified demo, set to 0 (adjust as needed)
    df["CompetitionOpen"] = 0
    
    # Promo2Active feature based on user input, using the current month mapping
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    if promo2 == 0 or promo_interval == "None":
         df["Promo2Active"] = 0
    else:
         month_abbr = month_map[month]
         df["Promo2Active"] = 1 if month_abbr in promo_interval else 0

    return df

# --- Step 5: Predict Sales When Triggered ---

# Inside your "Predict Sales" button section, after preprocessing:
if st.button("Predict Sales"):
    input_data = preprocess_input()
    st.write("**Input Data:**")
    st.dataframe(input_data)
    try:
        # Convert the input pandas DataFrame to a DMatrix
        input_data_dmatrix = xgb.DMatrix(input_data)
        prediction = model.predict(input_data_dmatrix)
        st.success(f"**Predicted Sales:** {prediction[0]:,.2f}")
    except Exception as e:
        st.error("Prediction error: " + str(e))

# --- Additional Information Display ---
st.markdown("---")
st.markdown("### Feature Details")
st.markdown("""
- **Open:** Whether the store is open (1 = open, 0 = closed).
- **StoreType_b & StoreType_d:** One-hot encoded indicators for store types.
- **Promo:** If a promotional campaign is active.
- **CompetitionDistance:** Distance to the nearest competitor store.
- **StateHoliday_0:** Indicator for a state holiday occurring.
- **DayOfWeek:** The day of the week (1=Monday, ..., 7=Sunday).
- **Store:** Store identifier.
- **Assortment_c:** One-hot encoded assortment feature.
- **Promo2:** Indicates if Promo2 is available.
- **Promo2Active:** Indicator based on the month and promo interval.
""")
st.markdown("**Note:** The forecast applies to the 2015-2016 date range.")
