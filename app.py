import streamlit as st
import pickle
import pandas as pd
import datetime
import xgboost as xgb

# Set page configuration for a cleaner look
st.set_page_config(
    page_title="Rossmann Sales Forecast",
    layout="centered"
)

# Load model and store data with caching
@st.cache_resource
def load_model():
    with open("./Model/xgb_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["features"]

@st.cache_data
def load_store_data():
    df = pd.read_csv("./Train-Test-Data/store.csv")
    # Fill nulls efficiently
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].max())
    numeric_cols = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 
                    'Promo2SinceWeek', 'Promo2SinceYear']
    df[numeric_cols] = df[numeric_cols].fillna(0).astype(float)
    df['PromoInterval'] = df['PromoInterval'].fillna('None')
    return df

# Load data at app startup
model, feature_order = load_model()
store_df = load_store_data()

# App header with minimal styling
st.title("Rossmann Sales Forecast")
st.markdown("##### Predict sales for Rossmann stores (2015-2016)")
st.divider()

# Create two columns for inputs
col1, col2 = st.columns(2)

# Left column inputs
with col1:
    store = st.number_input("Store ID", 1, 1115, value=1)
    date_input = st.date_input("Date", datetime.date(2015, 1, 1), 
                              min_value=datetime.date(2015, 1, 1), 
                              max_value=datetime.date(2016, 12, 31))
    # Calculate day of week automatically from date
    day_of_week = pd.to_datetime(date_input).dayofweek + 1  # +1 because pandas uses 0-6 and we want 1-7
    open_input = st.selectbox("Is the Store open?", [1, 0], 
                             format_func=lambda x: "Yes" if x == 1 else "No")

# Right column inputs
with col2:
    promo = st.selectbox("Is there any Promotion running?", [0, 1], 
                        format_func=lambda x: "Yes" if x == 1 else "No")
    state_holiday = st.selectbox("Type ofState Holiday", ["0", "a", "b", "c"], 
                               format_func=lambda x: "None" if x == "0" else f"Holiday {x.upper()}")
    school_holiday = st.selectbox("Is it a School Holiday?", [0, 1], 
                                format_func=lambda x: "Yes" if x == 1 else "No")
    # Empty space to align with left column
    st.write("")

# Add some space before the button
st.write("")
predict_btn = st.button("Predict Sales", type="primary", use_container_width=True)

# Prediction logic
if predict_btn:
    # Create a spinner for processing feedback
    with st.spinner("Calculating forecast..."):
        # Initialize pred_sales first to avoid NameError
        pred_sales = 0
        
        # Quick exit for closed stores
        if open_input == 0:
            pred_sales = 0
        else:
            # Process input data
            input_data = {
                'Store': store, 'DayOfWeek': day_of_week, 'Open': open_input,
                'Promo': promo, 'StateHoliday': state_holiday, 'SchoolHoliday': school_holiday
            }
            user_df = pd.DataFrame([input_data])
            date_dt = pd.to_datetime(date_input)
            user_df['Month'] = date_dt.month
            user_df['Year'] = date_dt.year
            
            # Merge with store data
            df = pd.merge(user_df, store_df, on='Store', how='left')
            
            # One-hot encode categoricals
            for col, values in {
                'StateHoliday': ['0', 'a', 'b', 'c'],
                'StoreType': ['a', 'b', 'c', 'd'],
                'Assortment': ['a', 'b', 'c'],
                'PromoInterval': ['None', 'Jan,Apr,Jul,Oct', 'Feb,May,Aug,Nov', 'Mar,Jun,Sept,Dec']
            }.items():
                for val in values:
                    df[f"{col}_{val}"] = (df[col] == val).astype(int)
            
            # Competition features
            df['CompetitionOpen'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + \
                                    (df['Month'] - df['CompetitionOpenSinceMonth'])
            df['CompetitionOpen'] = df['CompetitionOpen'].clip(lower=0)
            
            # Promo2Active feature
            month_dict = {i: m for i, m in enumerate(['Jan','Feb','Mar','Apr','May','Jun',
                                                    'Jul','Aug','Sep','Oct','Nov','Dec'], 1)}
            curr_month = month_dict[df['Month'].iloc[0]]
            df['Promo2Active'] = ((df['Promo2'] == 1) & 
                                 (df['PromoInterval'] != 'None') & 
                                 df['PromoInterval'].str.contains(curr_month)).astype(int)
            
            # Drop unused columns and reorder
            to_drop = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval',
                      'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                      'Promo2SinceWeek', 'Promo2SinceYear']
            df = df.drop(columns=to_drop).reindex(columns=feature_order)
            
            # Predict
            pred_sales = max(model.predict(xgb.DMatrix(df))[0], 0)
    
    # Display result in a more prominent way
    st.divider()
    st.metric(label=f"Predicted Sales for {date_input}", value=f"€{pred_sales:,.2f}")
    
    # Display store info in two columns
    st.subheader("Store Information")
    store_info_cols = st.columns(2)
    
    with store_info_cols[0]:
        st.write(f"**Store Type:** {store_df.loc[store_df['Store'] == store, 'StoreType'].values[0]}")
        st.write(f"**Assortment:** {store_df.loc[store_df['Store'] == store, 'Assortment'].values[0]}")
    
    with store_info_cols[1]:
        st.write(f"**Competition Distance:** {store_df.loc[store_df['Store'] == store, 'CompetitionDistance'].values[0]:.0f}m")
        promo2 = store_df.loc[store_df['Store'] == store, 'Promo2'].values[0]
        st.write(f"**Promo2 Participation:** {'Yes' if promo2 == 1 else 'No'}")