import streamlit as st
import pickle, pandas as pd, datetime, xgboost as xgb

# Load the saved model and feature order (saved as a dict)
@st.cache_resource
def load_model_bundle():
    with open("./Model/xgb_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["features"]

model, feature_order = load_model_bundle()

# Cache and load the store data with minimal preprocessing
@st.cache_data
def load_store_data():
    store_df = pd.read_csv("./Train-Test-Data/store.csv")
    store_df['CompetitionDistance'] = store_df['CompetitionDistance'].fillna(store_df['CompetitionDistance'].max()).astype(float)
    store_df['CompetitionOpenSinceMonth'] = store_df['CompetitionOpenSinceMonth'].fillna(0).astype(float)
    store_df['CompetitionOpenSinceYear'] = store_df['CompetitionOpenSinceYear'].fillna(0).astype(float)
    store_df['Promo2SinceWeek'] = store_df['Promo2SinceWeek'].fillna(0).astype(float)
    store_df['Promo2SinceYear'] = store_df['Promo2SinceYear'].fillna(0).astype(float)
    store_df['PromoInterval'] = store_df['PromoInterval'].fillna('None')
    return store_df

# App title and description
st.title("Rossmann Sales Forecast")
st.write("Predict sales for Rossmann stores (2015-2016).")

# User inputs (only the basic ones)
store = st.number_input("Store ID", 1, 1115, value=1)
day_of_week = st.number_input("Day of Week", 1, 7, value=3)
date_input = st.date_input("Date", datetime.date(2015, 1, 1), 
                           min_value=datetime.date(2015, 1, 1), max_value=datetime.date(2016, 12, 31))
open_input = st.selectbox("Store open?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
promo = st.selectbox("Promo running?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
state_holiday = st.selectbox("State Holiday", ["0", "a", "b", "c"])
school_holiday = st.selectbox("School Holiday?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

def preprocess_input():
    # Basic user input dataframe
    user_df = pd.DataFrame({
        'Store': [store],
        'DayOfWeek': [day_of_week],
        'Date': [pd.to_datetime(date_input)],
        'Open': [open_input],
        'Promo': [promo],
        'StateHoliday': [state_holiday],
        'SchoolHoliday': [school_holiday]
    })
    
    # Merge with store information
    store_df = load_store_data()
    df = pd.merge(user_df, store_df, on='Store', how='left')
    
    # One-hot encode categorical features
    for col in ['0', 'a', 'b', 'c']:
        df[f"StateHoliday_{col}"] = (df['StateHoliday'] == col).astype(int)
    for col in ['a', 'b', 'c', 'd']:
        df[f"StoreType_{col}"] = (df['StoreType'] == col).astype(int)
    for col in ['a', 'b', 'c']:
        df[f"Assortment_{col}"] = (df['Assortment'] == col).astype(int)
    for interval in ['None', 'Jan,Apr,Jul,Oct', 'Feb,May,Aug,Nov', 'Mar,Jun,Sept,Dec']:
        df[f"PromoInterval_{interval}"] = (df['PromoInterval'] == interval).astype(int)
    
    # Date-based features
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    
    # Competition features
    df['CompetitionOpen'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + (df['Month'] - df['CompetitionOpenSinceMonth'])
    df['CompetitionOpen'] = df['CompetitionOpen'].apply(lambda x: 0 if x < 0 else x)
    
    # Promo2Active feature
    month_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    df['Promo2Active'] = df.apply(lambda row: 0 if row['Promo2'] == 0 or row['PromoInterval'] == 'None'
                                  else (1 if month_map[row['Month']] in row['PromoInterval'] else 0), axis=1)
    
    # Drop original categorical columns and the Date column
    cols_to_drop = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval',
                    'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                    'Promo2SinceWeek', 'Promo2SinceYear', 'Date']
    df = df.drop(columns=cols_to_drop)
    
    # Reorder features to match training
    df = df.reindex(columns=feature_order)
    return df

if st.button("Predict Sales"):
    input_df = preprocess_input()
    dmat = xgb.DMatrix(input_df)
    pred = model.predict(dmat)
    # Enforce business logic: if the store is closed, sales = 0; also ensure non-negativity.
    pred_sales = 0 if open_input == 0 else max(pred[0], 0)
    st.success(f"Predicted Sales: {pred_sales:,.2f}")