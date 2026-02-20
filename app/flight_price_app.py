# Flight Price Prediction App

import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Flight Price Predictor", layout="wide")

st.title("✈️ Smart Flight Price Prediction System")
st.write("Enter flight details below to estimate ticket price.")

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_excel("flight_price.xlsx")

df = load_data()

# Feature Engineering 

df["Journey_day"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True).dt.day
df["Journey_month"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True).dt.month
df["Dep_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
df["Arrival_hour"] = pd.to_datetime(df["Arrival_Time"]).dt.hour

# SAFE duration conversion
def convert_duration(x):
    x = str(x)
    hours = 0
    mins = 0
    if "h" in x:
        hours = int(x.split("h")[0])
    if "m" in x:
        mins = int(x.split("h")[-1].replace("m", "").strip())
    return hours * 60 + mins

df["Duration_mins"] = df["Duration"].apply(convert_duration)

df["Total_Stops"] = df["Total_Stops"].map({
    "non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4
})

df.drop(["Date_of_Journey", "Dep_Time", "Arrival_Time", "Duration", "Route"], axis=1, inplace=True)

df.dropna(inplace=True)

df = pd.get_dummies(df, drop_first=True)

# Model Training

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# USER INPUT SECTION

st.subheader("📝 Enter Flight Details")

col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox("Airline", 
        ["IndiGo", "Air India", "Jet Airways", "SpiceJet", 
         "Vistara", "GoAir", "Multiple carriers"]
    )

    source = st.selectbox("Source", 
        ["Delhi", "Kolkata", "Mumbai", "Chennai"]
    )

    destination = st.selectbox("Destination", 
        ["Cochin", "Delhi", "New Delhi", "Hyderabad", "Kolkata"]
    )

with col2:
    journey_date = st.date_input("Journey Date")
    dep_time = st.time_input("Departure Time")
    arrival_time = st.time_input("Arrival Time")

    total_stops = st.selectbox("Total Stops", 
        ["non-stop", "1 stop", "2 stops", "3 stops"]
    )

# PREDICTION

if st.button("Predict Flight Price 💰"):

    journey_day = journey_date.day
    journey_month = journey_date.month
    dep_hour = dep_time.hour
    arrival_hour = arrival_time.hour

    # Midnight safe duration calculation
    dep_datetime = datetime.datetime.combine(datetime.date.today(), dep_time)
    arr_datetime = datetime.datetime.combine(datetime.date.today(), arrival_time)

    if arr_datetime < dep_datetime:
        arr_datetime += datetime.timedelta(days=1)

    duration_mins = int((arr_datetime - dep_datetime).total_seconds() / 60)

    stop_mapping = {
        "non-stop": 0,
        "1 stop": 1,
        "2 stops": 2,
        "3 stops": 3
    }

    stops = stop_mapping[total_stops]

    # Create base input dictionary
    input_dict = {
        "Journey_day": journey_day,
        "Journey_month": journey_month,
        "Dep_hour": dep_hour,
        "Arrival_hour": arrival_hour,
        "Total_Stops": stops,
        "Duration_mins": duration_mins
    }

    # Add missing dummy columns
    for col in X.columns:
        if col not in input_dict:
            input_dict[col] = 0

    input_df = pd.DataFrame([input_dict])

    # Set correct airline/source/destination column to 1
    for col in input_df.columns:
        if airline in col:
            input_df[col] = 1
        if source in col:
            input_df[col] = 1
        if destination in col:
            input_df[col] = 1

    input_df = input_df[X.columns]

    # Predict
    prediction = model.predict(input_df)[0]

    st.success(f"💵 Estimated Ticket Price: ₹ {int(prediction):,}")

    st.info("Prediction generated using Random Forest Regression model.")
