import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

st.title("AI Hyperlocal Air Quality Predictor")

st.write("This system predicts Air Quality Index using environmental data")

data = pd.read_csv("dataset.csv")

X = data[['temperature','humidity','wind_speed','traffic']]
y = data['AQI']

model = RandomForestRegressor()
model.fit(X,y)

st.sidebar.header("Enter Environmental Data")

temp = st.sidebar.slider("Temperature",0,50)
humidity = st.sidebar.slider("Humidity",0,100)
wind = st.sidebar.slider("Wind Speed",0,20)
traffic = st.sidebar.slider("Traffic Density",0,100)

if st.button("Predict AQI"):

    prediction = model.predict([[temp,humidity,wind,traffic]])

    st.success(f"Predicted AQI: {prediction[0]}")

st.subheader("Pollution Hotspots")

map_data = pd.DataFrame({
    "lat":[12.97,12.98,12.99],
    "lon":[77.59,77.60,77.61],
    "AQI":[80,120,70]
})

fig = px.scatter_mapbox(
    map_data,
    lat="lat",
    lon="lon",
    color="AQI",
    size="AQI",
    zoom=10,
    mapbox_style="open-street-map"
)

st.plotly_chart(fig)