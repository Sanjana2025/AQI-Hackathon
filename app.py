import requests
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor # type: ignore
import plotly.express as px # type: ignore

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
    aqi = int(prediction[0])

    st.success(f"Predicted AQI: {aqi}")

    st.subheader("Health Advice")

    if aqi <= 50:
        st.success("Air Quality is Good. Safe for outdoor activities.")
    elif aqi <= 100:
        st.warning("Moderate air quality. Sensitive people should be careful.")
    elif aqi <= 150:
        st.warning("Unhealthy for sensitive groups.")
    else:
        st.error("Unhealthy air quality. Wear a mask and avoid outdoor exercise.")

else:
    st.error("Unhealthy air quality. Wear a mask and avoid outdoor exercise.")
    st.subheader("Real-Time AQI (API Data)")

api_key = "5450534f048e34909856e1f0e3df64b5"

url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat=12.97&lon=77.59&appid={api_key}"

try:
    response = requests.get(url)
    api_data = response.json()

    if "list" in api_data:
        live_aqi = api_data["list"][0]["main"]["aqi"]
        st.write("Live AQI from API:", live_aqi)

except:
    st.write("API data not available")
    st.subheader("Location Based AQI")

city = st.text_input("Enter City Name")

if city:
    st.write(f"Showing AQI information for {city}")
    st.subheader("AQI Trend Over Time")

trend_fig = px.line(data, y="AQI", title="AQI Trend")
st.plotly_chart(trend_fig)

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
st.subheader("💬 AQI Chatbot")

user_input = st.text_input("Ask me about AQI or pollution:")

if user_input:
    if "bangalore" in user_input.lower():
        st.write("The AQI in Bangalore is usually moderate. You can check live AQI above.")
    elif "health" in user_input.lower():
        st.write("If AQI is above 150, avoid outdoor exercise and wear a mask.")
    elif "aqi" in user_input.lower():
        st.write("AQI (Air Quality Index) measures how polluted the air is.")
    elif "pollution" in user_input.lower():
        st.write("Major pollution sources include traffic, industries, and construction.")
    else:
        st.write("Sorry, I can only answer questions about AQI and pollution.")