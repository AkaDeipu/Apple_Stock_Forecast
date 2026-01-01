import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import date

# =================================
# PAGE CONFIG
# =================================
st.set_page_config(
    page_title="Apple Stock Price Forecast",
    layout="wide"
)

# =================================
# LOAD MODEL
# =================================
@st.cache_resource
def load_lstm():
    return load_model("stock_model.h5", compile=False)

model = load_lstm()

# =================================
# SIDEBAR
# =================================
st.sidebar.title("🍎 Apple Stock Forecast")
st.sidebar.caption("LSTM Model | 2012–2019 Data")

st.sidebar.markdown(
    """
    **Objective:**  
    Predict Apple stock prices for the **next 30 days**
    using historical closing price data.
    """
)

st.sidebar.markdown(
    """
    **Model Details**
    - Lookback window: 30 days  
    - Forecast horizon: 30 days  
    - Feature used: Close price  
    - Scaling: MinMaxScaler  
    """
)

# =================================
# MAIN TITLE
# =================================
st.markdown(
    "<h1 style='text-align:center;'>Apple Stock Price Forecast (Next 30 Days)</h1>",
    unsafe_allow_html=True
)

# =================================
# INPUT METHOD SELECTION
# =================================
st.subheader("📥 Select Input Method")

input_method = st.radio(
    "Choose how you want to provide data:",
    ("Date Range (Recommended)", "Manual Price Input")
)

prices = None

# =================================
# DATE RANGE INPUT
# =================================
if input_method == "Date Range (Recommended)":
    st.subheader("📅 Select Historical Date Range")

    start_date = st.date_input("Start Date", date(2018, 1, 1))
    end_date = st.date_input("End Date", date(2019, 12, 31))

    if st.button("Predict Next 30 Days"):
        data = yf.download(
            "AAPL",
            start=start_date,
            end=end_date,
            progress=False
        )

        if data.empty:
            st.error("❌ No data found for selected date range.")
            st.stop()

        prices = data["Close"].values

        if len(prices) < 30:
            st.error("❌ Please select at least 30 trading days.")
            st.stop()

        prices = prices[-30:]

# =================================
# MANUAL PRICE INPUT
# =================================
if input_method == "Manual Price Input":
    st.subheader("✍️ Enter Last 30 Closing Prices")

    prices_input = st.text_area(
        "Enter exactly 30 closing prices (comma separated)",
        placeholder="e.g. 145.2, 146.1, 147.0, ..."
    )

    if st.button("Predict Next 30 Days"):
        try:
            prices = np.array(
                [float(p.strip()) for p in prices_input.split(",") if p.strip()]
            )

            if len(prices) != 30:
                st.error("❌ Please enter exactly 30 price values.")
                st.stop()

        except ValueError:
            st.error("❌ Please enter valid numeric values.")
            st.stop()

# =================================
# PREDICTION LOGIC
# =================================
if prices is not None:
    try:
        # Scaling
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))

        # Recursive Forecast
        sequence = scaled_prices.flatten()
        predictions_scaled = []

        for _ in range(30):
            X = sequence.reshape(1, 30, 1)
            next_scaled = model.predict(X, verbose=0)[0][0]
            predictions_scaled.append(next_scaled)
            sequence = np.append(sequence[1:], next_scaled)

        # Inverse scaling
        predictions = scaler.inverse_transform(
            np.array(predictions_scaled).reshape(-1, 1)
        ).flatten()

        forecast_df = pd.DataFrame({
            "Day": range(1, 31),
            "Predicted Close Price": predictions
        })

        # Visualization
        st.subheader("🔮 Forecasted Prices for the Next 30 Days")

        fig = px.line(
            forecast_df,
            x="Day",
            y="Predicted Close Price",
            markers=True
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📄 Forecast Values")
        st.dataframe(forecast_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# =================================
# FOOTER
# =================================
st.caption("Apple Stock Price Prediction using LSTM | Streamlit Deployment")