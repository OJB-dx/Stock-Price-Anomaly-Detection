# yfinaice installation

# =======================
# importing modules

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import streamlit as st

# =======================
# Streamlit App UI
# =======================
st.set_page_config(page_title="Stock Anomaly Detection", layout="wide")

st.title("Stock Anomaly Detection using Isolation Forest")

# Sidebar inputs
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., CVX, AAPL, TSLA):", "CVX")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))
contamination = st.sidebar.slider("Anomaly proportion (contamination)", 0.01, 0.1, 0.02, 0.01)

if st.sidebar.button("Run Detection"):
    # 1. Download stock data
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[features].dropna()

    # 2. Train Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly_if'] = iso.fit_predict(df[features])  # -1 = anomaly, 1 = normal
    anomalies = df[df['anomaly_if'] == -1]

    # 3. Plot results
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df.index, df['Close'], label="Close Price", color="blue")
    ax.scatter(anomalies.index, anomalies['Close'], color="red", marker="x", s=100, label="Anomaly")
    ax.set_title(f"Anomaly Detection on {ticker} Stock ({start_date} → {end_date})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()

    # 4. Show chart + anomalies table
    st.pyplot(fig)
    st.subheader("📌 Detected Anomalies")
    st.dataframe(anomalies[['Open', 'High', 'Low', 'Close', 'Volume']])
