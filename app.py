import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
import traceback

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "Stock Market.csv"
N_STEPS = 30

st.set_page_config(page_title="XGBoost Stock Forecast", layout="wide")
st.title("📈 Stock Forecasting (Best Model: XGBoost)")

# Load XGBoost Model
try:
    xgb_model = XGBRegressor()
    xgb_model.load_model(str(MODEL_DIR / "xgb_model.json"))
    st.sidebar.success("XGBoost Model Loaded Successfully")
except Exception as e:
    st.sidebar.error("❌ Failed to load XGBoost model")
    st.sidebar.write(str(e))
    xgb_model = None

# Load CSV data
try:
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True).set_index('Date').sort_index()
    st.subheader("Historical Closing Price")
    st.line_chart(df['Close'])
except Exception as e:
    st.error("Error loading dataset")
    st.write(e)
    st.stop()

# Feature creation function
def create_features(data):
    df = data.copy()
    df['MA7'] = df['Close'].rolling(7).mean()
    df['MA30'] = df['Close'].rolling(30).mean()
    df['Volatility7'] = df['Close'].rolling(7).std()
    df['Range'] = df['High'] - df['Low']
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df['Lag3'] = df['Close'].shift(3)
    return df.dropna()

# Recursive forecasting
def xgb_recursive_forecast(model, df, steps):
    last_data = create_features(df)
    preds = []

    for _ in range(steps):
        row = last_data.iloc[-1]

        features = {
            "Open": row["Open"],
            "High": row["High"],
            "Low": row["Low"],
            "Volume": row["Volume"],
            "MA7": last_data['Close'].rolling(7).mean().iloc[-1],
            "MA30": last_data['Close'].rolling(30).mean().iloc[-1],
            "Volatility7": last_data['Close'].rolling(7).std().iloc[-1],
            "Range": row["High"] - row["Low"],
            "Lag1": row["Close"],
            "Lag2": last_data.iloc[-2]["Close"],
            "Lag3": last_data.iloc[-3]["Close"]
        }

        X = pd.DataFrame([features])
        pred = model.predict(X)[0]

        next_date = last_data.index[-1] + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:  # skip weekends
            next_date += pd.Timedelta(days=1)

        new_row = pd.DataFrame({
            "Close": pred,
            "Open": pred,
            "High": pred,
            "Low": pred,
            "Volume": row["Volume"]
        }, index=[next_date])

        last_data = pd.concat([last_data, new_row])
        preds.append((next_date, float(pred)))

    return pd.DataFrame(preds, columns=["Date", "Forecast"]).set_index("Date")["Forecast"]

# Button to generate forecast
if st.button("Generate 30-Day Forecast (XGBoost)"):
    if xgb_model is None:
        st.error("Model not loaded.")
    else:
        try:
            forecast_series = xgb_recursive_forecast(xgb_model, df, N_STEPS)
            st.subheader("📊 30-Day Forecast")
            st.line_chart(forecast_series)
            st.write(forecast_series)

            # Download option
            csv = forecast_series.to_csv().encode("utf-8")
            st.download_button("Download Forecast CSV", csv, "xgboost_forecast_30_days.csv")

        except Exception as e:
            st.error("Error generating forecast")
            st.text(traceback.format_exc())
