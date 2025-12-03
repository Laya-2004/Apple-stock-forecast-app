# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from xgboost import XGBRegressor
import io
import traceback

# -----------------------
# Paths and constants
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "Stock Market.csv"
N_STEPS = 30  # forecast length

st.set_page_config(page_title="Stock Forecast (ARIMA + XGBoost)", layout="wide")

# -----------------------
# Helper functions
# -----------------------
def load_models():
    arima_model = None
    xgb_model = None
    arima_msg = ""
    xgb_msg = ""
    try:
        arima_model = joblib.load(MODEL_DIR / "arima_model.pkl")
        arima_msg = "ARIMA loaded"
    except Exception as e:
        arima_msg = f"ARIMA load error: {e}"
    try:
        xgb_model = XGBRegressor()
        xgb_model.load_model(str(MODEL_DIR / "xgb_model.json"))
        xgb_msg = "XGBoost loaded"
    except Exception as e:
        xgb_msg = f"XGBoost load error: {e}"
    return arima_model, xgb_model, arima_msg, xgb_msg

def df_load(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['Date'], dayfirst=True)
    else:
        df = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True)
    df = df.sort_values("Date").set_index("Date")
    return df

def arima_forecast_to_df(arima_model, df, n_steps=N_STEPS):
    """
    Returns a pandas Series (indexed by business days) of length n_steps.
    """
    # get raw forecast (prefer get_forecast to preserve index if available)
    try:
        fc_obj = arima_model.get_forecast(steps=n_steps)
        fc_mean = fc_obj.predicted_mean
    except Exception:
        raw = arima_model.forecast(n_steps)
        fc_mean = pd.Series(np.array(raw).ravel())

    # if forecast index is not datetime, attach business-day dates starting after last df date
    if not isinstance(fc_mean.index, pd.DatetimeIndex):
        start = df.index[-1] + pd.Timedelta(days=1)
        future_dates = pd.date_range(start=start, periods=len(fc_mean), freq='B')
        fc_mean = pd.Series(fc_mean.values, index=future_dates)
    else:
        # ensure business-day frequency for display (if index has no freq, map to B starting from last date)
        if fc_mean.index.freq is None:
            start = df.index[-1] + pd.Timedelta(days=1)
            future_dates = pd.date_range(start=start, periods=len(fc_mean), freq='B')
            fc_mean = pd.Series(fc_mean.values, index=future_dates)

    fc_mean.name = "Forecast"
    return fc_mean

def xgb_recursive_forecast(xgb_model, df, n_steps=N_STEPS):
    """
    Recursive multi-step forecasting using the XGBoost model and engineered features.
    Returns DataFrame indexed by business days with a Forecast column.
    """
    data = df.copy()
    # Feature engineering (must match training)
    data['MA7'] = data['Close'].rolling(7).mean()
    data['MA30'] = data['Close'].rolling(30).mean()
    data['Volatility7'] = data['Close'].rolling(7).std()
    data['Range'] = data['High'] - data['Low']
    data['Lag1'] = data['Close'].shift(1)
    data['Lag2'] = data['Close'].shift(2)
    data['Lag3'] = data['Close'].shift(3)

    # be defensive: fill or drop NaNs sensibly
    # If too many rows are lost, we will raise a friendly error
    data = data.dropna()
    if data.shape[0] < 3:
        raise ValueError("Not enough data after feature creation. Provide more history or reduce rolling windows.")

    last = data.copy()
    preds = []
    for i in range(n_steps):
        last_row = last.iloc[-1]

        features = {
            "Open": last_row["Open"],
            "High": last_row["High"],
            "Low": last_row["Low"],
            "Volume": last_row["Volume"],
            "MA7": last['Close'].rolling(7).mean().iloc[-1],
            "MA30": last['Close'].rolling(30).mean().iloc[-1],
            "Volatility7": last['Close'].rolling(7).std().iloc[-1],
            "Range": last_row["High"] - last_row["Low"],
            "Lag1": last_row["Close"],
            "Lag2": last.iloc[-2]["Close"],
            "Lag3": last.iloc[-3]["Close"]
        }

        X = pd.DataFrame([features])
        pred = xgb_model.predict(X)[0]

        # next business day
        next_date = last.index[-1] + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:  # skip weekends
            next_date += pd.Timedelta(days=1)

        new_row = pd.DataFrame({
            "Close": pred,
            "Open": pred,
            "High": pred,
            "Low": pred,
            "Volume": last_row["Volume"]
        }, index=[next_date])

        last = pd.concat([last, new_row])
        preds.append((next_date, float(pred)))

    xgb_df = pd.DataFrame(preds, columns=["Date", "Forecast"]).set_index("Date")
    return xgb_df['Forecast']

def df_to_csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf)
    return buf.getvalue().encode('utf-8')

# -----------------------
# App UI
# -----------------------
st.title("📈 Stock Forecasting — ARIMA & XGBoost")
st.write("Upload a CSV to override the default dataset, or use the provided one.")

# Sidebar: load models
arima_model, xgb_model, arima_msg, xgb_msg = load_models()
st.sidebar.write("**Model status**")
st.sidebar.write(arima_msg)
st.sidebar.write(xgb_msg)

# Upload data
uploaded = st.file_uploader("Upload Stock CSV (optional). Must include Date, Open, High, Low, Close, Volume", type=["csv"])
try:
    df = df_load(uploaded)
except Exception as e:
    st.error("Error loading CSV: " + str(e))
    st.stop()

st.subheader("Historical Close Price")
st.line_chart(df['Close'])

# Layout: two columns
col1, col2 = st.columns(2)

# -----------------------
# ARIMA forecast UI
# -----------------------
with col1:
    st.subheader("🔮 ARIMA Forecast (30 days)")
    if arima_model is None:
        st.warning("ARIMA model not loaded. Place models/arima_model.pkl in the project.")
    else:
        if st.button("Generate ARIMA Forecast"):
            try:
                arima_series = arima_forecast_to_df(arima_model, df, N_STEPS)
                st.line_chart(arima_series)
                st.write(pd.DataFrame({"Forecast": arima_series}))
                csv_bytes = df_to_csv_bytes(pd.DataFrame({"Date": arima_series.index, "Forecast": arima_series.values}).set_index("Date"))
                st.download_button("Download ARIMA CSV", data=csv_bytes, file_name="arima_forecast_30.csv")
            except Exception as e:
                st.error("Error generating ARIMA forecast: " + str(e))
                st.text(traceback.format_exc())

# -----------------------
# XGBoost forecast UI
# -----------------------
with col2:
    st.subheader("⚡ XGBoost Forecast (30 days)")
    if xgb_model is None:
        st.warning("XGBoost model not loaded. Place models/xgb_model.json in the project.")
    else:
        if st.button("Generate XGBoost Forecast"):
            try:
                xgb_series = xgb_recursive_forecast(xgb_model, df, N_STEPS)
                st.line_chart(xgb_series)
                st.write(pd.DataFrame({"Forecast": xgb_series}))
                csv_bytes = df_to_csv_bytes(pd.DataFrame({"Date": xgb_series.index, "Forecast": xgb_series.values}).set_index("Date"))
                st.download_button("Download XGBoost CSV", data=csv_bytes, file_name="xgb_forecast_30.csv")
            except Exception as e:
                st.error("Error generating XGBoost forecast: " + str(e))
                st.text(traceback.format_exc())

# -----------------------
# Footer: model info & instructions
# -----------------------
st.markdown("---")
st.caption("Notes: \n- Models are loaded from the models/ folder (arima_model.pkl, xgb_model.json). \n- ARIMA should be a fitted ARIMAResults object saved with joblib.dump(arima_fit, 'models/arima_model.pkl'). \n- XGBoost should be saved with xgb_model.save_model('models/xgb_model.json').")
