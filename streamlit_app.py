import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
import os

# Cache the model loading to speed up app loading
@st.cache_resource
def load_trained_model(model_path):
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.error(f"Model file not found at path: {model_path}")
        return None

def calculate_moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

def create_dataset(data, look_back=100):
    X = []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
    return np.array(X)

# Remaining functions unchanged ...

def main():
    st.sidebar.title('Aeon Stock Price Predict')
    stock_symbol = st.sidebar.text_input('Enter Stock Ticker Symbol (e.g., MSFT):')
    start_date = st.sidebar.date_input('Select Start Date:', datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input('Select End Date:', datetime.now())
    selected_model = st.sidebar.radio("Select Model", ("Neural Network",))

    if stock_symbol:
        with st.spinner('Fetching stock data...'):
            stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

        if not stock_data.empty and len(stock_data) > 100:
            st.subheader(f'Stock Data for {stock_symbol}')
            st.write(stock_data.tail())

            # Handle missing columns
            required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
            missing_columns = [col for col in required_columns if col not in stock_data.columns]

            if missing_columns:
                st.error(f"The following columns are missing from the stock data: {missing_columns}")
                return

            # Handle NaN values in stock data
            stock_data.dropna(inplace=True)

            # Calculate moving averages
            stock_data['MA100'] = calculate_moving_average(stock_data['Close'], 100)
            stock_data['MA200'] = calculate_moving_average(stock_data['Close'], 200)

            # Display charts if data is available
            if len(stock_data) >= 2:
                display_charts(stock_data)
            else:
                st.error("Not enough data to display charts.")

            # Load the model and perform prediction if selected
            if selected_model == "Neural Network":
                with st.spinner('Loading the prediction model...'):
                    model_path = 'Models/neural_network_forecaster.keras'  # Ensure this path is correct
                    model = load_trained_model(model_path)

                if model is not None:
                    scaler, y_pred = prepare_and_predict(stock_data, model)
                    display_prediction_chart(stock_data, y_pred)
                    display_evaluation_metrics(stock_data, y_pred)
                    perform_and_display_forecasting(stock_data, model, scaler)
                else:
                    st.error("Model loading failed.")
        else:
            st.error("Failed to fetch stock data or insufficient data points. Please check the ticker symbol and try again.")
    else:
        st.info("Enter a stock ticker symbol to begin.")

if __name__ == '__main__':
    main()

