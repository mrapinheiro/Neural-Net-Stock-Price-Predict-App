import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
import plotly.graph_objs as go

# Cache the model loading to speed up app loading
@st.cache_resource
def load_trained_model(model_path):
    return load_model(model_path)

def calculate_moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

def create_dataset(data, look_back=100):
    X = []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
    return np.array(X)

def display_charts(stock_data):
    if stock_data.empty:
        st.error("No data available to display charts.")
        return

    # Price vs MA100
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
    if 'MA100' in stock_data.columns and stock_data['MA100'].notnull().any():
        fig1.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data['MA100'], mode='lines', name='MA100'))
    fig1.update_layout(title='Close Price vs MA100')
    st.plotly_chart(fig1, use_container_width=True)

    # Price vs MA100 vs MA200
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
    if 'MA100' in stock_data.columns and stock_data['MA100'].notnull().any():
        fig2.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data['MA100'], mode='lines', name='MA100'))
    if 'MA200' in stock_data.columns and stock_data['MA200'].notnull().any():
        fig2.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data['MA200'], mode='lines', name='MA200'))
    fig2.update_layout(title='Close Price vs MA100 vs MA200')
    st.plotly_chart(fig2, use_container_width=True)

    # Candlestick chart
    candlestick_fig = go.Figure(data=[go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'], high=stock_data['High'],
        low=stock_data['Low'], close=stock_data['Close'])])
    candlestick_fig.update_layout(title='Candlestick Chart')
    st.plotly_chart(candlestick_fig, use_container_width=True)

    # Volume plot
    volume_fig = go.Figure(data=[go.Bar(
        x=stock_data.index, y=stock_data['Volume'])])
    volume_fig.update_layout(title='Volume Plot')
    st.plotly_chart(volume_fig, use_container_width=True)

def prepare_and_predict(stock_data, model):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(stock_data['Close']).reshape(-1, 1))
    x_pred = create_dataset(scaled_data)
    if x_pred.size == 0:
        st.error("Not enough data to make predictions.")
        return None, None
    x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1])  # Adjusting the reshape to match model input
    y_pred = model.predict(x_pred)
    y_pred = scaler.inverse_transform(y_pred)
    return scaler, y_pred

def display_prediction_chart(stock_data, y_pred):
    if y_pred is None:
        st.error("Prediction data is not available.")
        return
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=stock_data.index[100:], y=stock_data['Close'][100:], mode='lines', name='Actual Price'))
    fig3.add_trace(go.Scatter(
        x=stock_data.index[100:], y=y_pred.flatten(), mode='lines', name='Predicted Price'))
    fig3.update_layout(title='Actual vs Predicted Prices')
    st.plotly_chart(fig3, use_container_width=True)

def display_evaluation_metrics(stock_data, y_pred):
    if y_pred is None:
        st.error("Cannot display evaluation metrics without predictions.")
        return
    y_true = stock_data['Close'].values[100:]
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    st.subheader('Model Evaluation')
    st.write(f'Mean Absolute Error (MAE): {mae:.2f}')
    st.write(f'Mean Squared Error (MSE): {mse:.2f}')

def perform_and_display_forecasting(stock_data, model, scaler):
    if scaler is None:
        st.error("Scaler is not available for forecasting.")
        return
    forecast_period_days = 30
    last_100_days = stock_data['Close'].tail(100).values.reshape(-1, 1)
    last_100_days_scaled = scaler.transform(last_100_days)

    forecasted_prices_scaled = []
    for _ in range(forecast_period_days):
        x_forecast = last_100_days_scaled[-100:].reshape(1, 100)
        y_forecast_scaled = model.predict(x_forecast)
        forecasted_prices_scaled.append(y_forecast_scaled[0, 0])
        last_100_days_scaled = np.append(last_100_days_scaled, y_forecast_scaled, axis=0)

    forecasted_prices = scaler.inverse_transform(np.array(forecasted_prices_scaled).reshape(-1, 1))

    forecast_dates = pd.date_range(start=stock_data.index[-1] + timedelta(days=1),
                                   periods=forecast_period_days, freq='D')
    forecast_df = pd.DataFrame(data=forecasted_prices, index=forecast_dates, columns=['Forecast'])

    # Display Forecast
    st.subheader('30-Day Forecast')
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecasted Price'))
    fig4.update_layout(title='Stock Price Forecast for the Next 30 Days')
    st.plotly_chart(fig4, use_container_width=True)

def main():
    st.sidebar.title('Aeon Stock Price Predict')
    stock_symbol = st.sidebar.text_input('Enter Stock Ticker Symbol (e.g., MSFT):')
    start_date = st.sidebar.date_input('Select Start Date:', datetime.now() - timedelta(days=730))
    end_date = st.sidebar.date_input('Select End Date:', datetime.now())
    selected_model = st.sidebar.radio("Select Model", ("Neural Network",))

    if stock_symbol:
        with st.spinner('Fetching stock data...'):
            stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

        if not stock_data.empty:
            st.subheader(f'Stock Data for {stock_symbol}')
            st.write(stock_data.tail())

            data_length = len(stock_data)
            if data_length < 200:
                st.error("Not enough data to calculate MA200. Please select a longer date range.")
                return

            stock_data['MA100'] = calculate_moving_average(stock_data['Close'], 100)
            stock_data['MA200'] = calculate_moving_average(stock_data['Close'], 200)
            stock_data.dropna(inplace=True)

            display_charts(stock_data)

            if selected_model == "Neural Network":
                with st.spinner('Loading the prediction model...'):
                    model_path = 'Models/NN_model.keras'  # Ensure this path is correct
                    model = load_trained_model(model_path)

                scaler, y_pred = prepare_and_predict(stock_data, model)
                display_prediction_chart(stock_data, y_pred)
                display_evaluation_metrics(stock_data, y_pred)
                perform_and_display_forecasting(stock_data, model, scaler)
        else:
            st.error("Failed to fetch stock data. Please check the ticker symbol and try again.")
    else:
        st.info("Enter a stock ticker symbol to begin.")

if __name__ == '__main__':
    main()
