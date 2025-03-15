# Aeon Quant Stock Price Prediction App

> **Disclaimer**: Stock price prediction is inherently uncertain, and the predictions made by this app are for educational purposes only. They should not be used for investment decisions.

A Streamlit application that predicts stock prices using a Neural Network model (LSTM) and visualizes stock data with interactive charts.

![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)

## Quick Links

- **Live Demo**: [Aeon Quant on Streamlit](https://aeonquant.streamlit.app/)
- **Last Model Training**: October 23, 2024

## Overview

The Aeon Quant Stock Price Prediction App provides a user-friendly interface for technical analysis and predictive modeling of stock prices. The app leverages deep learning techniques (LSTM neural networks) to forecast future stock price movements.

## Features

- **Real-time Stock Data Retrieval**:
  - Fetch historical market data via Yahoo Finance API
  - Flexible date range selection
  - Support for any publicly traded ticker symbol

- **Comprehensive Technical Analysis**:
  - Interactive price charts with moving averages (MA100, MA200)
  - Candlestick pattern visualization
  - Trading volume analysis
  - Key technical indicators

- **Advanced Predictive Modeling**:
  - Neural Network (LSTM) predictions based on historical patterns
  - Comparison of actual vs. predicted prices
  - Performance metrics display (MAE, MSE)
  - 30-day price forecast visualization

## Technical Implementation

### Architecture

- **Frontend**: Streamlit web application
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Model**: TensorFlow LSTM neural network
- **Visualization**: Plotly interactive charts

### Model Details

The stock prediction model utilizes a sequential LSTM (Long Short-Term Memory) neural network with:
- Two LSTM layers with 50 units each
- Dropout layers (0.2) to prevent overfitting
- Adam optimizer with MSE loss function
- 100-day lookback window for pattern recognition

## Installation and Usage

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/mrapinheiro/neural-net-stock-price-predictor.git
   cd aeon-quant
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run streamlit_app.py
   ```

4. Access the app in your browser at `http://localhost:8501`

### Environment Variables

- `TF_CPP_MIN_LOG_LEVEL=3`: Suppresses TensorFlow logs (set in the app)

## Future Enhancements

- Sentiment analysis integration using financial news data
- Additional prediction models for comparison
- Portfolio optimization tools
- Backtesting functionality
- Multi-asset correlation analysis

## Dependencies

- yfinance==0.2.18
- streamlit==1.25.0
- scikit-learn==1.3.0
- tensorflow==2.13.0
- plotly==5.15.0
- numpy==1.23.5
- pandas==1.5.3

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

*Built with ❤️ by Vortex Legacy*
