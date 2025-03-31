#!/usr/bin/env python3
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from broker_api import BrokerAPI  # Replace with your broker's API SDK
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
BASE_URL = "https://api.broker.com"  # Replace with your broker's API URL

# Initialize broker API
broker = BrokerAPI(api_key=API_KEY, api_secret=API_SECRET, base_url=BASE_URL)

def fetch_market_data(symbol, interval, lookback):
    """Fetch historical market data."""
    try:
        data = broker.get_historical_data(symbol=symbol, interval=interval, lookback=lookback)
        return pd.DataFrame(data)
    except Exception as e:
        logging.error(f"Error fetching market data: {e}")
        return pd.DataFrame()

def feature_engineering(data):
    """Generate features for the model."""
    try:
        data['SMA_10'] = data['close'].rolling(window=10).mean()
        data['SMA_50'] = data['close'].rolling(window=50).mean()
        data['RSI'] = calculate_rsi(data['close'])
        data['target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)
        data = data.dropna()
        return data
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        return pd.DataFrame()

def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index (RSI)."""
    try:
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        logging.error(f"Error calculating RSI: {e}")
        return pd.Series()

def train_model(data):
    """Train the AI model."""
    try:
        features = ['SMA_10', 'SMA_50', 'RSI']
        X = data[features]
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logging.info(f"Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return None

def execute_trade(signal, symbol, quantity):
    """Execute trades based on signal."""
    try:
        if signal == "BUY":
            broker.place_order(symbol=symbol, quantity=quantity, side="buy")
            logging.info(f"BUY order placed for {quantity} of {symbol}")
        elif signal == "SELL":
            broker.place_order(symbol=symbol, quantity=quantity, side="sell")
            logging.info(f"SELL order placed for {quantity} of {symbol}")
    except Exception as e:
        logging.error(f"Error executing trade: {e}")

def trading_bot(symbol, interval, lookback, quantity):
    """Main function for the trading bot."""
    data = fetch_market_data(symbol, interval, lookback)
    if data.empty:
        logging.error("No market data available. Exiting.")
        return

    data = feature_engineering(data)
    if data.empty:
        logging.error("Feature engineering failed. Exiting.")
        return

    model = train_model(data)
    if model is None:
        logging.error("Model training failed. Exiting.")
        return

    while True:
        try:
            # Fetch latest market data
            latest_data = fetch_market_data(symbol, interval, lookback)
            if latest_data.empty:
                logging.warning("No new market data available. Skipping iteration.")
                time.sleep(interval_to_seconds(interval))
                continue

            latest_data = feature_engineering(latest_data)
            if latest_data.empty:
                logging.warning("Feature engineering failed on latest data. Skipping iteration.")
                time.sleep(interval_to_seconds(interval))
                continue
            
            # Predict the next action
            features = ['SMA_10', 'SMA_50', 'RSI']
            latest_features = latest_data.iloc[-1][features].values.reshape(1, -1)
            prediction = model.predict(latest_features)

            # Execute trade based on prediction
            if prediction[0] == 1:
                logging.info("BUY Signal Detected")
                execute_trade("BUY", symbol, quantity)
            else:
                logging.info("SELL Signal Detected")
                execute_trade("SELL", symbol, quantity)

            # Sleep until next interval
            time.sleep(interval_to_seconds(interval))
        except Exception as e:
            logging.error(f"Error in trading loop: {e}")
            time.sleep(interval_to_seconds(interval))

def interval_to_seconds(interval):
    """Convert interval to seconds."""
    conversion = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}
    return conversion.get(interval, 60)

if __name__ == "__main__":
    trading_bot(symbol="AAPL", interval="1m", lookback=100, quantity=10)
