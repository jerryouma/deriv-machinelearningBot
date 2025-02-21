import websocket
import json
import pandas as pd
import time
import logging
import numpy as np
import sqlite3
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# WebSocket Parameters
APP_ID = "67625"
SYMBOL = "frxAUDJPY"
CURRENCY = "USD"
INITIAL_STAKE = 0.50
API_TOKEN = "QQRAPUXRJB6qqfT"
DB_FILE = "forex_knowledge.db"
MODEL_FILE = "forex_model_weights.pkl"  # File to save/load the model weights

# Global Variables
current_stake = INITIAL_STAKE
model = None
PROFITS = 0
PAYOUT_RATIO = 1.8
DESIRED_PROFIT = 0.5
last_balance = None
wins = 0
losses = 0
total_loss = 0
max_consecutive_losses = 0

##############################
# Model Persistence Functions
##############################
def save_model(model):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    logging.info("Model saved to disk.")

def load_model():
    global model
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        logging.info("Model loaded from disk.")
    else:
        logging.info("No saved model found.")

##############################
# Technical Indicator Helpers
##############################
def calculate_rsi(prices, period=14):
    """Calculate the Relative Strength Index (RSI) for a list of prices."""
    if len(prices) < period + 1:
        return None
    gains = []
    losses_list = []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(change if change > 0 else 0)
        losses_list.append(-change if change < 0 else 0)
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses_list[-period:]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(prices, period):
    """Calculate the Exponential Moving Average (EMA) for a list of prices."""
    if len(prices) < period:
        return None
    sma = sum(prices[:period]) / period
    ema = sma
    multiplier = 2 / (period + 1)
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return ema

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """
    Calculate MACD as the difference between two EMAs (short and long) 
    and then compute the signal line.
    Returns a tuple: (MACD value at the latest point, Signal line).
    """
    if len(prices) < long_period:
        return None, None
    ema_short = calculate_ema(prices, short_period)
    ema_long = calculate_ema(prices, long_period)
    if ema_short is None or ema_long is None:
        return None, None
    macd_line = ema_short - ema_long

    macd_series = []
    for i in range(long_period - 1, len(prices)):
        sub_prices = prices[:i+1]
        ema_short_i = calculate_ema(sub_prices, short_period)
        ema_long_i = calculate_ema(sub_prices, long_period)
        if ema_short_i is not None and ema_long_i is not None:
            macd_series.append(ema_short_i - ema_long_i)
    signal_line = calculate_ema(macd_series, signal_period) if len(macd_series) >= signal_period else None
    return macd_line, signal_line

def calculate_bollinger_bands(prices, period=20, num_std=2):
    """
    Calculate Bollinger Bands for the latest 'period' data points.
    Returns the SMA, upper band, and lower band.
    """
    if len(prices) < period:
        return None, None, None
    window = prices[-period:]
    sma = sum(window) / period
    std = np.std(window)
    upper = sma + num_std * std
    lower = sma - num_std * std
    return sma, upper, lower

def calculate_stochastic(prices, period=14):
    """
    Calculate the Stochastic Oscillator %K for the latest 'period' data points.
    """
    if len(prices) < period:
        return None
    window = prices[-period:]
    lowest = min(window)
    highest = max(window)
    if highest - lowest == 0:
        return 0
    return (prices[-1] - lowest) / (highest - lowest) * 100

##############################
# Database & API Functions
##############################
def init_db():
    logging.info("Initializing database...")
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trading_patterns (
                    id INTEGER PRIMARY KEY,
                    features TEXT,
                    target INTEGER
                )''')
    conn.commit()
    conn.close()
    logging.info("Database initialized.")

def fetch_balance():
    logging.info("Fetching balance...")
    url = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"
    ws = websocket.create_connection(url)
    ws.send(json.dumps({"authorize": API_TOKEN}))
    ws.recv()  # authorization response
    ws.send(json.dumps({"balance": 1}))
    response = json.loads(ws.recv())
    ws.close()
    balance = float(response.get("balance", {}).get("balance", 0))
    logging.info(f"Current balance: {balance}")
    return balance

def fetch_tick_data(count=100):
    logging.info(f"Fetching tick data (count={count})...")
    url = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"
    ws = websocket.create_connection(url)
    ws.send(json.dumps({"authorize": API_TOKEN}))
    ws.recv()
    ws.send(json.dumps({"ticks_history": SYMBOL, "count": count, "end": "latest"}))
    response = json.loads(ws.recv())
    ws.close()
    
    if "error" in response:
        logging.error(f"Tick data fetch error: {response['error']['message']}")
        return None
    
    logging.info("Tick data fetched successfully.")
    prices = list(map(float, response["history"]["prices"]))
    return pd.DataFrame({"price": prices})

def training_data_exists():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM trading_patterns")
    count = c.fetchone()[0]
    conn.close()
    return count > 0

def generate_training_data_from_history():
    """
    Fetch historical tick data and generate training samples using a sliding window approach.
    For each window of size 100, the target is defined by comparing the window's last price
    to the price 15 ticks later.
    """
    logging.info("Generating training data from historical tick data...")
    df = fetch_tick_data(500)  # Adjust count as needed.
    if df is None:
        logging.error("Could not fetch historical tick data.")
        return
    prices = df["price"].tolist()
    window_size = 100
    offset = 15  # offset ticks to determine trade outcome
    samples_generated = 0
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    for i in range(0, len(prices) - window_size - offset + 1):
        window = prices[i:i+window_size]
        target_index = i + window_size - 1 + offset
        if target_index >= len(prices):
            break
        target = 1 if prices[target_index] > window[-1] else 0
        
        # Compute features (8 features)
        last_price = window[-1]
        pct_change = (window[-1] - window[-2]) / window[-2] if len(window) >= 2 else 0
        sma5 = sum(window[-5:]) / 5 if len(window) >= 5 else last_price
        rsi14 = calculate_rsi(window, 14)
        macd_val, signal_line = calculate_macd(window)
        # Bollinger Bands feature: normalized distance from SMA
        bb_sma, bb_upper, bb_lower = calculate_bollinger_bands(window, period=20)
        if bb_sma is None:
            bollinger_percent = 0
        else:
            std = (bb_upper - bb_sma) if (bb_upper - bb_sma) != 0 else 1
            bollinger_percent = (window[-1] - bb_sma) / std
        stochastic_k = calculate_stochastic(window, period=14)
        
        # Ensure all indicators are available
        if (rsi14 is None or macd_val is None or signal_line is None or stochastic_k is None):
            continue
        
        features = [last_price, pct_change, sma5, rsi14, macd_val, signal_line, bollinger_percent, stochastic_k]
        c.execute("INSERT INTO trading_patterns (features, target) VALUES (?, ?)", (str(features), target))
        samples_generated += 1

    conn.commit()
    conn.close()
    logging.info(f"Generated and saved {samples_generated} training samples from historical data.")

##############################
# Feature & Prediction Functions
##############################
def get_latest_features():
    """
    Fetch the latest tick data and compute the feature vector (8 features):
    [last_price, pct_change, sma5, rsi14, macd_val, signal_line, bollinger_percent, stochastic_k]
    """
    df = fetch_tick_data(100)
    if df is None:
        return None
    prices = df["price"].tolist()
    if len(prices) < 26:
        return None
    last_price = prices[-1]
    pct_change = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0
    sma5 = sum(prices[-5:]) / 5 if len(prices) >= 5 else last_price
    rsi14 = calculate_rsi(prices, 14)
    macd_val, signal_line = calculate_macd(prices)
    
    bb_sma, bb_upper, bb_lower = calculate_bollinger_bands(prices, period=20)
    if bb_sma is None:
        bollinger_percent = 0
    else:
        std = (bb_upper - bb_sma) if (bb_upper - bb_sma) != 0 else 1
        bollinger_percent = (last_price - bb_sma) / std
    stochastic_k = calculate_stochastic(prices, period=14)
    
    if (rsi14 is None or macd_val is None or signal_line is None or stochastic_k is None):
        return None
    return [last_price, pct_change, sma5, rsi14, macd_val, signal_line, bollinger_percent, stochastic_k]

def predict_next_move():
    """
    Compute the feature vector from the latest tick data and return both the features and the prediction.
    """
    logging.info("Predicting next move...")
    features = get_latest_features()
    if features is None or model is None:
        logging.warning("Insufficient data or model not trained.")
        return None, None
    prediction = model.predict([features])[0]
    logging.info(f"Predicted move: {'CALL' if prediction == 1 else 'PUT'}")
    return features, prediction

##############################
# Training Sample Recording & Model Update
##############################
def record_training_sample(features, target):
    """
    Record a new training sample into the database.
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO trading_patterns (features, target) VALUES (?, ?)", (str(features), target))
    conn.commit()
    conn.close()
    logging.info("Recorded new training sample.")

##############################
# Model Training with Safeguards
##############################
def train_model():
    global model
    logging.info("Training model with validation...")
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM trading_patterns", conn)
    conn.close()
    
    if df.empty:
        logging.warning("No training data available.")
        return
    
    # Convert stored features and target values
    df["features_converted"] = df["features"].apply(lambda x: eval(x))
    df["target_converted"] = pd.to_numeric(df["target"], errors='coerce')
    
    # Drop rows with invalid data
    df = df.dropna(subset=["target_converted", "features_converted"])
    
    # Build X and y from the cleaned DataFrame
    X = pd.DataFrame(df["features_converted"].tolist())
    y = df["target_converted"].astype(int)
    
    if len(set(y)) < 2:
        logging.warning("Not enough data diversity for training.")
        return

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE on training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    candidate_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        # use_label_encoder=False
    )
    candidate_model.fit(X_train_res, y_train_res)

    # Evaluate candidate model on the validation set
    y_val_pred = candidate_model.predict(X_val)
    candidate_accuracy = accuracy_score(y_val, y_val_pred)
    logging.info(f"Candidate model validation accuracy: {candidate_accuracy:.2f}")

    # If no existing model, deploy candidate model directly.
    if model is None:
        logging.info("No existing model found; deploying candidate model.")
        model = candidate_model
        save_model(model)
    else:
        # Evaluate current model on the same validation set
        current_y_val_pred = model.predict(X_val)
        current_accuracy = accuracy_score(y_val, current_y_val_pred)
        logging.info(f"Current model validation accuracy: {current_accuracy:.2f}")

        # Deploy candidate model if its accuracy is equal or improved.
        if candidate_accuracy >= current_accuracy:
            logging.info("Candidate model performs as well as or better than the current model. Updating model.")
            model = candidate_model
            save_model(model)
        else:
            logging.warning("Candidate model performs worse than the current model. Retaining existing model.")

    logging.info("Model training and validation complete.")

##############################
# Trading Execution Functions
##############################
def martingale_strategy(result):
    global current_stake, wins, losses, total_loss, max_consecutive_losses, PROFITS
    logging.info(f"Martingale strategy processing result: {result}")
    if result == "win":
        wins += 1
        losses = 0
        total_loss = 0
        PROFITS += current_stake * (PAYOUT_RATIO - 1)
        current_stake = INITIAL_STAKE
    else:
        losses += 1
        wins = 0
        total_loss += current_stake
        PROFITS -= current_stake
        # Updated formula for next stake after a loss:
        current_stake = round(total_loss * 100 / 86, 2)
        max_consecutive_losses = max(max_consecutive_losses, losses)
    logging.info(f"Wins: {wins}, Losses: {losses}, Next Stake: {current_stake}, Max Losing Streak: {max_consecutive_losses}, Profit: {PROFITS:.2f}")

def execute_trade():
    global last_balance
    logging.info("Executing trade...")
    features, prediction = predict_next_move()
    if features is None or prediction is None:
        logging.warning("Trade skipped due to missing prediction.")
        return
    
    contract_type = "CALL" if prediction == 1 else "PUT"
    logging.info(f"Placing {contract_type} trade...")
    
    url = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"
    ws = websocket.create_connection(url)
    ws.send(json.dumps({"authorize": API_TOKEN}))
    ws.recv()
    trade_request = {
        "buy": 1,
        "price": current_stake,
        "parameters": {
            "amount": current_stake,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": CURRENCY,
            "duration": 15,
            "duration_unit": "m",
            "symbol": SYMBOL
        }
    }
    ws.send(json.dumps(trade_request))
    response = json.loads(ws.recv())
    ws.close()
    
    if "error" in response:
        logging.error(f"Trade error: {response['error']['message']}")
        return
    
    logging.info("Trade placed successfully, waiting for result...")
    time.sleep(900)  # Wait 15 minutes for the trade to complete.
    new_balance = fetch_balance()
    trade_result = "win" if new_balance > last_balance else "loss"
    
    # Determine correct target:
    correct_target = prediction if trade_result == "win" else 1 - prediction
    
    # Record the training sample based on the features used for the trade.
    record_training_sample(features, correct_target)
    
    # Retrain the model with the updated data.
    train_model()
    
    last_balance = new_balance
    logging.info(f"Trade result: {trade_result}")
    martingale_strategy(trade_result)

##############################
# Main Function
##############################
def main():
    global last_balance
    logging.info("Starting trading bot...")
    init_db()
    
    # If no training data exists, generate it from historical data.
    if not training_data_exists():
        logging.info("No training data found. Generating training data from historical ticks...")
        generate_training_data_from_history()
    
    load_model()
    train_model()
    last_balance = fetch_balance()
    
    while True:
        logging.info("Starting new trade cycle...")
        execute_trade()
        logging.info("Trade cycle completed.")
        time.sleep(2)

if __name__ == "__main__":
    main()
