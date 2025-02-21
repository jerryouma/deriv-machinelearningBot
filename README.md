# Forex Trading Bot with Machine Learning

This project is a machine learning-based forex trading bot designed to connect to the Deriv platform via WebSockets. It collects historical tick data, computes technical indicators, generates training samples, trains an XGBoost model, and executes trades using a martingale strategy.

## Features

- **Technical Analysis:**  
  Computes key technical indicators such as:

  - Relative Strength Index (RSI)
  - Exponential Moving Average (EMA)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
  - Stochastic Oscillator

- **Machine Learning:**  
  Utilizes [XGBoost](https://xgboost.readthedocs.io/) for model training with [SMOTE](https://imbalanced-learn.org/stable/) for handling class imbalance.

- **Automated Trading:**  
  Connects to the Deriv platform via WebSockets to fetch tick data, execute trades, and update the model based on trade outcomes.

- **Martingale Strategy:**  
  Adjusts trading stakes based on previous wins/losses to manage risk.

- **Data Persistence:**  
  Stores training samples in an SQLite database and saves/loads the trained model using pickle.

## Prerequisites

Before running the bot, ensure you have:

- **Python 3.12** (or a compatible version)
- **Pip**

Install the required Python packages using pip:

```bash
pip install python-dotenv websocket-client pandas numpy xgboost imbalanced-learn scikit-learn
Installation
Clone or Download the Repository:
Ensure that all project files, including forexmodel.py, .env, and the SQLite database (if it already exists), are in the same directory.

Navigate to the Project Directory:

bash
Copy
Edit
cd C:\Users\Dell\DerivMachinelearning
Configuration
Create a .env file in the project root (if it doesn't already exist) with the following content:

env
Copy
Edit
APP_ID=67625
API_TOKEN=QQRAPUXRJB6qqfT
DB_FILE=forex_knowledge.db
MODEL_FILE=forex_model_weights.pkl
Replace 67625 and QQRAPUXRJB6qqfT with your actual Deriv credentials.
Ensure the file is named exactly .env (with no extension).
Running the Bot
Once you have installed the dependencies and configured the .env file, run the trading bot with:

bash
Copy
Edit
python forexmodel.py
The bot will perform the following actions:

Initialization:

Load environment variables.
Initialize the SQLite database.
Training Data Generation:

If no training data is available, it will fetch historical tick data and generate training samples using a sliding window approach.
Model Training:

Load an existing model (if available) or train a new one using the collected data.
Trading Loop:

Fetch the latest tick data.
Compute the feature vector.
Predict the next trade (CALL or PUT).
Execute the trade via the Deriv WebSocket API.
Update the model with trade outcomes and adjust the stake using the martingale strategy.
Developer Information
Name: Pioneer Jerry
Email: jerryouma9@gmail.com
Phone: +254727057394
Troubleshooting
Missing Environment Variables:
If you encounter errors about missing environment variables, verify that:

The .env file is located in the same directory as forexmodel.py.
The file uses the correct format (using = instead of :).
Dependency Issues:
Ensure all required packages are installed. Use pip list to check installed packages.

WebSocket/API Connection Errors:
Make sure your network connection is active and that your Deriv credentials are valid.

Model Training Warnings:
The bot logs warnings if there’s insufficient data diversity or if a candidate model underperforms compared to the current model.

Code Structure
forexmodel.py:
Main script that handles data collection, feature extraction, model training, and trade execution.

.env:
Configuration file containing your API credentials and file paths.

forex_knowledge.db:
SQLite database file used for storing training samples.

forex_model_weights.pkl:
File used to save and load the machine learning model.

Disclaimer
WARNING: This trading bot is provided for educational purposes only. Trading in financial markets involves risk. The performance of this bot is not guaranteed, and you should use it at your own risk. Always conduct your own research before trading.
```
