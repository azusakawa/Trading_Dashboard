"""
This module provides prediction services, including loading pre-trained AI models,
fetching real-time or historical financial data, calculating technical indicators,
and generating trading signals/predictions for both futures (TXF) and forex markets.

It integrates with external APIs (FinMind, TwelveData) and utilizes scikit-learn
models and TA-Lib for analysis.
"""

import pandas as pd
import numpy as np
import requests
from twelvedata import TDClient
import ta # Technical Analysis library
import time
import random
from datetime import datetime, timedelta, timezone
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
import joblib # For loading pre-trained models

# Local utility and data update modules
from utils.feature_utils import calculate_features
from data.data_updater import update_futures_data, update_forex_data

# Removed logging imports and configuration as per previous changes.

# --- Global Variables and Settings ---
# Directory where pre-trained AI models are stored.
MODEL_DIR = "models"

# Import sensitive API keys and configuration from src.config.
# This ensures that sensitive information is kept separate and not hardcoded.
try:
    from src.config import TD_API_KEY, FINMIND_TOKEN, SYMBOLS_COLS
    
except ImportError:
    print("❌ Error: config.py file not found or missing required settings.")
    # Exit the application if critical configuration is missing.
    exit()



# --- Service Initialization ---
# Initialize TwelveData client with the API key.
td = TDClient(apikey=TD_API_KEY)
# Base URL for FinMind API.
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
# Authorization headers for FinMind API requests.
headers = {"Authorization": "Bearer " + FINMIND_TOKEN}



# --- 模型載入 ---
ai_models = {}
scalers = {}
selected_features_dict = {}

def load_model_artifacts(model_path: str, model_type: str, symbol: str = None):
    """
    Loads pre-trained model artifacts (model, scaler, selected features) from a joblib file.
    
    Args:
        model_path (str): The absolute or relative path to the joblib file containing the model artifacts.
        model_type (str): A descriptive string for the type of model (e.g., "TXF", "forex").
        symbol (str, optional): The specific symbol for which the model is being loaded (e.g., "AUDUSD").
                                 Defaults to None for models like TXF that don't have a symbol.
                                 
    Returns:
        dict or None: A dictionary containing the loaded artifacts if successful, otherwise None.
    """
    try:
        artifacts = joblib.load(model_path)
        print(f"DEBUG: Successfully loaded {model_type} model for {symbol if symbol else 'TXF'} from {model_path}")
        return artifacts
    except FileNotFoundError:
        print(f"WARNING: Model file not found for {model_type} {symbol if symbol else 'TXF'}: {model_path}")
    except Exception as e:
        print(f"ERROR: Failed to load {model_type} model for {symbol if symbol else 'TXF'} from {model_path}. Error: {e}")
    return None

def _load_all_models():
    """
    Loads all AI models (TXF and Forex) and their associated scalers and selected features.
    This function can be called to initially load models or to hot-reload them.
    """
    global ai_models, scalers, selected_features_dict
    ai_models = {}
    scalers = {}
    selected_features_dict = {}

    # Load TXF (Taiwan Futures) model artifacts.
    MODEL_FILE_TXF = os.path.join(MODEL_DIR, "txf_model.joblib")
    artifacts_txf = load_model_artifacts(MODEL_FILE_TXF, "TXF")
    if artifacts_txf:
        ai_models['txf'] = artifacts_txf.get('model')
        scalers['txf'] = artifacts_txf.get('scaler')
        selected_features_dict['txf'] = artifacts_txf.get('selected_features')
    else:
        print("WARNING: TXF model artifacts could not be loaded. Predictions for TXF may not work.")

    # Load Forex models for each specified symbol.
    ai_models['forex'] = {}
    scalers['forex'] = {}
    selected_features_dict['forex'] = {}
    for symbol in SYMBOLS_COLS:
        MODEL_FILE_FOREX = os.path.join(MODEL_DIR, f"forex_{symbol}_model.joblib")
        artifacts_forex = load_model_artifacts(MODEL_FILE_FOREX, "forex", symbol)
        if artifacts_forex:
            ai_models['forex'][symbol] = artifacts_forex.get('model')
            scalers['forex'][symbol] = artifacts_forex.get('scaler')
            selected_features_dict['forex'][symbol] = artifacts_forex.get('selected_features')
        else:
            # If logging is re-enabled, a warning could be logged here for missing forex models.
            pass

# Initial load of all models when the module is imported.
_load_all_models()

def reload_models():
    """
    Public function to trigger the reloading of all AI models.
    """
    print("INFO: Reloading all AI models...")
    _load_all_models()
    print("INFO: AI models reloaded successfully.")

# --- 核心邏輯函式 ---
def get_bband_advice(group_df: pd.DataFrame) -> tuple[str, float]:
    """
    Generates trading advice (BUY/SELL/NEUTRAL) and a stop-loss level based on Bollinger Bands (BBand).
    
    This function calculates Bollinger Bands and Average True Range (ATR) for the given DataFrame.
    It then determines a trading signal based on price crossing the upper or lower Bollinger Band.
    
    Args:
        group_df (pd.DataFrame): A DataFrame containing historical price data (at least 'close', 'high', 'low').
                                 It should have enough data points for Bollinger Bands (20 periods) and ATR (14 periods).
                                 
    Returns:
        tuple[str, float]: A tuple containing:
                           - str: The trading advice ("BBAND_BUY", "BBAND_SELL", or "NEUTRAL").
                           - float: The calculated stop-loss level, or np.nan if no signal or insufficient data.
    """
    high_col: str = 'high'
    low_col: str = 'low'
    
    bband_advice_key = "NEUTRAL"
    bband_stop_loss = np.nan

    # Ensure enough data for Bollinger Bands (window=20) and ATR (window=14).
    # Need at least 20 for BBands + 1 for previous candle to check crosses.
    if len(group_df) < 21:
        print(f"DEBUG: get_bband_advice - Insufficient data ({len(group_df)} points) for BBand calculation. Returning NEUTRAL.")
        return bband_advice_key, bband_stop_loss

    # Calculate Bollinger Bands (20-period, 2 standard deviations).
    bband = ta.volatility.BollingerBands(group_df['close'], window=20, window_dev=2)
    group_df['bband_h'] = bband.bollinger_hband() # Upper band
    group_df['bband_l'] = bband.bollinger_lband() # Lower band
    group_df['bband_m'] = bband.bollinger_mavg() # Middle band (20-period SMA)
    
    # Calculate Average True Range (ATR) for stop-loss calculation.
    group_df['atr'] = ta.volatility.average_true_range(group_df[high_col], group_df[low_col], group_df['close'], window=14)

    # Get the latest and previous data points for signal generation.
    latest = group_df.iloc[-1]
    previous = group_df.iloc[-2]

    # Check for valid indicator values to avoid errors with NaN.
    if pd.isna(latest['bband_l']) or pd.isna(latest['bband_h']) or pd.isna(latest['atr']) or \
       pd.isna(previous['bband_l']) or pd.isna(previous['bband_h']):
        print(f"DEBUG: get_bband_advice - NaN values in latest/previous BBands or ATR. Returning NEUTRAL.")
        print(f"  Latest: close={latest['close']:.2f}, bband_l={latest['bband_l']:.2f}, bband_h={latest['bband_h']:.2f}, atr={latest['atr']:.2f}")
        print(f"  Previous: close={previous['close']:.2f}, bband_l={previous['bband_l']:.2f}, bband_h={previous['bband_h']:.2f}")
        return bband_advice_key, bband_stop_loss

    # Debugging output for signal conditions
    # print(f"DEBUG: get_bband_advice - Latest close: {latest['close']:.2f}, Previous close: {previous['close']:.2f}")
    # print(f"  Latest BBand_L: {latest['bband_l']:.2f}, Previous BBand_L: {previous['bband_l']:.2f}")
    # print(f"  Latest BBand_H: {latest['bband_h']:.2f}, Previous BBand_H: {previous['bband_h']:.2f}")
    # print(f"  Latest ATR: {latest['atr']:.2f}")

    # Buy signal: Current close price crosses above the lower Bollinger Band from below.
    buy_condition = (latest['close'] > latest['bband_l'] and previous['close'] < previous['bband_l'])
    # print(f"  Buy condition (latest.close > latest.bband_l ({latest['close'] > latest['bband_l']}) AND previous.close < previous.bband_l ({previous['close'] < previous['bband_l']})) : {buy_condition}")
    if buy_condition:
        bband_advice_key = "BBAND_BUY"
        # Stop-loss set 2 * ATR below the latest close price.
        bband_stop_loss = latest['close'] - (2 * latest['atr'])
        print(f"  -> BUY signal generated. Stop Loss: {bband_stop_loss:.2f}")

    # Sell signal: Current close price crosses below the upper Bollinger Band from above.
    elif (latest['close'] < latest['bband_h'] and previous['close'] > previous['bband_h']):
        bband_advice_key = "BBAND_SELL"
        # Stop-loss set 2 * ATR above the latest close price.
        bband_stop_loss = latest['close'] + (2 * latest['atr'])
        print(f"  -> SELL signal generated. Stop Loss: {bband_stop_loss:.2f}")
    else:
        print(f"  -> No clear BUY/SELL signal. Returning NEUTRAL.")
                
    return bband_advice_key, bband_stop_loss

def get_forex_sub_category(symbol: str) -> str:
    """
    Determines the sub-category for forex symbols based on their base currency.
    For a 6-character forex symbol (e.g., "AUDUSD"), it extracts the first three characters
    (the base currency) and appends " Pairs" to form the sub-category.
    
    Args:
        symbol (str): The forex symbol (e.g., "AUDUSD", "EURJPY").
        
    Returns:
        str: The sub-category string (e.g., "AUD Pairs", "EUR Pairs"), or "Other Forex"
             if the symbol format is not recognized.
    """
    if len(symbol) == 6: # Assuming standard 6-character forex symbols like AUDUSD
        base_currency = symbol[:3]
        return f"{base_currency} Pairs"
    return "Other Forex"

def run_pipeline(mode: str, data_source: pd.DataFrame) -> pd.DataFrame:
    """
    Main pipeline to process financial data, apply trading strategies, and generate a summary
    of signals and key indicators for display on the frontend.
    
    This function iterates through symbols (or the single TXF data), calculates various
    technical indicators, and aggregates them into a summary DataFrame.
    
    Args:
        mode (str): The type of data being processed ("forex" or "txf"). This determines
                    how data is grouped and categorized.
        data_source (pd.DataFrame): The input DataFrame containing historical price data
                                    for one or more symbols.
                                    
    Returns:
        pd.DataFrame: A summary DataFrame with one row per symbol, containing:
                      - 'symbol': The trading symbol (e.g., "AUDUSD", "TXF").
                      - 'category': "Forex" or "Futures".
                      - 'sub_category': (For Forex) e.g., "AUD Pairs", or None for Futures.
                      - 'latest_close': The most recent closing price.
                      - 'bband_advice': Trading advice from Bollinger Bands (e.g., "BBAND_BUY").
                      - 'bband_stop_loss': Calculated stop-loss level for Bollinger Bands.
                      - 'quote_currency': The quote currency of the symbol (e.g., "USD", "TXF").
                      - Various technical indicators (RSI, MACD, EMA, CCI, ADX, Stochastics, ATR, BBW).
    """
    high_col: str = 'high'
    low_col: str = 'low'

    df = data_source
    # Group data by symbol for forex, or treat the entire DataFrame as a single group for TXF.
    groupby_obj = df.groupby("symbol") if mode == "forex" else [("TXF", df)]
    summary_rows = []

    # Lookback period for technical indicator calculations. Increased to ensure enough data for Ichimoku (if used).
    LOOKBACK_PERIOD = 200 # Increased lookback period for more robust indicator calculation

    for sym, group in groupby_obj:
        if group.empty:
            continue

        # Ensure enough data for technical indicators (e.g., 20 for BBands, 14 for ATR, 26 for MACD)
        # We need at least 26 data points for MACD, and 20 for BBands, plus one for previous candle.
        # So, ensure at least 27 data points for reliable calculation.
        if len(group) < 27:
            print(f"WARNING: Insufficient data for {sym} ({len(group)} points). Skipping BBand advice calculation.")
            bband_advice_key = "NEUTRAL"
            bband_stop_loss = np.nan
            # Fill other indicators with NaN if data is insufficient
            rsi_val, macd_val, ema_val, cci_val, adx_val, stoch_k_val, atr_val, bbw_val = [np.nan] * 8
        else:
            # Slice the group to the lookback period to speed up feature calculation and avoid unnecessary computations.
            processed_group = group.tail(LOOKBACK_PERIOD).copy()
            
            # Calculate Bollinger Bands (20-period, 2 standard deviations).
            bband = ta.volatility.BollingerBands(processed_group['close'], window=20, window_dev=2)
            processed_group['bband_h'] = bband.bollinger_hband() # Upper band
            processed_group['bband_l'] = bband.bollinger_lband() # Lower band
            processed_group['bband_m'] = bband.bollinger_mavg() # Middle band (20-period SMA)
            
            # Calculate Average True Range (ATR) for stop-loss calculation.
            processed_group['atr'] = ta.volatility.average_true_range(processed_group[high_col], processed_group[low_col], processed_group['close'], window=14)

            # Get trading advice from the Bollinger Bands strategy.
            bband_advice_key, bband_stop_loss = get_bband_advice(processed_group)

            # Calculate other indicators for display
            rsi_val = ta.momentum.rsi(processed_group['close']).iloc[-1] if len(processed_group) >= 14 else np.nan
            macd_val = ta.trend.macd(processed_group['close']).iloc[-1] if len(processed_group) >= 26 else np.nan
            ema_val = ta.trend.ema_indicator(processed_group['close']).iloc[-1] if len(processed_group) >= 10 else np.nan
            cci_val = ta.trend.cci(processed_group['high'], processed_group['low'], processed_group['close']).iloc[-1] if len(processed_group) >= 14 else np.nan
            adx_val = ta.trend.adx(processed_group['high'], processed_group['low'], processed_group['close']).iloc[-1] if len(processed_group) >= 14 else np.nan
            stoch_k_val = ta.momentum.stoch(processed_group['high'], processed_group['low'], processed_group['close']).iloc[-1] if len(processed_group) >= 14 else np.nan
            atr_val = processed_group['atr'].iloc[-1] if 'atr' in processed_group.columns else np.nan
            bbw_val = (processed_group['bband_h'].iloc[-1] - processed_group['bband_l'].iloc[-1]) / processed_group['bband_m'].iloc[-1] if 'bband_h' in processed_group.columns and 'bband_l' in processed_group.columns and 'bband_m' in processed_group.columns else np.nan

        # Determine the quote currency based on the mode.
        if mode == "forex":
            quote_currency = sym[-3:]
        else: # For TXF
            quote_currency = "TXF"

        # Append a summary row for the current symbol with calculated indicators and advice.
        summary_rows.append({
            "symbol": sym,
            "category": "Forex" if mode == "forex" else "Futures",
            "sub_category": get_forex_sub_category(sym) if mode == "forex" else None, # Add sub_category for forex.
            "latest_close": group['close'].iloc[-1],
            "bband_advice": bband_advice_key, # Key for translation on the frontend.
            "bband_stop_loss": bband_stop_loss,
            "quote_currency": quote_currency,
            # Calculated technical indicators.
            "rsi": rsi_val,
            "macd": macd_val,
            "ema": ema_val,
            "cci": cci_val,
            "adx": adx_val,
            "stoch_k": stoch_k_val,
            "atr": atr_val,
            "bbw": bbw_val,
        })
        
    return pd.DataFrame(summary_rows)

def get_predictions() -> pd.DataFrame:
    """
    Main function to orchestrate the prediction process.
    It reads the latest forex and futures data from their respective Excel files,
    processes them through the `run_pipeline` to generate trading signals and indicators,
    and then combines the results into a single DataFrame.

    This function handles file reading errors gracefully, returning empty DataFrames
    if a file is not found or an error occurs during processing.

    Returns:
        pd.DataFrame: A combined DataFrame containing summary predictions and indicators
                      for both forex and futures, ready to be sent to the frontend.
                      Returns an empty DataFrame if no data can be processed.
    """
    data_forex = None
    data_futures = None

    # Attempt to read forex data from Forex.xlsx.
    try:
        forex_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Forex.xlsx')
        data_forex = pd.read_excel(forex_path)
    except FileNotFoundError:
        print(f"WARNING: Forex.xlsx not found at {forex_path}. Skipping forex predictions.")
        pass # Continue without forex data if file not found.
    except Exception as e:
        print(f"ERROR: Failed to read Forex.xlsx: {e}. Skipping forex predictions.")
        pass # Continue without forex data if an error occurs.

    # Attempt to read futures data from TXF.xlsx.
    try:
        txf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'TXF.xlsx')
        data_futures = pd.read_excel(txf_path)
    except FileNotFoundError:
        print(f"WARNING: TXF.xlsx not found at {txf_path}. Skipping futures predictions.")
        pass # Continue without futures data if file not found.
    except Exception as e:
        print(f"ERROR: Failed to read TXF.xlsx: {e}. Skipping futures predictions.")
        pass # Continue without futures data if an error occurs.

    summary_forex = pd.DataFrame()
    summary_futures = pd.DataFrame()

    # Process forex data if available and not empty.
    if data_forex is not None and not data_forex.empty:
        # Convert relevant columns to numeric, coercing errors to NaN.
        for col in ['open', 'high', 'low', 'close']:
            if col in data_forex.columns:
                data_forex[col] = pd.to_numeric(data_forex[col], errors='coerce')
        # Drop rows where essential price data is missing.
        data_forex = data_forex.dropna(subset=['open', 'high', 'low', 'close'])

        if not data_forex.empty:
            try:
                summary_forex = run_pipeline("forex", data_forex)
            except Exception as e:
                print(f"ERROR: Error processing forex data pipeline: {e}")
                summary_forex = pd.DataFrame() # Ensure it's an empty DataFrame on error.
    
    # Process futures data if available and not empty.
    if data_futures is not None and not data_futures.empty:
        # Convert relevant columns to numeric, coercing errors to NaN.
        for col in ['open', 'high', 'low', 'close']:
            if col in data_futures.columns:
                data_futures[col] = pd.to_numeric(data_futures[col], errors='coerce')
        # Drop rows where essential price data is missing.
        data_futures = data_futures.dropna(subset=['open', 'high', 'low', 'close'])

        if not data_futures.empty:
            try:
                summary_futures = run_pipeline("txf", data_futures)
            except Exception as e:
                print(f"ERROR: Error processing futures data pipeline: {e}")
                summary_futures = pd.DataFrame() # Ensure it's an empty DataFrame on error.

    # Concatenate all prediction results (forex and futures) into a single DataFrame.
    combined_summary = pd.concat([summary_forex, summary_futures], ignore_index=True)
    return combined_summary

def get_historical_data_for_symbol(symbol: str) -> pd.DataFrame:
    """
    Retrieves historical closing price data for a specific symbol (TXF or Forex) from the Excel files.
    This function is primarily used to provide data for charting on the frontend.

    Args:
        symbol (str): The trading symbol for which to retrieve historical data (e.g., "TXF", "AUDUSD").

    Returns:
        pd.DataFrame: A DataFrame with two columns: 'time' (datetime) and 'close' (float),
                      containing the historical closing prices for the specified symbol.
                      Returns an empty DataFrame if the symbol is not found or data is empty/unavailable.
    """
    data_forex = None
    data_futures = None

    # Attempt to read forex data from Forex.xlsx.
    try:
        forex_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Forex.xlsx')
        data_forex = pd.read_excel(forex_path)
    except FileNotFoundError:
        print(f"WARNING: Forex.xlsx not found at {forex_path}. Cannot retrieve forex historical data.")
        pass
    except Exception as e:
        print(f"ERROR: Failed to read Forex.xlsx for historical data: {e}.")
        pass

    # Attempt to read futures data from TXF.xlsx.
    try:
        txf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'TXF.xlsx')
        data_futures = pd.read_excel(txf_path)
    except FileNotFoundError:
        print(f"WARNING: TXF.xlsx not found at {txf_path}. Cannot retrieve futures historical data.")
        pass
    except Exception as e:
        print(f"ERROR: Failed to read TXF.xlsx for historical data: {e}.")
        pass

    if symbol == "TXF" and data_futures is not None and not data_futures.empty:
        # For TXF, ensure 'date' column is datetime and sort by date.
        data_futures['date'] = pd.to_datetime(data_futures['date'])
        data_futures = data_futures.sort_values(by='date')
        # Rename 'date' to 'time' for consistency with frontend charting libraries.
        return data_futures.rename(columns={'date': 'time'})[['time', 'close']]
    elif symbol != "TXF" and data_forex is not None and not data_forex.empty:
        # For forex symbols, ensure 'datetime' column is datetime and sort by datetime.
        data_forex['datetime'] = pd.to_datetime(data_forex['datetime'])
        data_forex = data_forex.sort_values(by='datetime')
        # Filter for the specific forex symbol and rename 'datetime' to 'time'.
        return data_forex[data_forex['symbol'] == symbol].rename(columns={'datetime': 'time'})[['time', 'close']]
    
    # Return an empty DataFrame if the symbol is not found or data is unavailable.
    return pd.DataFrame()

