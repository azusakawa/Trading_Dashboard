
"""
This module provides utility functions for calculating various technical indicators
from financial time series data. These indicators are crucial for feature engineering
in machine learning models for financial prediction.
"""

import pandas as pd
import ta # Technical Analysis library
import numpy as np

def calculate_features(df: pd.DataFrame, selected_features: list[str] = None) -> pd.DataFrame:
    """
    Calculates a comprehensive set of technical indicators for the given DataFrame.
    
    This function takes a DataFrame with financial time series data (e.g., open, high, low, close, volume)
    and adds new columns for various technical indicators. It handles cases where there isn't
    enough data for a specific indicator calculation by assigning NaN and printing a warning.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing at least 'high', 'low', and 'close' columns.
                           Optionally, 'open' and 'volume' columns can be present for certain indicators.
        selected_features (list[str], optional): A list of specific technical indicator names to calculate.
                                                 If None, all supported indicators will be calculated.
                                                 
    Returns:
        pd.DataFrame: The original DataFrame with new columns added for the calculated technical indicators.
                      Returns an empty DataFrame if the input DataFrame becomes empty after data cleaning.
    """
    df_copy = df.copy()

    # Standardize column names for high and low prices.
    high_col, low_col = 'high', 'low'

    # Ensure relevant columns are numeric. Coerce non-numeric values to NaN.
    numeric_cols = ['open', high_col, low_col, 'close']
    if 'volume' in df_copy.columns and df_copy['volume'].notna().any():
        numeric_cols.append('volume')
    df_copy[numeric_cols] = df_copy[numeric_cols].apply(pd.to_numeric, errors='coerce')
    # Drop rows where essential numeric data (open, high, low, close) is missing.
    df_copy = df_copy.dropna(subset=numeric_cols)

    if df_copy.empty:
        print("WARNING: DataFrame is empty after numeric conversion and dropping NaNs. Cannot calculate features.")
        return pd.DataFrame()

    # Define the list of features to calculate. If not specified, calculate all supported features.
    if selected_features is None:
        selected_features = ['rsi', 'macd', 'ema', 'cci', 'adx', 'stoch_k', 'obv',
                             'macd_hist', 'atr', 'bband_h', 'bband_l', 'bband_m',
                             'ichimoku_a', 'ichimoku_b', 'ichimoku_base', 'ichimoku_conv',
                             'ichimoku_leading_a', 'ichimoku_leading_b', 'vwap', 'hl_pct_range',
                             'plus_di', 'minus_di'] # Added plus_di and minus_di here

    # --- Momentum Indicators ---
    if 'rsi' in selected_features:
        # Relative Strength Index (RSI): Requires at least 14 periods of data.
        if len(df_copy) >= 14:
            df_copy['rsi'] = ta.momentum.RSIIndicator(df_copy['close']).rsi()
        else:
            df_copy['rsi'] = np.nan
            print("WARNING: Insufficient data for RSI calculation (need >= 14 periods).")

    if 'macd' in selected_features or 'macd_hist' in selected_features:
        # Moving Average Convergence Divergence (MACD): Requires at least 34 periods for standard settings.
        if len(df_copy) >= 34:
            macd_indicator = ta.trend.MACD(df_copy['close'])
            df_copy['macd'] = macd_indicator.macd()
            df_copy['macd_hist'] = macd_indicator.macd_diff() # MACD Histogram
        else:
            df_copy['macd'] = np.nan
            df_copy['macd_hist'] = np.nan
            print("WARNING: Insufficient data for MACD calculation (need >= 34 periods).")

    if 'stoch_k' in selected_features:
        # Stochastic Oscillator (%K): Requires at least 14 periods.
        if len(df_copy) >= 14:
            df_copy['stoch_k'] = ta.momentum.StochasticOscillator(df_copy[high_col], df_copy[low_col], df_copy['close']).stoch()
        else:
            df_copy['stoch_k'] = np.nan
            print("WARNING: Insufficient data for Stochastic K calculation (need >= 14 periods).")

    # --- Trend Indicators ---
    if 'ema' in selected_features:
        # Exponential Moving Average (EMA): Requires at least 14 periods for a common setting.
        if len(df_copy) >= 14:
            df_copy['ema'] = ta.trend.EMAIndicator(df_copy['close']).ema_indicator()
        else:
            df_copy['ema'] = np.nan
            print("WARNING: Insufficient data for EMA calculation (need >= 14 periods).")

    if 'cci' in selected_features:
        # Commodity Channel Index (CCI): Requires at least 20 periods for a common setting.
        if len(df_copy) >= 20:
            df_copy['cci'] = ta.trend.CCIIndicator(df_copy[high_col], df_copy[low_col], df_copy['close'], window=20).cci()
        else:
            df_copy['cci'] = np.nan
            print("WARNING: Insufficient data for CCI calculation (need >= 20 periods).")

    if 'adx' in selected_features or 'plus_di' in selected_features or 'minus_di' in selected_features:
        # Average Directional Index (ADX), Positive Directional Indicator (+DI), Negative Directional Indicator (-DI).
        # ADX typically uses a 14-period window.
        if len(df_copy) >= 14:
            adx_indicator = ta.trend.ADXIndicator(df_copy[high_col], df_copy[low_col], df_copy['close'], window=14)
            df_copy['adx'] = adx_indicator.adx()
            df_copy['plus_di'] = adx_indicator.adx_pos()
            df_copy['minus_di'] = adx_indicator.adx_neg()
        else:
            df_copy['adx'] = np.nan
            df_copy['plus_di'] = np.nan
            df_copy['minus_di'] = np.nan
            print("WARNING: Insufficient data for ADX calculation (need >= 14 periods).")

    if 'ichimoku_a' in selected_features or 'ichimoku_b' in selected_features or \
       'ichimoku_base' in selected_features or 'ichimoku_conv' in selected_features or \
       'ichimoku_leading_a' in selected_features or 'ichimoku_leading_b' in selected_features:
        # Ichimoku Kinko Hyo: Requires at least 52 periods for standard settings.
        if len(df_copy) >= 52:
            ichimoku = ta.trend.IchimokuIndicator(df_copy[high_col], df_copy[low_col], window1=9, window2=26, window3=52)
            df_copy['ichimoku_conv'] = ichimoku.ichimoku_conversion_line() # Tenkan-sen
            df_copy['ichimoku_base'] = ichimoku.ichimoku_base_line()     # Kijun-sen
            df_copy['ichimoku_a'] = ichimoku.ichimoku_a()               # Senkou Span A
            df_copy['ichimoku_b'] = ichimoku.ichimoku_b()               # Senkou Span B
            df_copy['ichimoku_leading_a'] = ichimoku.ichimoku_a().shift(26) # Leading Span A (shifted forward)
            df_copy['ichimoku_leading_b'] = ichimoku.ichimoku_b().shift(26) # Leading Span B (shifted forward)
        else:
            df_copy['ichimoku_conv'] = np.nan
            df_copy['ichimoku_base'] = np.nan
            df_copy['ichimoku_a'] = np.nan
            df_copy['ichimoku_b'] = np.nan
            df_copy['ichimoku_leading_a'] = np.nan
            df_copy['ichimoku_leading_b'] = np.nan
            print("WARNING: Insufficient data for Ichimoku calculation (need >= 52 periods).")

    # --- Volatility Indicators ---
    if 'atr' in selected_features:
        # Average True Range (ATR): Requires at least 14 periods.
        if len(df_copy) >= 14:
            df_copy['atr'] = ta.volatility.AverageTrueRange(df_copy[high_col], df_copy[low_col], df_copy['close']).average_true_range()
        else:
            df_copy['atr'] = np.nan
            print("WARNING: Insufficient data for ATR calculation (need >= 14 periods).")

    if 'bband_h' in selected_features or 'bband_l' in selected_features or 'bband_m' in selected_features:
        # Bollinger Bands: Requires at least 20 periods for standard settings.
        if len(df_copy) >= 20:
            bband = ta.volatility.BollingerBands(df_copy['close'], window=20, window_dev=2)
            df_copy['bband_h'] = bband.bollinger_hband() # Upper Band
            df_copy['bband_l'] = bband.bollinger_lband() # Lower Band
            df_copy['bband_m'] = bband.bollinger_mavg()  # Middle Band (SMA)
        else:
            df_copy['bband_h'] = np.nan
            df_copy['bband_l'] = np.nan
            df_copy['bband_m'] = np.nan
            print("WARNING: Insufficient data for Bollinger Bands calculation (need >= 20 periods).")

    # --- Volume Indicators ---
    if 'obv' in selected_features:
        # On-Balance Volume (OBV): Requires 'volume' column.
        if 'volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['volume']) and not df_copy['volume'].isnull().all():
            df_copy['obv'] = ta.volume.OnBalanceVolumeIndicator(df_copy['close'], df_copy['volume']).on_balance_volume()
        else:
            df_copy['obv'] = np.nan
            print("WARNING: Insufficient data or missing volume for OBV calculation.")

    if 'vwap' in selected_features and 'volume' in df_copy.columns:
        # Volume Weighted Average Price (VWAP): Requires 'volume' column and at least 1 period.
        if len(df_copy) >= 1:
            df_copy['vwap'] = ta.volume.VolumeWeightedAveragePrice(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume']).volume_weighted_average_price()
        else:
            df_copy['vwap'] = np.nan
            print("WARNING: Insufficient data for VWAP calculation (need >= 1 period).")

    # --- Custom Features ---
    if 'hl_pct_range' in selected_features:
        # High-Low Percentage Range: Calculates the percentage range between high and low prices relative to close.
        df_copy['hl_pct_range'] = ((df_copy[high_col] - df_copy[low_col]) / df_copy['close']) * 100

    return df_copy
