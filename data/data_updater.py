"""
This module is responsible for updating financial data, including Taiwan Futures (TXF) and Forex data.
It fetches data from FinMind and TwelveData APIs and stores it in Excel files.

Dependencies:
- FinMind API (for futures data)
- TwelveData API (for forex data)

Configuration:
- FINMIND_TOKEN: API token for FinMind, loaded from src.config.
- TD_API_KEY: API key for TwelveData, loaded from src.config.
- SYMBOLS_COLS: List of forex symbols to fetch, loaded from src.config.

Output:
- TXF.xlsx: Contains the latest Taiwan Futures (TXF) near-month contract data.
- Forex.xlsx: Contains the latest forex data for specified symbols.
"""
import pandas as pd
import requests
from datetime import datetime, timedelta
import sys
import os

# Add the project root to sys.path to allow importing modules from src.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import FINMIND_TOKEN, SYMBOLS_COLS, TD_API_KEY
import sys
import io
import time

# Removed logging imports and configuration as per previous changes.

# Force UTF-8 encoding for stdout and stderr to ensure proper display of messages.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
TWELVEDATA_URL = "https://api.twelvedata.com/time_series"
headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}

def update_futures_data():
    """
    Fetches the latest Taiwan Futures (TXF) daily data from FinMind API.
    It processes the raw data to extract only the near-month contract for each trading day
    and saves the cleaned data to 'TXF.xlsx' in the 'data' directory.

    Steps:
    1. Constructs the API request to FinMind for 'TaiwanFuturesDaily' dataset.
    2. Handles potential network errors or empty responses from the API.
    3. Renames columns to a consistent format (e.g., 'Trading_Volume' to 'volume').
    4. Filters data to include only 'after_market' trading sessions.
    5. Calculates the near-month contract by finding the contract with the smallest
       absolute time difference between the trading date and the contract expiration date.
    6. Saves the processed near-month futures data to '../data/TXF.xlsx'.

    Returns:
        pd.DataFrame or None: A DataFrame containing the near-month TXF data if successful,
                              otherwise None if data fetching or processing fails.
    """
    print("正在獲取最新期貨資料...")
    today = datetime.now().strftime("%Y-%m-%d")
    three_years_ago = (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")

    params = {
        "dataset": "TaiwanFuturesDaily",
        "data_id": "TX",
        "start_date": three_years_ago,
        "end_date": today
    }
    try:
        resp = requests.get(FINMIND_URL, headers=headers, params=params)
        resp.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        data = resp.json()['data']
        if not data:
            print("WARNING: 未能從 FinMind 獲取期貨資料。請檢查 FINMIND_TOKEN 或網路連線。")
            return None
        df = pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: 獲取期貨資料時發生錯誤: {e}")
        return None

    # Rename columns for consistency and easier access.
    df.rename(columns={'Trading_Volume': 'volume', 'Trading_Value': 'value', 'Start_Price': 'open', 'End_Price': 'close', 'max': 'high', 'min': 'low'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    
    # Filter for 'after_market' trading sessions to get the main trading data.
    # print("DataFrame before trading_session filter:") # Debugging print, can be removed.
    # print(df.head()) # Debugging print, can be removed.
    # print(df.shape) # Debugging print, can be removed.
    df = df[df['trading_session'] == 'after_market']
    # print("DataFrame after trading_session filter:") # Debugging print, can be removed.
    # print(df.head()) # Debugging print, can be removed.
    # print(df.shape) # Debugging print, can be removed.
    
    # Clean contract_date to ensure it's in YYYYMM format before conversion to datetime.
    df['contract_date'] = df['contract_date'].astype(str).str.replace(r'[^0-9]', '', regex=True).str.strip()
    df['contract_date'] = df['contract_date'].str[:6] # Ensure it's exactly 6 characters long.
    df['contract_date'] = pd.to_datetime(df['contract_date'], format='%Y%m')
    
    # Calculate the absolute difference between the 'date' (trading date) and 'contract_date' (expiration month).
    # This helps in identifying the near-month contract.
    df['time_diff'] = (df['date'] - df['contract_date']).abs()

    # Group by 'date' and find the index of the minimum 'time_diff' for each date.
    # This selects the contract closest to expiration for each trading day.
    idx = df.groupby('date')['time_diff'].idxmin()

    # Select the rows corresponding to these indices, which represent the near-month contracts.
    near_month_df = df.loc[idx]

    # Drop the temporary 'time_diff' column as it's no longer needed.
    near_month_df = near_month_df.drop(columns=['time_diff'])
    
    # Define the output path for the Excel file.
    output_path_txf = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'TXF.xlsx')
    # Ensure the directory for the output file exists.
    os.makedirs(os.path.dirname(output_path_txf), exist_ok=True)
    # Save the DataFrame to an Excel file without the index.
    near_month_df.to_excel(output_path_txf, index=False)
    print("期貨資料已獲取並儲存至 ../data/TXF.xlsx。")
    return near_month_df

def update_forex_data():
    """
    Fetches the latest forex data for predefined symbols from TwelveData API using requests.
    It iterates through a list of forex symbols, fetches their historical data,
    and concatenates them into a single DataFrame, which is then saved to 'Forex.xlsx'.

    Steps:
    1. Iterates through each symbol defined in SYMBOLS_COLS.
    2. Formats the symbol for the TwelveData API (e.g., AUDUSD becomes AUD/USD).
    3. Constructs and executes the TwelveData API GET request.
    4. Handles potential errors or empty responses for each symbol.
    5. Adds a delay between API requests to comply with API rate limits.
    6. Concatenates all fetched forex data into a single DataFrame.
    7. Sorts the combined DataFrame by datetime.
    8. Saves the processed forex data to '../data/Forex.xlsx'.

    Returns:
        pd.DataFrame or None: A DataFrame containing all fetched forex data if successful,
                              otherwise None if no data could be fetched.
    """
    print("正在獲取最新外匯資料 (TwelveData via requests)...")
    all_forex_data = []
    for symbol in SYMBOLS_COLS:
        formatted_symbol = f"{symbol[:3]}/{symbol[3:]}"
        print(f"正在獲取 {formatted_symbol} 的數據...")
        
        params = {
            "symbol": formatted_symbol,
            "interval": "4h",
            "outputsize": 4999,
            "apikey": TD_API_KEY
        }
        
        try:
            resp = requests.get(TWELVEDATA_URL, params=params)
            resp.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            json_data = resp.json()

            if json_data.get('status') == 'error':
                print(f"ERROR: TwelveData API returned an error for {symbol}: {json_data.get('message')}")
                time.sleep(10) # Still sleep to avoid overwhelming the API with bad requests
                continue

            if 'values' in json_data and json_data['values']:
                df = pd.DataFrame(json_data['values'])
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['symbol'] = symbol
                # The API returns 'open', 'high', 'low', 'close', 'volume' which are the desired column names.
                # Let's ensure all numeric columns are float type
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col])
                all_forex_data.append(df)
                print(f"成功獲取 {symbol} 的數據，共 {len(df)} 條記錄。")
            else:
                print(f"WARNING: 未能從 TwelveData 獲取 {symbol} 的數據，數據為空。")

        except requests.exceptions.RequestException as e:
            print(f"ERROR: 獲取 {symbol} 的數據時發生網路錯誤: {e}")
        except ValueError:  # Catches JSON decoding errors
            print(f"ERROR: 無法解析來自 TwelveData 的 {symbol} 的 JSON 回應。")
        except Exception as e:
            print(f"ERROR: 獲取 {symbol} 的數據時發生未知錯誤: {e}")
            
        time.sleep(10)  # Add a 10-second delay between requests to avoid hitting API rate limits.

    if not all_forex_data:
        print("ERROR: 未能獲取任何外匯數據 (TwelveData)。")
        return None

    forex_df = pd.concat(all_forex_data, ignore_index=True)
    forex_df = forex_df.sort_values(by='datetime', ascending=True)
    
    output_path_forex = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Forex.xlsx')
    os.makedirs(os.path.dirname(output_path_forex), exist_ok=True)
    forex_df.to_excel(output_path_forex, index=False)
    print("所有外匯數據已更新並儲存至 ../data/Forex.xlsx")
    return forex_df