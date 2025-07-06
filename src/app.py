import eventlet
eventlet.monkey_patch()

import sys
import os
import pandas as pd
import numpy as np
import threading
import time

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from services.prediction_service import get_predictions

all_data = {} # Initialize global variable

# --- Pre-load all historical data into memory ---
def _load_all_chart_data():
    """
    Loads all historical data from Excel files into a dictionary of DataFrames.
    This is done once at startup to improve performance.
    """
    global all_data # Declare all_data as global
    data = {}
    try:
        # Load TXF data
        txf_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'TXF.xlsx')
        txf_df = pd.read_excel(txf_file_path)
        txf_df['time'] = pd.to_datetime(txf_df['date']).dt.strftime('%Y-%m-%d')
        data['TXF'] = txf_df

        # Load Forex data
        forex_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Forex.xlsx')
        forex_df = pd.read_excel(forex_file_path)
        forex_df['time'] = pd.to_datetime(forex_df['datetime']).dt.strftime('%Y-%m-%d')
        
        # Split Forex data by symbol
        for symbol in forex_df['symbol'].unique():
            data[symbol] = forex_df[forex_df['symbol'] == symbol].copy()

        print("Successfully loaded all chart data into memory.")
    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}")
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
    all_data = data # Assign loaded data to the global all_data

# Load data at application startup
_load_all_chart_data()



# --- Chart Data Function ---
def get_chart_data(symbol: str) -> list:
    """
    Retrieves historical data from the pre-loaded in-memory cache.
    """
    print(f"DEBUG: Retrieving chart data for symbol: {symbol}")
    
    df = all_data.get(symbol.upper())
    
    if df is None or df.empty:
        print(f"DEBUG: No data found for symbol {symbol} in cache.")
        return []

    # Ensure required columns exist
    required_cols = ['time', 'open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        print(f"DEBUG: Missing required columns for {symbol}. Found: {df.columns.tolist()}")
        return []

    # Format for Lightweight Charts
    chart_data = df[required_cols].to_dict('records')

    # Get the latest prediction price for the symbol
    try:
        predictions_df = get_predictions()
        if not predictions_df.empty and 'symbol' in predictions_df.columns:
            symbol_prediction = predictions_df[predictions_df['symbol'] == symbol]
            if not symbol_prediction.empty and 'prediction_price' in symbol_prediction.columns:
                latest_prediction = symbol_prediction['prediction_price'].iloc[0]
                if chart_data and pd.notna(latest_prediction):
                    chart_data[-1]['prediction_price'] = latest_prediction
    except Exception as e:
        print(f"Error retrieving or applying prediction for {symbol}: {e}")

    return chart_data

# Initialize Flask App
app = Flask(__name__, template_folder='../templates', static_folder='../static')
CORS(app)
app.config['SECRET_KEY'] = 'your_secret_key_here'
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/api/reload_chart_data', methods=['POST'])
def reload_chart_data_endpoint():
    """
    API endpoint to trigger the hot-reloading of chart data.
    """
    global all_data
    try:
        _load_all_chart_data()
        socketio.emit('chart_data_updated', {'status': 'success', 'message': 'Chart data reloaded.'})
        return jsonify({'status': 'success', 'message': 'Chart data reloaded successfully.'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to reload chart data: {e}'}), 500

thread = None
thread_lock = threading.Lock()

def format_predictions_for_frontend(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    df = df.replace({np.nan: None})
    results = {}
    for _, row in df.iterrows():
        category = row.get('category', 'Other')
        symbol = row.get('symbol')
        sub_category = row.get('sub_category')
        if symbol:
            if category not in results:
                results[category] = {}
            data_payload = row.drop(['category', 'symbol'] + (['sub_category'] if 'sub_category' in row else [])).to_dict()
            if category == "Forex" and sub_category:
                if sub_category not in results[category]:
                    results[category][sub_category] = {}
                results[category][sub_category][symbol] = data_payload
            else:
                results[category][symbol] = data_payload
    return results

def background_thread():
    count = 0
    while True:
        socketio.sleep(10)
        count += 1
        print(f"Emitting data update: {count}")
        try:
            predictions_df = get_predictions()
            formatted_predictions = format_predictions_for_frontend(predictions_df)
            socketio.emit('data_update', formatted_predictions)
        except Exception as e:
            print(f"Error in background thread: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/symbols')
def get_symbols():
    """
    Returns a list of available symbols based on the .joblib models in the models directory.
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    symbols = []
    for filename in os.listdir(models_dir):
        if filename.endswith('_model.joblib'):
            # Extract symbol from filename, e.g., 'forex_AUDUSD_model.joblib' -> 'AUDUSD'
            symbol = filename.replace('forex_', '').replace('_model.joblib', '').upper()
            if symbol == 'TXF': # Handle TXF separately if needed
                symbols.append(symbol)
            elif len(symbol) == 6: # Assuming forex symbols are 6 characters (e.g., AUDUSD)
                symbols.append(symbol)
    return jsonify(sorted(symbols))

@app.route('/api/reload_models', methods=['POST'])
def reload_models_endpoint():
    """
    API endpoint to trigger the hot-reloading of AI models.
    """
    try:
        from services.prediction_service import reload_models
        reload_models()
        return jsonify({'status': 'success', 'message': 'Models reloaded successfully.'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to reload models: {e}'}), 500

@app.route('/chart_data')
def chart_data_route():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Symbol parameter is missing'}), 400
    
    chart_data = get_chart_data(symbol)
    if not chart_data:
        return jsonify({'error': f'No data found for symbol: {symbol}'}), 404
        
    return jsonify(chart_data)

@socketio.on('connect')
def test_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 5000)), app)