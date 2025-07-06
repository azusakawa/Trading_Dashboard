"""
This module centralizes all configuration settings for the application.
It includes sensitive API keys, font paths, lists of financial symbols,
logging configurations, and parameters for AI model training and selection.

Sensitive information like API keys are loaded from environment variables for security best practices.
"""

import os

# --- API Key Settings ---
# API keys are loaded from environment variables for security.
# If environment variables are not set, default placeholder values are used.
# IMPORTANT: Replace default values with actual keys or ensure environment variables are set in production.
TD_API_KEY = os.environ.get("TD_API_KEY", "YOUR_TD_API_KEY_HERE") # TwelveData API Key
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "YOUR_FINMIND_TOKEN_HERE") # FinMind API Token

# --- Font Path Settings ---
# Paths to font files used for rendering text, especially for charts or UI elements.
# It's recommended to set these via environment variables in production environments
# or ensure the paths are valid on the deployment system.
# The application will attempt to find suitable font paths; if not found, a default font will be used.
FONT_PATH_REGULAR = os.environ.get("FONT_PATH_REGULAR", "") # e.g., "C:/Windows/Fonts/msjh.ttc" or "/System/Library/Fonts/PingFang.ttc"
FONT_PATH_BOLD = os.environ.get("FONT_PATH_BOLD", "") # e.g., "C:/Windows/Fonts/msjhbd.ttc" or "/System/Library/Fonts/PingFang.ttc"

# --- Forex Symbols ---
# A list of forex currency pairs for which data will be fetched and models will be trained.
# This list typically does not need to be loaded from environment variables.
SYMBOLS_COLS = [
    'AUDCAD','AUDCHF','AUDJPY','AUDNZD','AUDUSD','CADCHF','CADJPY','CHFJPY','EURAUD','EURCAD','EURCHF',
    'EURGBP','EURJPY','EURNZD','EURUSD','GBPAUD','GBPCAD','GBPCHF','GBPJPY','GBPNZD','GBPUSD','USDCAD',
    'USDCHF','USDJPY','NZDCAD','NZDCHF','NZDJPY','NZDUSD'
]

# --- Logging Configuration ---
# LOG_LEVEL: The minimum level of messages to log (e.g., INFO, DEBUG, WARNING, ERROR).
#            Defaults to INFO, can be overridden by the LOG_LEVEL environment variable.
# LOG_FORMAT: The format string for log messages.
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.environ.get("LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# --- Model Training Parameters ---
# MLP_HIDDEN_LAYER_SIZES: A tuple defining the number of neurons in each hidden layer of the MLPClassifier.
#                         Loaded from environment variable as a comma-separated string (e.g., "100,50").
# MLP_MAX_ITER: The maximum number of iterations for the MLPClassifier to converge.
# PLR_THRESHOLD: The threshold for Piecewise Linear Representation (PLR) segmentation.
#                Used to identify significant turning points in price data for generating target signals.
# FEATURE_SELECTOR_N_FEATURES: The number of top features to select during feature selection.
# FEATURE_SELECTION_METHOD: The method used for feature selection ('sfs' for SequentialFeatureSelector or 'rf_importance' for RandomForest importance).
MLP_HIDDEN_LAYER_SIZES = tuple(map(int, os.environ.get("MLP_HIDDEN_LAYER_SIZES", "10").split(",")))
MLP_MAX_ITER = int(os.environ.get("MLP_MAX_ITER", "1000"))
PLR_THRESHOLD = float(os.environ.get("PLR_THRESHOLD", "0.001"))
FEATURE_SELECTOR_N_FEATURES = int(os.environ.get("FEATURE_SELECTOR_N_FEATURES", "4"))
FEATURE_SELECTION_METHOD = os.environ.get("FEATURE_SELECTION_METHOD", "sfs") # sfs or rf_importance

# --- Model Selection ---
# MODEL_TYPE: Specifies the type of AI model to be trained ('MLP' for MLPClassifier or 'RandomForest' for RandomForestClassifier).
MODEL_TYPE = os.environ.get("MODEL_TYPE", "MLP") # MLP or RandomForest

# --- Hyperparameter Tuning Settings ---
# PERFORM_HYPERPARAMETER_TUNING: A boolean flag to enable or disable hyperparameter tuning for models.
#                                Loaded from environment variable (e.g., "True" or "False").
# MLP_PARAM_GRID: A dictionary defining the hyperparameter grid for GridSearchCV when tuning MLPClassifier.
PERFORM_HYPERPARAMETER_TUNING = os.environ.get("PERFORM_HYPERPARAMETER_TUNING", "False").lower() == "true"
MLP_PARAM_GRID = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001]
}