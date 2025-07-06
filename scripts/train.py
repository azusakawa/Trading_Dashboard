"""
This script provides a structured, class-based approach for training and saving AI models
for both Taiwan Futures (TXF) and Forex data. It encapsulates the training logic within
a `ModelTrainer` class and orchestrates the overall process using a `TrainingPipeline` class.

Key Classes:
- ModelTrainer: Handles the end-to-end process for a single model, including data preparation,
  feature selection, training, evaluation, and artifact saving.
- TrainingPipeline: Manages the overall workflow, fetching data and initiating training
  for TXF and each Forex symbol.
"""

import pandas as pd
import numpy as np
import sys
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# Add project root to sys.path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.feature_utils as feature_utils
from data.data_updater import update_futures_data, update_forex_data
try:
    from src.config import (
        MLP_HIDDEN_LAYER_SIZES, MLP_MAX_ITER, PLR_THRESHOLD,
        FEATURE_SELECTOR_N_FEATURES, FEATURE_SELECTION_METHOD, MODEL_TYPE,
        PERFORM_HYPERPARAMETER_TUNING, MLP_PARAM_GRID, SYMBOLS_COLS
    )
except ImportError:
    print("❌ Error: config.py file not found or missing required settings.")
    exit()

# --- Constants ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


class ModelTrainer:
    """
    Encapsulates the logic for training, evaluating, and saving a single model.
    """
    def __init__(self, data_df: pd.DataFrame, mode: str, specific_symbol: str = None):
        """
        Initializes the ModelTrainer.

        Args:
            data_df (pd.DataFrame): The input DataFrame with historical price data.
            mode (str): The type of model, either "txf" or "forex".
            specific_symbol (str, optional): The specific symbol, required for forex.
        """
        self.df = data_df.copy()
        self.mode = mode
        self.specific_symbol = specific_symbol
        self.model_name = f"{self.mode}{f'_{self.specific_symbol}' if self.specific_symbol else ''}"
        
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.selected_features = []
        self.scaler = None
        self.model = None

    def _prepare_data(self):
        """
        Performs feature engineering and generates the target variable.
        """
        print(f"[{self.model_name}] Step 1/5: Preparing data and calculating features...")
        
        self.df = feature_utils.calculate_features(self.df)
        if self.df.empty or 'close' not in self.df.columns or self.df['close'].isnull().all():
            raise ValueError("Data is empty or 'close' column is missing after feature calculation.")

        # Generate target signal using Piecewise Linear Representation (PLR)
        points = self._plr_segmentation(self.df['close'].values, threshold=PLR_THRESHOLD)
        self.df['target_signal'] = self._generate_trade_signals(points, len(self.df))

        # Drop rows with NaN values
        required_cols = ['rsi', 'macd', 'ema', 'cci', 'adx', 'stoch_k', 'target_signal']
        self.df.dropna(subset=required_cols, inplace=True)
        if self.df.empty:
            raise ValueError("Insufficient data after dropping NaNs.")

    def _select_features(self):
        """
        Performs feature selection based on the configured method.
        """
        print(f"[{self.model_name}] Step 2/5: Performing feature selection...")
        
        feature_pool = [
            'rsi', 'macd', 'ema', 'cci', 'adx', 'stoch_k', 'macd_hist', 'atr', 
            'bband_h', 'bband_l', 'bband_m', 'ichimoku_conv', 'ichimoku_base', 
            'ichimoku_a', 'ichimoku_b', 'hl_pct_range', 'plus_di', 'minus_di'
        ]
        # Filter out columns that are not in the dataframe
        feature_pool = [col for col in feature_pool if col in self.df.columns]
        
        X_all = self.df[feature_pool].dropna(axis=1)
        y_all = self.df['target_signal']

        if X_all.empty:
            raise ValueError("Feature set is empty after preparation.")

        if FEATURE_SELECTION_METHOD == 'sfs':
            lr = LinearRegression()
            sfs = SequentialFeatureSelector(lr, n_features_to_select=FEATURE_SELECTOR_N_FEATURES, direction='forward', n_jobs=-1)
            sfs.fit(X_all, y_all)
            self.selected_features = list(X_all.columns[sfs.get_support()])
        elif FEATURE_SELECTION_METHOD == 'rf_importance':
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_all, y_all)
            importances = pd.Series(rf.feature_importances_, index=X_all.columns)
            self.selected_features = importances.nlargest(FEATURE_SELECTOR_N_FEATURES).index.tolist()
        else:
            raise ValueError(f"Invalid feature selection method: {FEATURE_SELECTION_METHOD}")
            
        print(f"[{self.model_name}] Selected features: {self.selected_features}")

    def _train_model(self):
        """
        Splits data, scales features, and trains the model with optional hyperparameter tuning.
        """
        print(f"[{self.model_name}] Step 3/5: Training the model...")
        
        X = self.df[self.selected_features]
        y = self.df['target_signal']
        
        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        self.scaler = MinMaxScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

        # Initialize model based on type
        if MODEL_TYPE == "MLP":
            base_model = MLPClassifier(max_iter=MLP_MAX_ITER, random_state=42, early_stopping=True)
            params = {'hidden_layer_sizes': MLP_HIDDEN_LAYER_SIZES}
        elif MODEL_TYPE == "RandomForest":
            base_model = RandomForestClassifier(random_state=42)
            params = {'n_estimators': 100}
        else:
            raise ValueError(f"Invalid model type specified: {MODEL_TYPE}")

        # Cross-validation
        cv_scores = cross_val_score(base_model.set_params(**params), self.X_train_scaled, self.y_train, cv=5, scoring='accuracy')
        print(f"[{self.model_name}] Cross-validation accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

        # Hyperparameter tuning or standard training
        if PERFORM_HYPERPARAMETER_TUNING and MODEL_TYPE == "MLP":
            print(f"[{self.model_name}] Performing hyperparameter tuning for MLP...")
            grid_search = GridSearchCV(base_model, MLP_PARAM_GRID, cv=3, n_jobs=-1, verbose=1)
            grid_search.fit(self.X_train_scaled, self.y_train)
            self.model = grid_search.best_estimator_
            print(f"[{self.model_name}] Best MLP parameters: {grid_search.best_params_}")
        else:
            self.model = base_model.set_params(**params)
            self.model.fit(self.X_train_scaled, self.y_train)

    def _evaluate_model(self):
        """
        Evaluates the trained model on the test set and prints performance metrics.
        """
        print(f"[{self.model_name}] Step 4/5: Evaluating model performance...")
        
        accuracy = self.model.score(self.X_test_scaled, self.y_test)
        y_pred = self.model.predict(self.X_test_scaled)
        y_prob = self.model.predict_proba(self.X_test_scaled)[:, 1]

        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        # Handle cases with only one class in y_test for roc_auc_score
        if len(np.unique(self.y_test)) > 1:
            auc = roc_auc_score(self.y_test, y_prob, average='weighted')
        else:
            auc = float('nan') # Not applicable

        print(f"[{self.model_name}] Test Set Performance:")
        print(f"  Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, AUC-ROC: {auc:.2f}")

    def _save_artifacts(self):
        """
        Saves the trained model, scaler, and selected features to a joblib file.
        """
        model_filename = os.path.join(MODEL_DIR, f"{self.model_name}_model.joblib")
        print(f"[{self.model_name}] Step 5/5: Saving artifacts to {model_filename}...")
        
        artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'selected_features': self.selected_features
        }
        joblib.dump(artifacts, model_filename)
        print(f"✅ [{self.model_name}] Artifacts successfully saved!")

    def run(self):
        """
        Executes the full training pipeline for the model.
        """
        try:
            self._prepare_data()
            self._select_features()
            self._train_model()
            self._evaluate_model()
            self._save_artifacts()
        except ValueError as e:
            print(f"❌ Error training model {self.model_name}: {e}")
            return False
        return True

    @staticmethod
    def _plr_segmentation(price_series: np.ndarray, threshold: float) -> list[int]:
        """Performs Piecewise Linear Representation (PLR)."""
        points = [0]
        for i in range(1, len(price_series) - 1):
            prev = price_series[points[-1]]
            curr = price_series[i]
            if abs(curr - prev) / prev > threshold:
                points.append(i)
        points.append(len(price_series) - 1)
        return points

    @staticmethod
    def _generate_trade_signals(breakpoints: list[int], length: int) -> np.ndarray:
        """Generates a binary trade signal array from breakpoints."""
        signals = np.zeros(length)
        signals[breakpoints] = 1
        return signals


class TrainingPipeline:
    """
    Orchestrates the entire model training pipeline, including data fetching
    and managing individual model trainers.
    """
    def _run_txf_training(self):
        """
        Runs the training process for the TXF model.
        """
        print("--- Training Futures Model (TXF) ---")
        txf_data = update_futures_data()
        if txf_data is not None and not txf_data.empty:
            trainer = ModelTrainer(txf_data, mode="txf")
            trainer.run()
        else:
            print("Skipping TXF training due to data fetching failure.")

    def _run_forex_training(self):
        """
        Runs the training process for all Forex models.
        """
        print("--- Training Forex Models ---")
        forex_data = update_forex_data()
        if forex_data is not None and not forex_data.empty:
            for symbol in SYMBOLS_COLS:
                symbol_df = forex_data[forex_data['symbol'] == symbol]
                if not symbol_df.empty:
                    print(f"\n--- Training Forex Model: {symbol} ---")
                    trainer = ModelTrainer(symbol_df, mode="forex", specific_symbol=symbol)
                    trainer.run()
                else:
                    print(f"\n--- Skipping Forex Model: {symbol} (no data) ---")
        else:
            print("Skipping Forex training due to data fetching failure.")

    def run(self):
        """
        Executes the full training pipeline for all models.
        """
        self._run_txf_training()
        self._run_forex_training()
        print("\n--- All training pipelines finished. ---")


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()