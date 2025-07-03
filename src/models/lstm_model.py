"""
LSTM Trading Model

This module implements LSTM-based models for time series prediction
and trading signal generation in financial markets.

Author: Arthur Riche
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import warnings


class LSTMTradingModel:
    """
    LSTM-based trading model for time series prediction and signal generation.
    
    This class implements a Long Short-Term Memory neural network for
    predicting financial time series and generating trading signals.
    """
    
    def __init__(self, sequence_length: int = 60, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2, 
                 learning_rate: float = 0.001):
        """
        Initialize the LSTM trading model.
        
        Args:
            sequence_length: Number of time steps to look back
            hidden_size: Number of hidden units in LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for optimization
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Model parameters (simplified implementation)
        self.weights = {}
        self.biases = {}
        self.is_trained = False
        
        # Initialize model parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize model weights and biases."""
        # Simplified parameter initialization
        # In a real implementation, this would use TensorFlow or PyTorch
        
        # Input layer to first LSTM layer
        self.weights['input'] = np.random.randn(self.sequence_length, self.hidden_size) * 0.01
        self.biases['input'] = np.zeros(self.hidden_size)
        
        # LSTM layers
        for i in range(self.num_layers):
            layer_name = f'lstm_{i}'
            if i == 0:
                input_size = self.sequence_length
            else:
                input_size = self.hidden_size
            
            # LSTM gates: input, forget, cell, output
            self.weights[f'{layer_name}_i'] = np.random.randn(input_size, self.hidden_size) * 0.01
            self.weights[f'{layer_name}_f'] = np.random.randn(input_size, self.hidden_size) * 0.01
            self.weights[f'{layer_name}_c'] = np.random.randn(input_size, self.hidden_size) * 0.01
            self.weights[f'{layer_name}_o'] = np.random.randn(input_size, self.hidden_size) * 0.01
            
            self.biases[f'{layer_name}_i'] = np.zeros(self.hidden_size)
            self.biases[f'{layer_name}_f'] = np.zeros(self.hidden_size)
            self.biases[f'{layer_name}_c'] = np.zeros(self.hidden_size)
            self.biases[f'{layer_name}_o'] = np.zeros(self.hidden_size)
        
        # Output layer
        self.weights['output'] = np.random.randn(self.hidden_size, 1) * 0.01
        self.biases['output'] = np.zeros(1)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function."""
        return np.tanh(x)
    
    def _lstm_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through LSTM layers.
        
        Args:
            x: Input sequence of shape (sequence_length, features)
            
        Returns:
            np.ndarray: Output from the last LSTM layer
        """
        # Simplified LSTM implementation
        # In practice, use TensorFlow or PyTorch for efficiency
        
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        
        for t in range(self.sequence_length):
            # Input gate
            i_t = self._sigmoid(np.dot(x[t], self.weights['lstm_0_i']) + self.biases['lstm_0_i'])
            
            # Forget gate
            f_t = self._sigmoid(np.dot(x[t], self.weights['lstm_0_f']) + self.biases['lstm_0_f'])
            
            # Cell state
            c_tilde = self._tanh(np.dot(x[t], self.weights['lstm_0_c']) + self.biases['lstm_0_c'])
            c = f_t * c + i_t * c_tilde
            
            # Output gate
            o_t = self._sigmoid(np.dot(x[t], self.weights['lstm_0_o']) + self.biases['lstm_0_o'])
            h = o_t * self._tanh(c)
        
        return h
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training/prediction.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            tuple: (X, y) where X is sequences and y is targets
        """
        # Normalize the data
        data_normalized = (data - data.mean()) / data.std()
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(data_normalized)):
            X.append(data_normalized.iloc[i-self.sequence_length:i].values)
            y.append(data_normalized.iloc[i].values[0])  # Predict next price
        
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame, epochs: int = 100, batch_size: int = 32, 
              validation_split: float = 0.2) -> Dict[str, List[float]]:
        """
        Train the LSTM model.
        
        Args:
            data: Training data DataFrame
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            
        Returns:
            dict: Training history with loss and validation loss
        """
        # Prepare data
        X, y = self._prepare_data(data)
        
        # Split into training and validation sets
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        history = {'loss': [], 'val_loss': []}
        
        # Simplified training loop
        # In practice, use TensorFlow or PyTorch optimizers
        for epoch in range(epochs):
            # Forward pass
            train_predictions = []
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_predictions = []
                
                for x in batch_X:
                    h = self._lstm_forward(x)
                    pred = np.dot(h, self.weights['output']) + self.biases['output']
                    batch_predictions.append(pred[0])
                
                train_predictions.extend(batch_predictions)
            
            # Calculate loss (MSE)
            train_loss = np.mean((np.array(train_predictions) - y_train[:len(train_predictions)])**2)
            
            # Validation
            val_predictions = []
            for x in X_val:
                h = self._lstm_forward(x)
                pred = np.dot(h, self.weights['output']) + self.biases['output']
                val_predictions.append(pred[0])
            
            val_loss = np.mean((np.array(val_predictions) - y_val)**2)
            
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        self.is_trained = True
        return history
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X, _ = self._prepare_data(data)
        predictions = []
        
        for x in X:
            h = self._lstm_forward(x)
            pred = np.dot(h, self.weights['output']) + self.biases['output']
            predictions.append(pred[0])
        
        return np.array(predictions)
    
    def generate_trading_signals(self, data: pd.DataFrame, 
                                threshold: float = 0.02) -> pd.Series:
        """
        Generate trading signals based on model predictions.
        
        Args:
            data: DataFrame with price data
            threshold: Threshold for signal generation
            
        Returns:
            pd.Series: Trading signals (1 for buy, -1 for sell, 0 for hold)
        """
        predictions = self.predict(data)
        
        # Calculate price changes
        price_changes = data.iloc[self.sequence_length:].pct_change().values.flatten()
        
        # Generate signals based on predicted vs actual changes
        signals = np.zeros(len(predictions))
        
        for i in range(len(predictions)):
            if i < len(price_changes):
                predicted_change = predictions[i] - data.iloc[self.sequence_length + i - 1].values[0]
                actual_change = price_changes[i]
                
                # Buy signal if predicted change is significantly positive
                if predicted_change > threshold:
                    signals[i] = 1
                # Sell signal if predicted change is significantly negative
                elif predicted_change < -threshold:
                    signals[i] = -1
        
        return pd.Series(signals, index=data.index[self.sequence_length:])
    
    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            data: Test data DataFrame
            
        Returns:
            dict: Performance metrics
        """
        predictions = self.predict(data)
        actual = data.iloc[self.sequence_length:].values.flatten()
        
        # Calculate metrics
        mse = np.mean((predictions - actual)**2)
        mae = np.mean(np.abs(predictions - actual))
        rmse = np.sqrt(mse)
        
        # Calculate directional accuracy
        pred_direction = np.diff(predictions) > 0
        actual_direction = np.diff(actual) > 0
        directional_accuracy = np.mean(pred_direction == actual_direction)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
    
    def save_model(self, filepath: str):
        """Save model parameters to file."""
        import pickle
        
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load model parameters from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.sequence_length = model_data['sequence_length']
        self.hidden_size = model_data['hidden_size']
        self.num_layers = model_data['num_layers']
        self.dropout = model_data['dropout']
        self.learning_rate = model_data['learning_rate']
        self.is_trained = model_data['is_trained']


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.02)
    data = pd.DataFrame({'price': prices}, index=dates)
    
    # Initialize and train model
    model = LSTMTradingModel(sequence_length=30, hidden_size=64, num_layers=1)
    
    print("Training LSTM model...")
    history = model.train(data, epochs=50, batch_size=16)
    
    # Generate predictions and signals
    predictions = model.predict(data)
    signals = model.generate_trading_signals(data, threshold=0.01)
    
    # Evaluate model
    metrics = model.evaluate(data)
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nGenerated {len(signals[signals != 0])} trading signals")
    print(f"Buy signals: {len(signals[signals == 1])}")
    print(f"Sell signals: {len(signals[signals == -1])}") 