"""
Market Data Loader

Provides functions to fetch and preprocess market data for ML trading.

Author: Arthur Riche
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class MarketDataLoader:
    """Load and preprocess market data for machine learning models."""
    
    def __init__(self):
        self.data_cache = {}
    
    def load_data(self, symbol: str, start_date: str = '2020-01-01', 
                  end_date: str = '2024-01-01', interval: str = '1d') -> pd.DataFrame:
        """
        Load market data for a given symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval ('1d', '1h', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        # For now, generate synthetic data
        return self._generate_synthetic_data(start_date, end_date)
    
    def _generate_synthetic_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic market data for testing."""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # Generate realistic price series
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, n_days)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
            'Close': prices,
            'Volume': np.random.lognormal(10, 0.5, n_days)
        }, index=dates)
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data."""
        # Moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # RSI
        data['RSI'] = self._calculate_rsi(data['Close'])
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Bollinger Bands
        data['BB_Upper'] = data['SMA_20'] + 2 * data['Close'].rolling(window=20).std()
        data['BB_Lower'] = data['SMA_20'] - 2 * data['Close'].rolling(window=20).std()
        data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['SMA_20']
        
        # Volatility
        data['Volatility'] = data['Close'].pct_change().rolling(window=20).std()
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_multiple_symbols(self, symbols: List[str], start_date: str = '2020-01-01',
                           end_date: str = '2024-01-01') -> Dict[str, pd.DataFrame]:
        """Load data for multiple symbols."""
        data_dict = {}
        for symbol in symbols:
            data_dict[symbol] = self.load_data(symbol, start_date, end_date)
        return data_dict
    
    def preprocess_for_ml(self, data: pd.DataFrame, target_column: str = 'Close',
                         sequence_length: int = 60) -> tuple:
        """
        Preprocess data for machine learning models.
        
        Args:
            data: Input DataFrame
            target_column: Column to predict
            sequence_length: Number of time steps for sequences
            
        Returns:
            tuple: (X, y) where X is sequences and y is targets
        """
        # Remove NaN values
        data_clean = data.dropna()
        
        # Normalize features
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 
                          'SMA_50', 'RSI', 'MACD', 'Volatility', 'Volume_Ratio']
        
        # Use only available columns
        available_features = [col for col in feature_columns if col in data_clean.columns]
        
        # Normalize
        data_normalized = (data_clean[available_features] - data_clean[available_features].mean()) / data_clean[available_features].std()
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(data_normalized)):
            X.append(data_normalized.iloc[i-sequence_length:i].values)
            y.append(data_clean[target_column].iloc[i])
        
        return np.array(X), np.array(y) 