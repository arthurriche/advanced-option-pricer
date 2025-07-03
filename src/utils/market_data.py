"""
Market Data Utilities

Fetch and process market data for option pricing.

Author: Arthur Riche
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

def get_sample_data(symbol: str = 'AAPL', start_date: str = '2020-01-01', 
                   end_date: str = '2024-01-01') -> pd.DataFrame:
    """
    Generate sample market data for testing.
    
    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Generate realistic price series
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.normal(0, 0.005, n_days)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        'Close': prices,
        'Volume': np.random.lognormal(10, 0.5, n_days)
    })
    
    data.set_index('Date', inplace=True)
    return data

def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns from price series.
    
    Args:
        prices: Price series
        
    Returns:
        Log returns
    """
    return np.log(prices / prices.shift(1))

def calculate_volatility(returns: pd.Series, window: int = 252) -> pd.Series:
    """
    Calculate rolling volatility.
    
    Args:
        returns: Return series
        window: Rolling window size
        
    Returns:
        Rolling volatility
    """
    return returns.rolling(window=window).std() * np.sqrt(252)

def get_risk_free_rate() -> float:
    """
    Get current risk-free rate (simplified).
    
    Returns:
        Risk-free rate
    """
    return 0.05  # 5% for demonstration

def get_option_chain(symbol: str, expiration_date: str) -> pd.DataFrame:
    """
    Generate sample option chain data.
    
    Args:
        symbol: Stock symbol
        expiration_date: Option expiration date
        
    Returns:
        DataFrame with option chain
    """
    # Generate sample strikes around current price
    current_price = 100
    strikes = np.arange(current_price * 0.7, current_price * 1.3, 5)
    
    # Generate sample option prices
    call_prices = []
    put_prices = []
    
    for strike in strikes:
        # Simplified option pricing
        moneyness = current_price / strike
        time_to_expiry = 0.25  # 3 months
        
        if moneyness > 1:  # In-the-money call
            call_price = current_price - strike + 2
        else:  # Out-of-the-money call
            call_price = max(0.1, 2 * (1 - moneyness))
        
        if moneyness < 1:  # In-the-money put
            put_price = strike - current_price + 2
        else:  # Out-of-the-money put
            put_price = max(0.1, 2 * (moneyness - 1))
        
        call_prices.append(call_price)
        put_prices.append(put_price)
    
    option_chain = pd.DataFrame({
        'Strike': strikes,
        'Call_Price': call_prices,
        'Put_Price': put_prices,
        'Call_Volume': np.random.randint(100, 1000, len(strikes)),
        'Put_Volume': np.random.randint(100, 1000, len(strikes))
    })
    
    return option_chain 