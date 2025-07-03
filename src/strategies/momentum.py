"""
Momentum Trading Strategies

Implements momentum-based trading strategies with ML confirmation.

Author: Arthur Riche
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class MomentumStrategy:
    """Momentum trading strategy with ML confirmation."""
    
    def __init__(self, lookback_period: int = 20, threshold: float = 0.02):
        self.lookback_period = lookback_period
        self.threshold = threshold
    
    def calculate_momentum(self, prices: pd.Series) -> pd.Series:
        """Calculate momentum indicator."""
        return prices.pct_change(self.lookback_period)
    
    def generate_signals(self, data: pd.DataFrame, ml_predictions: np.ndarray = None) -> pd.Series:
        """
        Generate trading signals based on momentum and ML predictions.
        
        Args:
            data: Market data with technical indicators
            ml_predictions: ML model predictions (optional)
            
        Returns:
            Series with trading signals (1: buy, -1: sell, 0: hold)
        """
        signals = pd.Series(0, index=data.index)
        
        # Calculate momentum
        momentum = self.calculate_momentum(data['Close'])
        
        # Generate signals based on momentum
        signals[momentum > self.threshold] = 1  # Buy signal
        signals[momentum < -self.threshold] = -1  # Sell signal
        
        # If ML predictions are available, use them for confirmation
        if ml_predictions is not None:
            # Align predictions with data
            pred_returns = np.diff(ml_predictions) / ml_predictions[:-1]
            pred_signals = np.sign(pred_returns)
            
            # Combine momentum and ML signals
            for i in range(len(signals)):
                if i < len(pred_signals):
                    if signals.iloc[i] == 1 and pred_signals[i] > 0:
                        signals.iloc[i] = 1  # Confirm buy
                    elif signals.iloc[i] == -1 and pred_signals[i] < 0:
                        signals.iloc[i] = -1  # Confirm sell
                    else:
                        signals.iloc[i] = 0  # No confirmation
        
        return signals
    
    def backtest(self, data: pd.DataFrame, signals: pd.Series, 
                 initial_capital: float = 100000) -> Dict:
        """
        Backtest the momentum strategy.
        
        Args:
            data: Market data
            signals: Trading signals
            initial_capital: Initial capital
            
        Returns:
            Dictionary with backtest results
        """
        capital = initial_capital
        position = 0
        trades = []
        
        for i in range(1, len(data)):
            current_price = data['Close'].iloc[i]
            signal = signals.iloc[i]
            
            if signal == 1 and position <= 0:  # Buy signal
                if position < 0:  # Close short position
                    capital += position * current_price
                position = capital / current_price
                trades.append({
                    'date': data.index[i],
                    'action': 'buy',
                    'price': current_price,
                    'capital': capital
                })
            
            elif signal == -1 and position >= 0:  # Sell signal
                if position > 0:  # Close long position
                    capital = position * current_price
                position = -capital / current_price
                trades.append({
                    'date': data.index[i],
                    'action': 'sell',
                    'price': current_price,
                    'capital': capital
                })
        
        # Close final position
        if position != 0:
            capital = position * data['Close'].iloc[-1]
        
        # Calculate performance metrics
        returns = pd.Series([trade['capital'] for trade in trades]).pct_change()
        
        results = {
            'final_capital': capital,
            'total_return': (capital - initial_capital) / initial_capital,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0,
            'max_drawdown': self._calculate_max_drawdown(trades),
            'num_trades': len(trades),
            'trades': trades
        }
        
        return results
    
    def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown from trades."""
        if not trades:
            return 0
        
        capitals = [trade['capital'] for trade in trades]
        peak = capitals[0]
        max_dd = 0
        
        for capital in capitals:
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd 