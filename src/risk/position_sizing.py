"""
Position Sizing Module

Implements Kelly Criterion and Risk Parity for position sizing.

Author: Arthur Riche
Date: 2024
"""

import numpy as np
import pandas as pd

def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate Kelly Criterion optimal position size.
    
    Args:
        win_rate: Probability of winning
        avg_win: Average win amount
        avg_loss: Average loss amount
        
    Returns:
        Optimal fraction of capital to risk
    """
    if avg_loss == 0:
        return 0
    
    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    
    # Constrain to reasonable bounds
    return max(0, min(kelly_fraction, 0.25))

def risk_parity_weights(returns: pd.DataFrame, target_vol: float = 0.15) -> np.ndarray:
    """
    Calculate Risk Parity weights.
    
    Args:
        returns: DataFrame of asset returns
        target_vol: Target portfolio volatility
        
    Returns:
        Array of weights
    """
    # Calculate covariance matrix
    cov_matrix = returns.cov()
    
    # Calculate inverse volatility weights
    vols = np.sqrt(np.diag(cov_matrix))
    inv_vol_weights = 1 / vols
    inv_vol_weights = inv_vol_weights / np.sum(inv_vol_weights)
    
    # Scale to target volatility
    portfolio_vol = np.sqrt(inv_vol_weights.T @ cov_matrix @ inv_vol_weights)
    scaled_weights = inv_vol_weights * (target_vol / portfolio_vol)
    
    return scaled_weights

def calculate_position_size(capital: float, risk_per_trade: float, 
                          stop_loss_pct: float) -> float:
    """
    Calculate position size based on risk per trade.
    
    Args:
        capital: Available capital
        risk_per_trade: Maximum risk per trade (as fraction of capital)
        stop_loss_pct: Stop loss percentage
        
    Returns:
        Position size in currency units
    """
    risk_amount = capital * risk_per_trade
    position_size = risk_amount / stop_loss_pct
    return position_size 