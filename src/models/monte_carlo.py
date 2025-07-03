"""
Monte Carlo Option Pricing

Implements Monte Carlo simulation for option pricing with support for exotic options.

Author: Arthur Riche
Date: 2024
"""

import numpy as np
from typing import Literal, Callable, Optional

def monte_carlo_option_price(S: float, K: float, T: float, r: float, sigma: float,
                           n_paths: int, option_type: Literal['call', 'put'],
                           payoff_func: Optional[Callable] = None,
                           n_steps: int = 252) -> tuple:
    """
    Price options using Monte Carlo simulation.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        n_paths: Number of simulation paths
        option_type: 'call' or 'put'
        payoff_func: Custom payoff function (optional)
        n_steps: Number of time steps per path
        
    Returns:
        Tuple of (option_price, standard_error)
    """
    dt = T / n_steps
    
    # Generate random paths
    np.random.seed(42)  # For reproducibility
    Z = np.random.normal(0, 1, (n_paths, n_steps))
    
    # Simulate stock price paths
    S_paths = np.zeros((n_paths, n_steps + 1))
    S_paths[:, 0] = S
    
    for i in range(n_steps):
        S_paths[:, i + 1] = S_paths[:, i] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i]
        )
    
    # Calculate payoffs
    if payoff_func is not None:
        payoffs = payoff_func(S_paths, K)
    else:
        if option_type == 'call':
            payoffs = np.maximum(S_paths[:, -1] - K, 0)
        else:  # put
            payoffs = np.maximum(K - S_paths[:, -1], 0)
    
    # Discount payoffs
    discounted_payoffs = payoffs * np.exp(-r * T)
    
    # Calculate option price and standard error
    option_price = np.mean(discounted_payoffs)
    standard_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
    
    return option_price, standard_error

def asian_option_payoff(S_paths: np.ndarray, K: float, option_type: Literal['call', 'put']) -> np.ndarray:
    """Calculate payoff for Asian options (average price)."""
    avg_price = np.mean(S_paths, axis=1)
    if option_type == 'call':
        return np.maximum(avg_price - K, 0)
    else:
        return np.maximum(K - avg_price, 0)

def barrier_option_payoff(S_paths: np.ndarray, K: float, barrier: float, 
                         option_type: Literal['call', 'put'],
                         barrier_type: Literal['up', 'down']) -> np.ndarray:
    """Calculate payoff for barrier options."""
    # Check if barrier is hit
    if barrier_type == 'up':
        barrier_hit = np.any(S_paths >= barrier, axis=1)
    else:  # down
        barrier_hit = np.any(S_paths <= barrier, axis=1)
    
    # Calculate vanilla option payoff
    if option_type == 'call':
        vanilla_payoff = np.maximum(S_paths[:, -1] - K, 0)
    else:
        vanilla_payoff = np.maximum(K - S_paths[:, -1], 0)
    
    # Apply barrier condition
    return np.where(barrier_hit, vanilla_payoff, 0)

def lookback_option_payoff(S_paths: np.ndarray, K: float, option_type: Literal['call', 'put']) -> np.ndarray:
    """Calculate payoff for lookback options."""
    if option_type == 'call':
        max_price = np.max(S_paths, axis=1)
        return max_price - K
    else:
        min_price = np.min(S_paths, axis=1)
        return K - min_price

def monte_carlo_greeks(S: float, K: float, T: float, r: float, sigma: float,
                      n_paths: int, option_type: Literal['call', 'put']) -> dict:
    """
    Calculate Greeks using Monte Carlo with finite differences.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        n_paths: Number of simulation paths
        option_type: 'call' or 'put'
        
    Returns:
        Dictionary with Greeks
    """
    # Delta calculation
    dS = S * 0.01
    price_up, _ = monte_carlo_option_price(S + dS, K, T, r, sigma, n_paths, option_type)
    price_down, _ = monte_carlo_option_price(S - dS, K, T, r, sigma, n_paths, option_type)
    delta = (price_up - price_down) / (2 * dS)
    
    # Gamma calculation
    price_center, _ = monte_carlo_option_price(S, K, T, r, sigma, n_paths, option_type)
    gamma = (price_up + price_down - 2 * price_center) / (dS ** 2)
    
    # Vega calculation
    dsigma = sigma * 0.01
    price_vega_up, _ = monte_carlo_option_price(S, K, T, r, sigma + dsigma, n_paths, option_type)
    price_vega_down, _ = monte_carlo_option_price(S, K, T, r, sigma - dsigma, n_paths, option_type)
    vega = (price_vega_up - price_vega_down) / (2 * dsigma)
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega
    } 