"""
Binomial Tree Option Pricing

Implements binomial tree method for option pricing with support for American options.

Author: Arthur Riche
Date: 2024
"""

import numpy as np
from typing import Literal

def binomial_tree_price(S: float, K: float, T: float, r: float, sigma: float, 
                       N: int, option_type: Literal['call', 'put'], 
                       american: bool = False) -> float:
    """
    Price options using binomial tree method.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        N: Number of time steps
        option_type: 'call' or 'put'
        american: True for American options, False for European
        
    Returns:
        Option price
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Create stock price tree
    stock_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock_tree[i, j] = S * (u ** (i - j)) * (d ** j)
    
    # Create option value tree
    option_tree = np.zeros((N + 1, N + 1))
    
    # Calculate terminal payoffs
    for j in range(N + 1):
        if option_type == 'call':
            option_tree[N, j] = max(stock_tree[N, j] - K, 0)
        else:  # put
            option_tree[N, j] = max(K - stock_tree[N, j], 0)
    
    # Backward induction
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            # Risk-neutral valuation
            option_tree[i, j] = np.exp(-r * dt) * (
                p * option_tree[i + 1, j] + (1 - p) * option_tree[i + 1, j + 1]
            )
            
            # Early exercise for American options
            if american:
                if option_type == 'call':
                    intrinsic_value = max(stock_tree[i, j] - K, 0)
                else:  # put
                    intrinsic_value = max(K - stock_tree[i, j], 0)
                option_tree[i, j] = max(option_tree[i, j], intrinsic_value)
    
    return option_tree[0, 0]

def binomial_tree_greeks(S: float, K: float, T: float, r: float, sigma: float,
                        N: int, option_type: Literal['call', 'put']) -> dict:
    """
    Calculate Greeks using finite differences on binomial tree.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        N: Number of time steps
        option_type: 'call' or 'put'
        
    Returns:
        Dictionary with Greeks
    """
    # Delta calculation
    dS = S * 0.01
    price_up = binomial_tree_price(S + dS, K, T, r, sigma, N, option_type)
    price_down = binomial_tree_price(S - dS, K, T, r, sigma, N, option_type)
    delta = (price_up - price_down) / (2 * dS)
    
    # Gamma calculation
    price_center = binomial_tree_price(S, K, T, r, sigma, N, option_type)
    gamma = (price_up + price_down - 2 * price_center) / (dS ** 2)
    
    # Theta calculation
    dt = T * 0.01
    price_theta = binomial_tree_price(S, K, T - dt, r, sigma, N, option_type)
    theta = (price_theta - price_center) / dt
    
    # Vega calculation
    dsigma = sigma * 0.01
    price_vega_up = binomial_tree_price(S, K, T, r, sigma + dsigma, N, option_type)
    price_vega_down = binomial_tree_price(S, K, T, r, sigma - dsigma, N, option_type)
    vega = (price_vega_up - price_vega_down) / (2 * dsigma)
    
    # Rho calculation
    dr = r * 0.01
    price_rho_up = binomial_tree_price(S, K, T, r + dr, sigma, N, option_type)
    price_rho_down = binomial_tree_price(S, K, T, r - dr, sigma, N, option_type)
    rho = (price_rho_up - price_rho_down) / (2 * dr)
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    } 