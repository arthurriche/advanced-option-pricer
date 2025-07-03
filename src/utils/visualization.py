"""
Visualization Utilities

Create charts and plots for option pricing analysis.

Author: Arthur Riche
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def plot_option_prices(strikes: List[float], prices: List[float], 
                      option_type: str = 'call', title: str = None) -> None:
    """
    Plot option prices vs strike prices.
    
    Args:
        strikes: List of strike prices
        prices: List of option prices
        option_type: 'call' or 'put'
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, prices, 'b-', linewidth=2, label=f'{option_type.title()} Option')
    plt.xlabel('Strike Price')
    plt.ylabel('Option Price')
    plt.title(title or f'{option_type.title()} Option Prices')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_vol_surface(grid_K: np.ndarray, grid_T: np.ndarray, 
                    grid_vol: np.ndarray, title: str = 'Volatility Surface') -> None:
    """
    Plot 3D volatility surface.
    
    Args:
        grid_K: Strike price grid
        grid_T: Time to expiration grid
        grid_vol: Volatility grid
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(grid_K, grid_T, grid_vol, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Time to Expiration')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(title)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()

def plot_greeks(strikes: List[float], greeks: dict, title: str = 'Option Greeks') -> None:
    """
    Plot option Greeks vs strike prices.
    
    Args:
        strikes: List of strike prices
        greeks: Dictionary with Greek values
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)
    
    # Delta
    axes[0, 0].plot(strikes, greeks['delta'], 'b-', linewidth=2)
    axes[0, 0].set_title('Delta')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gamma
    axes[0, 1].plot(strikes, greeks['gamma'], 'r-', linewidth=2)
    axes[0, 1].set_title('Gamma')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Theta
    axes[1, 0].plot(strikes, greeks['theta'], 'g-', linewidth=2)
    axes[1, 0].set_title('Theta')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Vega
    axes[1, 1].plot(strikes, greeks['vega'], 'm-', linewidth=2)
    axes[1, 1].set_title('Vega')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_payoff_diagram(strikes: List[float], payoffs: List[float], 
                       title: str = 'Option Payoff Diagram') -> None:
    """
    Plot option payoff diagram.
    
    Args:
        strikes: List of strike prices
        payoffs: List of payoffs
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, payoffs, 'b-', linewidth=2, label='Payoff')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=strikes[len(strikes)//2], color='r', linestyle='--', alpha=0.5, label='Current Price')
    plt.xlabel('Stock Price at Expiration')
    plt.ylabel('Payoff')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_hedging_pnl(time_steps: List[float], pnl: List[float], 
                    title: str = 'Hedging P&L') -> None:
    """
    Plot hedging P&L over time.
    
    Args:
        time_steps: List of time steps
        pnl: List of P&L values
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, pnl, 'b-', linewidth=2, label='P&L')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time')
    plt.ylabel('P&L')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_var_analysis(returns: np.ndarray, var_levels: List[float], 
                     title: str = 'VaR Analysis') -> None:
    """
    Plot VaR analysis with return distribution.
    
    Args:
        returns: Array of returns
        var_levels: List of VaR levels
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Plot return distribution
    plt.subplot(1, 2, 1)
    plt.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=np.percentile(returns, 5), color='red', linestyle='--', 
                label='95% VaR')
    plt.axvline(x=np.percentile(returns, 1), color='darkred', linestyle='--', 
                label='99% VaR')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.title('Return Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot VaR levels
    plt.subplot(1, 2, 2)
    confidence_levels = [0.95, 0.99]
    var_values = [np.percentile(returns, (1-cl)*100) for cl in confidence_levels]
    plt.bar(confidence_levels, var_values, color=['red', 'darkred'], alpha=0.7)
    plt.xlabel('Confidence Level')
    plt.ylabel('VaR')
    plt.title('VaR by Confidence Level')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_comparison_models(strikes: List[float], prices_dict: dict, 
                          title: str = 'Model Comparison') -> None:
    """
    Plot comparison of different pricing models.
    
    Args:
        strikes: List of strike prices
        prices_dict: Dictionary with model names and prices
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (model_name, prices) in enumerate(prices_dict.items()):
        plt.plot(strikes, prices, color=colors[i % len(colors)], 
                linewidth=2, label=model_name)
    
    plt.xlabel('Strike Price')
    plt.ylabel('Option Price')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show() 