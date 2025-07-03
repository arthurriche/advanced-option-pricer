"""
Volatility Utilities

Calculate and fit volatility surfaces and implied volatility.

Author: Arthur Riche
Date: 2024
"""

import numpy as np
from typing import Tuple, List

def implied_volatility(price: float, S: float, K: float, T: float, r: float, 
                      option_type: str = 'call', tolerance: float = 1e-6, 
                      max_iterations: int = 100) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Args:
        price: Option price
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        option_type: 'call' or 'put'
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
        
    Returns:
        Implied volatility
    """
    # Initial guess
    sigma = 0.3
    
    for i in range(max_iterations):
        # Calculate option price with current volatility
        if option_type == 'call':
            option_price = call_price(S, K, T, r, sigma)
            vega = call_vega(S, K, T, r, sigma)
        else:  # put
            option_price = put_price(S, K, T, r, sigma)
            vega = put_vega(S, K, T, r, sigma)
        
        # Calculate difference
        diff = price - option_price
        
        # Check convergence
        if abs(diff) < tolerance:
            return sigma
        
        # Update volatility using Newton-Raphson
        sigma = sigma + diff / vega
        
        # Ensure volatility is positive
        sigma = max(sigma, 0.001)
    
    return sigma

def volatility_surface(strikes: List[float], maturities: List[float], 
                      S: float, r: float, option_type: str = 'call') -> np.ndarray:
    """
    Generate volatility surface.
    
    Args:
        strikes: List of strike prices
        maturities: List of time to expiration
        S: Current stock price
        r: Risk-free rate
        option_type: 'call' or 'put'
        
    Returns:
        Volatility surface matrix
    """
    surface = np.zeros((len(maturities), len(strikes)))
    
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            # Generate sample option price (in practice, use market data)
            if option_type == 'call':
                price = call_price(S, K, T, r, 0.3)  # Use 30% vol as base
            else:
                price = put_price(S, K, T, r, 0.3)
            
            # Add some noise to simulate market prices
            price += np.random.normal(0, 0.01)
            price = max(price, 0.01)  # Ensure positive price
            
            # Calculate implied volatility
            try:
                surface[i, j] = implied_volatility(price, S, K, T, r, option_type)
            except:
                surface[i, j] = 0.3  # Default if calculation fails
    
    return surface

def fit_volatility_surface(strikes: List[float], maturities: List[float], 
                          implied_vols: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit volatility surface using polynomial interpolation.
    
    Args:
        strikes: List of strike prices
        maturities: List of time to expiration
        implied_vols: Matrix of implied volatilities
        
    Returns:
        Tuple of fitted parameters
    """
    # Convert to log-moneyness and time
    S = strikes[len(strikes)//2]  # Use middle strike as reference
    log_moneyness = np.log(np.array(strikes) / S)
    time_sqrt = np.sqrt(np.array(maturities))
    
    # Create grid
    X, Y = np.meshgrid(log_moneyness, time_sqrt)
    
    # Fit polynomial surface
    # This is a simplified approach; in practice, use more sophisticated methods
    coeffs = np.polyfit(X.flatten(), implied_vols.flatten(), 3)
    
    return coeffs, log_moneyness, time_sqrt

def volatility_skew(strikes: List[float], implied_vols: List[float]) -> float:
    """
    Calculate volatility skew.
    
    Args:
        strikes: List of strike prices
        implied_vols: List of implied volatilities
        
    Returns:
        Skew measure
    """
    # Find ATM strike (closest to current price)
    S = strikes[len(strikes)//2]
    atm_idx = len(strikes) // 2
    
    # Calculate skew as difference between OTM and ITM implied vol
    if atm_idx > 0 and atm_idx < len(strikes) - 1:
        skew = implied_vols[atm_idx + 1] - implied_vols[atm_idx - 1]
    else:
        skew = 0
    
    return skew

def term_structure_volatility(maturities: List[float], implied_vols: List[float]) -> np.ndarray:
    """
    Calculate volatility term structure.
    
    Args:
        maturities: List of time to expiration
        implied_vols: List of implied volatilities
        
    Returns:
        Term structure parameters
    """
    # Fit exponential function to term structure
    # vol(T) = a + b * exp(-c * T)
    time_array = np.array(maturities)
    vol_array = np.array(implied_vols)
    
    # Simple linear fit for demonstration
    coeffs = np.polyfit(time_array, vol_array, 2)
    
    return coeffs

# Helper functions for option pricing
def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate call option price."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm_cdf(d1) - K * np.exp(-r*T) * norm_cdf(d2)

def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate put option price."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def call_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate call option vega."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * np.sqrt(T) * norm_pdf(d1)

def put_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate put option vega (same as call)."""
    return call_vega(S, K, T, r, sigma)

def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))

def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi) 