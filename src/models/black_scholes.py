"""
Black-Scholes-Merton Option Pricing Model

This module implements the Black-Scholes-Merton option pricing model
with Greeks calculations and implied volatility functionality.

Author: Arthur Riche
Date: 2024
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Union, Optional
import warnings


class BlackScholesModel:
    """
    Black-Scholes-Merton option pricing model implementation.
    
    This class provides methods for pricing European options and calculating
    Greeks using the Black-Scholes-Merton formula.
    """
    
    def __init__(self):
        """Initialize the Black-Scholes model."""
        pass
    
    def _d1_d2(self, S: float, K: float, T: float, r: float, sigma: float) -> tuple:
        """
        Calculate d1 and d2 parameters for Black-Scholes formula.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            
        Returns:
            tuple: (d1, d2) parameters
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def price_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Price a European call option using Black-Scholes formula.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            
        Returns:
            float: Call option price
        """
        if T <= 0:
            return max(S - K, 0)
        
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def price_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Price a European put option using Black-Scholes formula.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            
        Returns:
            float: Put option price
        """
        if T <= 0:
            return max(K - S, 0)
        
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    
    def delta(self, S: float, K: float, T: float, r: float, sigma: float, 
              option_type: str = 'call') -> float:
        """
        Calculate option delta (first derivative with respect to stock price).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            float: Option delta
        """
        if T <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            return norm.cdf(d1)
        elif option_type == 'put':
            return norm.cdf(d1) - 1
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def gamma(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate option gamma (second derivative with respect to stock price).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            
        Returns:
            float: Option gamma
        """
        if T <= 0:
            return 0.0
        
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    
    def theta(self, S: float, K: float, T: float, r: float, sigma: float, 
              option_type: str = 'call') -> float:
        """
        Calculate option theta (derivative with respect to time).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            float: Option theta (per year)
        """
        if T <= 0:
            return 0.0
        
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        
        theta_term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        
        if option_type == 'call':
            theta_term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            theta_term2 = -r * K * np.exp(-r * T) * norm.cdf(-d2)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        return theta_term1 + theta_term2
    
    def vega(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate option vega (derivative with respect to volatility).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            
        Returns:
            float: Option vega
        """
        if T <= 0:
            return 0.0
        
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        vega = S * np.sqrt(T) * norm.pdf(d1)
        return vega
    
    def rho(self, S: float, K: float, T: float, r: float, sigma: float, 
            option_type: str = 'call') -> float:
        """
        Calculate option rho (derivative with respect to interest rate).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            float: Option rho
        """
        if T <= 0:
            return 0.0
        
        _, d2 = self._d1_d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            return K * T * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            return -K * T * np.exp(-r * T) * norm.cdf(-d2)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def all_greeks(self, S: float, K: float, T: float, r: float, sigma: float, 
                   option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all Greeks for the option.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            dict: Dictionary containing all Greeks
        """
        return {
            'delta': self.delta(S, K, T, r, sigma, option_type),
            'gamma': self.gamma(S, K, T, r, sigma),
            'theta': self.theta(S, K, T, r, sigma, option_type),
            'vega': self.vega(S, K, T, r, sigma),
            'rho': self.rho(S, K, T, r, sigma, option_type)
        }
    
    def implied_volatility(self, option_price: float, S: float, K: float, 
                          T: float, r: float, option_type: str = 'call', 
                          tolerance: float = 1e-6, max_iterations: int = 100) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            option_price: Market price of the option
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            option_type: 'call' or 'put'
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
            
        Returns:
            float: Implied volatility
        """
        if T <= 0:
            raise ValueError("Time to expiration must be positive")
        
        # Initial guess for volatility
        sigma = 0.3
        
        for i in range(max_iterations):
            if option_type == 'call':
                price = self.price_call(S, K, T, r, sigma)
                vega = self.vega(S, K, T, r, sigma)
            elif option_type == 'put':
                price = self.price_put(S, K, T, r, sigma)
                vega = self.vega(S, K, T, r, sigma)
            else:
                raise ValueError("option_type must be 'call' or 'put'")
            
            diff = option_price - price
            
            if abs(diff) < tolerance:
                return sigma
            
            if abs(vega) < 1e-10:
                warnings.warn("Vega too small, convergence may be slow")
                sigma += 0.01
                continue
            
            sigma = sigma + diff / vega
            
            # Ensure volatility stays positive
            sigma = max(sigma, 1e-6)
        
        raise ValueError(f"Failed to converge after {max_iterations} iterations")
    
    def price_option(self, S: float, K: float, T: float, r: float, sigma: float, 
                     option_type: str = 'call') -> float:
        """
        Price an option (call or put) using Black-Scholes formula.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            float: Option price
        """
        if option_type == 'call':
            return self.price_call(S, K, T, r, sigma)
        elif option_type == 'put':
            return self.price_put(S, K, T, r, sigma)
        else:
            raise ValueError("option_type must be 'call' or 'put'")


# Convenience functions
def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Convenience function for pricing call options."""
    pricer = BlackScholesModel()
    return pricer.price_call(S, K, T, r, sigma)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Convenience function for pricing put options."""
    pricer = BlackScholesModel()
    return pricer.price_put(S, K, T, r, sigma)


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                     option_type: str = 'call') -> Dict[str, float]:
    """Convenience function for calculating all Greeks."""
    pricer = BlackScholesModel()
    return pricer.all_greeks(S, K, T, r, sigma, option_type)


# Example usage
if __name__ == "__main__":
    # Example: Price a call option
    S = 100.0  # Current stock price
    K = 100.0  # Strike price
    T = 1.0    # Time to expiration (1 year)
    r = 0.05   # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    
    pricer = BlackScholesModel()
    
    # Price call and put options
    call_price = pricer.price_call(S, K, T, r, sigma)
    put_price = pricer.price_put(S, K, T, r, sigma)
    
    print(f"Call Option Price: ${call_price:.4f}")
    print(f"Put Option Price: ${put_price:.4f}")
    
    # Calculate Greeks
    greeks = pricer.all_greeks(S, K, T, r, sigma, 'call')
    print("\nGreeks for Call Option:")
    for greek, value in greeks.items():
        print(f"{greek.capitalize()}: {value:.6f}")
    
    # Calculate implied volatility
    implied_vol = pricer.implied_volatility(call_price, S, K, T, r, 'call')
    print(f"\nImplied Volatility: {implied_vol:.4f}") 