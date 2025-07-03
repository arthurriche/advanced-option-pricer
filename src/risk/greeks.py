"""
Greeks Calculator for Options

This module provides comprehensive calculation of option Greeks:
- Delta: Sensitivity to underlying price
- Gamma: Sensitivity to underlying price changes
- Theta: Sensitivity to time decay
- Vega: Sensitivity to volatility
- Rho: Sensitivity to interest rate
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings


class GreeksCalculator:
    """
    Calculator for option Greeks using analytical and numerical methods.
    
    Supports both European and American options with various pricing models.
    """
    
    def __init__(self, method: str = 'analytical'):
        """
        Initialize Greeks calculator.
        
        Args:
            method: Calculation method ('analytical', 'numerical')
        """
        self.method = method.lower()
        if self.method not in ['analytical', 'numerical']:
            raise ValueError("Method must be 'analytical' or 'numerical'")
    
    def calculate_greeks(self,
                        S: float,
                        K: float,
                        T: float,
                        r: float,
                        sigma: float,
                        option_type: str = 'call',
                        model: str = 'black_scholes') -> Dict[str, float]:
        """
        Calculate all Greeks for an option.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            model: Pricing model to use
            
        Returns:
            Dictionary containing all Greeks
        """
        if self.method == 'analytical':
            return self._analytical_greeks(S, K, T, r, sigma, option_type)
        else:
            return self._numerical_greeks(S, K, T, r, sigma, option_type, model)
    
    def _analytical_greeks(self,
                          S: float,
                          K: float,
                          T: float,
                          r: float,
                          sigma: float,
                          option_type: str = 'call') -> Dict[str, float]:
        """Calculate Greeks using analytical Black-Scholes formulas."""
        
        # Calculate d1 and d2
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Standard normal PDF and CDF
        def norm_pdf(x):
            return np.exp(-0.5*x**2) / np.sqrt(2*np.pi)
        
        def norm_cdf(x):
            return 0.5 * (1 + np.math.erf(x/np.sqrt(2)))
        
        # Calculate Greeks
        if option_type.lower() == 'call':
            delta = norm_cdf(d1)
            gamma = norm_pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-S * norm_pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r*T) * norm_cdf(d2))
            vega = S * np.sqrt(T) * norm_pdf(d1)
            rho = K * T * np.exp(-r*T) * norm_cdf(d2)
        else:  # put
            delta = norm_cdf(d1) - 1
            gamma = norm_pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-S * norm_pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r*T) * norm_cdf(-d2))
            vega = S * np.sqrt(T) * norm_pdf(d1)
            rho = -K * T * np.exp(-r*T) * norm_cdf(-d2)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def _numerical_greeks(self,
                         S: float,
                         K: float,
                         T: float,
                         r: float,
                         sigma: float,
                         option_type: str = 'call',
                         model: str = 'black_scholes') -> Dict[str, float]:
        """Calculate Greeks using finite difference methods."""
        
        # Import pricing models
        try:
            from ..models.black_scholes import BlackScholesModel
            from ..models.binomial_tree import BinomialTreeModel
            from ..models.monte_carlo import MonteCarloModel
        except ImportError:
            # Fallback for direct execution
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
            from src.models.black_scholes import BlackScholesModel
            from src.models.binomial_tree import BinomialTreeModel
            from src.models.monte_carlo import MonteCarloModel
        
        # Select pricing model
        if model == 'black_scholes':
            pricer = BlackScholesModel()
        elif model == 'binomial':
            pricer = BinomialTreeModel()
        elif model == 'monte_carlo':
            pricer = MonteCarloModel()
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # Perturbation sizes
        dS = S * 0.01
        dT = T * 0.01
        dsigma = sigma * 0.01
        dr = r * 0.01
        
        # Base price
        base_price = pricer.price_option(S, K, T, r, sigma, option_type)
        
        # Delta (first derivative with respect to S)
        price_up = pricer.price_option(S + dS, K, T, r, sigma, option_type)
        price_down = pricer.price_option(S - dS, K, T, r, sigma, option_type)
        delta = (price_up - price_down) / (2 * dS)
        
        # Gamma (second derivative with respect to S)
        gamma = (price_up + price_down - 2 * base_price) / (dS**2)
        
        # Theta (derivative with respect to T)
        price_theta = pricer.price_option(S, K, T + dT, r, sigma, option_type)
        theta = (price_theta - base_price) / dT
        
        # Vega (derivative with respect to sigma)
        price_vega_up = pricer.price_option(S, K, T, r, sigma + dsigma, option_type)
        price_vega_down = pricer.price_option(S, K, T, r, sigma - dsigma, option_type)
        vega = (price_vega_up - price_vega_down) / (2 * dsigma)
        
        # Rho (derivative with respect to r)
        price_rho_up = pricer.price_option(S, K, T, r + dr, sigma, option_type)
        price_rho_down = pricer.price_option(S, K, T, r - dr, sigma, option_type)
        rho = (price_rho_up - price_rho_down) / (2 * dr)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def calculate_portfolio_greeks(self,
                                 positions: list) -> Dict[str, float]:
        """
        Calculate portfolio-level Greeks.
        
        Args:
            positions: List of dictionaries with position details
                     [{'S': 100, 'K': 100, 'T': 1, 'r': 0.05, 'sigma': 0.2,
                       'option_type': 'call', 'quantity': 1}, ...]
        
        Returns:
            Portfolio Greeks
        """
        portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
        
        for position in positions:
            greeks = self.calculate_greeks(
                position['S'], position['K'], position['T'],
                position['r'], position['sigma'], position['option_type']
            )
            
            quantity = position.get('quantity', 1)
            
            for greek in portfolio_greeks:
                portfolio_greeks[greek] += greeks[greek] * quantity
        
        return portfolio_greeks
    
    def delta_hedge_ratio(self,
                         S: float,
                         K: float,
                         T: float,
                         r: float,
                         sigma: float,
                         option_type: str = 'call') -> float:
        """
        Calculate delta hedge ratio (number of shares to short).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Delta hedge ratio
        """
        greeks = self.calculate_greeks(S, K, T, r, sigma, option_type)
        return -greeks['delta']  # Negative because we short shares
    
    def gamma_exposure(self,
                      S: float,
                      K: float,
                      T: float,
                      r: float,
                      sigma: float,
                      option_type: str = 'call',
                      position_size: float = 1.0) -> float:
        """
        Calculate gamma exposure for position sizing.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            position_size: Size of the position
            
        Returns:
            Gamma exposure
        """
        greeks = self.calculate_greeks(S, K, T, r, sigma, option_type)
        return greeks['gamma'] * position_size * (S**2) / 100  # Scaled by S^2/100


def greeks_example():
    """Example usage of Greeks calculator."""
    print("=== Option Greeks Calculator Example ===\n")
    
    # Model parameters
    S = 100.0  # Current stock price
    K = 100.0  # Strike price
    T = 1.0    # Time to maturity
    r = 0.05   # Risk-free rate
    sigma = 0.2  # Volatility
    
    # Create calculator
    calculator = GreeksCalculator('analytical')
    
    # Calculate Greeks for call option
    print("European Call Option Greeks:")
    call_greeks = calculator.calculate_greeks(S, K, T, r, sigma, 'call')
    for greek, value in call_greeks.items():
        print(f"{greek.capitalize()}: {value:.6f}")
    
    print("\nEuropean Put Option Greeks:")
    put_greeks = calculator.calculate_greeks(S, K, T, r, sigma, 'put')
    for greek, value in put_greeks.items():
        print(f"{greek.capitalize()}: {value:.6f}")
    
    # Portfolio example
    print("\nPortfolio Greeks Example:")
    positions = [
        {'S': 100, 'K': 100, 'T': 1, 'r': 0.05, 'sigma': 0.2, 
         'option_type': 'call', 'quantity': 1},
        {'S': 100, 'K': 95, 'T': 1, 'r': 0.05, 'sigma': 0.2, 
         'option_type': 'put', 'quantity': 2}
    ]
    
    portfolio_greeks = calculator.calculate_portfolio_greeks(positions)
    for greek, value in portfolio_greeks.items():
        print(f"Portfolio {greek.capitalize()}: {value:.6f}")
    
    # Delta hedge ratio
    hedge_ratio = calculator.delta_hedge_ratio(S, K, T, r, sigma, 'call')
    print(f"\nDelta Hedge Ratio: {hedge_ratio:.6f}")


if __name__ == "__main__":
    greeks_example() 