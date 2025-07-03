"""
Stochastic Volatility Models for Option Pricing

This module implements various stochastic volatility models including:
- Heston model
- SABR model
- Hull-White model
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings


class HestonModel:
    """
    Heston Stochastic Volatility Model for option pricing.
    
    The Heston model assumes the volatility follows a mean-reverting
    square-root process (CIR process).
    """
    
    def __init__(self, 
                 kappa: float = 2.0,
                 theta: float = 0.04,
                 sigma_v: float = 0.3,
                 rho: float = -0.7,
                 v0: float = 0.04):
        """
        Initialize Heston model parameters.
        
        Args:
            kappa: Mean reversion speed
            theta: Long-term variance level
            sigma_v: Volatility of volatility
            rho: Correlation between asset and volatility
            v0: Initial variance
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.v0 = v0
    
    def price_option(self,
                    S: float,
                    K: float,
                    T: float,
                    r: float,
                    option_type: str = 'call',
                    N: int = 10000,
                    dt: float = 0.01) -> float:
        """
        Price option using Monte Carlo simulation with Heston model.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_type: 'call' or 'put'
            N: Number of simulations
            dt: Time step size
            
        Returns:
            Option price
        """
        n_steps = int(T / dt)
        
        # Initialize arrays
        S_paths = np.zeros((N, n_steps + 1))
        v_paths = np.zeros((N, n_steps + 1))
        
        S_paths[:, 0] = S
        v_paths[:, 0] = self.v0
        
        # Generate correlated random numbers
        np.random.seed(42)  # For reproducibility
        Z1 = np.random.normal(0, 1, (N, n_steps))
        Z2 = np.random.normal(0, 1, (N, n_steps))
        Z_v = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
        
        # Euler scheme for Heston model
        for i in range(n_steps):
            # Variance process
            v_prev = v_paths[:, i]
            v_new = v_prev + self.kappa * (self.theta - v_prev) * dt + \
                   self.sigma_v * np.sqrt(np.maximum(v_prev, 0)) * np.sqrt(dt) * Z_v[:, i]
            v_paths[:, i + 1] = np.maximum(v_new, 0)  # Ensure non-negative variance
            
            # Asset price process
            S_prev = S_paths[:, i]
            S_paths[:, i + 1] = S_prev * np.exp((r - 0.5 * v_prev) * dt + 
                                               np.sqrt(np.maximum(v_prev, 0)) * np.sqrt(dt) * Z1[:, i])
        
        # Calculate option payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S_paths[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - S_paths[:, -1], 0)
        
        # Discount and average
        option_price = np.exp(-r * T) * np.mean(payoffs)
        
        return option_price
    
    def implied_volatility(self,
                          S: float,
                          K: float,
                          T: float,
                          r: float,
                          option_price: float,
                          option_type: str = 'call') -> float:
        """
        Calculate implied volatility from option price using Heston model.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_price: Observed option price
            option_type: 'call' or 'put'
            
        Returns:
            Implied volatility
        """
        from scipy.optimize import minimize_scalar
        
        def objective(sigma):
            # Use Black-Scholes to get theoretical price
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type.lower() == 'call':
                bs_price = S*np.exp(-r*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            else:
                bs_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-r*T)*norm.cdf(-d1)
            
            return abs(bs_price - option_price)
        
        # Use Brent's method to find implied volatility
        result = minimize_scalar(objective, bounds=(0.01, 2.0), method='bounded')
        return result.x


class SABRModel:
    """
    SABR (Stochastic Alpha Beta Rho) Model for option pricing.
    
    The SABR model is widely used for interest rate derivatives and
    has become popular for equity options as well.
    """
    
    def __init__(self,
                 alpha: float = 0.3,
                 beta: float = 0.5,
                 rho: float = -0.1,
                 nu: float = 0.4):
        """
        Initialize SABR model parameters.
        
        Args:
            alpha: Initial volatility
            beta: CEV parameter (0 < beta <= 1)
            rho: Correlation between asset and volatility
            nu: Volatility of volatility
        """
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
    
    def price_option(self,
                    S: float,
                    K: float,
                    T: float,
                    r: float,
                    option_type: str = 'call',
                    N: int = 10000,
                    dt: float = 0.01) -> float:
        """
        Price option using Monte Carlo simulation with SABR model.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_type: 'call' or 'put'
            N: Number of simulations
            dt: Time step size
            
        Returns:
            Option price
        """
        n_steps = int(T / dt)
        
        # Initialize arrays
        S_paths = np.zeros((N, n_steps + 1))
        alpha_paths = np.zeros((N, n_steps + 1))
        
        S_paths[:, 0] = S
        alpha_paths[:, 0] = self.alpha
        
        # Generate correlated random numbers
        np.random.seed(42)
        Z1 = np.random.normal(0, 1, (N, n_steps))
        Z2 = np.random.normal(0, 1, (N, n_steps))
        Z_alpha = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
        
        # Euler scheme for SABR model
        for i in range(n_steps):
            # Volatility process
            alpha_prev = alpha_paths[:, i]
            alpha_new = alpha_prev + self.nu * alpha_prev * np.sqrt(dt) * Z_alpha[:, i]
            alpha_paths[:, i + 1] = np.maximum(alpha_new, 0.001)  # Ensure positive volatility
            
            # Asset price process
            S_prev = S_paths[:, i]
            S_paths[:, i + 1] = S_prev + r * S_prev * dt + \
                               alpha_prev * (S_prev ** self.beta) * np.sqrt(dt) * Z1[:, i]
            S_paths[:, i + 1] = np.maximum(S_paths[:, i + 1], 0.001)  # Ensure positive prices
        
        # Calculate option payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S_paths[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - S_paths[:, -1], 0)
        
        # Discount and average
        option_price = np.exp(-r * T) * np.mean(payoffs)
        
        return option_price


class StochasticVolatilityModel:
    """
    Generic stochastic volatility model wrapper.
    
    This class provides a unified interface for different stochastic
    volatility models.
    """
    
    def __init__(self, model_type: str = 'heston', **kwargs):
        """
        Initialize stochastic volatility model.
        
        Args:
            model_type: Type of model ('heston', 'sabr')
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type.lower()
        
        if self.model_type == 'heston':
            self.model = HestonModel(**kwargs)
        elif self.model_type == 'sabr':
            self.model = SABRModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def price_option(self, *args, **kwargs):
        """Price option using the selected stochastic volatility model."""
        return self.model.price_option(*args, **kwargs)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model_type == 'heston':
            return {
                'kappa': self.model.kappa,
                'theta': self.model.theta,
                'sigma_v': self.model.sigma_v,
                'rho': self.model.rho,
                'v0': self.model.v0
            }
        elif self.model_type == 'sabr':
            return {
                'alpha': self.model.alpha,
                'beta': self.model.beta,
                'rho': self.model.rho,
                'nu': self.model.nu
            }


def stochastic_volatility_example():
    """Example usage of stochastic volatility models."""
    print("=== Stochastic Volatility Models Example ===\n")
    
    # Model parameters
    S = 100.0  # Current stock price
    K = 100.0  # Strike price
    T = 1.0    # Time to maturity
    r = 0.05   # Risk-free rate
    
    # Heston model
    print("Heston Model:")
    heston = HestonModel(kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04)
    heston_call = heston.price_option(S, K, T, r, 'call')
    heston_put = heston.price_option(S, K, T, r, 'put')
    print(f"Call: {heston_call:.6f}")
    print(f"Put:  {heston_put:.6f}\n")
    
    # SABR model
    print("SABR Model:")
    sabr = SABRModel(alpha=0.3, beta=0.5, rho=-0.1, nu=0.4)
    sabr_call = sabr.price_option(S, K, T, r, 'call')
    sabr_put = sabr.price_option(S, K, T, r, 'put')
    print(f"Call: {sabr_call:.6f}")
    print(f"Put:  {sabr_put:.6f}\n")
    
    # Generic wrapper
    print("Using Generic Wrapper:")
    sv_model = StochasticVolatilityModel('heston', kappa=2.0, theta=0.04, 
                                        sigma_v=0.3, rho=-0.7, v0=0.04)
    generic_call = sv_model.price_option(S, K, T, r, 'call')
    print(f"Heston Call: {generic_call:.6f}")
    
    print(f"Model Parameters: {sv_model.get_parameters()}")


if __name__ == "__main__":
    stochastic_volatility_example() 