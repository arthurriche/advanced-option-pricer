"""
Value at Risk (VaR) Calculator

This module provides comprehensive VaR calculations for options and portfolios
using various methods including historical simulation, Monte Carlo, and parametric approaches.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings


class ValueAtRisk:
    """
    Value at Risk calculator for options and portfolios.
    
    Supports multiple VaR calculation methods:
    - Historical simulation
    - Monte Carlo simulation
    - Parametric (normal distribution)
    - Delta-Gamma approximation
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize VaR calculator.
        
        Args:
            confidence_level: VaR confidence level (e.g., 0.95 for 95% VaR)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def historical_var(self,
                      returns: np.ndarray,
                      portfolio_value: float,
                      holding_period: int = 1) -> float:
        """
        Calculate VaR using historical simulation.
        
        Args:
            returns: Historical returns array
            portfolio_value: Current portfolio value
            holding_period: Holding period in days
            
        Returns:
            VaR value
        """
        # Sort returns in ascending order
        sorted_returns = np.sort(returns)
        
        # Find the percentile corresponding to alpha
        percentile_index = int(self.alpha * len(sorted_returns))
        var_return = sorted_returns[percentile_index]
        
        # Calculate VaR
        var = portfolio_value * abs(var_return) * np.sqrt(holding_period)
        
        return var
    
    def parametric_var(self,
                      returns: np.ndarray,
                      portfolio_value: float,
                      holding_period: int = 1,
                      distribution: str = 'normal') -> float:
        """
        Calculate VaR using parametric method.
        
        Args:
            returns: Historical returns array
            portfolio_value: Current portfolio value
            holding_period: Holding period in days
            distribution: Distribution assumption ('normal', 't')
            
        Returns:
            VaR value
        """
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if distribution.lower() == 'normal':
            # Standard normal quantile
            z_score = self._normal_quantile(self.alpha)
        elif distribution.lower() == 't':
            # Student's t quantile
            df = len(returns) - 1
            z_score = self._t_quantile(self.alpha, df)
        else:
            raise ValueError("Distribution must be 'normal' or 't'")
        
        # Calculate VaR
        var_return = mean_return - z_score * std_return
        var = portfolio_value * abs(var_return) * np.sqrt(holding_period)
        
        return var
    
    def monte_carlo_var(self,
                       portfolio_value: float,
                       expected_return: float,
                       volatility: float,
                       holding_period: int = 1,
                       n_simulations: int = 10000) -> float:
        """
        Calculate VaR using Monte Carlo simulation.
        
        Args:
            portfolio_value: Current portfolio value
            expected_return: Expected return
            volatility: Volatility
            holding_period: Holding period in days
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            VaR value
        """
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(expected_return, volatility, n_simulations)
        
        # Calculate portfolio values
        portfolio_values = portfolio_value * (1 + returns * np.sqrt(holding_period))
        
        # Calculate losses
        losses = portfolio_value - portfolio_values
        
        # Sort losses and find VaR
        sorted_losses = np.sort(losses)
        var_index = int(self.confidence_level * n_simulations)
        var = sorted_losses[var_index]
        
        return var
    
    def option_var(self,
                  S: float,
                  K: float,
                  T: float,
                  r: float,
                  sigma: float,
                  option_type: str = 'call',
                  position_size: float = 1.0,
                  method: str = 'delta_gamma') -> float:
        """
        Calculate VaR for option positions.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            position_size: Size of the position
            method: VaR method ('delta_gamma', 'monte_carlo')
            
        Returns:
            Option VaR
        """
        if method.lower() == 'delta_gamma':
            return self._delta_gamma_option_var(S, K, T, r, sigma, option_type, position_size)
        elif method.lower() == 'monte_carlo':
            return self._monte_carlo_option_var(S, K, T, r, sigma, option_type, position_size)
        else:
            raise ValueError("Method must be 'delta_gamma' or 'monte_carlo'")
    
    def _delta_gamma_option_var(self,
                               S: float,
                               K: float,
                               T: float,
                               r: float,
                               sigma: float,
                               option_type: str,
                               position_size: float) -> float:
        """Calculate option VaR using Delta-Gamma approximation."""
        
        # Import Greeks calculator
        from .greeks import GreeksCalculator
        
        # Calculate Greeks
        greeks_calc = GreeksCalculator('analytical')
        greeks = greeks_calc.calculate_greeks(S, K, T, r, sigma, option_type)
        
        # Calculate option price
        from ..models.black_scholes import BlackScholesModel
        pricer = BlackScholesModel()
        option_price = pricer.price_option(S, K, T, r, sigma, option_type)
        
        # Delta-Gamma approximation parameters
        delta = greeks['delta']
        gamma = greeks['gamma']
        
        # Price change approximation
        # dP = delta * dS + 0.5 * gamma * dS^2
        
        # For VaR calculation, we need to find the worst-case scenario
        # This involves solving a quadratic equation
        
        # Standard normal quantile
        z_score = self._normal_quantile(self.alpha)
        
        # Stock price change (assuming normal distribution)
        dS = z_score * S * sigma * np.sqrt(1/252)  # Daily volatility
        
        # Option price change
        dP = delta * dS + 0.5 * gamma * dS**2
        
        # VaR
        var = abs(dP) * position_size
        
        return var
    
    def _monte_carlo_option_var(self,
                               S: float,
                               K: float,
                               T: float,
                               r: float,
                               sigma: float,
                               option_type: str,
                               position_size: float,
                               n_simulations: int = 10000) -> float:
        """Calculate option VaR using Monte Carlo simulation."""
        
        # Import option pricer
        from ..models.black_scholes import BlackScholesModel
        pricer = BlackScholesModel()
        
        # Current option price
        current_price = pricer.price_option(S, K, T, r, sigma, option_type)
        
        # Generate stock price scenarios
        np.random.seed(42)
        daily_returns = np.random.normal(0, sigma * np.sqrt(1/252), n_simulations)
        future_prices = S * np.exp(daily_returns)
        
        # Calculate future option prices
        future_prices_option = []
        for future_S in future_prices:
            # Adjust time to maturity
            future_T = T - 1/252  # One day later
            if future_T > 0:
                future_price = pricer.price_option(future_S, K, future_T, r, sigma, option_type)
                future_prices_option.append(future_price)
            else:
                # Option expired
                if option_type.lower() == 'call':
                    future_price = max(future_S - K, 0)
                else:
                    future_price = max(K - future_S, 0)
                future_prices_option.append(future_price)
        
        # Calculate losses
        losses = (current_price - np.array(future_prices_option)) * position_size
        
        # Sort losses and find VaR
        sorted_losses = np.sort(losses)
        var_index = int(self.confidence_level * n_simulations)
        var = sorted_losses[var_index]
        
        return var
    
    def portfolio_var(self,
                     positions: List[Dict],
                     correlation_matrix: Optional[np.ndarray] = None,
                     method: str = 'monte_carlo') -> float:
        """
        Calculate portfolio VaR.
        
        Args:
            positions: List of position dictionaries
            correlation_matrix: Correlation matrix between assets
            method: VaR calculation method
            
        Returns:
            Portfolio VaR
        """
        if method.lower() == 'monte_carlo':
            return self._monte_carlo_portfolio_var(positions, correlation_matrix)
        else:
            raise ValueError("Method must be 'monte_carlo'")
    
    def _monte_carlo_portfolio_var(self,
                                  positions: List[Dict],
                                  correlation_matrix: Optional[np.ndarray] = None,
                                  n_simulations: int = 10000) -> float:
        """Calculate portfolio VaR using Monte Carlo simulation."""
        
        n_assets = len(positions)
        
        # Extract position data
        weights = np.array([pos['weight'] for pos in positions])
        volatilities = np.array([pos['volatility'] for pos in positions])
        expected_returns = np.array([pos.get('expected_return', 0) for pos in positions])
        
        # Generate correlated returns
        np.random.seed(42)
        if correlation_matrix is None:
            # Assume uncorrelated returns
            returns = np.random.normal(expected_returns, volatilities, (n_simulations, n_assets))
        else:
            # Generate correlated returns using Cholesky decomposition
            L = np.linalg.cholesky(correlation_matrix)
            uncorrelated_returns = np.random.normal(0, 1, (n_simulations, n_assets))
            correlated_returns = uncorrelated_returns @ L.T
            returns = correlated_returns * volatilities + expected_returns
        
        # Calculate portfolio returns
        portfolio_returns = np.sum(returns * weights, axis=1)
        
        # Calculate portfolio values
        portfolio_values = 1 + portfolio_returns
        
        # Calculate losses
        losses = 1 - portfolio_values
        
        # Sort losses and find VaR
        sorted_losses = np.sort(losses)
        var_index = int(self.confidence_level * n_simulations)
        var = sorted_losses[var_index]
        
        return var
    
    def _normal_quantile(self, alpha: float) -> float:
        """Calculate standard normal quantile."""
        # Approximation for normal quantile
        # For alpha = 0.05, this gives approximately 1.645
        if alpha == 0.05:
            return 1.645
        elif alpha == 0.01:
            return 2.326
        elif alpha == 0.025:
            return 1.96
        else:
            # Use approximation formula
            return np.sqrt(-2 * np.log(alpha))
    
    def _t_quantile(self, alpha: float, df: int) -> float:
        """Calculate Student's t quantile."""
        # Approximation for t quantile
        # For large degrees of freedom, t approaches normal
        if df > 30:
            return self._normal_quantile(alpha)
        else:
            # Use approximation
            return self._normal_quantile(alpha) * (1 + 1/(4*df))


def var_example():
    """Example usage of VaR calculator."""
    print("=== Value at Risk Calculator Example ===\n")
    
    # Create VaR calculator
    var_calc = ValueAtRisk(confidence_level=0.95)
    
    # Historical VaR example
    print("Historical VaR Example:")
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)  # Daily returns
    portfolio_value = 1000000  # $1M portfolio
    historical_var = var_calc.historical_var(returns, portfolio_value)
    print(f"Historical VaR (95%): ${historical_var:,.2f}")
    
    # Parametric VaR example
    print("\nParametric VaR Example:")
    parametric_var = var_calc.parametric_var(returns, portfolio_value)
    print(f"Parametric VaR (95%): ${parametric_var:,.2f}")
    
    # Monte Carlo VaR example
    print("\nMonte Carlo VaR Example:")
    mc_var = var_calc.monte_carlo_var(portfolio_value, 0.001, 0.02)
    print(f"Monte Carlo VaR (95%): ${mc_var:,.2f}")
    
    # Option VaR example
    print("\nOption VaR Example:")
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    option_var = var_calc.option_var(S, K, T, r, sigma, 'call', position_size=100)
    print(f"Option VaR (95%): ${option_var:,.2f}")
    
    # Portfolio VaR example
    print("\nPortfolio VaR Example:")
    positions = [
        {'weight': 0.6, 'volatility': 0.15, 'expected_return': 0.08},
        {'weight': 0.4, 'volatility': 0.25, 'expected_return': 0.12}
    ]
    correlation_matrix = np.array([[1.0, 0.3], [0.3, 1.0]])
    portfolio_var = var_calc.portfolio_var(positions, correlation_matrix)
    print(f"Portfolio VaR (95%): {portfolio_var:.4f}")


if __name__ == "__main__":
    var_example() 