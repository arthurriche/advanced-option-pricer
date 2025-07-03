"""
Hedging Strategies for Options

This module provides various hedging strategies for option positions:
- Delta hedging
- Delta-Gamma hedging
- Dynamic hedging
- Static hedging
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings


class HedgingStrategy:
    """
    Base class for option hedging strategies.
    
    Provides common functionality for different hedging approaches.
    """
    
    def __init__(self, rebalance_frequency: str = 'daily'):
        """
        Initialize hedging strategy.
        
        Args:
            rebalance_frequency: How often to rebalance ('daily', 'weekly', 'monthly')
        """
        self.rebalance_frequency = rebalance_frequency
        self.hedge_history = []
    
    def calculate_hedge_ratio(self, *args, **kwargs) -> float:
        """Calculate hedge ratio - to be implemented by subclasses."""
        raise NotImplementedError
    
    def rebalance_hedge(self, *args, **kwargs) -> Dict:
        """Rebalance hedge position - to be implemented by subclasses."""
        raise NotImplementedError


class DeltaHedging(HedgingStrategy):
    """
    Delta hedging strategy for option positions.
    
    Maintains a delta-neutral portfolio by dynamically adjusting
    the underlying asset position.
    """
    
    def __init__(self, rebalance_frequency: str = 'daily'):
        super().__init__(rebalance_frequency)
        self.hedge_position = 0.0
    
    def calculate_hedge_ratio(self,
                            S: float,
                            K: float,
                            T: float,
                            r: float,
                            sigma: float,
                            option_type: str = 'call') -> float:
        """
        Calculate delta hedge ratio.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Number of shares to hold for delta neutrality
        """
        from .greeks import GreeksCalculator
        
        greeks_calc = GreeksCalculator('analytical')
        greeks = greeks_calc.calculate_greeks(S, K, T, r, sigma, option_type)
        
        # Delta hedge ratio is negative of option delta
        return -greeks['delta']
    
    def rebalance_hedge(self,
                       S: float,
                       K: float,
                       T: float,
                       r: float,
                       sigma: float,
                       option_type: str = 'call',
                       option_quantity: float = 1.0) -> Dict:
        """
        Rebalance delta hedge position.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            option_quantity: Number of options
            
        Returns:
            Dictionary with hedge information
        """
        # Calculate required hedge ratio
        required_hedge = self.calculate_hedge_ratio(S, K, T, r, sigma, option_type)
        required_hedge *= option_quantity
        
        # Calculate hedge adjustment
        hedge_adjustment = required_hedge - self.hedge_position
        
        # Update hedge position
        self.hedge_position = required_hedge
        
        # Record hedge action
        hedge_info = {
            'timestamp': len(self.hedge_history),
            'stock_price': S,
            'required_hedge': required_hedge,
            'hedge_adjustment': hedge_adjustment,
            'hedge_cost': hedge_adjustment * S
        }
        
        self.hedge_history.append(hedge_info)
        
        return hedge_info
    
    def simulate_hedging(self,
                        initial_S: float,
                        K: float,
                        T: float,
                        r: float,
                        sigma: float,
                        option_type: str = 'call',
                        option_quantity: float = 1.0,
                        n_periods: int = 252,
                        price_path: Optional[np.ndarray] = None) -> Dict:
        """
        Simulate delta hedging over time.
        
        Args:
            initial_S: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            option_quantity: Number of options
            n_periods: Number of periods to simulate
            price_path: Optional predefined price path
            
        Returns:
            Dictionary with simulation results
        """
        # Generate price path if not provided
        if price_path is None:
            np.random.seed(42)
            returns = np.random.normal(r - 0.5*sigma**2, sigma, n_periods)
            price_path = initial_S * np.exp(np.cumsum(returns))
        
        # Initialize tracking variables
        current_S = initial_S
        current_T = T
        total_cost = 0.0
        hedge_positions = []
        
        # Calculate initial option price
        from ..models.black_scholes import BlackScholesModel
        pricer = BlackScholesModel()
        initial_option_price = pricer.price_option(initial_S, K, T, r, sigma, option_type)
        
        # Simulate hedging
        for i in range(n_periods):
            # Update time to maturity
            current_T = T - i/252
            
            if current_T <= 0:
                break
            
            # Rebalance hedge
            hedge_info = self.rebalance_hedge(
                current_S, K, current_T, r, sigma, option_type, option_quantity
            )
            
            total_cost += hedge_info['hedge_cost']
            hedge_positions.append(hedge_info['required_hedge'])
            
            # Update stock price for next period
            if i < len(price_path) - 1:
                current_S = price_path[i + 1]
        
        # Calculate final option value
        final_option_price = pricer.price_option(current_S, K, current_T, r, sigma, option_type)
        if current_T <= 0:
            # Option expired
            if option_type.lower() == 'call':
                final_option_price = max(current_S - K, 0)
            else:
                final_option_price = max(K - current_S, 0)
        
        # Calculate P&L
        option_pnl = (final_option_price - initial_option_price) * option_quantity
        total_pnl = option_pnl - total_cost
        
        return {
            'initial_option_price': initial_option_price,
            'final_option_price': final_option_price,
            'option_pnl': option_pnl,
            'hedging_cost': total_cost,
            'total_pnl': total_pnl,
            'hedge_positions': hedge_positions,
            'price_path': price_path[:len(hedge_positions)]
        }


class DeltaGammaHedging(HedgingStrategy):
    """
    Delta-Gamma hedging strategy.
    
    Uses both the underlying asset and another option to hedge
    delta and gamma risk simultaneously.
    """
    
    def __init__(self, rebalance_frequency: str = 'daily'):
        super().__init__(rebalance_frequency)
        self.hedge_position = 0.0
        self.gamma_hedge_position = 0.0
    
    def calculate_hedge_ratios(self,
                              S: float,
                              K1: float,
                              T1: float,
                              r: float,
                              sigma: float,
                              option_type1: str = 'call',
                              K2: float = None,
                              T2: float = None,
                              option_type2: str = 'call') -> Tuple[float, float]:
        """
        Calculate delta and gamma hedge ratios.
        
        Args:
            S: Current stock price
            K1: Strike price of option to hedge
            T1: Time to maturity of option to hedge
            r: Risk-free rate
            sigma: Volatility
            option_type1: Type of option to hedge
            K2: Strike price of hedging option
            T2: Time to maturity of hedging option
            option_type2: Type of hedging option
            
        Returns:
            Tuple of (delta_hedge_ratio, gamma_hedge_ratio)
        """
        from .greeks import GreeksCalculator
        
        greeks_calc = GreeksCalculator('analytical')
        
        # Calculate Greeks for option to hedge
        greeks1 = greeks_calc.calculate_greeks(S, K1, T1, r, sigma, option_type1)
        
        # Calculate Greeks for hedging option
        greeks2 = greeks_calc.calculate_greeks(S, K2, T2, r, sigma, option_type2)
        
        # Solve system of equations:
        # delta1 + w1 * delta2 + w2 * 1 = 0
        # gamma1 + w1 * gamma2 = 0
        
        # Gamma hedge ratio
        gamma_hedge_ratio = -greeks1['gamma'] / greeks2['gamma']
        
        # Delta hedge ratio (including gamma hedge)
        delta_hedge_ratio = -(greeks1['delta'] + gamma_hedge_ratio * greeks2['delta'])
        
        return delta_hedge_ratio, gamma_hedge_ratio
    
    def rebalance_hedge(self,
                       S: float,
                       K1: float,
                       T1: float,
                       r: float,
                       sigma: float,
                       option_type1: str = 'call',
                       K2: float = None,
                       T2: float = None,
                       option_type2: str = 'call',
                       option_quantity: float = 1.0) -> Dict:
        """
        Rebalance delta-gamma hedge position.
        
        Args:
            S: Current stock price
            K1: Strike price of option to hedge
            T1: Time to maturity of option to hedge
            r: Risk-free rate
            sigma: Volatility
            option_type1: Type of option to hedge
            K2: Strike price of hedging option
            T2: Time to maturity of hedging option
            option_type2: Type of hedging option
            option_quantity: Number of options to hedge
            
        Returns:
            Dictionary with hedge information
        """
        # Calculate hedge ratios
        delta_hedge, gamma_hedge = self.calculate_hedge_ratios(
            S, K1, T1, r, sigma, option_type1, K2, T2, option_type2
        )
        
        # Scale by option quantity
        delta_hedge *= option_quantity
        gamma_hedge *= option_quantity
        
        # Calculate adjustments
        delta_adjustment = delta_hedge - self.hedge_position
        gamma_adjustment = gamma_hedge - self.gamma_hedge_position
        
        # Update positions
        self.hedge_position = delta_hedge
        self.gamma_hedge_position = gamma_hedge
        
        # Record hedge action
        hedge_info = {
            'timestamp': len(self.hedge_history),
            'stock_price': S,
            'delta_hedge': delta_hedge,
            'gamma_hedge': gamma_hedge,
            'delta_adjustment': delta_adjustment,
            'gamma_adjustment': gamma_adjustment,
            'total_cost': delta_adjustment * S
        }
        
        self.hedge_history.append(hedge_info)
        
        return hedge_info


class StaticHedging(HedgingStrategy):
    """
    Static hedging strategy using portfolio of standard options.
    
    Constructs a hedge that doesn't require rebalancing by using
    a combination of standard options.
    """
    
    def __init__(self):
        super().__init__('static')
    
    def construct_static_hedge(self,
                              S: float,
                              K: float,
                              T: float,
                              r: float,
                              sigma: float,
                              option_type: str = 'call',
                              n_strikes: int = 10) -> Dict:
        """
        Construct static hedge using portfolio of standard options.
        
        Args:
            S: Current stock price
            K: Strike price of option to hedge
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            n_strikes: Number of strikes to use in hedge
            
        Returns:
            Dictionary with hedge portfolio
        """
        # Generate strikes around current price
        strike_range = 0.5  # 50% range
        strikes = np.linspace(S * (1 - strike_range), S * (1 + strike_range), n_strikes)
        
        # Calculate option prices and Greeks for each strike
        from ..models.black_scholes import BlackScholesModel
        from .greeks import GreeksCalculator
        
        pricer = BlackScholesModel()
        greeks_calc = GreeksCalculator('analytical')
        
        hedge_portfolio = []
        
        for strike in strikes:
            # Calculate option price and Greeks
            option_price = pricer.price_option(S, strike, T, r, sigma, option_type)
            greeks = greeks_calc.calculate_greeks(S, strike, T, r, sigma, option_type)
            
            hedge_portfolio.append({
                'strike': strike,
                'price': option_price,
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'quantity': 0.0  # To be determined
            })
        
        # Solve for hedge quantities (simplified approach)
        # This is a simplified version - in practice, you'd use optimization
        
        return {
            'strikes': strikes,
            'portfolio': hedge_portfolio,
            'total_cost': sum([pos['price'] * pos['quantity'] for pos in hedge_portfolio])
        }


def hedging_example():
    """Example usage of hedging strategies."""
    print("=== Option Hedging Strategies Example ===\n")
    
    # Model parameters
    S = 100.0  # Current stock price
    K = 100.0  # Strike price
    T = 1.0    # Time to maturity
    r = 0.05   # Risk-free rate
    sigma = 0.2  # Volatility
    
    # Delta hedging example
    print("Delta Hedging Example:")
    delta_hedger = DeltaHedging()
    
    # Calculate hedge ratio
    hedge_ratio = delta_hedger.calculate_hedge_ratio(S, K, T, r, sigma, 'call')
    print(f"Delta Hedge Ratio: {hedge_ratio:.6f}")
    
    # Simulate hedging
    simulation = delta_hedger.simulate_hedging(S, K, T, r, sigma, 'call', n_periods=50)
    print(f"Initial Option Price: ${simulation['initial_option_price']:.4f}")
    print(f"Final Option Price: ${simulation['final_option_price']:.4f}")
    print(f"Option P&L: ${simulation['option_pnl']:.4f}")
    print(f"Hedging Cost: ${simulation['hedging_cost']:.4f}")
    print(f"Total P&L: ${simulation['total_pnl']:.4f}")
    
    # Delta-Gamma hedging example
    print("\nDelta-Gamma Hedging Example:")
    dg_hedger = DeltaGammaHedging()
    
    # Use a different strike for hedging option
    K2 = 105.0
    delta_hedge, gamma_hedge = dg_hedger.calculate_hedge_ratios(
        S, K, T, r, sigma, 'call', K2, T, 'call'
    )
    print(f"Delta Hedge Ratio: {delta_hedge:.6f}")
    print(f"Gamma Hedge Ratio: {gamma_hedge:.6f}")
    
    # Static hedging example
    print("\nStatic Hedging Example:")
    static_hedger = StaticHedging()
    static_hedge = static_hedger.construct_static_hedge(S, K, T, r, sigma, 'call')
    print(f"Number of strikes used: {len(static_hedge['strikes'])}")
    print(f"Total hedge cost: ${static_hedge['total_cost']:.4f}")


if __name__ == "__main__":
    hedging_example() 