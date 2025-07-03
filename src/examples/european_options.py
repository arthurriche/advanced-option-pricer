"""
European Options Pricing Examples

This module demonstrates various European option pricing methods
and their applications in quantitative finance.
"""

import numpy as np
from typing import Dict, List, Tuple
import warnings


def european_option_example():
    """Comprehensive example of European option pricing."""
    print("=== European Options Pricing Example ===\n")
    
    # Model parameters
    S = 100.0  # Current stock price
    K = 100.0  # Strike price
    T = 1.0    # Time to maturity (years)
    r = 0.05   # Risk-free rate
    sigma = 0.2  # Volatility
    
    print(f"Model Parameters:")
    print(f"Stock Price (S): ${S}")
    print(f"Strike Price (K): ${K}")
    print(f"Time to Maturity (T): {T} years")
    print(f"Risk-free Rate (r): {r:.1%}")
    print(f"Volatility (σ): {sigma:.1%}\n")
    
    # Import pricing models
    try:
        from ..models.black_scholes import BlackScholesModel
        from ..models.binomial_tree import BinomialTreeModel
        from ..models.monte_carlo import MonteCarloModel
        from ..risk.greeks import GreeksCalculator
        
        # Black-Scholes pricing
        print("1. Black-Scholes Model:")
        bs_model = BlackScholesModel()
        
        call_price_bs = bs_model.price_option(S, K, T, r, sigma, 'call')
        put_price_bs = bs_model.price_option(S, K, T, r, sigma, 'put')
        
        print(f"   Call Option Price: ${call_price_bs:.4f}")
        print(f"   Put Option Price:  ${put_price_bs:.4f}")
        
        # Verify put-call parity
        pcp_check = call_price_bs - put_price_bs - S + K * np.exp(-r * T)
        print(f"   Put-Call Parity Check: {pcp_check:.6f} (should be ~0)\n")
        
        # Binomial tree pricing
        print("2. Binomial Tree Model:")
        bin_model = BinomialTreeModel()
        
        call_price_bin = bin_model.price_option(S, K, T, r, sigma, 'call')
        put_price_bin = bin_model.price_option(S, K, T, r, sigma, 'put')
        
        print(f"   Call Option Price: ${call_price_bin:.4f}")
        print(f"   Put Option Price:  ${put_price_bin:.4f}")
        print(f"   Difference from BS: ${abs(call_price_bin - call_price_bs):.6f}\n")
        
        # Monte Carlo pricing
        print("3. Monte Carlo Model:")
        mc_model = MonteCarloModel()
        
        call_price_mc = mc_model.price_option(S, K, T, r, sigma, 'call')
        put_price_mc = mc_model.price_option(S, K, T, r, sigma, 'put')
        
        print(f"   Call Option Price: ${call_price_mc:.4f}")
        print(f"   Put Option Price:  ${put_price_mc:.4f}")
        print(f"   Difference from BS: ${abs(call_price_mc - call_price_bs):.6f}\n")
        
        # Greeks calculation
        print("4. Option Greeks (Black-Scholes):")
        greeks_calc = GreeksCalculator('analytical')
        
        call_greeks = greeks_calc.calculate_greeks(S, K, T, r, sigma, 'call')
        put_greeks = greeks_calc.calculate_greeks(S, K, T, r, sigma, 'put')
        
        print("   Call Option Greeks:")
        for greek, value in call_greeks.items():
            print(f"     {greek.capitalize()}: {value:.6f}")
        
        print("   Put Option Greeks:")
        for greek, value in put_greeks.items():
            print(f"     {greek.capitalize()}: {value:.6f}")
        
        # Sensitivity analysis
        print("\n5. Sensitivity Analysis:")
        sensitivity_analysis(S, K, T, r, sigma)
        
        # Volatility smile example
        print("\n6. Volatility Smile Example:")
        volatility_smile_example(S, K, T, r)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all required modules are available.")


def sensitivity_analysis(S: float, K: float, T: float, r: float, sigma: float):
    """Analyze option price sensitivity to various parameters."""
    
    try:
        from ..models.black_scholes import BlackScholesModel
        
        bs_model = BlackScholesModel()
        base_call = bs_model.price_option(S, K, T, r, sigma, 'call')
        
        # Stock price sensitivity
        S_range = np.linspace(80, 120, 9)
        call_prices = []
        
        for s in S_range:
            price = bs_model.price_option(s, K, T, r, sigma, 'call')
            call_prices.append(price)
        
        print("   Stock Price Sensitivity:")
        for s, price in zip(S_range, call_prices):
            print(f"     S=${s:.0f}: ${price:.4f}")
        
        # Volatility sensitivity
        sigma_range = np.linspace(0.1, 0.4, 7)
        vol_prices = []
        
        for vol in sigma_range:
            price = bs_model.price_option(S, K, T, r, vol, 'call')
            vol_prices.append(price)
        
        print("\n   Volatility Sensitivity:")
        for vol, price in zip(sigma_range, vol_prices):
            print(f"     σ={vol:.1%}: ${price:.4f}")
        
        # Time to maturity sensitivity
        T_range = np.linspace(0.1, 2.0, 8)
        time_prices = []
        
        for t in T_range:
            price = bs_model.price_option(S, K, t, r, sigma, 'call')
            time_prices.append(price)
        
        print("\n   Time to Maturity Sensitivity:")
        for t, price in zip(T_range, time_prices):
            print(f"     T={t:.1f}y: ${price:.4f}")
            
    except ImportError:
        print("   Black-Scholes model not available for sensitivity analysis.")


def volatility_smile_example(S: float, K: float, T: float, r: float):
    """Demonstrate volatility smile effect."""
    
    try:
        from ..models.black_scholes import BlackScholesModel
        
        bs_model = BlackScholesModel()
        
        # Different strikes
        strikes = np.linspace(80, 120, 9)
        implied_vols = []
        
        # Assume market prices follow a volatility smile
        for strike in strikes:
            # Create a simple volatility smile
            moneyness = strike / S
            if moneyness < 0.9:  # Out-of-the-money puts
                vol = 0.25 + 0.1 * (0.9 - moneyness)
            elif moneyness > 1.1:  # Out-of-the-money calls
                vol = 0.25 + 0.1 * (moneyness - 1.1)
            else:  # At-the-money
                vol = 0.20
            
            # Calculate option price with this volatility
            option_price = bs_model.price_option(S, strike, T, r, vol, 'call')
            
            # Calculate implied volatility (simplified)
            implied_vols.append(vol)
        
        print("   Volatility Smile:")
        for strike, vol in zip(strikes, implied_vols):
            moneyness = strike / S
            print(f"     K=${strike:.0f} (Moneyness={moneyness:.2f}): σ={vol:.1%}")
            
    except ImportError:
        print("   Black-Scholes model not available for volatility smile example.")


def option_strategies_example():
    """Demonstrate common option strategies."""
    print("\n=== Option Strategies Example ===\n")
    
    S = 100.0
    K1, K2 = 95.0, 105.0
    T = 0.5
    r = 0.05
    sigma = 0.2
    
    try:
        from ..models.black_scholes import BlackScholesModel
        
        bs_model = BlackScholesModel()
        
        # Bull spread
        print("1. Bull Spread (Long Call Spread):")
        long_call = bs_model.price_option(S, K1, T, r, sigma, 'call')
        short_call = bs_model.price_option(S, K2, T, r, sigma, 'call')
        bull_spread_cost = long_call - short_call
        
        print(f"   Buy call at K=${K1}: ${long_call:.4f}")
        print(f"   Sell call at K=${K2}: ${short_call:.4f}")
        print(f"   Net cost: ${bull_spread_cost:.4f}")
        
        # Maximum profit and loss
        max_profit = K2 - K1 - bull_spread_cost
        max_loss = bull_spread_cost
        print(f"   Maximum profit: ${max_profit:.4f}")
        print(f"   Maximum loss: ${max_loss:.4f}\n")
        
        # Straddle
        print("2. Straddle:")
        call_price = bs_model.price_option(S, K1, T, r, sigma, 'call')
        put_price = bs_model.price_option(S, K1, T, r, sigma, 'put')
        straddle_cost = call_price + put_price
        
        print(f"   Buy call: ${call_price:.4f}")
        print(f"   Buy put:  ${put_price:.4f}")
        print(f"   Total cost: ${straddle_cost:.4f}")
        print(f"   Breakeven points: ${K1 - straddle_cost:.2f} and ${K1 + straddle_cost:.2f}\n")
        
        # Butterfly spread
        print("3. Butterfly Spread:")
        K_mid = (K1 + K2) / 2
        long_call1 = bs_model.price_option(S, K1, T, r, sigma, 'call')
        long_call2 = bs_model.price_option(S, K2, T, r, sigma, 'call')
        short_calls = 2 * bs_model.price_option(S, K_mid, T, r, sigma, 'call')
        butterfly_cost = long_call1 + long_call2 - short_calls
        
        print(f"   Buy call at K=${K1}: ${long_call1:.4f}")
        print(f"   Buy call at K=${K2}: ${long_call2:.4f}")
        print(f"   Sell 2 calls at K=${K_mid}: ${short_calls:.4f}")
        print(f"   Net cost: ${butterfly_cost:.4f}")
        
    except ImportError:
        print("Black-Scholes model not available for strategies example.")


def risk_management_example():
    """Demonstrate option risk management techniques."""
    print("\n=== Risk Management Example ===\n")
    
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    
    try:
        from ..risk.greeks import GreeksCalculator
        from ..risk.var import ValueAtRisk
        from ..risk.hedging import DeltaHedging
        
        # Portfolio with multiple options
        positions = [
            {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 'option_type': 'call', 'quantity': 1},
            {'S': S, 'K': K*0.95, 'T': T, 'r': r, 'sigma': sigma, 'option_type': 'put', 'quantity': 2},
            {'S': S, 'K': K*1.05, 'T': T, 'r': r, 'sigma': sigma, 'option_type': 'call', 'quantity': 1}
        ]
        
        # Portfolio Greeks
        print("1. Portfolio Greeks:")
        greeks_calc = GreeksCalculator('analytical')
        portfolio_greeks = greeks_calc.calculate_portfolio_greeks(positions)
        
        for greek, value in portfolio_greeks.items():
            print(f"   {greek.capitalize()}: {value:.6f}")
        
        # Portfolio VaR
        print("\n2. Portfolio Value at Risk:")
        var_calc = ValueAtRisk(confidence_level=0.95)
        
        # Simulate returns for VaR calculation
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)
        portfolio_value = 10000  # $10k portfolio
        
        historical_var = var_calc.historical_var(returns, portfolio_value)
        print(f"   Historical VaR (95%): ${historical_var:,.2f}")
        
        # Delta hedging example
        print("\n3. Delta Hedging Example:")
        delta_hedger = DeltaHedging()
        
        # Calculate hedge ratio for a call option
        hedge_ratio = delta_hedger.calculate_hedge_ratio(S, K, T, r, sigma, 'call')
        print(f"   Delta hedge ratio: {hedge_ratio:.6f}")
        print(f"   Number of shares to short: {abs(hedge_ratio):.2f}")
        
    except ImportError as e:
        print(f"Risk management modules not available: {e}")


if __name__ == "__main__":
    european_option_example()
    option_strategies_example()
    risk_management_example() 