"""
Comprehensive test suite for the Advanced Option Pricer system.

This module contains unit tests for all major components including:
- Black-Scholes-Merton model
- Binomial tree model
- Monte Carlo simulation
- Greeks calculations
- Risk metrics
- Hedging strategies
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Import the modules to test
from src.models.black_scholes import BlackScholesPricer
from src.models.binomial_tree import BinomialTreePricer
from src.models.monte_carlo import MonteCarloPricer
from src.risk.greeks import GreeksCalculator
from src.risk.var import VaRCalculator
from src.risk.hedging import DynamicHedger


class TestBlackScholesModel(unittest.TestCase):
    """Test cases for Black-Scholes-Merton option pricing model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pricer = BlackScholesPricer()
        self.S = 100.0  # Current stock price
        self.K = 100.0  # Strike price
        self.T = 1.0    # Time to maturity (years)
        self.r = 0.05   # Risk-free rate
        self.sigma = 0.2  # Volatility
        
    def test_call_option_pricing(self):
        """Test European call option pricing."""
        call_price = self.pricer.price_call(
            S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma
        )
        
        # Basic sanity checks
        self.assertGreater(call_price, 0)
        self.assertIsInstance(call_price, float)
        
        # Test put-call parity
        put_price = self.pricer.price_put(
            S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma
        )
        parity_check = call_price - put_price - self.S + self.K * np.exp(-self.r * self.T)
        self.assertAlmostEqual(parity_check, 0, places=6)
        
    def test_put_option_pricing(self):
        """Test European put option pricing."""
        put_price = self.pricer.price_put(
            S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma
        )
        
        self.assertGreater(put_price, 0)
        self.assertIsInstance(put_price, float)
        
    def test_greeks_calculation(self):
        """Test Greeks calculation for call options."""
        greeks = self.pricer.all_greeks(
            S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma, option_type='call'
        )
        
        expected_keys = ['delta', 'gamma', 'theta', 'vega', 'rho']
        for key in expected_keys:
            self.assertIn(key, greeks)
            self.assertIsInstance(greeks[key], float)
            
        # Test delta bounds for call option
        self.assertGreaterEqual(greeks['delta'], 0)
        self.assertLessEqual(greeks['delta'], 1)
        
        # Test gamma is always positive
        self.assertGreater(greeks['gamma'], 0)
        
    def test_implied_volatility(self):
        """Test implied volatility calculation."""
        # Calculate option price first
        call_price = self.pricer.price_call(
            S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma
        )
        
        # Calculate implied volatility
        implied_vol = self.pricer.implied_volatility(
            price=call_price, S=self.S, K=self.K, T=self.T, r=self.r, option_type='call'
        )
        
        self.assertAlmostEqual(implied_vol, self.sigma, places=4)
        
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # At-the-money option
        atm_price = self.pricer.price_call(
            S=self.K, K=self.K, T=self.T, r=self.r, sigma=self.sigma
        )
        self.assertGreater(atm_price, 0)
        
        # Deep in-the-money call
        itm_price = self.pricer.price_call(
            S=self.K * 1.5, K=self.K, T=self.T, r=self.r, sigma=self.sigma
        )
        self.assertGreater(itm_price, self.K * 0.5 - self.K * np.exp(-self.r * self.T))
        
        # Deep out-of-the-money call
        otm_price = self.pricer.price_call(
            S=self.K * 0.5, K=self.K, T=self.T, r=self.r, sigma=self.sigma
        )
        self.assertLess(otm_price, 1.0)  # Should be very small


class TestBinomialTreeModel(unittest.TestCase):
    """Test cases for Binomial Tree option pricing model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pricer = BinomialTreePricer()
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2
        self.n_steps = 100
        
    def test_binomial_pricing(self):
        """Test binomial tree option pricing."""
        call_price = self.pricer.price_call(
            S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma, n_steps=self.n_steps
        )
        
        self.assertGreater(call_price, 0)
        self.assertIsInstance(call_price, float)
        
    def test_convergence_to_black_scholes(self):
        """Test that binomial tree converges to Black-Scholes as n_steps increases."""
        bs_pricer = BlackScholesPricer()
        bs_price = bs_pricer.price_call(
            S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma
        )
        
        # Test with different numbers of steps
        for n_steps in [50, 100, 200]:
            binomial_price = self.pricer.price_call(
                S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma, n_steps=n_steps
            )
            # Should converge to Black-Scholes price
            self.assertAlmostEqual(binomial_price, bs_price, places=2)
            
    def test_american_options(self):
        """Test American option pricing with early exercise."""
        american_price = self.pricer.price_american_call(
            S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma, n_steps=self.n_steps
        )
        
        european_price = self.pricer.price_call(
            S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma, n_steps=self.n_steps
        )
        
        # American call should be worth at least as much as European call
        self.assertGreaterEqual(american_price, european_price)


class TestMonteCarloModel(unittest.TestCase):
    """Test cases for Monte Carlo option pricing model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pricer = MonteCarloPricer()
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2
        self.n_simulations = 10000
        
    def test_monte_carlo_pricing(self):
        """Test Monte Carlo option pricing."""
        call_price = self.pricer.price_call(
            S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma, 
            n_simulations=self.n_simulations
        )
        
        self.assertGreater(call_price, 0)
        self.assertIsInstance(call_price, float)
        
    def test_convergence(self):
        """Test that Monte Carlo converges with more simulations."""
        prices = []
        for n_sims in [1000, 5000, 10000]:
            price = self.pricer.price_call(
                S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma, 
                n_simulations=n_sims
            )
            prices.append(price)
            
        # Prices should be reasonably close (within 5%)
        for i in range(1, len(prices)):
            self.assertAlmostEqual(prices[i], prices[i-1], delta=prices[i-1] * 0.05)
            
    def test_path_dependent_options(self):
        """Test path-dependent option pricing (Asian options)."""
        asian_price = self.pricer.price_asian_call(
            S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma,
            n_simulations=self.n_simulations, n_steps=252
        )
        
        self.assertGreater(asian_price, 0)
        self.assertIsInstance(asian_price, float)


class TestGreeksCalculator(unittest.TestCase):
    """Test cases for Greeks calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = GreeksCalculator()
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2
        
    def test_delta_calculation(self):
        """Test delta calculation."""
        delta = self.calculator.delta(
            S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma, option_type='call'
        )
        
        self.assertIsInstance(delta, float)
        self.assertGreaterEqual(delta, 0)
        self.assertLessEqual(delta, 1)
        
    def test_gamma_calculation(self):
        """Test gamma calculation."""
        gamma = self.calculator.gamma(
            S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma
        )
        
        self.assertIsInstance(gamma, float)
        self.assertGreater(gamma, 0)
        
    def test_theta_calculation(self):
        """Test theta calculation."""
        theta = self.calculator.theta(
            S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma, option_type='call'
        )
        
        self.assertIsInstance(theta, float)
        # Theta should be negative for long options (time decay)
        self.assertLess(theta, 0)


class TestVaRCalculator(unittest.TestCase):
    """Test cases for Value at Risk calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = VaRCalculator()
        
    def test_historical_var(self):
        """Test historical VaR calculation."""
        # Create sample returns
        returns = np.random.normal(0.001, 0.02, 1000)
        
        var_95 = self.calculator.historical_var(returns, confidence_level=0.95)
        var_99 = self.calculator.historical_var(returns, confidence_level=0.99)
        
        self.assertIsInstance(var_95, float)
        self.assertIsInstance(var_99, float)
        self.assertLess(var_99, var_95)  # 99% VaR should be more negative than 95% VaR
        
    def test_parametric_var(self):
        """Test parametric VaR calculation."""
        returns = np.random.normal(0.001, 0.02, 1000)
        
        var_95 = self.calculator.parametric_var(returns, confidence_level=0.95)
        
        self.assertIsInstance(var_95, float)
        self.assertLess(var_95, 0)  # VaR should be negative (loss)


class TestDynamicHedger(unittest.TestCase):
    """Test cases for dynamic hedging strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hedger = DynamicHedger()
        
    def test_delta_hedging(self):
        """Test delta hedging strategy."""
        # Mock portfolio data
        portfolio = {
            'positions': [{'asset': 'AAPL', 'quantity': 100, 'price': 150}],
            'options': [{'type': 'call', 'strike': 150, 'quantity': 1, 'price': 10}]
        }
        
        hedge_ratio = self.hedger.calculate_delta_hedge(portfolio)
        
        self.assertIsInstance(hedge_ratio, float)
        self.assertIsInstance(hedge_ratio, (int, float))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete option pricing system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bs_pricer = BlackScholesPricer()
        self.binomial_pricer = BinomialTreePricer()
        self.mc_pricer = MonteCarloPricer()
        self.greeks_calc = GreeksCalculator()
        
    def test_end_to_end_pricing(self):
        """Test complete end-to-end option pricing workflow."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        
        # Price using different methods
        bs_price = self.bs_pricer.price_call(S, K, T, r, sigma)
        binomial_price = self.binomial_pricer.price_call(S, K, T, r, sigma, n_steps=100)
        mc_price = self.mc_pricer.price_call(S, K, T, r, sigma, n_simulations=10000)
        
        # All prices should be reasonably close
        self.assertAlmostEqual(bs_price, binomial_price, places=2)
        self.assertAlmostEqual(bs_price, mc_price, delta=bs_price * 0.05)
        
        # Calculate Greeks
        greeks = self.greeks_calc.all_greeks(S, K, T, r, sigma, option_type='call')
        
        # Verify Greeks are reasonable
        self.assertGreater(greeks['delta'], 0)
        self.assertLess(greeks['delta'], 1)
        self.assertGreater(greeks['gamma'], 0)
        self.assertLess(greeks['theta'], 0)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
