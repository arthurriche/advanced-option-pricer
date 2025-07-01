"""
Unit Tests for Option Pricing Models

Tests for Black-Scholes, Binomial Tree, Monte Carlo, and other pricing models.

Author: Arthur Riche
Date: 2024
"""

import unittest
import numpy as np
from src.models.black_scholes import BlackScholesPricer
from src.models.binomial_tree import binomial_tree_price
from src.models.monte_carlo import monte_carlo_option_price

class TestBlackScholes(unittest.TestCase):
    def setUp(self):
        self.pricer = BlackScholesPricer()
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2

    def test_call_price(self):
        call_price = self.pricer.price_call(self.S, self.K, self.T, self.r, self.sigma)
        self.assertGreater(call_price, 0)
        self.assertIsInstance(call_price, float)

    def test_put_price(self):
        put_price = self.pricer.price_put(self.S, self.K, self.T, self.r, self.sigma)
        self.assertGreater(put_price, 0)
        self.assertIsInstance(put_price, float)

    def test_put_call_parity(self):
        call_price = self.pricer.price_call(self.S, self.K, self.T, self.r, self.sigma)
        put_price = self.pricer.price_put(self.S, self.K, self.T, self.r, self.sigma)
        parity_check = call_price - put_price - self.S + self.K * np.exp(-self.r * self.T)
        self.assertAlmostEqual(parity_check, 0, places=6)

    def test_greeks(self):
        greeks = self.pricer.all_greeks(self.S, self.K, self.T, self.r, self.sigma, 'call')
        self.assertIn('delta', greeks)
        self.assertIn('gamma', greeks)
        self.assertIn('theta', greeks)
        self.assertIn('vega', greeks)
        self.assertIn('rho', greeks)

class TestBinomialTree(unittest.TestCase):
    def test_european_call(self):
        price = binomial_tree_price(100, 100, 1, 0.05, 0.2, N=50, option_type='call')
        self.assertGreater(price, 0)

    def test_european_put(self):
        price = binomial_tree_price(100, 100, 1, 0.05, 0.2, N=50, option_type='put')
        self.assertGreater(price, 0)

class TestMonteCarlo(unittest.TestCase):
    def test_call_price(self):
        price = monte_carlo_option_price(100, 100, 1, 0.05, 0.2, n_paths=1000, option_type='call')
        self.assertGreater(price, 0)

    def test_put_price(self):
        price = monte_carlo_option_price(100, 100, 1, 0.05, 0.2, n_paths=1000, option_type='put')
        self.assertGreater(price, 0)

if __name__ == '__main__':
    unittest.main() 