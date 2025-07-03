"""
Unit Tests for Risk Management Modules

Tests for Greeks, VaR, and hedging calculations.

Author: Arthur Riche
Date: 2024
"""

import unittest
import numpy as np
from src.risk.greeks import delta, gamma, theta, vega, rho, all_greeks
from src.risk.var import historical_var, parametric_var, historical_cvar, parametric_cvar
from src.risk.hedging import delta_hedge, gamma_hedge

class TestGreeks(unittest.TestCase):
    def setUp(self):
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2

    def test_delta(self):
        call_delta = delta(self.S, self.K, self.T, self.r, self.sigma, 'call')
        put_delta = delta(self.S, self.K, self.T, self.r, self.sigma, 'put')
        self.assertGreater(call_delta, 0)
        self.assertLess(put_delta, 0)

    def test_gamma(self):
        call_gamma = gamma(self.S, self.K, self.T, self.r, self.sigma)
        put_gamma = gamma(self.S, self.K, self.T, self.r, self.sigma)
        self.assertGreater(call_gamma, 0)
        self.assertEqual(call_gamma, put_gamma)  # Gamma is same for calls and puts

    def test_all_greeks(self):
        greeks = all_greeks(self.S, self.K, self.T, self.r, self.sigma, 'call')
        self.assertIn('delta', greeks)
        self.assertIn('gamma', greeks)
        self.assertIn('theta', greeks)
        self.assertIn('vega', greeks)
        self.assertIn('rho', greeks)

class TestVaR(unittest.TestCase):
    def test_historical_var(self):
        returns = np.random.normal(0, 0.02, 1000)
        var = historical_var(returns, 0.95)
        self.assertLess(var, 0)  # VaR should be negative

    def test_parametric_var(self):
        var = parametric_var(0.001, 0.02, 0.95)
        self.assertLess(var, 0)

class TestHedging(unittest.TestCase):
    def test_delta_hedge(self):
        hedge_amount = delta_hedge(0.5, 0)
        self.assertEqual(hedge_amount, -0.5)

    def test_gamma_hedge(self):
        hedge_amount = gamma_hedge(0.01, 0, 100, 0)
        self.assertEqual(hedge_amount, -1.0)

if __name__ == '__main__':
    unittest.main() 