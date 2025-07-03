"""
Integration Tests

Tests for integration between different modules of the option pricing system.

Author: Arthur Riche
Date: 2024
"""

import unittest
import numpy as np
from src.models.black_scholes import BlackScholesModel
from src.risk.greeks import all_greeks
from src.risk.var import historical_var
from src.utils.market_data import get_sample_data

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.model = BlackScholesModel()
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2

    def test_pricing_and_greeks_consistency(self):
        # Test that pricing and Greeks are consistent
        call_price = self.model.price_call(self.S, self.K, self.T, self.r, self.sigma)
        greeks = all_greeks(self.S, self.K, self.T, self.r, self.sigma, 'call')
        
        # Basic consistency checks
        self.assertGreater(call_price, 0)
        self.assertGreater(greeks['delta'], 0)
        self.assertGreater(greeks['gamma'], 0)

    def test_market_data_integration(self):
        # Test market data integration
        data = get_sample_data()
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

    def test_risk_metrics_consistency(self):
        # Test that risk metrics are consistent
        returns = np.random.normal(0, 0.02, 1000)
        var = historical_var(returns, 0.95)
        self.assertLess(var, 0)

if __name__ == '__main__':
    unittest.main() 