"""
Unit Tests for Black-Scholes Model
"""
import unittest
import numpy as np
from src.models.black_scholes import BlackScholesModel

class TestBlackScholesModel(unittest.TestCase):
    def setUp(self):
        self.model = BlackScholesModel()
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2

    def test_call_price(self):
        call_price = self.model.price_option(self.S, self.K, self.T, self.r, self.sigma, 'call')
        self.assertGreater(call_price, 0)
        self.assertIsInstance(call_price, float)

    def test_put_price(self):
        put_price = self.model.price_option(self.S, self.K, self.T, self.r, self.sigma, 'put')
        self.assertGreater(put_price, 0)
        self.assertIsInstance(put_price, float)

    def test_put_call_parity(self):
        call_price = self.model.price_option(self.S, self.K, self.T, self.r, self.sigma, 'call')
        put_price = self.model.price_option(self.S, self.K, self.T, self.r, self.sigma, 'put')
        left_side = call_price - put_price
        right_side = self.S - self.K * np.exp(-self.r * self.T)
        self.assertAlmostEqual(left_side, right_side, places=4)

if __name__ == "__main__":
    unittest.main() 