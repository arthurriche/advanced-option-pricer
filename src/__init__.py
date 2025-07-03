"""
Advanced Option Pricer & Risk Management System

A comprehensive Python library for option pricing, risk management,
and quantitative finance applications.
"""

__version__ = "1.0.0"
__author__ = "Quantitative Finance Portfolio"
__email__ = "portfolio@example.com"

# Core imports
from .models.black_scholes import BlackScholesModel
from .models.binomial_tree import BinomialTreeModel
from .models.monte_carlo import MonteCarloModel
from .models.finite_difference import FiniteDifferenceModel
from .models.stochastic_vol import StochasticVolatilityModel

from .risk.greeks import GreeksCalculator
from .risk.var import ValueAtRisk
from .risk.hedging import HedgingStrategy

from .utils.market_data import MarketDataLoader
from .utils.volatility import VolatilitySurface
from .utils.visualization import OptionVisualizer

__all__ = [
    # Models
    'BlackScholesModel',
    'BinomialTreeModel', 
    'MonteCarloModel',
    'FiniteDifferenceModel',
    'StochasticVolatilityModel',
    
    # Risk Management
    'GreeksCalculator',
    'ValueAtRisk',
    'HedgingStrategy',
    
    # Utilities
    'MarketDataLoader',
    'VolatilitySurface',
    'OptionVisualizer'
] 