"""
Examples Package

This package contains example scripts demonstrating:
- European option pricing
- Exotic option pricing
- Portfolio risk management
"""

from .european_options import european_option_example
from .exotic_options import exotic_option_example
from .portfolio_risk import portfolio_risk_example

__all__ = [
    'european_option_example',
    'exotic_option_example', 
    'portfolio_risk_example'
] 