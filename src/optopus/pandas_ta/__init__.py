# -*- coding: utf-8 -*-
"""
pandas_ta - A Technical Analysis Library using Pandas and Numpy
"""
__author__ = "Kevin Johnson"

# Import constants and expose them
from ._constants import (
    Imports, Category, CANGLE_AGG, EXCHANGE_TZ, RATE, version
)
# Import core functionalities (including the 'ta' accessor registration)
from .core import *

# Define public API
__all__ = [
    'Imports',
    'Category',
    'CANGLE_AGG',
    'EXCHANGE_TZ',
    'RATE',
    'version',
    # Plus everything imported from .core via *
    # A more explicit list derived from core.py would be better practice
]

name = "pandas_ta"
