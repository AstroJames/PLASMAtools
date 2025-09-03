"""
PLASMAtools Scalar Analysis Module

Provides optimized functions for computing scalar operations such as root-mean-square (RMS) values.

This module is designed for high-performance spectral analysis in plasma physics and related fields.
"""

# Import main classes
from .operations import ScalarOperations

# Import core functions for advanced users
from .core_functions import (
    scalar_rms_np_core
)

# Version info
__version__ = "1.0.0"
__author__ = "James R. Beattie"

# Define public API
__all__ = [
    'ScalarOperations',
    # Core functions for advanced use
    'scalar_rms_np_core'
]