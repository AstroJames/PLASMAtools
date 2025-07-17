"""
PLASMAtools Derived Variables Module

Provides optimized functions for computing derived variables in plasma physics,
including vector potentials, helicities, and other related quantities.

This module is designed for high-performance spectral analysis in plasma physics and related fields.

Author: James R. Beattie

"""

# Import main classes
from .operations import DerivedVars

# Import core functions for advanced users
from .core_functions import (
    compute_vector_potential_2D_core,
    compute_vector_potential_3D_core,
    reconstruct_field_from_stream_2D_core
)

# Version info
__version__ = "1.0.0"
__author__ = "James R. Beattie"

# Define public API
__all__ = [
    'DerivedVars',
    # Core functions for advanced use
    'compute_vector_potential_2D_core',
    'compute_vector_potential_3D_core',
    'reconstruct_field_from_stream_2D_core'
]