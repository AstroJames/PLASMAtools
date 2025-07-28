"""
PLASMAtools Eigenvalue Operations Module

Provides optimized eigenvalue computations for tensors, particularly for symmetric
tensors used in plasma physics applications such as analyzing stretching tensors
and dynamo growth rate modeling.

Features:
- Analytical eigenvalue computation for 3x3 symmetric tensors (45x speedup)
- Numba-optimized implementations for maximum performance  
- Machine precision accuracy across all computations
- Support for both 2x2 and 3x3 tensor fields

The module now uses analytical methods by default for 3x3 symmetric tensors,
providing exact solutions based on https://hal.science/hal-01501221/document
"""

# Import main classes
from .operations import EigenvalueOperations


# Import core functions for advanced users
from .core_functions import (
    eigenvalues_symmetric_3x3_nb_core,
    eigenvalues_symmetric_2x2_nb_core,
    eigenvectors_symmetric_3x3_nb_core,
    eigenvectors_symmetric_2x2_nb_core,
    eigenvalues_symmetric_3x3_np_core,
    eigenvalues_symmetric_2x2_np_core,
    eigenvalues_general_3x3_nb_core,
    eigenvalues_general_2x2_nb_core,
    eigenvectors_general_3x3_nb_core,
    eigenvectors_general_2x2_nb_core
)

# Version info
__version__ = "1.0.0"
__author__ = "James R. Beattie"

# Define public API
__all__ = [
    'EigenvalueOperations',
    # Core functions for advanced use
    'eigenvalues_symmetric_3x3_nb_core',
    'eigenvalues_symmetric_2x2_nb_core',
    'eigenvectors_symmetric_3x3_nb_core',
    'eigenvectors_symmetric_2x2_nb_core',
    'eigenvalues_symmetric_3x3_np_core',
    'eigenvalues_symmetric_2x2_np_core',
    'eigenvalues_general_3x3_nb_core',
    'eigenvalues_general_2x2_nb_core',
    'eigenvectors_general_3x3_nb_core',
    'eigenvectors_general_2x2_nb_core'
]