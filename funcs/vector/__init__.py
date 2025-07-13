"""
PLASMAtools Vector Operations Module

Provides optimized vector operations for high-performance manipulation of vector fields,
including vector magnitude, dot product, cross product, normalization, and angle calculations.

This module is designed for high-performance analysis in plasma physics and related fields.
"""

# Import main classes
from .operations import VectorOperations


# Import core functions for advanced users
from .core_functions import (
    vector_magnitude_3D_nb_core,
    vector_magnitude_2D_nb_core,
    vector_dot_product_3D_nb_core,
    vector_dot_product_2D_nb_core,
    vector_cross_product_3D_nb_core,
    vector_cross_product_2D_nb_core,
    vector_normalize_3D_nb_core,
    vector_normalize_2D_nb_core,
    vector_triple_product_3D_nb_core,
    vector_angle_3D_nb_core,
    vector_magnitude_np_core,
    vector_dot_product_np_core,
    vector_cross_product_np_core,
    vector_normalize_np_core
)

# Version info
__version__ = "1.0.0"
__author__ = "James R. Beattie"

# Define public API
__all__ = [
    'VectorOperations',
    # Core functions for advanced use
    'vector_magnitude_3D_nb_core',
    'vector_magnitude_2D_nb_core',
    'vector_dot_product_3D_nb_core',
    'vector_dot_product_2D_nb_core',
    'vector_cross_product_3D_nb_core',
    'vector_cross_product_2D_nb_core',
    'vector_normalize_3D_nb_core',
    'vector_normalize_2D_nb_core',
    'vector_triple_product_3D_nb_core',
    'vector_angle_3D_nb_core',
    'vector_magnitude_np_core',
    'vector_dot_product_np_core',
    'vector_cross_product_np_core',
    'vector_normalize_np_core'
]