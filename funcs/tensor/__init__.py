"""
PLASMAtools Tensor Operations Module

Provides optimized tensor operations for high-performance for manipulating tensor fields,
including tensor decomposition, contractions, and tensor magnitude calculations.

This module is designed for high-performance analysis in plasma physics and related fields.
"""

# Import main classes
from .operations import TensorOperations


# Import core functions for advanced users
from .core_functions import (
    tensor_decomp_3D_nb_core,
    tensor_decomp_2D_nb_core,
    tensor_double_contraction_ij_ij_3D_nb_core,
    tensor_double_contraction_ji_ij_3D_nb_core,
    tensor_magnitude_3D_nb_core,
    vector_dot_tensor_i_ij_3D_nb_core,
    tensor_outer_product_3D_nb_core,
    tensor_transpose_3D_nb_core,
    tensor_invariants_3D_nb_core,
    tensor_magnitude_np_core,
    tensor_double_contraction_ij_ij_np_core,
    tensor_double_contraction_ji_ij_np_core,
    vector_dot_tensor_i_ij_np_core,
    tensor_transpose_np_core,
    tensor_outer_product_np_core,
    smooth_gradient_tensor_np_core,
    tensor_invariants_np_core
)

# Version info
__version__ = "1.0.0"
__author__ = "James R. Beattie"

# Define public API
__all__ = [
    'TensorOperations',
    # Core functions for advanced use
    'tensor_decomp_3D_nb_core',
    'tensor_decomp_2D_nb_core',
    'tensor_double_contraction_ij_ij_3D_nb_core',
    'tensor_double_contraction_ji_ij_3D_nb_core',
    'tensor_magnitude_3D_nb_core',
    'vector_dot_tensor_i_ij_3D_nb_core',
    'tensor_outer_product_3D_nb_core',
    'tensor_transpose_3D_nb_core',
    'tensor_invariants_3D_nb_core',
    'tensor_magnitude_np_core',
    'tensor_double_contraction_ij_ij_np_core',
    'tensor_double_contraction_ji_ij_np_core',
    'vector_dot_tensor_i_ij_np_core',
    'tensor_transpose_np_core',
    'tensor_outer_product_np_core',
    'smooth_gradient_tensor_np_core',
    'tensor_invariants_np_core'
]