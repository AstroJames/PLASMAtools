"""
PLASMAtools: Tensor Operations

This module provides optimized tensor operations using Numba for high-performance computing,
including tensor decomposition, contraction, and vector-tensor operations. It supports both 2D and 3D tensors.

This code is designed to be used with the Numba library for high-performance numerical computing in Python.

Author: James R. Beattie

"""

import numpy as np
from typing import Tuple, Union
from .core_functions import *


class TensorOperations:
    """
    A class to perform operations on tensor fields. 
    No data objects. Only methods.
    
    """
    def __init__(
        self, 
        use_numba: bool = True):
        """
        Initialize the TensorOperations class.

        Args:
            use_numba (bool, optional): use Numba core functions. Defaults to True.
        """
        self.use_numba = use_numba


    def tensor_magnitude(
        self, 
        tensor_field: np.ndarray) -> np.ndarray:
        """Optimized tensor magnitude using fused kernel"""
        if self.use_numba and tensor_field.ndim == 5:  # 3D
            return tensor_magnitude_3D_nb_core(tensor_field)
        else:
            # Fallback to einsum
            return tensor_magnitude_np_core(tensor_field)
    
    
    def tensor_double_contraction_ij_ij(
        self, 
        tensor_field_0: np.ndarray,
        tensor_field_1: np.ndarray) -> np.ndarray:
        """Optimized A_ij*B_ij contraction"""
        if self.use_numba and tensor_field_0.ndim == 5:  # 3D
            return tensor_double_contraction_ij_ij_3D_nb_core(tensor_field_0, tensor_field_1)
        else:
            return tensor_double_contraction_ij_ij_np_core(tensor_field_0, tensor_field_1)
    
    
    def tensor_double_contraction_ji_ij(
        self,
        tensor_field_0: np.ndarray,
        tensor_field_1: np.ndarray) -> np.ndarray:
        """Optimized A_ji*B_ij contraction"""
        if self.use_numba and tensor_field_0.ndim == 5:  # 3D
            return tensor_double_contraction_ji_ij_3D_nb_core(tensor_field_0, tensor_field_1)
        else:
            return tensor_double_contraction_ji_ij_np_core(tensor_field_0, tensor_field_1)
    
    
    def vector_dot_tensor_i_ij(
        self,
        vector_field: np.ndarray,
        tensor_field: np.ndarray) -> np.ndarray:
        """Optimized A_i*B_ij"""
        if self.use_numba and vector_field.ndim == 4:  # 3D
            return vector_dot_tensor_i_ij_3D_nb_core(vector_field, tensor_field)
        else:
            return vector_dot_tensor_i_ij_np_core(vector_field, tensor_field)
    
    
    def tensor_transpose(
        self, 
        tensor_field: np.ndarray) -> np.ndarray:
        """Optimized tensor transpose"""
        if self.use_numba and tensor_field.ndim == 5:  # 3D
            return tensor_transpose_3D_nb_core(tensor_field)
        else:
            return tensor_transpose_np_core(tensor_field)
    
    
    def tensor_outer_product(
        self,
        vector_field_0: np.ndarray,
        vector_field_1: np.ndarray) -> np.ndarray:
        """Optimized tensor outer product"""
        if self.use_numba and vector_field_0.ndim == 4:  # 3D
            return tensor_outer_product_3D_nb_core(vector_field_0, vector_field_1)
        else:
            return tensor_outer_product_np_core(vector_field_0, vector_field_1)
    
    
    def tensor_invariants(
        self, 
        tensor_field: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute all three tensor invariants
        """
        if self.use_numba and tensor_field.ndim == 5:
            return tensor_invariants_3D_nb_core(tensor_field)
        else:
            return tensor_invariants_np_core(tensor_field)
        
        
    def orthogonal_tensor_decomposition(
        self,
        tensor_field: np.ndarray,
        num_of_dims: int = 3,
        sym: bool = False,
        asym: bool = False,
        bulk: bool = False,
        all: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Decompose tensor into three orgthogonal tensors. (1) symmetric, (2) antisymmetric, (3) trace.
        """
        
        def _stub_like_3d(tensor_field: np.ndarray) -> np.ndarray:
            # Matches 5D rank, zero-length so it's cheap; dtype matches input
            return np.empty((0, 0, 0, 0, 0), dtype=tensor_field.dtype)

        def _stub_like_2d(tensor_field: np.ndarray) -> np.ndarray:
            # For a (2,2,Nx,Ny) style kernel; keep rank, zero-length spatial dims
            return np.empty((0, 0, 0, 0), dtype=tensor_field.dtype)
        
        if all:
            sym = asym = bulk = True
        
        # Early exit if nothing requested
        if not (sym or asym or bulk):
            raise ValueError("Must request at least one component")
        
        # Pre-allocate outputs
        outputs = {}
        if sym:
            outputs['sym'] = np.empty_like(tensor_field)
        if asym:
            outputs['asym'] = np.empty_like(tensor_field)
        if bulk:
            outputs['bulk'] = np.empty_like(tensor_field)
        
        # Use fused kernel
        if num_of_dims == 3:
            tensor_decomp_3D_nb_core(
                tensor_field,
                outputs.get('sym', _stub_like_3d(tensor_field)),
                outputs.get('asym', _stub_like_3d(tensor_field)),
                outputs.get('bulk', _stub_like_3d(tensor_field)),
                (sym, asym, bulk)
            )
        else:
            tensor_decomp_2D_nb_core(
                tensor_field,
                outputs.get('sym', _stub_like_2d(tensor_field)),
                outputs.get('asym', _stub_like_2d(tensor_field)),
                outputs.get('bulk', _stub_like_2d(tensor_field)),
                (sym, asym, bulk)
            )
        
        # Return in the same format as original
        if sum([sym, asym, bulk]) == 1:
            return next(iter(outputs.values()))
        else:
            result = []
            if sym:
                result.append(outputs['sym'])
            if asym:
                result.append(outputs['asym'])
            if bulk:
                result.append(outputs['bulk'])
            return tuple(result)