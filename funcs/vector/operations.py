"""
    PLASMAtools Vector Operations Module

    This module provides vector operations such as magnitude, dot product,
    cross product, normalization, and more, using Numba for performance optimization.
    It supports both 2D and 3D vector fields and can fall back to NumPy
    implementations when Numba is not used. 

    Author: James R. Beattie

"""

import numpy as np
from .constants import *
from .core_functions import *


class VectorOperations():
    """
    Vector Operations using Numba kernels
    """
    
    def __init__(
        self,
        num_of_dims: int = 3,
        use_numba: bool = True) -> None:
        
        self.num_of_dims = num_of_dims
        self.use_numba = use_numba
        
    
    def vector_magnitude(
        self, 
        vector_field: np.ndarray) -> np.ndarray:
        """
        Vector magnitude using fused kernel
        """
        out = np.zeros_like(vector_field[0,...])
        if self.use_numba:
            if self.num_of_dims == 3 and vector_field.ndim == 4:
                out = vector_magnitude_3D_nb_core(
                    vector_field)
                return out[np.newaxis,...]
            elif self.num_of_dims == 2 and vector_field.ndim == 3:
                out = vector_magnitude_2D_nb_core(
                    vector_field)
                return out[np.newaxis,...]
        else:
            out = vector_magnitude_np_core(vector_field)
            return out[np.newaxis,...]
    
    
    def vector_dot_product(
        self, 
        vector_field_1: np.ndarray,
        vector_field_2: np.ndarray) -> np.ndarray:
        """
        Vector dot product
        """
        out = np.zeros_like(vector_field_1[0,...])
        if self.use_numba:
            if self.num_of_dims == 3 and vector_field_1.ndim == 4:
                out = vector_dot_product_3D_nb_core(
                    vector_field_1,
                    vector_field_2)
                return out[np.newaxis,...] 
            elif self.num_of_dims == 2 and vector_field_1.ndim == 3:
                out = vector_dot_product_2D_nb_core(
                    vector_field_1,
                    vector_field_2)
                return out[np.newaxis,...]
        else:
            out = vector_dot_product_np_core(vector_field_1, vector_field_2)
            return out[np.newaxis,...]
    
    
    def vector_cross_product(
        self,
        vector_field_1: np.ndarray,
        vector_field_2: np.ndarray) -> np.ndarray:
        """
        Vector cross product
        """
        # preallocate
        out = np.zeros_like(vector_field_1)
        if self.num_of_dims == 1:
            raise ValueError("Vector cross product is not defined for 1D.")
        
        if self.use_numba:
            if self.num_of_dims == 3 and vector_field_1.ndim == 4:
                out = vector_cross_product_3D_nb_core(
                    vector_field_1,
                    vector_field_2)
                return out
            elif self.num_of_dims == 2 and vector_field_1.ndim == 3:
                out = vector_cross_product_2D_nb_core(
                    vector_field_1,
                    vector_field_2)
                return out
        else:
            # Fallback to original implementation
            out = vector_cross_product_np_core(vector_field_1, vector_field_2)
            return out
    
    
    def vector_normalize(
        self, 
        vector_field: np.ndarray, 
        epsilon: float = DEFAULT_EPS) -> np.ndarray:
        """
        Normalize vector field to unit vectors
        """
        out = np.zeros_like(vector_field)
        if self.use_numba and self.num_of_dims == 3 and vector_field.ndim == 4:
            out = vector_normalize_3D_nb_core(
                vector_field,
                epsilon=epsilon)
            return out
        else:
            out = vector_normalize_np_core(
                vector_field,
                epsilon=epsilon)
            return out
    
    
    def vector_triple_product(
        self, 
        vector_field_1: np.ndarray, 
        vector_field_2: np.ndarray, 
        vector_field_3: np.ndarray) -> np.ndarray:
        """
        Compute scalar triple product: vec1 · (vec2 x vec3)
        """
        out = np.zeros_like(vector_field_1[0,...])
        if self.use_numba and self.num_of_dims == 3 and vector_field_1.ndim == 4:
            out = vector_triple_product_3D_nb_core(
                vector_field_1,
                vector_field_2,
                vector_field_3)
            return out[np.newaxis,...]
        else:
            out = vector_triple_product_np_core(
                vector_field_1,
                vector_field_2,
                vector_field_3)
            return out[np.newaxis,...]
    
    
    def vector_angle(
        self, 
        vector_field_1: np.ndarray, 
        vector_field_2: np.ndarray, 
        epsilon: float = DEFAULT_EPS) -> np.ndarray:
        """
        Compute angle between two vector fields
        """
        out = np.zeros_like(vector_field_1[0,...])
        if self.use_numba and self.num_of_dims == 3 and vector_field_1.ndim == 4:
            out = vector_angle_3D_nb_core(
                vector_field_1, 
                vector_field_2, 
                epsilon=epsilon)
            return out[np.newaxis,...]
        else:
            out = vector_angle_np_core(
                vector_field_1,
                vector_field_2,
                epsilon=epsilon)
            return out[np.newaxis,...]
            

    def vector_projection(
        self, 
        vector_field_1: np.ndarray, 
        vector_field_2: np.ndarray) -> np.ndarray:
        """
        Project vector field 1 onto vector field 2
        """
        out = np.zeros_like(vector_field_1)
        if self.use_numba and self.num_of_dims == 3 and vector_field_1.ndim == 4:
            out = vector_projection_3D_nb_core(
                vector_field_1,
                vector_field_2)
            return out
        else:   
            out = vector_projection_np_core(vector_field_1,
                                             vector_field_2)      
            return out 
    
