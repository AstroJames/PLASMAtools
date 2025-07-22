"""

PLASMAtools: Numba-optimized derivatives for vector fields

This module provides Numba-optimized functions for computing derivatives of vector fields,
including gradient, curl, and divergence operations. It supports both 1D, 2D, and 3D vector fields with various boundary conditions.

This code is designed to be used with the Numba library for high-performance numerical computing in Python.

Author: James R. Beattie

"""

import numpy as np
from .core_functions import *
from typing import Union


class Derivative:
    """
    Derivative class using Numba for periodic finite-difference.
    Now with fused operations for curl and divergence.
    """
    def __init__(
        self, 
        stencil: int = 2) -> None:
        
        # Validate stencil order
        if stencil not in (2, 4, 6, 8):
            raise ValueError("Invalid stencil order: choose 2,4,6, or 8")
        self.stencil = stencil
        
        
        # first-derivative offsets & coeffs
        if stencil == 2:
            self.offsets1 = np.array([-1, +1],
                                     dtype=np.int32)
            self.coeffs1  = np.array([ -0.5, +0.5],
                                     dtype=np.float32)
            self.offsets2 = np.array([-1, 0, +1],
                                     dtype=np.int32)
            self.coeffs2  = np.array([1.0, -2.0, 1.0],
                                     dtype=np.float64)
        elif stencil == 4:
            self.offsets1 = np.array([-2, -1, +1, +2],
                                     dtype=np.int32)
            self.coeffs1  = np.array([+1/12, -8/12, +8/12, -1/12],
                                     dtype=np.float32)
            self.offsets2 = np.array([-2, -1, 0, +1, +2],
                                     dtype=np.int32)
            self.coeffs2  = np.array([-1, 16, -30, 16, -1],
                                     dtype=np.float32)/12.0
        elif stencil == 6:
            self.offsets1 = np.array([-3,-2,-1,+1,+2,+3],
                                     dtype=np.int32)
            self.coeffs1  = np.array([-1/60, +9/60, -45/60, +45/60, -9/60, +1/60],
                                     dtype=np.float32)
            self.offsets2 = np.array([-3,-2,-1,0,+1,+2,+3],
                                     dtype=np.int32)
            self.coeffs2  = np.array([-2, +27, -270, +490, -270, +27, -2],
                                     dtype=np.float32)/180.0
        else:  # stencil==8
            self.offsets1 = np.array([-4,-3,-2,-1,+1,+2,+3,+4],
                                     dtype=np.int32)
            self.coeffs1 = np.array([+1/280,-4/105,+1/5,-4/5,+4/5,-1/5,+4/105,-1/280],
                                    dtype=np.float32)
            self.offsets2 = np.array([-4,-3,-2,-1,0,+1,+2,+3,+4],
                                     dtype=np.int32)
            self.coeffs2  = np.array([-9, +128, -1008, +8064, -14350, +8064, -1008, +128, -9],
                                     dtype=np.float32)/5040.0


    def gradient(
        self,
        scalar_field       : np.ndarray,
        gradient_dir       : int,
        L                  : float = 1.0,
        derivative_order   : int = 1,
        boundary_condition : str = "periodic") -> np.ndarray:
        """
        Compute the 1st or 2nd derivative along one axis using Numba.
        """
        # map BC string to code for Numba
        if boundary_condition == 'periodic':
            bc_code = 0
        elif boundary_condition == 'dirichlet':
            bc_code = 1
        else: # neumann
            bc_code = 2
            
        if gradient_dir < 0 or gradient_dir >= scalar_field.ndim:
            raise ValueError("Invalid gradient_dir")
        scalar_field_shape = scalar_field.shape
        dr = self.compute_dr(L, scalar_field_shape, gradient_dir)
        # choose offsets/coeffs
        if derivative_order == 1:
            offsets, coeffs = self.offsets1, self.coeffs1
        else:
            offsets, coeffs = self.offsets2, self.coeffs2
        # allocate output
        out = np.empty_like(scalar_field)
        # dispatch by dimension
        if scalar_field.ndim == 1:
            return gradient_axis_numba_1d(
                scalar_field, 
                dr if derivative_order==1 else dr*dr,
                offsets, 
                coeffs, 
                bc_code, 
                out)
        elif scalar_field.ndim == 2:
            return gradient_axis_numba_2d(
                scalar_field, 
                gradient_dir,
                dr if derivative_order==1 else dr*dr,
                offsets, 
                coeffs, 
                bc_code, 
                out)
        elif scalar_field.ndim == 3:
            return gradient_axis_numba_3d(
                scalar_field, 
                gradient_dir,
                dr if derivative_order==1 else dr*dr,
                offsets, 
                coeffs, 
                bc_code, 
                out)
        else:
            raise NotImplementedError("gradient only implemented for ndim<=3")


    def gradient_tensor_fast(
        self, 
        f3: np.ndarray, 
        L: float) -> np.ndarray:
        """
        Compute the full 3x3 gradient tensor in a single fused Numba call.
        """
        offsets, coeffs = self.offsets1, self.coeffs1
        dr = self.compute_dr(
            L, 
            f3.shape, 
            1)
        N = f3.shape[X_GRID_VEC]
        out = np.empty((
            f3.shape[N_COORDS_VEC], 
            f3.shape[N_COORDS_VEC], N, N, N), 
                       dtype=f3.dtype)
        return gradient_tensor_fused(
            f3, 
            offsets, 
            coeffs, 
            dr, 
            out)


    def vector_curl_fast(
        self, 
        vector_field: np.ndarray,
        L: Union[float, np.ndarray],
        boundary_condition: str = "periodic") -> np.ndarray:
        """
        Compute curl of a 3D vector field using a single fused kernel.
        
        Args:
            vector_field: shape (3, Nx, Ny, Nz)
            L: domain size (scalar for cubic, or array [Lx, Ly, Lz])
            boundary_condition: "periodic" or "neumann"
            
        Returns:
            curl: shape (3, Nx, Ny, Nz)
        """
        if vector_field.shape[N_COORDS_VEC] != 3:
            raise ValueError("vector_field must have shape (3, ...)")
        
        # Handle domain size
        if isinstance(L, (int, float)):
            Lx = Ly = Lz = float(L)
        else:
            Lx, Ly, Lz = L[X], L[Y], L[Z]
        
        # Compute grid spacings
        dx = self.compute_dr(Lx,vector_field.shape[X_GRID_VEC:],X)
        dy = self.compute_dr(Ly,vector_field.shape[X_GRID_VEC:],Y)
        dz = self.compute_dr(Lz,vector_field.shape[X_GRID_VEC:],Z)
        
        # BC code
        bc_code = 0 if boundary_condition == "periodic" else 2
        
        # Allocate output
        out = np.empty_like(vector_field)
                
        # Call fused kernel
        return vector_curl_fused_3d(
            vector_field, 
            self.offsets1, 
            self.coeffs1,
            dx, dy, dz, bc_code, out)


    def vector_divergence_fast(
        self,
        vector_field: np.ndarray,
        L: Union[float, np.ndarray],
        boundary_condition: str = "periodic") -> np.ndarray:
        """
        Compute divergence of a 3D vector field using a single fused kernel.
        
        Args:
            vector_field: shape (3, Nx, Ny, Nz)
            L: domain size (scalar for cubic, or array [Lx, Ly, Lz])
            boundary_condition: "periodic" or "neumann"
            
        Returns:
            divergence: shape (Nx, Ny, Nz)
        """
        if vector_field.shape[N_COORDS_VEC] != 3:
            raise ValueError("vector_field must have shape (3, ...)")
        
        # Handle domain size
        if isinstance(L, (int, float)):
            Lx = Ly = Lz = float(L)
        else:
            Lx, Ly, Lz = L[X], L[Y], L[Z]
        
        # Compute grid spacings
        dx = self.compute_dr(Lx,vector_field.shape[X_GRID_VEC:],X)
        dy = self.compute_dr(Ly,vector_field.shape[X_GRID_VEC:],Y)
        dz = self.compute_dr(Lz,vector_field.shape[X_GRID_VEC:],Z)
        
        # BC code
        bc_code = 0 if boundary_condition == "periodic" else 2
        
        # Allocate output
        out = np.empty(
            vector_field.shape[X_GRID_VEC:],
            dtype=vector_field.dtype)
                
        # Call fused kernel
        return vector_divergence_fused_3d(
            vector_field, 
            self.offsets1, 
            self.coeffs1,
            dx, dy, dz, bc_code, out)


    def compute_dr(
        self, 
        L: float,
        shape: tuple,
        gradient_dir: int) -> float:
        return np.float32(L / shape[gradient_dir])