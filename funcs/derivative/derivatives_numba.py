"""

PLASMAtools: Numba-optimized derivatives for vector fields

This module provides Numba-optimized functions for computing derivatives of vector fields,
including gradient, curl, and divergence operations. It supports both 1D, 2D, and 3D vector fields with various boundary conditions.

This code is designed to be used with the Numba library for high-performance numerical computing in Python.

Author: James R. Beattie

"""

import numpy as np
from numba import njit, prange, types
from typing import Union

# Define common type signatures
float32_1d = types.float32[:]
float64_1d = types.float64[:]
float32_2d = types.float32[:, :]
float64_2d = types.float64[:, :]
float32_3d = types.float32[:, :, :]
float64_3d = types.float64[:, :, :]
float32_4d = types.float32[:, :, :, :]
float64_4d = types.float64[:, :, :, :]
float32_5d = types.float32[:, :, :, :, :]
float64_5d = types.float64[:, :, :, :, :]
int32_1d = types.int32[:]
int64_1d = types.int64[:]

# Existing signatures...
sig_1d_f32 = types.float32[:](float32_1d, types.float32, int64_1d, float64_1d, types.int32, float32_1d)
sig_1d_f64 = types.float64[:](float64_1d, types.float64, int64_1d, float64_1d, types.int32, float64_1d)
sig_2d_f32 = types.float32[:,:](float32_2d, types.int32, types.float32, int64_1d, float64_1d, types.int32, float32_2d)
sig_2d_f64 = types.float64[:,:](float64_2d, types.int32, types.float64, int64_1d, float64_1d, types.int32, float64_2d)
sig_3d_f32 = types.float32[:,:,:](float32_3d, types.int32, types.float32, int64_1d, float64_1d, types.int32, float32_3d)
sig_3d_f64 = types.float64[:,:,:](float64_3d, types.int32, types.float64, int64_1d, float64_1d, types.int32, float64_3d)
sig_tensor_f32 = types.float32[:,:,:,:,:](float32_4d, int64_1d, float64_1d, types.float32, float32_5d)
sig_tensor_f64 = types.float64[:,:,:,:,:](float64_4d, int64_1d, float64_1d, types.float64, float64_5d)

# New signatures for fused operations
sig_curl_3d_f32 = types.float32[:,:,:,:](float32_4d, int64_1d, float64_1d, types.float32, types.float32, types.float32, types.int32, float32_4d)
sig_curl_3d_f64 = types.float64[:,:,:,:](float64_4d, int64_1d, float64_1d, types.float64, types.float64, types.float64, types.int32, float64_4d)
sig_div_3d_f32 = types.float32[:,:,:](float32_4d, int64_1d, float64_1d, types.float32, types.float32, types.float32, types.int32, float32_3d)
sig_div_3d_f64 = types.float64[:,:,:](float64_4d, int64_1d, float64_1d, types.float64, types.float64, types.float64, types.int32, float64_3d)

# Existing gradient functions...
@njit([sig_1d_f32, sig_1d_f64], parallel=True, fastmath=True, cache=True)
def _gradient_axis_numba_1d(f, dr, offsets, coeffs, bc_code, out):
    N0 = f.shape[0]
    for i in prange(N0):
        s = 0.0
        for t in range(offsets.shape[0]):
            idx = i + offsets[t]
            if idx < 0 or idx >= N0:
                if bc_code == 0:
                    ii = idx % N0
                    fval = f[ii]
                elif bc_code == 1:
                    fval = 0.0
                else:
                    ii = 0 if idx < 0 else N0-1
                    fval = f[ii]
            else:
                fval = f[idx]
            s += coeffs[t] * fval
        out[i] = s / dr
    return out

@njit([sig_2d_f32, sig_2d_f64], parallel=True, fastmath=True, cache=True)
def _gradient_axis_numba_2d(f, axis, dr, offsets, coeffs, bc_code, out):
    N0, N1 = f.shape
    if axis == 0:
        for i in prange(N0):
            for j in range(N1):
                s = 0.0
                for t in range(offsets.shape[0]):
                    idx = i + offsets[t]
                    if idx < 0 or idx >= N0:
                        if bc_code == 0:
                            ii = idx % N0
                            fval = f[ii, j]
                        elif bc_code == 1:
                            fval = 0.0
                        else:
                            ii = 0 if idx < 0 else N0-1
                            fval = f[ii, j]
                    else:
                        fval = f[idx, j]
                    s += coeffs[t] * fval
                out[i, j] = s / dr
    else:
        for i in prange(N0):
            for j in range(N1):
                s = 0.0
                for t in range(offsets.shape[0]):
                    idx = j + offsets[t]
                    if idx < 0 or idx >= N1:
                        if bc_code == 0:
                            jj = idx % N1
                            fval = f[i, jj]
                        elif bc_code == 1:
                            fval = 0.0
                        else:
                            jj = 0 if idx < 0 else N1-1
                            fval = f[i, jj]
                    else:
                        fval = f[i, idx]
                    s += coeffs[t] * fval
                out[i, j] = s / dr
    return out

@njit([sig_3d_f32, sig_3d_f64], parallel=True, fastmath=True, cache=True)
def _gradient_axis_numba_3d(f, axis, dr, offsets, coeffs, bc_code, out):
    N0, N1, N2 = f.shape
    if axis == 0:
        for i in prange(N0):
            for j in range(N1):
                for k in range(N2):
                    s = 0.0
                    for t in range(offsets.shape[0]):
                        idx = i + offsets[t]
                        if idx < 0 or idx >= N0:
                            if bc_code == 0:
                                ii = idx % N0
                                fval = f[ii, j, k]
                            elif bc_code == 1:
                                fval = 0.0
                            else:
                                ii = 0 if idx < 0 else N0-1
                                fval = f[ii, j, k]
                        else:
                            fval = f[idx, j, k]
                        s += coeffs[t] * fval
                    out[i, j, k] = s / dr
    elif axis == 1:
        for i in prange(N0):
            for j in range(N1):
                for k in range(N2):
                    s = 0.0
                    for t in range(offsets.shape[0]):
                        idx = j + offsets[t]
                        if idx < 0 or idx >= N1:
                            if bc_code == 0:
                                jj = idx % N1
                                fval = f[i, jj, k]
                            elif bc_code == 1:
                                fval = 0.0
                            else:
                                jj = 0 if idx < 0 else N1-1
                                fval = f[i, jj, k]
                        else:
                            fval = f[i, idx, k]
                        s += coeffs[t] * fval
                    out[i, j, k] = s / dr
    else:
        for i in prange(N0):
            for j in range(N1):
                for k in range(N2):
                    s = 0.0
                    for t in range(offsets.shape[0]):
                        idx = k + offsets[t]
                        if idx < 0 or idx >= N2:
                            if bc_code == 0:
                                kk = idx % N2
                                fval = f[i, j, kk]
                            elif bc_code == 1:
                                fval = 0.0
                            else:
                                kk = 0 if idx < 0 else N2-1
                                fval = f[i, j, kk]
                        else:
                            fval = f[i, j, idx]
                        s += coeffs[t] * fval
                    out[i, j, k] = s / dr
    return out

@njit([sig_tensor_f32, sig_tensor_f64], parallel=True, fastmath=True, cache=True)
def gradient_tensor_fused(f3, offsets, coeffs, dr, out):
    """
    Fused Numba kernel to compute the 3x3 gradient tensor of a 3D vector field.
    """
    M, N, P, Q = f3.shape
    for comp in range(M):
        for coord in range(M):
            for i in prange(N):
                for j in range(P):
                    for k in range(Q):
                        s = 0.0
                        if coord == 0:
                            for t in range(offsets.shape[0]):
                                ii = (i + offsets[t]) % N
                                s += coeffs[t] * f3[comp, ii, j, k]
                        elif coord == 1:
                            for t in range(offsets.shape[0]):
                                jj = (j + offsets[t]) % P
                                s += coeffs[t] * f3[comp, i, jj, k]
                        else:
                            for t in range(offsets.shape[0]):
                                kk = (k + offsets[t]) % Q
                                s += coeffs[t] * f3[comp, i, j, kk]
                        out[comp, coord, i, j, k] = s / dr
    return out

# NEW: Fused curl computation
@njit([sig_curl_3d_f32, sig_curl_3d_f64], parallel=True, fastmath=True, cache=True)
def vector_curl_fused_3d(f3, offsets, coeffs, dx, dy, dz, bc_code, out):
    """
    Compute curl of a 3D vector field in a single fused kernel.
    f3: shape (3, Nx, Ny, Nz)
    out: shape (3, Nx, Ny, Nz) - preallocated
    
    curl = (∂Vz/∂y - ∂Vy/∂z, ∂Vx/∂z - ∂Vz/∂x, ∂Vy/∂x - ∂Vx/∂y)
    """
    _, Nx, Ny, Nz = f3.shape
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Component 0: ∂Vz/∂y - ∂Vy/∂z
                dVz_dy = 0.0
                dVy_dz = 0.0
                
                # ∂Vz/∂y
                for t in range(offsets.shape[0]):
                    idx = j + offsets[t]
                    if bc_code == 0:  # periodic
                        jj = idx % Ny
                    else:
                        jj = max(0, min(idx, Ny-1))
                    dVz_dy += coeffs[t] * f3[2, i, jj, k]
                
                # ∂Vy/∂z
                for t in range(offsets.shape[0]):
                    idx = k + offsets[t]
                    if bc_code == 0:  # periodic
                        kk = idx % Nz
                    else:
                        kk = max(0, min(idx, Nz-1))
                    dVy_dz += coeffs[t] * f3[1, i, j, kk]
                
                out[0, i, j, k] = dVz_dy / dy - dVy_dz / dz
                
                # Component 1: ∂Vx/∂z - ∂Vz/∂x
                dVx_dz = 0.0
                dVz_dx = 0.0
                
                # ∂Vx/∂z
                for t in range(offsets.shape[0]):
                    idx = k + offsets[t]
                    if bc_code == 0:  # periodic
                        kk = idx % Nz
                    else:
                        kk = max(0, min(idx, Nz-1))
                    dVx_dz += coeffs[t] * f3[0, i, j, kk]
                
                # ∂Vz/∂x
                for t in range(offsets.shape[0]):
                    idx = i + offsets[t]
                    if bc_code == 0:  # periodic
                        ii = idx % Nx
                    else:
                        ii = max(0, min(idx, Nx-1))
                    dVz_dx += coeffs[t] * f3[2, ii, j, k]
                
                out[1, i, j, k] = dVx_dz / dz - dVz_dx / dx
                
                # Component 2: ∂Vy/∂x - ∂Vx/∂y
                dVy_dx = 0.0
                dVx_dy = 0.0
                
                # ∂Vy/∂x
                for t in range(offsets.shape[0]):
                    idx = i + offsets[t]
                    if bc_code == 0:  # periodic
                        ii = idx % Nx
                    else:
                        ii = max(0, min(idx, Nx-1))
                    dVy_dx += coeffs[t] * f3[1, ii, j, k]
                
                # ∂Vx/∂y
                for t in range(offsets.shape[0]):
                    idx = j + offsets[t]
                    if bc_code == 0:  # periodic
                        jj = idx % Ny
                    else:
                        jj = max(0, min(idx, Ny-1))
                    dVx_dy += coeffs[t] * f3[0, i, jj, k]
                
                out[2, i, j, k] = dVy_dx / dx - dVx_dy / dy
    
    return out

# NEW: Fused divergence computation
@njit([sig_div_3d_f32, sig_div_3d_f64], parallel=True, fastmath=True, cache=True)
def vector_divergence_fused_3d(f3, offsets, coeffs, dx, dy, dz, bc_code, out):
    """
    Compute divergence of a 3D vector field in a single fused kernel.
    f3: shape (3, Nx, Ny, Nz)
    out: shape (Nx, Ny, Nz) - preallocated
    
    div = ∂Vx/∂x + ∂Vy/∂y + ∂Vz/∂z
    """
    _, Nx, Ny, Nz = f3.shape
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # ∂Vx/∂x
                dVx_dx = 0.0
                for t in range(offsets.shape[0]):
                    idx = i + offsets[t]
                    if bc_code == 0:  # periodic
                        ii = idx % Nx
                    else:
                        ii = max(0, min(idx, Nx-1))
                    dVx_dx += coeffs[t] * f3[0, ii, j, k]
                
                # ∂Vy/∂y
                dVy_dy = 0.0
                for t in range(offsets.shape[0]):
                    idx = j + offsets[t]
                    if bc_code == 0:  # periodic
                        jj = idx % Ny
                    else:
                        jj = max(0, min(idx, Ny-1))
                    dVy_dy += coeffs[t] * f3[1, i, jj, k]
                
                # ∂Vz/∂z
                dVz_dz = 0.0
                for t in range(offsets.shape[0]):
                    idx = k + offsets[t]
                    if bc_code == 0:  # periodic
                        kk = idx % Nz
                    else:
                        kk = max(0, min(idx, Nz-1))
                    dVz_dz += coeffs[t] * f3[2, i, j, kk]
                
                out[i, j, k] = dVx_dx / dx + dVy_dy / dy + dVz_dz / dz
    
    return out


class Derivative:
    """
    Derivative class using Numba for periodic finite-difference.
    Now with fused operations for curl and divergence.
    """
    def __init__(self, stencil: int = 2) -> None:
        if stencil not in (2, 4, 6, 8):
            raise ValueError("Invalid stencil order: choose 2,4,6, or 8")
        self.stencil = stencil
        # first-derivative offsets & coeffs
        if stencil == 2:
            self.offsets1 = np.array([-1, +1], dtype=np.int64)
            self.coeffs1  = np.array([ -0.5, +0.5], dtype=np.float64)
            self.offsets2 = np.array([-1, 0, +1], dtype=np.int64)
            self.coeffs2  = np.array([1.0, -2.0, 1.0], dtype=np.float64)
        elif stencil == 4:
            self.offsets1 = np.array([-2, -1, +1, +2], dtype=np.int64)
            self.coeffs1  = np.array([+1/12, -8/12, +8/12, -1/12], dtype=np.float64)
            self.offsets2 = np.array([-2, -1, 0, +1, +2], dtype=np.int64)
            self.coeffs2  = np.array([-1, 16, -30, 16, -1], dtype=np.float64)/12.0
        elif stencil == 6:
            self.offsets1 = np.array([-3,-2,-1,+1,+2,+3], dtype=np.int64)
            self.coeffs1  = np.array([-1/60, +9/60, -45/60, +45/60, -9/60, +1/60], dtype=np.float64)
            self.offsets2 = np.array([-3,-2,-1,0,+1,+2,+3], dtype=np.int64)
            self.coeffs2  = np.array([-2, +27, -270, +490, -270, +27, -2], dtype=np.float64)/180.0
        else:  # stencil==8
            self.offsets1 = np.array([-4,-3,-2,-1,+1,+2,+3,+4], dtype=np.int64)
            self.coeffs1 = np.array([+1/280,-4/105,+1/5,-4/5,+4/5,-1/5,+4/105,-1/280], dtype=np.float64)
            self.offsets2 = np.array([-4,-3,-2,-1,0,+1,+2,+3,+4], dtype=np.int64)
            self.coeffs2  = np.array([-9, +128, -1008, +8064, -14350, +8064, -1008, +128, -9], dtype=np.float64)/5040.0

    def gradient(self,
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
        else:
            bc_code = 2
        ndim = scalar_field.ndim
        if gradient_dir < 0 or gradient_dir >= ndim:
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
        if ndim == 1:
            return _gradient_axis_numba_1d(scalar_field, dr if derivative_order==1 else dr*dr,
                                           offsets, coeffs, bc_code, out)
        elif ndim == 2:
            return _gradient_axis_numba_2d(scalar_field, gradient_dir,
                                           dr if derivative_order==1 else dr*dr,
                                           offsets, coeffs, bc_code, out)
        elif ndim == 3:
            return _gradient_axis_numba_3d(scalar_field, gradient_dir,
                                           dr if derivative_order==1 else dr*dr,
                                           offsets, coeffs, bc_code, out)
        else:
            raise NotImplementedError("gradient only implemented for ndim<=3")

    def gradient_tensor_fast(self, f3: np.ndarray, L: float) -> np.ndarray:
        """
        Compute the full 3×3 gradient tensor in a single fused Numba call.
        """
        offsets, coeffs = self.offsets1, self.coeffs1
        dr = self.compute_dr(L, f3.shape, 1)
        N = f3.shape[1]
        out = np.empty((f3.shape[0], f3.shape[0], N, N, N), dtype=f3.dtype)
        return gradient_tensor_fused(f3, offsets, coeffs, dr, out)

    def vector_curl_fast(self, 
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
        if vector_field.shape[0] != 3:
            raise ValueError("vector_field must have shape (3, ...)")
        
        # Handle domain size
        if isinstance(L, (int, float)):
            Lx = Ly = Lz = float(L)
        else:
            Lx, Ly, Lz = L[0], L[1], L[2]
        
        # Compute grid spacings
        dx = Lx / vector_field.shape[1]
        dy = Ly / vector_field.shape[2]
        dz = Lz / vector_field.shape[3]
        
        # BC code
        bc_code = 0 if boundary_condition == "periodic" else 2
        
        # Allocate output
        out = np.empty_like(vector_field)
        
        # Call fused kernel
        return vector_curl_fused_3d(vector_field, self.offsets1, self.coeffs1,
                                   dx, dy, dz, bc_code, out)

    def vector_divergence_fast(self,
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
        if vector_field.shape[0] != 3:
            raise ValueError("vector_field must have shape (3, ...)")
        
        # Handle domain size
        if isinstance(L, (int, float)):
            Lx = Ly = Lz = float(L)
        else:
            Lx, Ly, Lz = L[0], L[1], L[2]
        
        # Compute grid spacings
        dx = Lx / vector_field.shape[1]
        dy = Ly / vector_field.shape[2]
        dz = Lz / vector_field.shape[3]
        
        # BC code
        bc_code = 0 if boundary_condition == "periodic" else 2
        
        # Allocate output
        out = np.empty(vector_field.shape[1:], dtype=vector_field.dtype)
        
        # Call fused kernel
        return vector_divergence_fused_3d(vector_field, self.offsets1, self.coeffs1,
                                         dx, dy, dz, bc_code, out)

    def compute_dr(self, L: float, shape: tuple, gradient_dir: int) -> float:
        return L / shape[gradient_dir]