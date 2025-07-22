import numpy as np
from numba import njit, prange
from .constants import *

##########################################################################################
# Core numba JIT functions for spectral operations
##########################################################################################

@njit([sig_1d_f32, sig_1d_f64], parallel=True, fastmath=True, cache=True)
def gradient_axis_numba_1d(f, dr, offsets, coeffs, bc_code, out):
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
def gradient_axis_numba_2d(f, axis, dr, offsets, coeffs, bc_code, out):
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
def gradient_axis_numba_3d(f, axis, dr, offsets, coeffs, bc_code, out):
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


@njit([sig_curl_3d_f32, sig_curl_3d_f64], 
      parallel=True, fastmath=True, cache=True)
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