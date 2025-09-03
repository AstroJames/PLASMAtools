from numba import njit, prange
import numpy as np
from scipy.ndimage import uniform_filter
from .constants import *
from typing import Tuple

##########################################################################################
# Core numba JIT functions for tensor operations
##########################################################################################

@njit([tensor_decomp_3d_sig_32,tensor_decomp_3d_sig_64], parallel=True, fastmath=True, cache=True)
def tensor_decomp_3D_nb_core(
    tensor_field, 
    out_sym, 
    out_asym, 
    out_bulk, 
    compute_flags):
    """
    Fused kernel for orthogonal tensor decomposition in 3D
    
    Args:
        tensor_field: Input tensor (3, 3, Nx, Ny, Nz)
        out_sym: Output symmetric part (3, 3, Nx, Ny, Nz)
        out_asym: Output antisymmetric part (3, 3, Nx, Ny, Nz)
        out_bulk: Output bulk part (3, 3, Nx, Ny, Nz)
        compute_flags: (compute_sym, compute_asym, compute_bulk)
    """
    compute_sym, compute_asym, compute_bulk = compute_flags
    Nx, Ny, Nz = tensor_field.shape[2], tensor_field.shape[3], tensor_field.shape[4]
    
    # Process each spatial point
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Compute trace (divergence)
                trace = (tensor_field[0, 0, i, j, k] + 
                        tensor_field[1, 1, i, j, k] + 
                        tensor_field[2, 2, i, j, k])
                trace_third = trace / 3.0
                
                # Process each tensor component
                for m in range(3):
                    for n in range(3):
                        T_mn = tensor_field[m, n, i, j, k]
                        T_nm = tensor_field[n, m, i, j, k]
                        
                        if compute_sym:
                            # Symmetric part: 0.5*(T_mn + T_nm) - (1/3)*trace*delta_mn
                            sym_part = 0.5 * (T_mn + T_nm)
                            if m == n:
                                sym_part -= trace_third
                            out_sym[m, n, i, j, k] = sym_part
                        
                        if compute_asym:
                            # Antisymmetric part: 0.5*(T_mn - T_nm)
                            out_asym[m, n, i, j, k] = 0.5 * (T_mn - T_nm)
                        
                        if compute_bulk:
                            # Bulk part: (1/3)*trace*delta_mn
                            if m == n:
                                out_bulk[m, n, i, j, k] = trace_third
                            else:
                                out_bulk[m, n, i, j, k] = 0.0


@njit([tensor_decomp_2d_sig_32, tensor_decomp_2d_sig_64], parallel=True, fastmath=True, cache=True)
def tensor_decomp_2D_nb_core(
    tensor_field,
    out_sym, 
    out_asym, 
    out_bulk, 
    compute_flags):
    """
    Fused kernel for orthogonal tensor decomposition in 2D
    """
    compute_sym, compute_asym, compute_bulk = compute_flags
    Nx, Ny = tensor_field.shape[2], tensor_field.shape[3]
    
    for i in prange(Nx):
        for j in range(Ny):
            # Compute trace
            trace = tensor_field[0, 0, i, j] + tensor_field[1, 1, i, j]
            trace_half = trace / 2.0
            
            for m in range(2):
                for n in range(2):
                    T_mn = tensor_field[m, n, i, j]
                    T_nm = tensor_field[n, m, i, j]
                    
                    if compute_sym:
                        sym_part = 0.5 * (T_mn + T_nm)
                        if m == n:
                            sym_part -= trace_half
                        out_sym[m, n, i, j] = sym_part
                    
                    if compute_asym:
                        out_asym[m, n, i, j] = 0.5 * (T_mn - T_nm)
                    
                    if compute_bulk:
                        if m == n:
                            out_bulk[m, n, i, j] = trace_half
                        else:
                            out_bulk[m, n, i, j] = 0.0
                            
@njit(sig_double_contraction_3d, parallel=True, fastmath=True, cache=True)
def tensor_double_contraction_ij_ij_3D_nb_core(
    tensor_a, 
    tensor_b):
    """
    Compute A_ij*B_ij for 3D tensors
    """
    Nx, Ny, Nz = tensor_a.shape[2], tensor_a.shape[3], tensor_a.shape[4]
    out = np.zeros((Nx, Ny, Nz), dtype=tensor_a.dtype)
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                sum_val = 0.0
                for m in range(3):
                    for n in range(3):
                        sum_val += tensor_a[m, n, i, j, k] * tensor_b[m, n, i, j, k]
                out[i, j, k] = sum_val
    
    return out 
                
                            
@njit(sig_double_contraction_3d, parallel=True, fastmath=True, cache=True)
def tensor_double_contraction_ji_ij_3D_nb_core(
    tensor_a,
    tensor_b):
    """
    Compute A_ji*B_ij for 3D tensors (with transpose on A)
    """
    Nx, Ny, Nz = tensor_a.shape[2], tensor_a.shape[3], tensor_a.shape[4]
    out = np.zeros((Nx, Ny, Nz), dtype=tensor_a.dtype)
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                sum_val = 0.0
                for m in range(3):
                    for n in range(3):
                        sum_val += tensor_a[n, m, i, j, k] * tensor_b[m, n, i, j, k]
                out[i, j, k] = sum_val
    
    return out


@njit(sig_tensor_magnitude_3d, parallel=True, fastmath=True, cache=True)
def tensor_magnitude_3D_nb_core(
    tensor_field):
    """
    Compute tensor magnitude sqrt(A_ij*A_ij) for 3D tensors
    """
    Nx, Ny, Nz = tensor_field.shape[2], tensor_field.shape[3], tensor_field.shape[4]
    out = np.zeros((Nx, Ny, Nz), dtype=tensor_field.dtype)
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                sum_val = 0.0
                for m in range(3):
                    for n in range(3):
                        val = tensor_field[m, n, i, j, k]
                        sum_val += val * val
                out[i, j, k] = np.sqrt(sum_val)
    
    return out


@njit(sig_vector_dot_tensor_3d, parallel=True, fastmath=True, cache=True)
def vector_dot_tensor_i_ij_3D_nb_core(
    vector_field,
    tensor_field):
    """
    Compute A_i*B_ij for 3D fields
    """
    Nx, Ny, Nz = vector_field.shape[1], vector_field.shape[2], vector_field.shape[3]
    out = np.zeros((3, Nx, Ny, Nz), dtype=vector_field.dtype)
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # For each output vector component
                for n in range(3):
                    sum_val = 0.0
                    for m in range(3):
                        sum_val += vector_field[m, i, j, k] * tensor_field[m, n, i, j, k]
                    out[n, i, j, k] = sum_val
    
    return out


@njit(sig_outer_product_3d, parallel=True, fastmath=True, cache=True)
def tensor_outer_product_3D_nb_core(
    vector_a,
    vector_b):
    """
    Compute A_i*B_j outer product for 3D vector fields
    """
    Nx, Ny, Nz = vector_a.shape[1], vector_a.shape[2], vector_a.shape[3]
    out = np.zeros((3, 3, Nx, Ny, Nz), dtype=vector_a.dtype)
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for m in range(3):
                    for n in range(3):
                        out[m, n, i, j, k] = vector_a[m, i, j, k] * vector_b[n, i, j, k]
    
    return out


@njit([sig_tensor_transpose_3d_32, sig_tensor_transpose_3d_64], parallel=True, fastmath=True, cache=True)
def tensor_transpose_3D_nb_core(tensor_field):
    """
    Compute tensor transpose for 3D tensor field
    """
    Nx, Ny, Nz = tensor_field.shape[2], tensor_field.shape[3], tensor_field.shape[4]
    
    out = np.empty_like(tensor_field)
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for m in range(3):
                    for n in range(3):
                        out[n, m, i, j, k] = tensor_field[m, n, i, j, k]
    
    return out


@njit(parallel=True, fastmath=True, cache=True)
def tensor_invariants_3D_nb_core(
    tensor_field):
    """
    Compute all three tensor invariants in a single pass
    I1 = trace(A)
    I2 = 0.5 * (trace(A)^2 - trace(A^2))
    I3 = det(A)
    """
    Nx, Ny, Nz = tensor_field.shape[2], tensor_field.shape[3], tensor_field.shape[4]
    I1 = np.zeros((Nx, Ny, Nz), dtype=tensor_field.dtype)
    I2 = np.zeros((Nx, Ny, Nz), dtype=tensor_field.dtype)
    I3 = np.zeros((Nx, Ny, Nz), dtype=tensor_field.dtype)
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Extract tensor components
                a00 = tensor_field[0, 0, i, j, k]
                a01 = tensor_field[0, 1, i, j, k]
                a02 = tensor_field[0, 2, i, j, k]
                a10 = tensor_field[1, 0, i, j, k]
                a11 = tensor_field[1, 1, i, j, k]
                a12 = tensor_field[1, 2, i, j, k]
                a20 = tensor_field[2, 0, i, j, k]
                a21 = tensor_field[2, 1, i, j, k]
                a22 = tensor_field[2, 2, i, j, k]
                
                # First invariant (trace)
                trace = a00 + a11 + a22
                I1[i, j, k] = trace
                
                # Trace of A^2
                trace_A2 = (a00*a00 + a01*a10 + a02*a20 +
                           a10*a01 + a11*a11 + a12*a21 +
                           a20*a02 + a21*a12 + a22*a22)
                
                # Second invariant
                I2[i, j, k] = 0.5 * (trace*trace - trace_A2)
                
                # Third invariant (determinant)
                I3[i, j, k] = (a00 * (a11*a22 - a12*a21) -
                              a01 * (a10*a22 - a12*a20) +
                              a02 * (a10*a21 - a11*a20))
    
    return I1, I2, I3


##########################################################################################
# Core numpy functions for tensor operations
##########################################################################################


def tensor_magnitude_np_core(
    tensor_field : np.ndarray) -> np.ndarray:
    """
    Compute the tensor magnitude of a tensor field.
    Args:
        tensor_field (np.ndarray) : (i,j),N,N,N array of tensor field, where 
                                    (i,j) are the tensor components and N is the number of grid
                                    points in each direction
    Returns:
        tensor_mag (np.ndarray) : N,N,N array of tensor magnitude of the tensor field
    """
    
    out = np.zeros_like(tensor_field[0,0,...])
    out = np.sqrt(
        tensor_double_contraction_ij_ij_np_core(tensor_field,
                                        tensor_field))
    
    return out


def tensor_double_contraction_ij_ij_np_core(
    tensor_field_0 : np.ndarray,
    tensor_field_1 : np.ndarray) -> np.ndarray:
    """
    Compute the A_ijA_ij scalar field from a tensor field.
    Args:
        tensor_field_0 (np.ndarray) : (i,j),N,N,N array of tensor field, where 
                                        (i,j) are the tensor components and N is the number of grid
                                        points in each direction
        tensor_field_1 (np.ndarray) : (i,j),N,N,N array of tensor field, where 
                                        (i,j) are the tensor components and N is the number of grid 
                                        points in each direction
    Returns:
        A_ijB_ij = a_11 b_11 + a_12 b_11 + ... a_nn b_nn: the contraction scalar field.
    """
    
    out = np.zeros_like(tensor_field_0[0,0,...])
    out = np.einsum('ij...,ij...->...',
                        tensor_field_0,
                        tensor_field_1)
        
    return out
    
    
def tensor_double_contraction_ji_ij_np_core(
    tensor_field_0 : np.ndarray,
    tensor_field_1 : np.ndarray) -> np.ndarray:
    """
    Compute the A_jiA_ij scalar field from a tensor field.
    Args:
        tensor_field_0 (np.ndarray) : (i,j),N,N,N array of tensor field, where 
                                        (i,j) are the tensor components and N is the number of grid
                                        points in each direction
        tensor_field_1 (np.ndarray) : (i,j),N,N,N array of tensor field, where 
                                        (i,j) are the tensor components and N is the number of grid 
                                        points in each direction
    Returns:
        A_jiB_ij = a_11 b_11 + a_21 b_11 + ... a_nn b_nn: the contraction scalar field.
    """
    
    out = np.zeros_like(tensor_field_0[0,0,...])
    out = np.einsum('ji...,ij...->...',
                    tensor_field_0,
                    tensor_field_1)
        
    return out
    

def vector_dot_tensor_i_ij_np_core(
    vector_field : np.ndarray,
    tensor_field : np.ndarray) -> np.ndarray:
    """
    Compute the A_iB_ij vector field from a tensor field.
    Args:
        vector (np.ndarray)         : i,N,N,N array of vector field, where 
                                        i, are the vector components and N is the number of grid 
                                        points in each direction
        tensor_field (np.ndarray)   : (i,j),N,N,N array of tensor field, where 
                                        (i,j) are the tensor components and N is the number of grid
                                        points in each direction
    Returns:
        A_iB_ij : a_1b_1j + a_2b_2j + ... contraction vector field.
    """
    
    out = np.zeros_like(vector_field)
    out = np.einsum('i...,ij...->j...',
                    vector_field,
                    tensor_field)
        
    return out


def tensor_transpose_np_core(
    tensor_field : np.ndarray) -> np.ndarray:
    """
    Compute the transpose of tensor field.    
    Args:
        tensor_field (np.ndarray): M,M,N,N,N array of tensor field, where 
        M is the tensor component and N is the number of grid points in each 
        direction
    Returns:
        the transpose A_ji of the A_ij tensor field
    """
    
    out = np.zeros_like(tensor_field)
    out = np.einsum('ij... -> ji...',
                    tensor_field)
        
    return out
    
    
def tensor_outer_product_np_core(
    vector_field_0 : np.ndarray,
    vector_field_1 : np.ndarray) -> np.ndarray:
    """
    Compute the A_iB_j tensor field from a vector field.
    Args:
        vector_field_0 (np.ndarray): 3,N,N,N array of vector field, where 
        3 is the vector component and N is the number of grid points in each 
        direction
        vector_field_1 (np.ndarray): 3,N,N,N array of vector field, where 
        3 is the vector component and N is the number of grid points in each 
        direction
    Returns:
        A_i_B_j: the A_iB_j tensor field
    """
        
    Nx, Ny, Nz = vector_field_0.shape[1], vector_field_0.shape[2], vector_field_0.shape[3]
    out = np.zeros((3, 3, Nx, Ny, Nz),
                   dtype=np.float32)
    out = np.einsum('i...,j...->ij...',
                    vector_field_0,
                    vector_field_1)
        
    return out
    
    
def smooth_gradient_tensor_np_core(
    tensor_field : np.ndarray, 
    smoothing_size : int = DEFAULT_SIGMA) -> np.ndarray:
    """
    Smooth a gradient tensor field by averaging over adjacent cells.
    Args:
        gradient_tensor (np.ndarray)    : The gradient tensor to smooth (shape: 3,3,N,N,N).
        smoothing_size (int)            : The size of the smoothing window (default: 10).
    Returns:
        smoothed tensor field (np.ndarray) : The smoothed gradient tensor.
    """

    out = np.zeros_like(tensor_field,
                        dtype=np.float32)
    out = uniform_filter(tensor_field, 
                         size  = smoothing_size, 
                         axes  = (-3, -2, -1), 
                         mode  = 'nearest')

    return out


def tensor_invariants_np_core(
    tensor_field : np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the trace, determinant and discriminant
    Args:
        tensor_field (np.ndarray): M,M,N,N,N array of tensor field, where 
        M is the tensor component and N is the number of grid points in each 
        direction
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    """
    trace = np.trace(tensor_field, axis1=0, axis2=1)
    A2 = np.einsum('ik...,kj...->ij...', tensor_field, tensor_field)
    trace_A2 = np.trace(A2, axis1=0, axis2=1)
    I1 = trace
    I2 = 0.5 * (trace**2 - trace_A2)
    I3 = np.linalg.det(np.moveaxis(tensor_field, [0, 1], [-2, -1]))

    return I1, I2, I3