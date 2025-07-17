
import numpy as np
from numba import njit, prange
from ..tensor import TensorOperations
from ..vector import VectorOperations
from ..scalar import ScalarOperations
from ..spectral import SpectralOperations


@njit(parallel=True, fastmath=True, cache=True)
def compute_vector_potential_2D_core(
    k: np.ndarray,
    field_fft: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray) -> np.ndarray:
    """
    Compute vector potential in 2D (stream function).
    
    Args:
        k: Wave vector array (2, nx, ny)
        field_fft: FFT of vector field (2, nx, ny)
        kx: x-component of wave vector (nx, ny)
        ky: y-component of wave vector (nx, ny)
    
    Returns:
        Stream function in Fourier space (nx, ny)
    """
    nx, ny = field_fft.shape[1:]
    psi_hat = np.zeros((nx, ny), dtype=field_fft.dtype)
    
    for x in prange(nx):
        for y in range(ny):
            k_norm_sq = kx[x, y]**2 + ky[x, y]**2
            
            if k_norm_sq > 0:
                # Cross product k × F (z-component)
                cross_z = k[0, x, y] * field_fft[1, x, y] - k[1, x, y] * field_fft[0, x, y]
                # Stream function: ψ = i(k × F)_z / |k|²
                psi_hat[x, y] = 1j * cross_z / k_norm_sq
            else:
                psi_hat[x, y] = 0.0 + 0.0j
    
    return psi_hat


@njit(parallel=True, fastmath=True, cache=True)
def compute_vector_potential_3D_core(
    k: np.ndarray,
    field_fft: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray) -> np.ndarray:
    """
    Compute vector potential in 3D.
    
    Args:
        k: Wave vector array (3, nx, ny, nz)
        field_fft: FFT of vector field (3, nx, ny, nz)
        kx: x-component of wave vector (nx, ny, nz)
        ky: y-component of wave vector (nx, ny, nz)
        kz: z-component of wave vector (nx, ny, nz)
    
    Returns:
        Vector potential in Fourier space (3, nx, ny, nz)
    """
    nx, ny, nz = field_fft.shape[1:]
    a_hat = np.zeros((3, nx, ny, nz), dtype=field_fft.dtype)
    
    for x in prange(nx):
        for y in range(ny):
            for z in range(nz):
                k_norm_sq = kx[x, y, z]**2 + ky[x, y, z]**2 + kz[x, y, z]**2
                
                if k_norm_sq > 0:
                    # Get k and field components
                    k_x = k[0, x, y, z]
                    k_y = k[1, x, y, z]
                    k_z = k[2, x, y, z]
                    
                    f_x = field_fft[0, x, y, z]
                    f_y = field_fft[1, x, y, z]
                    f_z = field_fft[2, x, y, z]
                    
                    # Cross product k × F
                    cross_x = k_y * f_z - k_z * f_y
                    cross_y = k_z * f_x - k_x * f_z
                    cross_z = k_x * f_y - k_y * f_x
                    
                    # Vector potential: A = i(k × F) / |k|²
                    a_hat[0, x, y, z] = 1j * cross_x / k_norm_sq
                    a_hat[1, x, y, z] = 1j * cross_y / k_norm_sq
                    a_hat[2, x, y, z] = 1j * cross_z / k_norm_sq
                else:
                    a_hat[0, x, y, z] = 0.0 + 0.0j
                    a_hat[1, x, y, z] = 0.0 + 0.0j
                    a_hat[2, x, y, z] = 0.0 + 0.0j
    
    return a_hat


@njit(parallel=True, fastmath=True, cache=True)
def reconstruct_field_from_stream_2D_core(
    psi: np.ndarray,
    dx: float,
    dy: float) -> np.ndarray:
    """
    Reconstruct 2D vector field from stream function using curl.
    F = ∇ x ψẑ = (∂ψ/∂y, -∂ψ/∂x)
    
    Args:
        psi: Stream function (nx, ny)
        dx: Grid spacing in x
        dy: Grid spacing in y
    
    Returns:
        Reconstructed vector field (2, nx, ny)
    """
    nx, ny = psi.shape
    field = np.zeros((2, nx, ny), dtype=psi.dtype)
    
    # Use central differences for interior points
    for x in prange(1, nx-1):
        for y in range(1, ny-1):
            # F_x = ∂ψ/∂y
            field[0, x, y] = (psi[x, y+1] - psi[x, y-1]) / (2.0 * dy)
            # F_y = -∂ψ/∂x
            field[1, x, y] = -(psi[x+1, y] - psi[x-1, y]) / (2.0 * dx)
    
    # Handle boundaries with one-sided differences
    # x boundaries
    for y in range(ny):
        # Left boundary (x=0)
        field[0, 0, y] = (psi[0, min(y+1, ny-1)] - psi[0, max(y-1, 0)]) / (2.0 * dy) if ny > 1 else 0.0
        field[1, 0, y] = -(psi[1, y] - psi[0, y]) / dx if nx > 1 else 0.0
        
        # Right boundary (x=nx-1)
        field[0, nx-1, y] = (psi[nx-1, min(y+1, ny-1)] - psi[nx-1, max(y-1, 0)]) / (2.0 * dy) if ny > 1 else 0.0
        field[1, nx-1, y] = -(psi[nx-1, y] - psi[nx-2, y]) / dx if nx > 1 else 0.0
    
    # y boundaries
    for x in range(nx):
        # Bottom boundary (y=0)
        field[0, x, 0] = (psi[x, 1] - psi[x, 0]) / dy if ny > 1 else 0.0
        field[1, x, 0] = -(psi[min(x+1, nx-1), 0] - psi[max(x-1, 0), 0]) / (2.0 * dx) if nx > 1 else 0.0
        
        # Top boundary (y=ny-1)
        field[0, x, ny-1] = (psi[x, ny-1] - psi[x, ny-2]) / dy if ny > 1 else 0.0
        field[1, x, ny-1] = -(psi[min(x+1, nx-1), ny-1] - psi[max(x-1, 0), ny-1]) / (2.0 * dx) if nx > 1 else 0.0
    
    return field