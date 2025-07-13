"""
Core Numba JIT compiled functions for spectral analysis.
These are the performance-critical numerical kernels.
"""
import numpy as np
from numba import njit, prange
import numba
from .constants import *

##########################################################################################
# Core numba JIT functions for spectral operations
##########################################################################################

@njit(sig_rad_dist_2D, parallel=True, fastmath=True, cache=True)
def compute_radial_distances_2D_core(
    shape : tuple) -> np.ndarray:
    """
    Parallel computation of radial distances from center
    """
    ny, nx = shape
    center_y = (ny - 1) / 2.0
    center_x = (nx - 1) / 2.0
    
    r = np.empty(shape, dtype=np.float32)
    
    for y in prange(ny):
        for x in range(nx):
            dy = y - center_y
            dx = x - center_x
            r[y, x] = np.sqrt(dy*dy + dx*dx)
    return r


@njit(sig_rad_dist_3D, parallel=True, fastmath=True, cache=True)
def compute_radial_distances_3D_core(
    shape : tuple) -> np.ndarray:
    """
    Parallel computation of radial distances from center
    """
    nz, ny, nx = shape
    center_z = (nz - 1) / 2.0
    center_y = (ny - 1) / 2.0
    center_x = (nx - 1) / 2.0
    
    r = np.empty(shape, dtype=np.float32)
    
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                dz = z - center_z
                dy = y - center_y
                dx = x - center_x
                r[z, y, x] = np.sqrt(dz*dz + dy*dy + dx*dx)
    return r


@njit([sig_sph_int_2d_32, sig_sph_int_2d_64], parallel=True, fastmath=True, cache=True)
def spherical_integrate_2D_core(
    data: np.ndarray,
    r: np.ndarray,
    bin_edges: np.ndarray,
    bins: int) -> np.ndarray:
    """
    2D spherical integration using thread-local accumulation pattern.
    """
    data_flat = data.ravel()
    r_flat = r.ravel()
    n_elements = len(data_flat)
    n_edges = len(bin_edges)
    
    # Get number of threads
    n_threads = numba.config.NUMBA_NUM_THREADS
    
    # Create thread-local accumulators
    local_sums = np.zeros((n_threads, bins), dtype=data.dtype)
    
    # Parallel binning
    for i in prange(n_elements):
        r_val = r_flat[i]
        
        # Binary search for bin
        if r_val < bin_edges[0]:
            bin_idx = 0
        elif r_val >= bin_edges[-1]:
            bin_idx = n_edges
        else:
            left = 0
            right = n_edges - 1
            
            while left < right:
                mid = (left + right) // 2
                if r_val < bin_edges[mid]:
                    right = mid
                else:
                    left = mid + 1
            
            bin_idx = left
        
        # Accumulate
        if 1 <= bin_idx <= bins:
            thread_id = numba.get_thread_id()
            local_sums[thread_id, bin_idx - 1] += data_flat[i]
    
    # Merge results
    radial_sum = np.zeros(bins, dtype=data.dtype)
    for i in range(bins):
        for t in range(n_threads):
            radial_sum[i] += local_sums[t, i]
    
    return radial_sum


@njit([sig_sph_int_32, sig_sph_int_64], parallel=True, fastmath=True, cache=True)
def spherical_integrate_3D_core(
    data : np.ndarray, 
    r : np.ndarray,
    bin_edges : np.ndarray,
    bins : int) -> np.ndarray:
    """
    Fully parallel version using thread-local accumulation
    """
    data_flat = data.ravel()
    r_flat = r.ravel()
    n_elements = len(data_flat)
    n_edges = len(bin_edges)
    
    # Get number of threads
    n_threads = numba.config.NUMBA_NUM_THREADS
    
    # Create thread-local accumulators with matching dtype
    local_sums = np.zeros((n_threads, bins), dtype=data.dtype)
    
    # Parallel binning and accumulation
    for i in prange(n_elements):
        r_val = r_flat[i]
        
        # Compute bin index
        if r_val < bin_edges[0]:
            bin_idx = 0
        elif r_val >= bin_edges[-1]:
            bin_idx = n_edges
        else:
            left = 0
            right = n_edges - 1
            
            while left < right:
                mid = (left + right) // 2
                if r_val < bin_edges[mid]:
                    right = mid
                else:
                    left = mid + 1
            
            bin_idx = left
        
        # Accumulate into thread-local array
        if 1 <= bin_idx <= bins:
            thread_id = numba.get_thread_id()
            local_sums[thread_id, bin_idx - 1] += data_flat[i]
    
    # Merge thread-local results with matching dtype
    radial_sum = np.zeros(bins, dtype=data.dtype)
    for i in range(bins):
        for t in range(n_threads):
            radial_sum[i] += local_sums[t, i]
    
    return radial_sum


@njit(parallel=True, fastmath=True, cache=True)
def compute_cylindrical_distances_core(
    shape: tuple) -> tuple:
    """
    Compute cylindrical distances (k_perp and k_para) in parallel.
    """
    nz, ny, nx = shape
    center_z = (nz - 1) / 2.0
    center_y = (ny - 1) / 2.0
    center_x = (nx - 1) / 2.0
    
    k_perp = np.empty(shape, dtype=np.float64)
    k_para = np.empty(shape, dtype=np.float64)
    
    for z in prange(nz):
        dz = z - center_z
        for y in range(ny):
            dy = y - center_y
            for x in range(nx):
                dx = x - center_x
                k_perp[z, y, x] = np.sqrt(dx*dx + dy*dy)
                k_para[z, y, x] = np.abs(dz)
    
    return k_perp, k_para


@njit([sig_cyl_int_32, sig_cyl_int_64], parallel=True, fastmath=True, cache=True)
def cylindrical_integrate_core(
    data: np.ndarray,
    k_perp: np.ndarray,
    k_para: np.ndarray,
    bin_edges_perp: np.ndarray,
    bin_edges_para: np.ndarray,
    bins_perp: int,
    bins_para: int) -> np.ndarray:
    """
    Cylindrical integration using thread-local accumulation.
    """
    nz, ny, nx = data.shape
    n_threads = numba.config.NUMBA_NUM_THREADS
    
    # Thread-local accumulators
    local_sums = np.zeros((n_threads, bins_perp, bins_para), dtype=data.dtype)
    
    # Parallel accumulation
    for z in prange(nz):
        thread_id = numba.get_thread_id()
        for y in range(ny):
            for x in range(nx):
                k_perp_val = k_perp[z, y, x]
                k_para_val = k_para[z, y, x]
                
                # Find bin indices
                # Perpendicular bin
                if k_perp_val < bin_edges_perp[0]:
                    bin_perp = -1
                elif k_perp_val >= bin_edges_perp[-1]:
                    bin_perp = bins_perp
                else:
                    left = 0
                    right = len(bin_edges_perp) - 1
                    while left < right:
                        mid = (left + right) // 2
                        if k_perp_val < bin_edges_perp[mid]:
                            right = mid
                        else:
                            left = mid + 1
                    bin_perp = left - 1
                
                # Parallel bin
                if k_para_val < bin_edges_para[0]:
                    bin_para = -1
                elif k_para_val >= bin_edges_para[-1]:
                    bin_para = bins_para
                else:
                    left = 0
                    right = len(bin_edges_para) - 1
                    while left < right:
                        mid = (left + right) // 2
                        if k_para_val < bin_edges_para[mid]:
                            right = mid
                        else:
                            left = mid + 1
                    bin_para = left - 1
                
                # Accumulate if in valid bin
                if 0 <= bin_perp < bins_perp and 0 <= bin_para < bins_para:
                    local_sums[thread_id, bin_perp, bin_para] += data[z, y, x]
    
    # Merge thread results
    cylindrical_sum = np.zeros((bins_perp, bins_para), dtype=data.dtype)
    for i in range(bins_perp):
        for j in range(bins_para):
            for t in range(n_threads):
                cylindrical_sum[i, j] += local_sums[t, i, j]
    
    return cylindrical_sum


@njit(sig_filter_32, parallel=True, fastmath=True, cache=True, 
      boundscheck=False, nogil=True, inline='always')
def compute_shell_filter_3D_core(
    data_real: np.ndarray,
    data_imag: np.ndarray,
    k_mag: np.ndarray,
    k_min: float,
    k_max: float,
    filter_type: int,
    sigma: float) -> tuple:
    """
    Apply shell filter in Fourier space using vectorized operations.
    """
    nz, ny, nx = data_real.shape
    out_real = np.empty_like(data_real)  # Use empty instead of zeros
    out_imag = np.empty_like(data_imag)
    
    if filter_type == 0:  # Tophat filter - optimized
        # Parallel outer loop with optimal scheduling
        for z in prange(nz):
            for y in range(ny):
                # Innermost loop - optimized for vectorization
                for x in range(nx):
                    k_val = k_mag[z, y, x]
                    # Branchless selection using boolean arithmetic
                    mask = (k_val > k_min) & (k_val <= k_max)
                    out_real[z, y, x] = data_real[z, y, x] if mask else 0.0
                    out_imag[z, y, x] = data_imag[z, y, x] if mask else 0.0
                    
    else:  # Gaussian filter - optimized
        # Pre-compute all constants
        k0 = (k_min + k_max) * 0.5
        inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)
        
        for z in prange(nz):
            for y in range(ny):
                for x in range(nx):
                    k_val = k_mag[z, y, x]
                    k_diff = k_val - k0
                    # Optimized exponential calculation
                    weight = np.exp(-k_diff * k_diff * inv_two_sigma_sq)
                    out_real[z, y, x] = data_real[z, y, x] * weight
                    out_imag[z, y, x] = data_imag[z, y, x] * weight
    
    return out_real, out_imag


@njit(fastmath=True)
def compute_shell_filter_2D_core(
    data_real: np.ndarray, 
    data_imag: np.ndarray,
    k_mag: np.ndarray, 
    k_min: float, 
    k_max: float) -> tuple:
    """
    Apply 2D shell filter in Fourier space.
    """
    ny, nx = data_real.shape
    out_real = np.zeros_like(data_real)
    out_imag = np.zeros_like(data_imag)
    
    for y in range(ny):
        for x in range(nx):
            k = k_mag[y, x]
            if k_min <= k <= k_max:
                out_real[y, x] = data_real[y, x]
                out_imag[y, x] = data_imag[y, x]
    
    return out_real, out_imag


@njit(parallel=True, fastmath=True)
def compute_k_magnitude_2D_core(
    kx: np.ndarray, 
    ky: np.ndarray) -> np.ndarray:
    """
    Compute 2D k magnitude in parallel.
    """
    ny, nx = kx.shape
    k_mag = np.empty((ny, nx), dtype=kx.dtype)
    
    for y in prange(ny):
        for x in range(nx):
            k_mag[y, x] = np.sqrt(kx[y, x]**2 + ky[y, x]**2)
    
    return k_mag


@njit(parallel=True, fastmath=True)
def compute_k_magnitude_3D_core(
    kx: np.ndarray, 
    ky: np.ndarray, 
    kz: np.ndarray) -> np.ndarray:
    """
    Compute k magnitude in parallel.
    """
    nz, ny, nx = kx.shape
    k_mag = np.empty((nz, ny, nx), dtype=kx.dtype)
    
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                k_mag[z, y, x] = np.sqrt(kx[z, y, x]**2 + 
                                        ky[z, y, x]**2 + 
                                        kz[z, y, x]**2)
    return k_mag


@njit([mixed_spec_2d_sig_32], parallel=True, fastmath=True, cache=True)
def compute_mixed_spectrum_2D_core(
    field1_fft: np.ndarray, 
    field2_fft: np.ndarray) -> np.ndarray:
    """
    Core function to compute 2D mixed variable power spectrum.
    
    Args:
        field1_fft: FFT of first field (2, ny, nx)
        field2_fft: FFT of second field (2, ny, nx)
    
    Returns:
        Mixed power spectrum (ny, nx)
    """
    ny, nx = field1_fft.shape[1:]
    mixed_spectrum = np.zeros((ny, nx), dtype=np.float32)
    
    for y in prange(ny):
        for x in range(nx):
            # Dot product: sum over vector components
            dot_real = 0.0
            dot_imag = 0.0
            for comp in range(2):  # 2D case
                f1 = field1_fft[comp, y, x]
                f2 = field2_fft[comp, y, x]
                dot_real += f1.real * f2.real + f1.imag * f2.imag
                dot_imag += f1.imag * f2.real - f1.real * f2.imag
            
            # Take magnitude
            mixed_spectrum[y, x] = np.sqrt(dot_real*dot_real + dot_imag*dot_imag)
    
    return mixed_spectrum


@njit([mixed_spec_sig_32], parallel=True, fastmath=True, cache=True)
def compute_mixed_spectrum_3D_core(
    field1_fft: np.ndarray, 
    field2_fft: np.ndarray) -> np.ndarray:
    """
    Core function to compute mixed variable power spectrum from FFT'd fields.
    Computes |field1(k) · field2*(k)| where · is dot product and * is complex conjugate.
    
    Args:
        field1_fft: FFT of first field (3, nz, ny, nx)
        field2_fft: FFT of second field (3, nz, ny, nx)
    
    Returns:
        Mixed power spectrum (nz, ny, nx)
    """
    nz, ny, nx = field1_fft.shape[1:]
    mixed_spectrum = np.zeros((nz, ny, nx), dtype=np.float32)
    
    # Compute dot product of field1_fft with complex conjugate of field2_fft
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                # Dot product: sum over vector components
                dot_real = 0.0
                dot_imag = 0.0
                for comp in range(3):
                    # field1 · field2* (complex conjugate)
                    f1 = field1_fft[comp, z, y, x]
                    f2 = field2_fft[comp, z, y, x]
                    # Complex multiplication: (a+bi)(c-di) = (ac+bd) + (bc-ad)i
                    dot_real += f1.real * f2.real + f1.imag * f2.imag
                    dot_imag += f1.imag * f2.real - f1.real * f2.imag
                
                # Take magnitude: |z| = sqrt(real^2 + imag^2)
                mixed_spectrum[z, y, x] = np.sqrt(dot_real*dot_real + dot_imag*dot_imag)
    
    return mixed_spectrum

