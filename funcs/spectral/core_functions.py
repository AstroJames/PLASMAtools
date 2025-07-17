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
    Shape: (nx, ny)
    """
    nx, ny = shape
    center_x = (nx - 1) / 2.0
    center_y = (ny - 1) / 2.0
    
    r = np.empty(shape, dtype=np.float32)
    
    for x in prange(nx):
        for y in range(ny):
            dx = x - center_x
            dy = y - center_y
            r[x, y] = np.sqrt(dx*dx + dy*dy)
    return r


@njit(sig_rad_dist_3D, parallel=True, fastmath=True, cache=True)
def compute_radial_distances_3D_core(
    shape : tuple) -> np.ndarray:
    """
    Parallel computation of radial distances from center
    Shape: (nx, ny, nz)
    """
    nx, ny, nz = shape
    center_x = (nx - 1) / 2.0
    center_y = (ny - 1) / 2.0
    center_z = (nz - 1) / 2.0
    
    r = np.empty(shape, dtype=np.float32)
    
    for x in prange(nx):
        for y in range(ny):
            for z in range(nz):
                dx = x - center_x
                dy = y - center_y
                dz = z - center_z
                r[x, y, z] = np.sqrt(dx*dx + dy*dy + dz*dz)
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
    Shape: (nx, ny, nz)
    """
    nx, ny, nz = shape
    center_x = (nx - 1) / 2.0
    center_y = (ny - 1) / 2.0
    center_z = (nz - 1) / 2.0
    
    k_perp = np.empty(shape, dtype=np.float64)
    k_para = np.empty(shape, dtype=np.float64)
    
    for x in prange(nx):
        dx = x - center_x
        for y in range(ny):
            dy = y - center_y
            for z in range(nz):
                dz = z - center_z
                k_perp[x, y, z] = np.sqrt(dx*dx + dy*dy)
                k_para[x, y, z] = np.abs(dz)
    
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
    Data shape: (nx, ny, nz)
    """
    nx, ny, nz = data.shape
    n_threads = numba.config.NUMBA_NUM_THREADS
    
    # Thread-local accumulators
    local_sums = np.zeros((n_threads, bins_perp, bins_para), dtype=data.dtype)
    
    # Parallel accumulation
    for x in prange(nx):
        thread_id = numba.get_thread_id()
        for y in range(ny):
            for z in range(nz):
                k_perp_val = k_perp[x, y, z]
                k_para_val = k_para[x, y, z]
                
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
                    local_sums[thread_id, bin_perp, bin_para] += data[x, y, z]
    
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
    Data shape: (nx, ny, nz)
    """
    nx, ny, nz = data_real.shape
    out_real = np.empty_like(data_real)  # Use empty instead of zeros
    out_imag = np.empty_like(data_imag)
    
    if filter_type == 0:  # Tophat filter - optimized
        # Parallel outer loop with optimal scheduling
        for x in prange(nx):
            for y in range(ny):
                # Innermost loop - optimized for vectorization
                for z in range(nz):
                    k_val = k_mag[x, y, z]
                    # Branchless selection using boolean arithmetic
                    mask = (k_val > k_min) & (k_val <= k_max)
                    out_real[x, y, z] = data_real[x, y, z] if mask else 0.0
                    out_imag[x, y, z] = data_imag[x, y, z] if mask else 0.0
                    
    else:  # Gaussian filter - optimized
        # Pre-compute all constants
        k0 = (k_min + k_max) * 0.5
        inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)
        
        for x in prange(nx):
            for y in range(ny):
                for z in range(nz):
                    k_val = k_mag[x, y, z]
                    k_diff = k_val - k0
                    # Optimized exponential calculation
                    weight = np.exp(-k_diff * k_diff * inv_two_sigma_sq)
                    out_real[x, y, z] = data_real[x, y, z] * weight
                    out_imag[x, y, z] = data_imag[x, y, z] * weight
    
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
    Data shape: (nx, ny)
    """
    nx, ny = data_real.shape
    out_real = np.zeros_like(data_real)
    out_imag = np.zeros_like(data_imag)
    
    for x in range(nx):
        for y in range(ny):
            k = k_mag[x, y]
            if k_min <= k <= k_max:
                out_real[x, y] = data_real[x, y]
                out_imag[x, y] = data_imag[x, y]
    
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
    Shape: (nx, ny, nz)
    """
    nx, ny, nz = kx.shape
    k_mag = np.empty((nx, ny, nz), dtype=kx.dtype)
    
    for x in prange(nx):
        for y in range(ny):
            for z in range(nz):
                k_mag[x, y, z] = np.sqrt(kx[x, y, z]**2 + 
                                        ky[x, y, z]**2 + 
                                        kz[x, y, z]**2)
    return k_mag


@njit([mixed_spec_2d_sig_32], parallel=True, fastmath=True, cache=True)
def compute_mixed_spectrum_2D_core(
    field1_fft: np.ndarray, 
    field2_fft: np.ndarray) -> np.ndarray:
    """
    Core function to compute 2D mixed variable power spectrum.
    
    Args:
        field1_fft: FFT of first field (2, nx, ny)
        field2_fft: FFT of second field (2, nx, ny)
    
    Returns:
        Mixed power spectrum (nx, ny)
    """
    nx, ny = field1_fft.shape[1:]
    mixed_spectrum = np.zeros((nx, ny), dtype=np.float32)
    
    for x in prange(nx):
        for y in range(ny):
            # Dot product: sum over vector components
            dot_real = 0.0
            dot_imag = 0.0
            for comp in range(2):  # 2D case
                f1 = field1_fft[comp, x, y]
                f2 = field2_fft[comp, x, y]
                dot_real += f1.real * f2.real + f1.imag * f2.imag
                dot_imag += f1.imag * f2.real - f1.real * f2.imag
            
            # Take magnitude
            mixed_spectrum[x, y] = np.sqrt(dot_real*dot_real + dot_imag*dot_imag)
    
    return mixed_spectrum


@njit([mixed_spec_sig_32], parallel=True, fastmath=True, cache=True)
def compute_mixed_spectrum_3D_core(
    field1_fft: np.ndarray, 
    field2_fft: np.ndarray) -> np.ndarray:
    """
    Core function to compute mixed variable power spectrum from FFT'd fields.
    Computes |field1(k) · field2*(k)| where · is dot product and * is complex conjugate.
    
    Args:
        field1_fft: FFT of first field (3, nx, ny, nz)
        field2_fft: FFT of second field (3, nx, ny, nz)
    
    Returns:
        Mixed power spectrum (nx, ny, nz)
    """
    nx, ny, nz = field1_fft.shape[1:]
    mixed_spectrum = np.zeros((nx, ny, nz), dtype=np.float32)
    
    # Compute dot product of field1_fft with complex conjugate of field2_fft
    for x in prange(nx):
        for y in range(ny):
            for z in range(nz):
                # Dot product: sum over vector components
                dot_real = 0.0
                dot_imag = 0.0
                for comp in range(3):
                    # field1 · field2* (complex conjugate)
                    f1 = field1_fft[comp, x, y, z]
                    f2 = field2_fft[comp, x, y, z]
                    # Complex multiplication: (a+bi)(c-di) = (ac+bd) + (bc-ad)i
                    dot_real += f1.real * f2.real + f1.imag * f2.imag
                    dot_imag += f1.imag * f2.real - f1.real * f2.imag
                
                # Take magnitude: |z| = sqrt(real^2 + imag^2)
                mixed_spectrum[x, y, z] = np.sqrt(dot_real*dot_real + dot_imag*dot_imag)
    
    return mixed_spectrum


@njit(parallel=True, fastmath=True, cache=True)
def compute_wave_numbers_reduced(
    spectral_shape, 
    L):
    """
    Compute wave numbers for reduced spectral space (real FFT)
    """
    Nx, Ny, Nz_half = spectral_shape
    
    kx = np.empty((Nx, Ny, Nz_half), dtype=np.float32)
    ky = np.empty((Nx, Ny, Nz_half), dtype=np.float32)
    kz = np.empty((Nx, Ny, Nz_half), dtype=np.float32)
    ksqr = np.empty((Nx, Ny, Nz_half), dtype=np.float32)
    
    # 1D wave number arrays
    kx_1d = np.empty(Nx, dtype=np.float32)
    ky_1d = np.empty(Ny, dtype=np.float32)
    kz_1d = np.empty(Nz_half, dtype=np.float32)
    
    # X frequencies (full range)
    for i in range(Nx):
        if i <= Nx // 2:
            kx_1d[i] = 2.0 * np.pi * i / L[0]
        else:
            kx_1d[i] = 2.0 * np.pi * (i - Nx) / L[0]
    
    # Y frequencies (full range)
    for j in range(Ny):
        if j <= Ny // 2:
            ky_1d[j] = 2.0 * np.pi * j / L[1]
        else:
            ky_1d[j] = 2.0 * np.pi * (j - Ny) / L[1]
    
    # Z frequencies (only positive for real FFT)
    for k in range(Nz_half):
        kz_1d[k] = 2.0 * np.pi * k / L[2]
    
    # Build 3D wave number arrays
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz_half):
                kx[i, j, k] = kx_1d[i]
                ky[i, j, k] = ky_1d[j]
                kz[i, j, k] = kz_1d[k]
                ksqr[i, j, k] = kx[i, j, k]*kx[i, j, k] + \
                    ky[i, j, k]*ky[i, j, k] + kz[i, j, k]*kz[i, j, k]
    
    # Handle zero mode
    ksqr[0, 0, 0] = 1.0
    
    return kx, ky, kz, ksqr


@njit(parallel=True, fastmath=True, cache=True)
def helmholtz_decomposition_3D_nb_core(
    Fhat, 
    kx, 
    ky, 
    kz, 
    ksqr):
    """
    Helmholtz decomposition in reduced spectral space
    """
    nx, ny, nz_half = Fhat.shape[1:]
    
    Fhat_irrot = np.zeros_like(Fhat)
    Fhat_solen = np.zeros_like(Fhat)
    
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz_half):
                k_x = kx[i, j, k]
                k_y = ky[i, j, k]
                k_z = kz[i, j, k]
                k_squared = ksqr[i, j, k]
                
                if k_squared > 0:
                    # Compute k·F
                    k_dot_F = (k_x * Fhat[0, i, j, k] + 
                              k_y * Fhat[1, i, j, k] + 
                              k_z * Fhat[2, i, j, k])
                    
                    # Irrotational component
                    k_dot_F_over_ksqr = k_dot_F / k_squared
                    Fhat_irrot[0, i, j, k] = k_dot_F_over_ksqr * k_x
                    Fhat_irrot[1, i, j, k] = k_dot_F_over_ksqr * k_y
                    Fhat_irrot[2, i, j, k] = k_dot_F_over_ksqr * k_z
                    
                    # Solenoidal component
                    Fhat_solen[0, i, j, k] = Fhat[0, i, j, k] - Fhat_irrot[0, i, j, k]
                    Fhat_solen[1, i, j, k] = Fhat[1, i, j, k] - Fhat_irrot[1, i, j, k]
                    Fhat_solen[2, i, j, k] = Fhat[2, i, j, k] - Fhat_irrot[2, i, j, k]
                else:
                    # Zero mode
                    for comp in range(3):
                        Fhat_irrot[comp, i, j, k] = 0.0 + 0.0j
                        Fhat_solen[comp, i, j, k] = 0.0 + 0.0j
    
    return Fhat_irrot, Fhat_solen
