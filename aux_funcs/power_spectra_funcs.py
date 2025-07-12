"""
    PLASMAtools: Power Spectra Functions

    Functions for calculating power spectra, and performing spherical and cylindrical integration.
    
    This module provides JIT-compiled functions for efficient computation of power spectra
    and integration in spherical and cylindrical coordinates. It supports both 2D and 3D data
    and can handle both real and complex data types. The functions are optimized for performance
    using Numba and can utilize multi-threading for parallel execution.
    
    Author: James R. Beattie
    Collaborators: Anne Noer Kolborg
"""

## ###############################################################
## IMPORTS
## ###############################################################

import numpy as np
import multiprocessing
from numba import njit, prange, types
import numba

pyfftw_import = False
try: 
    import pyfftw
    pyfftw_import = True
    pyfftw.interfaces.cache.enable()
    threads = multiprocessing.cpu_count()
except ImportError:
    print("pyfftw not installed, using scipy's serial fft")

if pyfftw_import:
    # Use pyfftw for faster FFTs if available
    pyfftw.config.NUM_THREADS = threads
    fftn = pyfftw.interfaces.numpy_fft.fftn
    ifftn = pyfftw.interfaces.numpy_fft.ifftn
    rfftn = pyfftw.interfaces.numpy_fft.rfftn
    irfftn = pyfftw.interfaces.numpy_fft.irfftn
    fftfreq = pyfftw.interfaces.numpy_fft.fftfreq
    fftshift = pyfftw.interfaces.numpy_fft.fftshift
else:
    # Use numpy's FFT functions
    fftn = np.fft.fftn
    ifftn = np.fft.ifftn
    rfftn = np.fft.rfftn
    irfftn = np.fft.irfftn
    fftfreq = np.fft.fftfreq
    fftshift = np.fft.fftshift
    
# Numb signatures for JIT compilation

sig_rad_dist_3D = types.float32[:,:,:](types.UniTuple(types.int64, 3))
sig_rad_dist_2D = types.float32[:,:](types.UniTuple(types.int64, 2))

sig_sph_int_32 = types.float32[:](types.float32[:,:,:], types.float32[:,:,:], types.float32[:], types.int64)
sig_sph_int_64 = types.float64[:](types.float64[:,:,:], types.float64[:,:,:], types.float64[:], types.int64)

sig_sph_int_2d_32 = types.float32[:](types.float32[:,:], types.float32[:,:], types.float32[:], types.int64)
sig_sph_int_2d_64 = types.float64[:](types.float64[:,:], types.float64[:,:], types.float64[:], types.int64)

sig_cyl_int_32 = types.float32[:,:](types.float32[:,:,:], types.float32[:,:,:], types.float32[:,:,:], 
                                    types.float32[:], types.float32[:], types.int64, types.int64)
sig_cyl_int_64 = types.float64[:,:](types.float64[:,:,:], types.float64[:,:,:], types.float64[:,:,:], 
                                     types.float64[:], types.float64[:], types.int64, types.int64)

sig_filter_32 = types.UniTuple(types.float32[:,:,:], 2)(
    types.float32[:,:,:], types.float32[:,:,:], types.float32[:,:,:],
    types.float32, types.float32, types.int64, types.float32
)

## ###############################################################
## Numba Core Spectral Functions
## ###############################################################

@njit(sig_rad_dist_3D, parallel=True, fastmath=True, cache=True)
def compute_radial_distances_3D(shape : tuple) -> np.ndarray:
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


@njit(sig_rad_dist_2D, parallel=True, fastmath=True, cache=True)
def compute_radial_distances_2D(shape : tuple) -> np.ndarray:
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


@njit([sig_sph_int_32, sig_sph_int_64], parallel=True, fastmath=True, cache=True)
def spherical_integrate_core_3D(data : np.ndarray, 
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


@njit([sig_sph_int_2d_32, sig_sph_int_2d_64], parallel=True, fastmath=True, cache=True)
def spherical_integrate_core_2D(data: np.ndarray,
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


@njit(parallel=True, fastmath=True, cache=True)
def compute_cylindrical_distances(shape: tuple) -> tuple:
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


@njit(parallel=True, fastmath=True)
def compute_cylindrical_distances(shape: tuple) -> tuple:
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
def cylindrical_integrate_core(data: np.ndarray,
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


@njit(sig_filter_32, parallel=True, fastmath=True, cache=True, boundscheck=False, nogil=True, inline='always')
def apply_shell_filter_3d_vectorized(data_real: np.ndarray,
                                data_imag: np.ndarray,
                                k_mag: np.ndarray,
                                k_min: float,
                                k_max: float,
                                filter_type: int,
                                sigma: float) -> tuple:
    """
    Ultra-optimized shell filter with maximum performance focus.
    
    Optimizations:
    - Minimal branching
    - Optimal memory access patterns  
    - Pre-computed constants
    - Aggressive compiler hints
    - Cache-friendly loop ordering
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

# @njit([filter_sig_32, filter_sig_64], parallel=True, fastmath=True, cache=True, boundscheck=False)
# def apply_shell_filter_3d_vectorized(data_real: np.ndarray,
#                                      data_imag: np.ndarray,
#                                      k_mag: np.ndarray,
#                                      k_min: float,
#                                      k_max: float,
#                                      filter_type: int,
#                                      sigma: float) -> tuple:
#     """
#     Vectorized shell filter optimized for better cache usage.
#     Processes y-x planes in parallel for better memory locality.
#     """
#     nz, ny, nx = data_real.shape
#     out_real = np.empty_like(data_real)
#     out_imag = np.empty_like(data_imag)
    
#     if filter_type == 0:  # tophat filter
#         for z in prange(nz):
#             for y in range(ny):
#                 # Inner loop optimization - better for vectorization
#                 for x in range(nx):
#                     k = k_mag[z, y, x]
#                     # Branchless version using multiplication
#                     mask = (k > k_min) & (k <= k_max)
#                     out_real[z, y, x] = data_real[z, y, x] * mask
#                     out_imag[z, y, x] = data_imag[z, y, x] * mask
#     else:  # gaussian filter
#         k0 = (k_min + k_max) * 0.5
#         norm = 1.0 / (2.0 * sigma * sigma)
        
#         for z in prange(nz):
#             for y in range(ny):
#                 for x in range(nx):
#                     k = k_mag[z, y, x]
#                     k_diff = k - k0
#                     weight = np.exp(-k_diff * k_diff * norm)
#                     out_real[z, y, x] = data_real[z, y, x] * weight
#                     out_imag[z, y, x] = data_imag[z, y, x] * weight
    
#     return out_real, out_imag


@njit(fastmath=True)
def apply_shell_filter_2d(data_real: np.ndarray, data_imag: np.ndarray,
                         k_mag: np.ndarray, k_min: float, k_max: float) -> tuple:
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
def compute_k_magnitude_3d(kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> np.ndarray:
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

@njit(parallel=True, fastmath=True)
def compute_k_magnitude_2d(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """
    Compute 2D k magnitude in parallel.
    """
    ny, nx = kx.shape
    k_mag = np.empty((ny, nx), dtype=kx.dtype)
    
    for y in prange(ny):
        for x in range(nx):
            k_mag[y, x] = np.sqrt(kx[y, x]**2 + ky[y, x]**2)
    
    return k_mag


## ###############################################################
## FFTW Plan Cache
## ###############################################################

class FFTWPlanCache:
    """Cache for FFTW plans to avoid recreation overhead."""
    
    def __init__(self, 
                 max_plans=10):
        self.plans = {}
        self.max_plans = max_plans
        self.enabled = pyfftw_import
        
    def get_fft_plan(self, 
                     shape, 
                     axes, 
                     forward=True, 
                     real=False):
        """Get or create an FFTW plan."""
        
        if not self.enabled:
            return None
            
        key = (shape, axes, forward, real)
        
        if key in self.plans:
            return self.plans[key]
        
        # Create new plan
        if len(self.plans) >= self.max_plans:
            # Remove oldest plan
            oldest_key = next(iter(self.plans))
            del self.plans[oldest_key]
        
        # Create aligned arrays for plan
        if real:
            if forward:
                input_array = pyfftw.empty_aligned(shape, dtype='float64')
                output_shape = list(shape)
                output_shape[axes[-1]] = shape[axes[-1]] // 2 + 1
                output_array = pyfftw.empty_aligned(output_shape, dtype='complex128')
                plan = pyfftw.FFTW(input_array, output_array, axes=axes,
                                  flags=['FFTW_MEASURE'], threads=threads)
            else:
                input_shape = list(shape)
                input_shape[axes[-1]] = shape[axes[-1]] // 2 + 1
                input_array = pyfftw.empty_aligned(input_shape, dtype='complex128')
                output_array = pyfftw.empty_aligned(shape, dtype='float64')
                plan = pyfftw.FFTW(input_array, output_array, axes=axes,
                                  direction='FFTW_BACKWARD',
                                  flags=['FFTW_MEASURE'], threads=threads)
        else:
            input_array = pyfftw.empty_aligned(shape, dtype='complex128')
            output_array = pyfftw.empty_aligned(shape, dtype='complex128')
            direction = 'FFTW_FORWARD' if forward else 'FFTW_BACKWARD'
            plan = pyfftw.FFTW(input_array, output_array, axes=axes,
                              direction=direction,
                              flags=['FFTW_MEASURE'], threads=threads)
        
        self.plans[key] = plan
        return plan
    
    def execute_fft(self, 
                    data, 
                    axes, 
                    forward=True, 
                    real=False, 
                    norm='forward'):
        
        """Execute FFT using cached plan if available."""
        if not self.enabled:
            # Fallback to numpy
            if real:
                if forward:
                    return rfftn(data, axes=axes, norm=norm)
                else:
                    return irfftn(data, axes=axes, norm=norm)
            else:
                if forward:
                    return fftn(data, axes=axes, norm=norm)
                else:
                    return ifftn(data, axes=axes, norm=norm)
        
        plan = self.get_fft_plan(data.shape, axes, forward, real)
        
        if plan is None:
            # Fallback if plan creation failed
            if real:
                if forward:
                    return rfftn(data, axes=axes, norm=norm)
                else:
                    return irfftn(data, axes=axes, norm=norm)
            else:
                if forward:
                    return fftn(data, axes=axes, norm=norm)
                else:
                    return ifftn(data, axes=axes, norm=norm)
        
        # Copy data to aligned array
        plan.input_array[:] = data
        
        # Execute
        result = plan()
        
        # Apply normalization
        if norm == 'forward' and forward:
            n = np.prod([data.shape[i] for i in axes])
            result = result / n
        elif norm == 'forward' and not forward:
            pass  # No normalization for inverse
        
        return result.copy()


class SpectraOperations:
    
    
    def __init__(self, 
                 cache_plans=False):
        """
        Initialize with optional FFTW plan caching.
        
        Args:
            cache_plans (bool): If True, use FFTW plan caching for faster repeated FFTs.
        Default is False, because caching is not always beneficial for single use FFTs. If
        you are running many FFTs, then caching is worth it, so set this to True.
        
        Author: James R. Beattie
        
        """
        self.fft_cache = FFTWPlanCache() if cache_plans else None
        
        
    def _do_fft(self, 
                data, 
                axes, 
                forward=True, 
                real=False, 
                norm='forward'):
        """
        Helper to use cached FFT plans if available.
        
        This ends up being a lot of overhead for small arrays, and if you're just
        running a single FFT, it is better to not cache the plans.
        If you are running many FFTs, then caching is worth it.
        
        """
        if self.fft_cache is not None:
            return self.fft_cache.execute_fft(data,
                                              axes,
                                              forward,
                                              real,
                                              norm)
        else:
            if real:
                if forward:
                    return rfftn(data,
                                 axes=axes,
                                 norm=norm)
                else:
                    return irfftn(data,
                                  axes=axes,
                                  norm=norm)
            else:
                if forward:
                    return fftn(data,
                                axes=axes,
                                norm=norm)
                else:
                    return ifftn(data,
                                 axes=axes,
                                 norm=norm)
    
    
    def compute_power_spectrum_3D(self, 
                                  field: np.ndarray) -> np.ndarray:
        """
        Computes the power spectrum using cached FFT plans.
        """
        assert len(field.shape) == 4, "Field should be 3D"
        
        # Ensure data is float32 for memory efficiency
        if field.dtype != np.float32:
            print(f"Converting {field.dtype} data to float32 for memory efficiency")
            field = field.astype(np.float32)
        
        # Use cached FFT
        field_fft = self._do_fft(field, 
                                 axes=(1, 2, 3), 
                                 forward=True, 
                                 real=np.isrealobj(field), norm='forward')
        
        # Compute power spectrum
        out = np.sum(np.abs(field_fft)**2,
                     axis=0)
        
        # Handle real FFT output shape
        if np.isrealobj(field) and field_fft.shape[-1] != field.shape[-1]:
            # Restore full spectrum by mirroring
            N = field.shape
            full_out = np.zeros((N[1], N[2], N[3]), dtype=out.dtype)
            full_out[:, :, :out.shape[-1]] = out
            # Mirror conjugate parts
            if N[3] % 2 == 0:
                full_out[:, :, -N[3]//2+1:] = out[:, :, 1:N[3]//2][:, :, ::-1]
            else:
                full_out[:, :, -N[3]//2:] = out[:, :, 1:N[3]//2+1][:, :, ::-1]
            out = full_out
        
        return fftshift(out, axes=(0, 1, 2))
    

    def compute_power_spectrum_2D(self, 
                                  field: np.ndarray) -> np.ndarray:
        """
        Computes 2D power spectrum using cached FFT plans.
        """
        assert len(field.shape) == 3, "Field should be 2D"
        
        # Ensure data is float32 for memory efficiency
        if field.dtype != np.float32:
            print(f"Converting {field.dtype} data to float32 for memory efficiency")
            field = field.astype(np.float32)

        field_fft = self._do_fft(field, axes=(1, 2), 
                                 forward=True,
                                 real=np.isrealobj(field),
                                 norm='forward')
        
        out = np.sum(np.abs(field_fft)**2, 
                     axis=0)
        
        # Handle real FFT
        if np.isrealobj(field) and field_fft.shape[-1] != field.shape[-1]:
            N = field.shape
            full_out = np.zeros((N[1], N[2]), dtype=out.dtype)
            full_out[:, :out.shape[-1]] = out
            if N[2] % 2 == 0:
                full_out[:, -N[2]//2+1:] = out[:, 1:N[2]//2][:, ::-1]
            else:
                full_out[:, -N[2]//2:] = out[:, 1:N[2]//2+1][:, ::-1]
            out = full_out
        
        return fftshift(out, axes=(0, 1))


    def compute_tensor_power_spectrum(self, 
                                      field: np.ndarray) -> np.ndarray:
        """
        Computes tensor power spectrum using cached FFT plans.
        """
        assert field.shape[:2] == (3, 3), "Field should be a 3D tensor field"
        
        # Ensure data is float32 for memory efficiency
        if field.dtype != np.float32:
            print(f"Converting {field.dtype} data to float32 for memory efficiency")
            field = field.astype(np.float32)
        
        field_fft = self._do_fft(field, 
                                 axes=(2, 3, 4), 
                                 forward=True,
                                 real=np.isrealobj(field),
                                 norm='forward')
        
        out = np.sum(np.abs(field_fft)**2, 
                     axis=(0, 1))
        
        # Handle real FFT
        if np.isrealobj(field) and field_fft.shape[-1] != field.shape[-1]:
            N = field.shape
            full_out = np.zeros((N[2], N[3], N[4]), dtype=out.dtype)
            full_out[:, :, :out.shape[-1]] = out
            if N[4] % 2 == 0:
                full_out[:, :, -N[4]//2+1:] = out[:, :, 1:N[4]//2][:, :, ::-1]
            else:
                full_out[:, :, -N[4]//2:] = out[:, :, 1:N[4]//2+1][:, :, ::-1]
            out = full_out
        
        return fftshift(out, axes=(0, 1, 2))

    
    def spherical_integrate_2D(self, 
                               data: np.ndarray, 
                               bins: int = None) -> tuple:
        """
        2D spherical integration using JIT-compiled functions.
        """
        N = data.shape[0]
        if not bins:
            bins = N // 2
            
        # Ensure data is float32 for memory efficiency
        if data.dtype != np.float32:
            print(f"Converting {data.dtype} data to float32 for memory efficiency")
            data = data.astype(np.float32)
        
        # Use JIT function for distances
        r = compute_radial_distances_2D(data.shape)
        bin_edges = np.linspace(0.5, bins, bins + 1)
        radial_sum = spherical_integrate_core_2D(data, 
                                                 r, 
                                                 bin_edges, 
                                                 bins)    
        k_modes = np.ceil((bin_edges[:-1] + bin_edges[1:]) / 2)
    
        return k_modes, radial_sum

    def spherical_integrate_3D(self, 
                               data: np.ndarray, 
                               bins: int = None) -> tuple:
        """
        3D spherical integration (already optimized).
        """
        N = data.shape[0]
        if not bins:
            bins = N // 2
            
        # Ensure data is float32 for memory efficiency
        if data.dtype != np.float32:
            print(f"Converting {data.dtype} data to float32 for memory efficiency")
            data = data.astype(np.float32)
        
        r = compute_radial_distances_3D(data.shape)
        bin_edges = np.linspace(0.5, bins, bins + 1, dtype=np.float32)
        radial_sum = spherical_integrate_core_3D(data, 
                                                 r, 
                                                 bin_edges, 
                                                 bins)
        k_modes = np.ceil((bin_edges[:-1] + bin_edges[1:]) / 2)
        
        return k_modes, radial_sum


    def spherical_integrate_2D(self,
                               data: np.ndarray, 
                               bins: int = None) -> tuple:
        """
        The spherical integrate function takes the 2D power spectrum and integrates
        over spherical shells of constant k. The result is a 1D power spectrum.
        
        Needs to be tested in detail
        
        Args:
            data: The 2D power spectrum
            bins: The number of bins to use for the radial integration. 
                If not specified, the Nyquist limit is used (as should always be the case, anyway).

        Returns:
            k_modes: The k modes corresponding to the radial integration
            radial_sum: The radial integration of the 3D power spectrum (including k^2 correction)
        """
        y, x = np.indices(data.shape)
        center = np.array([(i - 1) / 2.0 for i in data.shape])
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)

        N = data.shape[0]
        if not bins:
            bins = N // 2

        bin_edges = np.linspace(0.5, bins, bins+1)

        # Use np.digitize to assign each element to a bin
        bin_indices = np.digitize(r, bin_edges)

        # Compute the radial profile
        radial_sum = np.zeros(bins)
        for i in range(1, bins+1):
            mask = bin_indices == i
            radial_sum[i-1] = np.sum(data[mask])

        # Generate the spatial frequencies with dk=1
        # Now k_modes represent the bin centers
        k_modes = np.ceil((bin_edges[:-1] + bin_edges[1:])/2)

        return k_modes, radial_sum

    def cylindrical_integrate(self, 
                              data: np.ndarray, 
                              bins_perp: int = 0,
                              bins_para: int = 0) -> tuple:
        """
        Cylindrical integration using JIT-compiled functions.
        """
        N = data.shape[0]
        if bins_perp == 0:
            bins_perp = N // 2
        if bins_para == 0:
            bins_para = N // 2
        
        # Use JIT function for distances
        k_perp, k_para = compute_cylindrical_distances(data.shape)
        
        bin_edges_perp = np.linspace(0, bins_perp, bins_perp + 1)
        bin_edges_para = np.linspace(0, bins_para, bins_para + 1)
        
        # Use JIT function for integration
        cylindrical_sum = cylindrical_integrate_core(
            data, k_perp, k_para,
            bin_edges_perp, bin_edges_para,
            bins_perp, bins_para
        )
        
        k_perp_modes = (bin_edges_perp[:-1] + bin_edges_perp[1:]) / 2
        k_para_modes = (bin_edges_para[:-1] + bin_edges_para[1:]) / 2
        
        return k_perp_modes, k_para_modes, cylindrical_sum


    def cylindrical_integrate(self, 
                              data: np.ndarray, 
                              bins_perp: int = 0,
                              bins_para: int = 0) -> tuple:
        """
        Cylindrical integration using JIT-compiled functions.
        """
        N = data.shape[0]
        if bins_perp == 0:
            bins_perp = N // 2
        if bins_para == 0:
            bins_para = N // 2
        
        # Use JIT function for distances
        k_perp, k_para = compute_cylindrical_distances(data.shape)
        
        bin_edges_perp = np.linspace(0, bins_perp, bins_perp + 1)
        bin_edges_para = np.linspace(0, bins_para, bins_para + 1)
        
        # Use JIT function for integration
        cylindrical_sum = cylindrical_integrate_core(
            data, k_perp, k_para,
            bin_edges_perp, bin_edges_para,
            bins_perp, bins_para
        )
        
        k_perp_modes = (bin_edges_perp[:-1] + bin_edges_perp[1:]) / 2
        k_para_modes = (bin_edges_para[:-1] + bin_edges_para[1:]) / 2
        
        return k_perp_modes, k_para_modes, cylindrical_sum


    def extract_isotropic_shell_X(self, 
                                  vector_field: np.ndarray,
                                  k_minus_dk: float,
                                  k_plus_dk: float,
                                  filter: str = 'tophat',
                                  sigma: float = 10.0):
        """
        Extract shell using JIT-compiled filter application.
        """
        L = 1.0
        k_minus = 2 * np.pi / L * k_minus_dk
        k_plus = 2 * np.pi / L * k_plus_dk
        
        # FFT
        vector_field_fft = self._do_fft(vector_field, 
                                        axes=(1, 2, 3),
                                        forward=True, 
                                        real=False, 
                                        norm='forward')
        
        # Compute k magnitudes
        N = vector_field.shape[1]
        kx = 2 * np.pi * fftfreq(N, d=L/N)
        ky = 2 * np.pi * fftfreq(N, d=L/N)
        kz = 2 * np.pi * fftfreq(N, d=L/N)
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2).astype(np.float32)
        
        # Apply filter using JIT function
        filter_type = 0 if filter == 'tophat' else 1
        
        # Process each component
        for comp in range(vector_field.shape[0]):
            real_part = np.real(vector_field_fft[comp]).astype(np.float32)
            imag_part = np.imag(vector_field_fft[comp]).astype(np.float32)
            
            real_filtered, imag_filtered = self.apply_shell_filter_3d(
                real_part, imag_part, k_mag,
                k_minus, k_plus, filter_type, sigma
            )
            
            vector_field_fft[comp] = real_filtered + 1j * imag_filtered
        
        # Inverse FFT
        result = self._do_fft(vector_field_fft, 
                              axes=(1, 2, 3),
                              forward=False, 
                              real=False, 
                              norm='forward')
        
        return np.real(result)
    
    
    @staticmethod
    def extract_shell_X_2D(vector_field: np.ndarray,
                          k_minus_dk: float,
                          k_plus_dk: float,
                          L: list = [1.0, 1.0]) -> np.ndarray:
        """
        2D shell extraction (keeping original implementation for now).
        """
        N = vector_field.shape
        kx = 2 * np.pi * fftfreq(N[1], d=L[0]/N[1])
        ky = 2 * np.pi * fftfreq(N[2], d=L[1]/N[2])
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        
        # Create filter
        k_mag = np.sqrt(kx**2 + ky**2)
        mask = np.logical_and(k_mag >= k_minus_dk, k_mag <= k_plus_dk)
        mask = np.stack([mask, mask], axis=0)
        
        # Apply filter in Fourier space
        vector_fft = fftn(vector_field, axes=(1, 2), norm='forward')
        vector_fft *= mask
        
        # Inverse transform
        return np.real(ifftn(vector_fft, axes=(1, 2), norm='forward'))


    def apply_shell_filter_3d(self,
                              data_real: np.ndarray,
                              data_imag: np.ndarray,
                              k_mag: np.ndarray,
                              k_min: float,
                              k_max: float,
                              filter_type: int = 0,
                              sigma: float = 1.0) -> tuple:
        """
        Automatically choose the best filtering method based on array size and parameters.
        
        """
        
        return apply_shell_filter_3d_vectorized(
            data_real, data_imag, k_mag, k_min, k_max, filter_type, sigma
        )
        

    def extract_shell_X_2D(vector_field    : np.ndarray,
                        k_minus_dk      : float,
                        k_plus_dk       : float,
                        L               : list = [1.0,1.0] ) -> np.ndarray:
        """
        Extract and return the inverse FFT of a specific shell of a vector field.

        This method extracts the components of a vector field that fall within a 
        specified wavenumber shell, defined by the range `k_minus_dk < k < k_plus_dk`. 
        The shell is selected based on the direction specified in `self.direction` 
        ('parallel', 'perp', or 'iso'), and the inverse FFT of the filtered shell 
        is computed and returned.
        
        Author: James Beattie & Anne Noer Kolborg

        Args:
            vector_field (np.ndarray): The input vector field to be filtered and 
                                    transformed. It is assumed to be a 3D field.
            k_minus_dk (float): The lower bound of the wavenumber shell.
            k_plus_dk (float): The upper bound of the wavenumber shell.

        Returns:
            np.ndarray: The inverse FFT of the filtered vector field, limited to 
                        the specified wavenumber shell.

        Raises:
            Assertion error: If the input vector field is not 3D.
            ValueError: If `self.direction` is not recognized. Valid options are: 
                        'parallel', 'perp', 'iso'.

        Workflow:
            1. The method first determines the type of filter to apply based on 
            `self.direction`. The filter type is either 'parallel', 'perp', 
            or 'iso', corresponding to different wavenumber components.
            2. It then creates a mask using the filter, selecting the wavenumbers 
            that fall within the specified range (`k_minus_dk` to `k_plus_dk`).
            3. The vector field is transformed into Fourier space using `fftn`.
            4. The mask is applied to isolate the desired wavenumber components.
            5. The inverse FFT (`ifftn`) of the masked field is computed and returned.

        Example:
            # Example usage to extract a shell and compute the inverse FFT:
            filtered_field = self.extract_shell_X(vector_field, 0.5, 1.5)

        Notes:
            - The method assumes that `self.kx`, `self.ky`, and `self.kz` have been 
            initialized and correspond to the wavenumbers of the grid.
            - The extracted shell is in the form of a 3D numpy array, and the output 
            is also a 3D numpy array representing the spatial domain.

        References:
            This method is based on the transfer function code by Philip Grete:
            https://github.com/pgrete/energy-transfer-analysis
        """
        
        #assert np.shape(vector_field)[0] == 2, "Error: Vector field must be 2D."
        
        N = vector_field.shape
        kx = 2 * np.pi * fftfreq(N, d=L[0]/N[0]) / L[0]
        ky = 2 * np.pi * fftfreq(N, d=L[1]/N[1]) / L[1]
        
        def filter(kmin, kmax, kx, ky, filter_type):
            kx, ky = np.meshgrid(kx, ky, indexing='ij')
            # Define filter types
            filters = {
                'twod': np.sqrt(kx**2 + ky**2),
            }
            # Calculate the filter
            k_filter = filters[filter_type]
            mask = np.logical_and(k_filter >= kmin, k_filter <= kmax)
            return np.array([mask.astype(float), mask.astype(float)])

        # Inverse FFT with just the wavenumbers from the shell 
        return np.real(
            ifftn(
                filter(k_minus_dk, 
                    k_plus_dk, 
                    kx, 
                    ky, 
                    'twod') * fftn(
                    vector_field,
                    norm='forward',
                    axes=(1, 2)),
                axes=(1, 2),
                norm="forward"))


class GeneratedFields:
    """
    
    A class for generating various types of vector fields, including helical fields,
    isotropic power-law fields, and anisotropic power-law fields.
    
    """
    
    def __init__(self):
        """
        Initializes the GeneratedFields class.
        This class provides methods for generating vector fields with specific properties.
        
        TODO: need to fully implement the class and its methods.
        Currently, it contains a placeholder, with some basic methods, for the class.
        It is not fully functional yet.
        """
        pass
    
    @staticmethod
    def create_helical_field(N, 
                             k_index,
                             A_plus=1,
                             A_minus=0):
        """
        Creates a vector field with known helical components.

        Parameters:
        N (int): Size of the grid in each dimension.
        k_index (tuple): The index of the wavevector to be used.

        Returns:
        np.ndarray: The generated vector field.
        """
        # Create an empty field in Fourier space
        field_fft = np.zeros((N, N, N, 3), dtype=complex)

        # Generate the wavevector
        L = 1
        kx, ky, kz = np.meshgrid(fftfreq(N, d=L/N), 
                                fftfreq(N, d=L/N), 
                                fftfreq(N, d=L/N), indexing='ij')
        k = np.stack((kx, ky, kz), axis=-1)

        # Calculate h_plus and h_minus for the selected wavevector
        k_vector = k[k_index]
        k_norm = np.linalg.norm(k_vector,axis=-1,keepdims=True)
        k_hat = k_vector / k_norm

        z = np.array([0, 0, 1])
        e = np.cross(z, k_hat)
        e_norm = np.linalg.norm(e,axis=-1)
        e_hat = e / e_norm

        factor = 1/np.sqrt(2.0)
        e_cross_k = np.cross(e_hat,k_vector)
        e_cross_k_norm = np.linalg.norm(e_cross_k,axis=-1,keepdims=True)
        k_cross_e_cross_k = np.cross(k_vector, e_cross_k)
        k_cross_e_cross_k_norm = np.linalg.norm(k_cross_e_cross_k,axis=-1,keepdims=True)
        
        h_plus =  factor * e_cross_k / e_cross_k_norm + \
                    factor * 1j * k_cross_e_cross_k / k_cross_e_cross_k_norm
                    
        h_minus = factor * e_cross_k / e_cross_k_norm - \
                    factor * 1j * k_cross_e_cross_k / k_cross_e_cross_k_norm

        # Assign coefficients in Fourier space
        field_fft[k_index] = A_plus * h_plus + A_minus * h_minus

        # Perform inverse FFT to get the field in physical space
        field = ifftn(field_fft, axes=(0, 1, 2),norm="forward").real

        return field

    @staticmethod
    def generate_isotropic_powerlaw_field(size:  int,
                                          alpha: float = 5./3.) -> np.ndarray:
        """
        This computes a random field with a power-law power spectrum. The power spectrum
        is P(k) = k^-alpha. The field is generated in Fourier space, and then inverse
        transformed to real space.

        Author: James Beattie (2023)

        Args:
            size (int): the linear dimension of the 3D field
            alpha (float): the negative 1D power-law exponent used in Fourier space. Defaults to 5/3.
                            Note that I use the negative exponent, because the power spectrum is
                            proportional to k^-alpha, and note that I make the transformations between
                            3D Fourier transform exponent and 1D power spectrum exponent in the code.

        Returns:
            ifft field (np.ndarray): the inverse fft of the random field with a power-law power spectrum
        """
        # Create a grid of frequencies
        kx = np.fft.fftfreq(size)
        ky = np.fft.fftfreq(size)
        kz = np.fft.fftfreq(size)
        
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Calculate the magnitude of k for each frequency component
        k = np.sqrt(kx**2 + ky**2 + kz**2)
        k[0, 0, 0] = 1  # Avoid division by zero
        
        # Create a 3D grid of random complex numbers
        random_field = np.random.randn(size, size, size) + 1j * np.random.randn(size, size, size)
        
        # Adjust the amplitude of each frequency component to follow k^-5/3 (-11/3 in 3D)
        amplitude = np.where(k != 0, k**(-(alpha+2.0)/2.0), 0)
        adjusted_field = 10*random_field * amplitude

        return np.fft.ifftn(adjusted_field).real

    @staticmethod
    def generate_anisotropic_powerlaw_field(N:      int,
                                            alpha:  float = 5./3.,
                                            beta:   float = 5./3.,
                                            L:      float = 1.0) -> np.ndarray:
        """
        This computes a random field with a power-law power spectrum. The power spectrum
        is P(k) = k_perp^-alpha k_parallel^-beta. The field is generated in Fourier space, 
        and then inverse transformed to real space.
        
        Author: James Beattie

        Args:
            N (int): the linear dimension of the 3D field
            alpha (float): the negative 1D power-law exponent used in Fourier space for the 
                            perpendicular component. Defaults to 5/3.
                            Note that I make the transformations between 3D Fourier transform 
                            exponent and 1D power spectrum exponent in the code.
            beta (float): the negative 1D power-law exponent used in Fourier space for the 
                            parallel component. Defaults to 5/3.
            L (float): the physical size of the domain. Defaults to 1.0.

        Returns:
            ifft field (np.ndarray): the inverse fft of the random field with a power-law power spectrum
        """
        
        # Create a grid of frequencies
        kx = np.fft.fftfreq(N)
        ky = np.fft.fftfreq(N)
        kz = np.fft.rfftfreq(N)   
        
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Calculate the magnitude of k for each frequency component
        k_perp              = np.sqrt(kx**2 + ky**2)
        k_par               = np.abs(kz)
        k_perp[k_perp==0]   = np.inf   # Avoid division by zero
        k_par[k_par==0]     = np.inf   # Avoid division by zero
        
        
        # Adjust the amplitude of each frequency component to follow k^-5/3 (-11/3 in 3D)
        amplitude = k_perp**(-(alpha+1.0)/2.0)*k_par**(-beta/2.0)

        # Create a 3D grid of random complex numbers for phases
        rphase = np.exp(2j*np.pi*np.random.rand(*amplitude.shape))
        
        Fhalf = 10*rphase * amplitude
        Fhalf[0,0,0] = 0.0  # Set the zero mode to zero to avoid DC component
        
        # build full cube via Hermitian symmetry
        F = np.zeros((N, N, N), dtype=complex)
        F[:, :, :N//2+1] = Fhalf
        F[:, :, N//2+1:] = np.conj(Fhalf[:, :, 1:N//2][..., ::-1])

        return np.fft.ifftn(F,norm="forward").real
    
    @staticmethod
    def helical_decomposition(vector_field):
        """
        Performs a helical decomposition of a vector field.

        Parameters:
        velocity_field (array-like): The velocity field corresponding to each k, an array of shape (N, 3).

        Returns:
        u_plus (array): The component of the vector field in the direction of the right-handed helical component.
        u_minus (array): The component of the vector field in the direction of the left-handed helical component.
        
        TODO: this whole function needs to be updated to conform to 3,N,N,N vector fields instead of
            N,N,N,3 vector fields
        """
        # Convert inputs to numpy arrays
        vector_field = np.asarray(vector_field)
        
        # Take FFT of vector field
        vector_field_FFT = fftn(vector_field,
                                norm='forward',
                                axes=(0,1,2))
        N = vector_field.shape[0]  # Assuming a cubic domain
        L = 1  # The physical size of the domain
        kx = fftfreq(N, d=L/N)
        ky = fftfreq(N, d=L/N)
        kz = fftfreq(N, d=L/N)
        
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
        k = np.stack((kx, ky, kz), axis=-1)  # This will be of shape (N, N, N, 3)

        # Normalize k to get the unit wavevector
        k_norm = np.linalg.norm(k, axis=-1, keepdims=True)

        if np.any(np.isnan(k_norm)) or np.any(np.isinf(k_norm)):
            raise ValueError("NaN or Inf found in k_norm")    

        # Set the k_hat for zero wavevectors explicitly to zero (or some other appropriate value)
        k_hat = np.zeros_like(k)
        non_zero_indices = k_norm.squeeze() > 0  # Indices where the norm is non-zero
        k_hat[non_zero_indices] = k[non_zero_indices] / k_norm[non_zero_indices]
        
        if np.any(np.isnan(k_hat)) or np.any(np.isinf(k_hat)):
            raise ValueError("NaN or Inf found in k_hat")

        # Choose an arbitrary versor orthogonal to k
        z = np.array([0, 0, 1])
        e = np.cross(z, k_hat)
        e_norm = np.linalg.norm(e, axis=-1, keepdims=True)
        
        # Set the k_hat for zero wavevectors explicitly to zero (or some other appropriate value)
        e_hat = np.zeros_like(k)
        non_zero_indices = e_norm.squeeze() > 0  # Indices where the norm is non-zero
        e_hat[non_zero_indices] = e[non_zero_indices] / e_norm[non_zero_indices]

        # Ensure that e_hat is not a zero vector (which can happen if k is parallel to z)
        # In such a case, we can choose e_hat to be any vector orthogonal to k
        for i, e in enumerate(e_hat):
            if np.allclose(e, np.zeros_like(e)):
                # Choose a new e_hat that is not parallel to k
                if np.allclose(k_hat[i], np.array([1, 0, 0])) or np.allclose(k_hat[i], np.array([0, 0, 1])):
                    e_hat[i] = np.array([0, 1, 0])
                else:
                    e_hat[i] = np.array([1, 0, 0])

        # Calculate helical components    
        factor                 = 1/np.sqrt(2.0)
        e_cross_k              = np.cross(e_hat,k)
        e_cross_k_norm         = np.linalg.norm(e_cross_k,axis=-1,keepdims=True)
        k_cross_e_cross_k      = np.cross(k, e_cross_k)
        k_cross_e_cross_k_norm = np.linalg.norm(k_cross_e_cross_k,axis=-1,keepdims=True)
        
        h_plus =  factor * e_cross_k / e_cross_k_norm  + \
                    factor * 1j * k_cross_e_cross_k / k_cross_e_cross_k_norm
                    
        h_minus = factor * e_cross_k / e_cross_k_norm - \
                    factor * 1j * k_cross_e_cross_k / k_cross_e_cross_k_norm

        # test orthogonality 
        #print(np.abs(np.sum(h_plus * h_minus, axis=-1)))
        #print(np.sum(np.abs(h_minus * h_plus), axis=-1))

        # Project velocity field onto helical components
        u_plus = np.sum(vector_field_FFT * h_plus, axis=-1)
        u_minus = np.sum(vector_field_FFT * h_minus, axis=-1)

        # remove k = 0 mode   
        u_plus[np.isnan(u_plus)] = 0
        u_minus[np.isnan(u_minus)] = 0

        return u_plus, u_minus
    
    
    # def cylindrical_integrate(self,
    #                           data: np.ndarray, 
    #                           bins_perp: int = 0,
    #                           bins_para:  int = 0) -> tuple:
    #     """
    #     The cylindrical integrate function takes the 3D power spectrum and integrates
    #     over cylindrical shells of constant k in a plane, and then a 1D spectrum along
    #     the the remaining dimension. The result is a 2D power spectrum of k_perp and k_par
    #     modes.
        
        
    # Args:
    #         data     : The 3D power spectrum
    #         bins_perp: Number of bins for k_perpendicular
    #         bins_para: Number of bins for k_parallel

    #     Returns:
    #         k_perp_modes    : k_perpendicular modes
    #         k_para_modes    : k_parallel modes
    #         cylindrical_sum : cylindrical integration of the 3D power spectrum
    #                         note that k_perp is on axis 0, k_para is on axis 1
    #     """
        
    #     z, y, x = np.indices(data.shape)
    #     center  = np.array([(i - 1) / 2.0 for i in data.shape])
    #     k_perp  = np.sqrt((x - center[0])**2 + (y - center[1])**2)  # Cylindrical radius
    #     k_para  = np.abs(z - center[2])                             # Distance from the plane

    #     N = data.shape[0]
    #     if bins_perp == 0:
    #         bins_perp = N // 2
    #     if bins_para == 0:
    #         bins_para = N // 2

    #     # initailize cylindrical sum
    #     cylindrical_sum = np.zeros((bins_perp, bins_para))

    #     # define bin edges (note starting at 0.5 to avoid binning the zero mode)
    #     bin_edges_perp = np.linspace(0, bins_perp, bins_perp+1)
    #     bin_edges_para = np.linspace(0, bins_para, bins_para+1)

    #     # Vectorized bin assignment
    #     bin_indices_perp = np.digitize(k_perp, bin_edges_perp) - 1
    #     bin_indices_para = np.digitize(k_para, bin_edges_para) - 1

    #     # Create 2D linear indices
    #     linear_indices = bin_indices_perp + bin_indices_para * bins_perp

    #     # Use np.bincount for efficient summation
    #     cylindrical_sum = np.bincount(linear_indices.ravel(), 
    #                                 weights=data.ravel(), 
    #                                 minlength=bins_perp * bins_para) 
        
    #     # Ensure that the length matches the expected size
    #     cylindrical_sum = cylindrical_sum[:bins_perp * bins_para]
    #     cylindrical_sum = cylindrical_sum.reshape((bins_perp, bins_para),
    #                                             order='F')
    #     # k_perp are in the first axis, k_par are in the second axis
    #     k_perp_modes    = (bin_edges_perp[:-1] + bin_edges_perp[1:]) / 2
    #     k_para_modes    = (bin_edges_para[:-1] + bin_edges_para[1:]) / 2

    #     return k_perp_modes, k_para_modes, cylindrical_sum