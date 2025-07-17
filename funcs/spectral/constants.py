"""
Type signatures and constants for spectral analysis functions.
Centralizes all Numba type definitions.
"""
from numba import types
import numpy as np

##############################################################################
# Global constants
##############################################################################

TwoPi = 2.0 * np.pi  # 2 * pi constant
X, Y, Z = 0, 1, 2  # coordinate indices
N_COORDS_VEC, X_GRID_VEC, Y_GRID_VEC, Z_GRID_VEC = 0, 1, 2, 3  # vector grid dimensions
N_COORDS_TENS, M_COORDS_TENS, X_GRID_TENS, Y_GRID_TENS, Z_GRID_TENS = 0, 1, 2, 3, 4  # tensor grid dimensions
DEFAULT_BIN_MIN = 0.5  # the smallest k mode
DEFAULT_BINS_RATIO = 2  # N // 2
DEFAULT_SIGMA = 10.0    # for Gaussian bins


##############################################################################
# Type signatures for Numba functions
##############################################################################

# Distances
sig_rad_dist_3D = types.float32[:,:,:](
    types.UniTuple(types.int64, 3)
    )
sig_rad_dist_2D = types.float32[:,:](
    types.UniTuple(types.int64, 2)
    )

# Spherical Integration in 3D
sig_sph_int_32 = types.float32[:](
    types.float32[:,:,:],
    types.float32[:,:,:],
    types.float32[:],
    types.int64
    )
sig_sph_int_64 = types.float64[:](
    types.float64[:,:,:],
    types.float64[:,:,:],
    types.float64[:],
    types.int64
    )

# Spherical Integration 2D
sig_sph_int_2d_32 = types.float32[:](
    types.float32[:,:],
    types.float32[:,:],
    types.float32[:],
    types.int64
    )
sig_sph_int_2d_64 = types.float64[:](
    types.float64[:,:],
    types.float64[:,:],
    types.float64[:],
    types.int64
    )

# Cylindrical Integration
sig_cyl_int_32 = types.float32[:,:](
    types.float32[:,:,:],
    types.float32[:,:,:],
    types.float32[:,:,:], 
    types.float32[:],
    types.float32[:],
    types.int64,
    types.int64
    )
sig_cyl_int_64 = types.float64[:,:](
    types.float64[:,:,:],
    types.float64[:,:,:],
    types.float64[:,:,:], 
    types.float64[:],
    types.float64[:],
    types.int64,
    types.int64
    )

# Shell Filtering
sig_filter_32 = types.UniTuple(types.float32[:,:,:], 2)(
    types.float32[:,:,:],
    types.float32[:,:,:], 
    types.float32[:,:,:],
    types.float32, 
    types.float32,
    types.int64,
    types.float32
    )

# Mixed spectrum in 2D
mixed_spec_2d_sig_32 = types.float32[:,:](
    types.complex64[:,:,:],
    types.complex64[:,:,:]
    )
mixed_spec_2d_sig_64 = types.float64[:,:](
    types.complex128[:,:,:],
    types.complex128[:,:,:]
    )

# Mixed spectrum in 3D
mixed_spec_sig_32 = types.float32[:,:,:](
    types.complex64[:,:,:,:],
    types.complex64[:,:,:,:]
    )
mixed_spec_sig_64 = types.float64[:,:,:](
    types.complex128[:,:,:,:],
    types.complex128[:,:,:,:]
    )

# Type signatures for Helmholtz decomposition core functions
helmholtz_sig_32 = types.UniTuple(types.complex64[:,:,:,:], 2)(
    types.complex64[:,:,:,:],  # Fhat input
    types.float32[:,:,:],      # kx
    types.float32[:,:,:],      # ky  
    types.float32[:,:,:],      # kz
    types.float32[:,:,:]       # ksqr
)

helmholtz_sig_64 = types.UniTuple(types.complex128[:,:,:,:], 2)(
    types.complex128[:,:,:,:], # Fhat input
    types.float64[:,:,:],      # kx
    types.float64[:,:,:],      # ky
    types.float64[:,:,:],      # kz
    types.float64[:,:,:]       # ksqr
)