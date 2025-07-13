"""
Type signatures and constants for spectral analysis functions.
Centralizes all Numba type definitions.
"""
from numba import types

# Numb signatures for JIT compilation

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


##############################################################################
# Global constants
##############################################################################

DEFAULT_BINS_RATIO = 2  # N // 2
DEFAULT_SIGMA = 10.0    # for Gaussian bins