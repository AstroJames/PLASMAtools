from numba import types

##############################################################################
# Global constants
##############################################################################

# Constants
X, Y, Z = 0, 1, 2
DEFAULT_EPS = 1e-10


##############################################################################
# Type signatures for Numba functions
##############################################################################

# Vector dot product signatures
sig_dot_3d_f32 = types.float32[:,:,:]( # three-dimensional
    types.float32[:,:,:,:], 
    types.float32[:,:,:,:]
    )
sig_dot_3d_f64 = types.float64[:,:,:](
    types.float64[:,:,:,:], 
    types.float64[:,:,:,:]
    )
sig_dot_2d_f32 = types.float32[:,:]( # two-dimensional
    types.float32[:,:,:],
    types.float32[:,:,:]
    )
sig_dot_2d_f64 = types.float64[:,:](
    types.float64[:,:,:],
    types.float64[:,:,:]
    )

# Vector magnitude signatures
sig_mag_3d_f32 = types.float32[:,:,:]( # three-dimensional
    types.float32[:,:,:,:]
    )
sig_mag_3d_f64 = types.float64[:,:,:](
    types.float64[:,:,:,:]
    )
sig_mag_2d_f32 = types.float32[:,:]( # two-dimensional
    types.float32[:,:,:]
    )
sig_mag_2d_f64 = types.float64[:,:](
    types.float64[:,:,:]
    )

# Vector cross product signatures (3D returns vector, 2D returns scalar)
sig_cross_3d_f32 = types.float32[:,:,:,:]( # three-dimensional
    types.float32[:,:,:,:],
    types.float32[:,:,:,:]
    )
sig_cross_3d_f64 = types.float64[:,:,:,:](
    types.float64[:,:,:,:],
    types.float64[:,:,:,:]
    )
sig_cross_2d_f32 = types.float32[:,:]( # two-dimensional
    types.float32[:,:,:],
    types.float32[:,:,:]
    )
sig_cross_2d_f64 = types.float64[:,:](
    types.float64[:,:,:], types.float64[:,:,:]
    )

# Vector normalize signatures
sig_norm_3d_f32 = types.float32[:,:,:,:]( # three-dimensional
    types.float32[:,:,:,:],
    types.float32
    )
sig_norm_3d_f64 = types.float64[:,:,:,:](
    types.float64[:,:,:,:],
    types.float64
    )
sig_norm_2d_f32 = types.float32[:,:,:]( # two-dimensional
    types.float32[:,:,:],
    types.float32
    )
sig_norm_2d_f64 = types.float64[:,:,:](
    types.float64[:,:,:], 
    types.float64
    )

# Scalar triple product signatures
sig_triple_3d_f32 = types.float32[:,:,:]( # three-dimensional
    types.float32[:,:,:,:],
    types.float32[:,:,:,:], 
    types.float32[:,:,:,:]
    )
sig_triple_3d_f64 = types.float64[:,:,:](
    types.float64[:,:,:,:], 
    types.float64[:,:,:,:], 
    types.float64[:,:,:,:]
    )

# Vector angle signatures
sig_angle_3d_f32 = types.float32[:,:,:]( # three-dimensional
    types.float32[:,:,:,:],
    types.float32[:,:,:,:],
    types.float32
    )
sig_angle_3d_f64 = types.float64[:,:,:](
    types.float64[:,:,:,:],
    types.float64[:,:,:,:],
    types.float64
    )
sig_angle_2d_f32 = types.float32[:,:]( # two-dimensional
    types.float32[:,:,:],
    types.float32[:,:,:],
    types.float32
    )
sig_angle_2d_f64 = types.float64[:,:](
    types.float64[:,:,:],
    types.float64[:,:,:],
    types.float64
    )

# Extract vector component
sig_extract_component_3d_f32 = types.float32[:,:,:]( # three-dimensional
    types.float32[:,:,:,:],
    types.int32
    )
sig_extract_component_3d_f64 = types.float64[:,:,:](
    types.float64[:,:,:,:],
    types.int32
    )

# Vector projection signatures
sig_project_3d_f32 = types.float32[:,:,:,:]( # three-dimensional
    types.float32[:,:,:,:], 
    types.float32[:,:,:,:]
    )
sig_project_3d_f64 = types.float64[:,:,:,:](
    types.float64[:,:,:,:],
    types.float64[:,:,:,:]
    )
