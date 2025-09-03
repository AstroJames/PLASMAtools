import numpy as np
from numba import types

##############################################################################
# Global constants
##############################################################################

TwoPi = 2.0 * np.pi  # 2 * pi constant
X, Y, Z = 0, 1, 2  # coordinate indices
N_COORDS_VEC, X_GRID_VEC, Y_GRID_VEC, Z_GRID_VEC = 0, 1, 2, 3  # vector grid dimensions
N_COORDS_TENS, M_COORDS_TENS, X_GRID_TENS, Y_GRID_TENS, Z_GRID_TENS = 0, 1, 2, 3, 4  # tensor grid dimensions


##############################################################################
# Type signatures for Numba functions
##############################################################################

# Define for common type signatures
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

# 1D gradient signatures
sig_1d_f32 = types.float32[:](
    types.float32[:], 
    types.float32, 
    types.int32[:], 
    types.float64[:], 
    types.int32, 
    types.float32[:]
    )
sig_1d_f64 = types.float64[:](
    types.float64[:], 
    types.float64, 
    types.int64[:], 
    types.float64[:], 
    types.int32, 
    types.float64[:]
    )

# 2D gradient signatures
sig_2d_f32 = types.float32[:,:](
    float32_2d, 
    types.int32, 
    types.float32, 
    int32_1d, 
    float32_1d, 
    types.int32, 
    float32_2d
    )
sig_2d_f64 = types.float64[:,:](
    float64_2d, 
    types.int32, 
    types.float64, 
    int64_1d, 
    float64_1d, 
    types.int32, 
    float64_2d
    )

# 3D gradient signatures
sig_3d_f32 = types.float32[:,:,:](
    float32_3d, 
    types.int32, 
    types.float32, 
    int32_1d, 
    float32_1d, 
    types.int32, 
    float32_3d
    )
sig_3d_f64 = types.float64[:,:,:](
    float64_3d, 
    types.int32, 
    types.float64, 
    int64_1d, 
    float64_1d, 
    types.int32, 
    float64_3d
    )

# gradient tensor signatures
sig_tensor_f32 = types.float32[:,:,:,:,:](
    float32_4d, 
    int32_1d, 
    float32_1d, 
    types.float32, 
    float32_5d
    )
sig_tensor_f64 = types.float64[:,:,:,:,:](
    float64_4d, 
    int64_1d, 
    float64_1d, 
    types.float64, 
    float64_5d
    )

# vector curl signatures
sig_curl_3d_f32 = types.float32[:,:,:,:](
    float32_4d, 
    int32_1d, 
    float32_1d, 
    types.float32, 
    types.float32, 
    types.float32, 
    types.int32, 
    float32_4d
    )
sig_curl_3d_f64 = types.float64[:,:,:,:](
    float32_4d,
    int32_1d,
    float32_1d,
    types.float64,
    types.float64,
    types.float64,
    types.int32,
    float64_4d
    )

# vector divergence signatures
sig_div_3d_f32 = types.float32[:,:,:](
    float32_4d, 
    int32_1d, 
    float32_1d, 
    types.float32, 
    types.float32, 
    types.float32, 
    types.int32, 
    float32_3d
    )
sig_div_3d_f64 = types.float64[:,:,:](
    float64_4d, 
    int32_1d, 
    float32_1d, 
    types.float64, 
    types.float64,
    types.float64,
    types.int32,
    float64_3d
    )