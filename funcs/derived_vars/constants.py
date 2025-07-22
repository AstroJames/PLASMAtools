"""
    Type signatures and constants for derived variables analysis functions.
    Centralizes all Numba type definitions.
"""
from numba import types
import numpy as np

##############################################################################
# Global constants
##############################################################################

boundary_lookup = {0: 'periodic', 
                   1: 'neumann', 
                   2: 'dirichlet'}

TwoPi = 2.0 * np.pi  # 2 * pi constant
X,Y,Z = 0, 1, 2  # coordinate indices
N_COORDS_VEC, X_GRID_VEC, Y_GRID_VEC, Z_GRID_VEC = 0, 1, 2, 3  # vector grid dimensions
N_COORDS_TENS, M_COORDS_TENS, X_GRID_TENS, Y_GRID_TENS, Z_GRID_TENS = 0, 1, 2, 3, 4  # tensor grid dimensions


##############################################################################
# Type signatures for Numba functions
##############################################################################

# Define common array types
complex64_2d = types.complex64[:, :]
complex64_3d = types.complex64[:, :, :]
complex64_4d = types.complex64[:, :, :, :]

complex128_2d = types.complex128[:, :]
complex128_3d = types.complex128[:, :, :]
complex128_4d = types.complex128[:, :, :, :]

float32_2d = types.float32[:, :]
float32_3d = types.float32[:, :, :]
float32_4d = types.float32[:, :, :, :]

float64_2d = types.float64[:, :]
float64_3d = types.float64[:, :, :]
float64_4d = types.float64[:, :, :, :]

# Signatures for compute_vector_potential_2D_core
# Input: k (2, nx, ny), field_fft (2, nx, ny), kx (nx, ny), ky (nx, ny)
# Output: psi_hat (nx, ny) - complex
sig_vector_potential_2D_f32 = complex64_2d(
    float32_3d,     # k: (2, nx, ny)
    complex64_3d,   # field_fft: (2, nx, ny)
    float32_2d,     # kx: (nx, ny)
    float32_2d      # ky: (nx, ny)
)

sig_vector_potential_2D_f64 = complex128_2d(
    float64_3d,     # k: (2, nx, ny)
    complex128_3d,  # field_fft: (2, nx, ny)
    float64_2d,     # kx: (nx, ny)
    float64_2d      # ky: (nx, ny)
)

# Mixed precision versions (common case: float64 k with complex64 field_fft)
sig_vector_potential_2D_mixed_32 = complex64_2d(
    float64_3d,     # k: (2, nx, ny) - often computed in float64
    complex64_3d,   # field_fft: (2, nx, ny)
    float64_2d,     # kx: (nx, ny)
    float64_2d      # ky: (nx, ny)
)

sig_vector_potential_2D_mixed_64 = complex128_2d(
    float64_3d,     # k: (2, nx, ny)
    complex64_3d,   # field_fft: (2, nx, ny) - input data might be float32
    float64_2d,     # kx: (nx, ny)
    float64_2d      # ky: (nx, ny)
)

# Signatures for compute_vector_potential_3D_core
# Input: k (3, nx, ny, nz), field_fft (3, nx, ny, nz), kx (nx, ny, nz), ky (nx, ny, nz), kz (nx, ny, nz)
# Output: a_hat (3, nx, ny, nz) - complex
sig_vector_potential_3D_f32 = complex64_4d(
    float32_4d,     # k: (3, nx, ny, nz)
    complex64_4d,   # field_fft: (3, nx, ny, nz)
    float32_3d,     # kx: (nx, ny, nz)
    float32_3d,     # ky: (nx, ny, nz)
    float32_3d      # kz: (nx, ny, nz)
)

sig_vector_potential_3D_f64 = complex128_4d(
    float64_4d,     # k: (3, nx, ny, nz)
    complex128_4d,  # field_fft: (3, nx, ny, nz)
    float64_3d,     # kx: (nx, ny, nz)
    float64_3d,     # ky: (nx, ny, nz)
    float64_3d      # kz: (nx, ny, nz)
)

# Mixed precision versions for 3D
sig_vector_potential_3D_mixed_32 = complex64_4d(
    float64_4d,     # k: (3, nx, ny, nz)
    complex64_4d,   # field_fft: (3, nx, ny, nz)
    float64_3d,     # kx: (nx, ny, nz)
    float64_3d,     # ky: (nx, ny, nz)
    float64_3d      # kz: (nx, ny, nz)
)

sig_vector_potential_3D_mixed_64 = complex128_4d(
    float64_4d,     # k: (3, nx, ny, nz)
    complex64_4d,   # field_fft: (3, nx, ny, nz)
    float64_3d,     # kx: (nx, ny, nz)
    float64_3d,     # ky: (nx, ny, nz)
    float64_3d      # kz: (nx, ny, nz)
)

# Your specific case: float32 k with complex64 field_fft and float64 wave vectors
sig_vector_potential_3D_mixed_f32_k = complex64_4d(
    float32_4d,     # k: (3, nx, ny, nz) - float32
    complex64_4d,   # field_fft: (3, nx, ny, nz) - complex64
    float64_3d,     # kx: (nx, ny, nz) - float64
    float64_3d,     # ky: (nx, ny, nz) - float64
    float64_3d      # kz: (nx, ny, nz) - float64
)

# Signatures for reconstruct_field_from_stream_2D_core
# Input: psi (nx, ny), dx (float), dy (float)
# Output: field (2, nx, ny) - same type as input
sig_reconstruct_2D_f32 = float32_3d(
    float32_2d,     # psi: (nx, ny)
    types.float32,  # dx
    types.float32   # dy
)

sig_reconstruct_2D_f64 = float64_3d(
    float64_2d,     # psi: (nx, ny)
    types.float64,  # dx
    types.float64   # dy
)

# Complex versions (if psi is complex from inverse FFT)
sig_reconstruct_2D_complex_f32 = complex64_3d(
    complex64_2d,   # psi: (nx, ny) - complex
    types.float32,  # dx
    types.float32   # dy
)

sig_reconstruct_2D_complex_f64 = complex128_3d(
    complex128_2d,  # psi: (nx, ny) - complex
    types.float64,  # dx
    types.float64   # dy
)