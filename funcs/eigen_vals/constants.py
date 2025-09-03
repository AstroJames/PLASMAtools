from numba import types

##############################################################################
# Global constants
##############################################################################

X, Y, Z = 0, 1, 2  # indexes
EPSILON = 1e-10    # Small value for numerical stability

##############################################################################
# Type signatures for Numba functions
##############################################################################

# Signatures for 3D symmetric 3x3 eigenvalue computation
eigenvalues_symmetric_3x3_3d_sig_32 = types.void(
    types.float32[:,:,:,:,:],     # tensor_field: (3, 3, Nx, Ny, Nz)
    types.float32[:,:,:,:],       # eigenvalues: (3, Nx, Ny, Nz)
)
eigenvalues_symmetric_3x3_3d_sig_64 = types.void(
    types.float64[:,:,:,:,:],     # tensor_field: (3, 3, Nx, Ny, Nz)
    types.float64[:,:,:,:],       # eigenvalues: (3, Nx, Ny, Nz)
)

# Signatures for 2D symmetric 2x2 eigenvalue computation
eigenvalues_symmetric_2x2_2d_sig_32 = types.void(
    types.float32[:,:,:,:],       # tensor_field: (2, 2, Nx, Ny)
    types.float32[:,:,:],         # eigenvalues: (2, Nx, Ny)
)
eigenvalues_symmetric_2x2_2d_sig_64 = types.void(
    types.float64[:,:,:,:],       # tensor_field: (2, 2, Nx, Ny)
    types.float64[:,:,:],         # eigenvalues: (2, Nx, Ny)
)

# Signatures for eigenvector computation (3D)
eigenvectors_symmetric_3x3_3d_sig_32 = types.void(
    types.float32[:,:,:,:,:],     # tensor_field: (3, 3, Nx, Ny, Nz)
    types.float32[:,:,:,:],       # eigenvalues: (3, Nx, Ny, Nz)
    types.float32[:,:,:,:,:],     # eigenvectors: (3, 3, Nx, Ny, Nz)
)
eigenvectors_symmetric_3x3_3d_sig_64 = types.void(
    types.float64[:,:,:,:,:],     # tensor_field: (3, 3, Nx, Ny, Nz)
    types.float64[:,:,:,:],       # eigenvalues: (3, Nx, Ny, Nz)
    types.float64[:,:,:,:,:],     # eigenvectors: (3, 3, Nx, Ny, Nz)
)

# Signatures for general (non-symmetric) 3x3 eigenvalue computation
eigenvalues_general_3x3_3d_sig_32 = types.void(
    types.float32[:,:,:,:,:],     # tensor_field: (3, 3, Nx, Ny, Nz)
    types.float32[:,:,:,:],       # eigenvalues_real: (3, Nx, Ny, Nz)
    types.float32[:,:,:,:],       # eigenvalues_imag: (3, Nx, Ny, Nz)
)
eigenvalues_general_3x3_3d_sig_64 = types.void(
    types.float64[:,:,:,:,:],     # tensor_field: (3, 3, Nx, Ny, Nz)
    types.float64[:,:,:,:],       # eigenvalues_real: (3, Nx, Ny, Nz)
    types.float64[:,:,:,:],       # eigenvalues_imag: (3, Nx, Ny, Nz)
)

# Signatures for general (non-symmetric) 2x2 eigenvalue computation
eigenvalues_general_2x2_2d_sig_32 = types.void(
    types.float32[:,:,:,:],       # tensor_field: (2, 2, Nx, Ny)
    types.float32[:,:,:],         # eigenvalues_real: (2, Nx, Ny)
    types.float32[:,:,:],         # eigenvalues_imag: (2, Nx, Ny)
)
eigenvalues_general_2x2_2d_sig_64 = types.void(
    types.float64[:,:,:,:],       # tensor_field: (2, 2, Nx, Ny)
    types.float64[:,:,:],         # eigenvalues_real: (2, Nx, Ny)
    types.float64[:,:,:],         # eigenvalues_imag: (2, Nx, Ny)
)

# Signatures for general (non-symmetric) 3x3 eigenvector computation
eigenvectors_general_3x3_3d_sig_32 = types.void(
    types.float32[:,:,:,:,:],     # tensor_field: (3, 3, Nx, Ny, Nz)
    types.float32[:,:,:,:],       # eigenvalues_real: (3, Nx, Ny, Nz)
    types.float32[:,:,:,:],       # eigenvalues_imag: (3, Nx, Ny, Nz)
    types.float32[:,:,:,:,:],     # eigenvectors_real: (3, 3, Nx, Ny, Nz)
    types.float32[:,:,:,:,:],     # eigenvectors_imag: (3, 3, Nx, Ny, Nz)
)
eigenvectors_general_3x3_3d_sig_64 = types.void(
    types.float64[:,:,:,:,:],     # tensor_field: (3, 3, Nx, Ny, Nz)
    types.float64[:,:,:,:],       # eigenvalues_real: (3, Nx, Ny, Nz)
    types.float64[:,:,:,:],       # eigenvalues_imag: (3, Nx, Ny, Nz)
    types.float64[:,:,:,:,:],     # eigenvectors_real: (3, 3, Nx, Ny, Nz)
    types.float64[:,:,:,:,:],     # eigenvectors_imag: (3, 3, Nx, Ny, Nz)
)

# Signatures for general (non-symmetric) 2x2 eigenvector computation
eigenvectors_general_2x2_2d_sig_32 = types.void(
    types.float32[:,:,:,:],       # tensor_field: (2, 2, Nx, Ny)
    types.float32[:,:,:],         # eigenvalues_real: (2, Nx, Ny)
    types.float32[:,:,:],         # eigenvalues_imag: (2, Nx, Ny)
    types.float32[:,:,:,:],       # eigenvectors_real: (2, 2, Nx, Ny)
    types.float32[:,:,:,:],       # eigenvectors_imag: (2, 2, Nx, Ny)
)
eigenvectors_general_2x2_2d_sig_64 = types.void(
    types.float64[:,:,:,:],       # tensor_field: (2, 2, Nx, Ny)
    types.float64[:,:,:],         # eigenvalues_real: (2, Nx, Ny)
    types.float64[:,:,:],         # eigenvalues_imag: (2, Nx, Ny)
    types.float64[:,:,:,:],       # eigenvectors_real: (2, 2, Nx, Ny)
    types.float64[:,:,:,:],       # eigenvectors_imag: (2, 2, Nx, Ny)
)

# Signatures for symmetric 3x3 analytical eigenvalue computation (from hal-01501221)
eigenvalues_symmetric_analytical_3x3_3d_sig_32 = types.void(
    types.float32[:,:,:,:,:],     # tensor_field: (3, 3, Nx, Ny, Nz)
    types.float32[:,:,:,:],       # eigenvalues: (3, Nx, Ny, Nz)
)
eigenvalues_symmetric_analytical_3x3_3d_sig_64 = types.void(
    types.float64[:,:,:,:,:],     # tensor_field: (3, 3, Nx, Ny, Nz)
    types.float64[:,:,:,:],       # eigenvalues: (3, Nx, Ny, Nz)
)