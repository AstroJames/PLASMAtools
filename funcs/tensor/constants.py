from numba import types

##############################################################################
# Global constants
##############################################################################

X,Y,Z = 0, 1, 2 # indexes
DEFAULT_SIGMA = 10.0    # for Gaussian bins


##############################################################################
# Type signatures for Numba functions
##############################################################################

# Signatures for 3D tensor decomposition
tensor_decomp_3d_sig_32 = types.void(
    types.float32[:,:,:,:,:],     # tensor_field: (3, 3, Nx, Ny, Nz)
    types.float32[:,:,:,:,:],     # out_sym: (3, 3, Nx, Ny, Nz)
    types.float32[:,:,:,:,:],     # out_asym: (3, 3, Nx, Ny, Nz)
    types.float32[:,:,:,:,:],     # out_bulk: (3, 3, Nx, Ny, Nz)
    types.UniTuple(types.boolean, 3)  # compute_flags: (bool, bool, bool)
)
tensor_decomp_3d_sig_64 = types.void(
    types.float64[:,:,:,:,:],     # tensor_field: (3, 3, Nx, Ny, Nz)
    types.float64[:,:,:,:,:],     # out_sym: (3, 3, Nx, Ny, Nz)
    types.float64[:,:,:,:,:],     # out_asym: (3, 3, Nx, Ny, Nz)
    types.float64[:,:,:,:,:],     # out_bulk: (3, 3, Nx, Ny, Nz)
    types.UniTuple(types.boolean, 3)  # compute_flags: (bool, bool, bool)
)

# Signatures for 2D tensor decomposition
tensor_decomp_2d_sig_32 = types.void(
    types.float32[:,:,:,:],       # tensor_field: (2, 2, Nx, Ny)
    types.float32[:,:,:,:],       # out_sym: (2, 2, Nx, Ny)
    types.float32[:,:,:,:],       # out_asym: (2, 2, Nx, Ny)
    types.float32[:,:,:,:],       # out_bulk: (2, 2, Nx, Ny)
    types.UniTuple(types.boolean, 3)  # compute_flags: (bool, bool, bool)
)
tensor_decomp_2d_sig_64 = types.void(
    types.float64[:,:,:,:],       # tensor_field: (2, 2, Nx, Ny)
    types.float64[:,:,:,:],       # out_sym: (2, 2, Nx, Ny)
    types.float64[:,:,:,:],       # out_asym: (2, 2, Nx, Ny)
    types.float64[:,:,:,:],       # out_bulk: (2, 2, Nx, Ny)
    types.UniTuple(types.boolean, 3)  # compute_flags: (bool, bool, bool)
)

# Signatures for tensor double contraction
sig_double_contraction_3d = types.float32[:,:,:](
    types.float32[:,:,:,:,:], 
    types.float32[:,:,:,:,:]
    )
sig_double_contraction_2d = types.float32[:,:](
    types.float32[:,:,:,:], 
    types.float32[:,:,:,:]
    )

# Signatures for tensor vector dot product
sig_vector_dot_tensor_3d = types.float32[:,:,:,:](
    types.float32[:,:,:,:], 
    types.float32[:,:,:,:,:]
    )
sig_vector_dot_tensor_2d = types.float32[:,:,:](
    types.float32[:,:,:], 
    types.float32[:,:,:,:]
    )

# Signatures for tensor magnitude calculations
sig_tensor_magnitude_3d = types.float32[:,:,:](
    types.float32[:,:,:,:,:]
    )
sig_tensor_magnitude_2d = types.float32[:,:](
    types.float32[:,:,:,:]
    )

# Signatures for tensor outer product
sig_outer_product_3d = types.float32[:,:,:,:,:](
    types.float32[:,:,:,:], 
    types.float32[:,:,:,:]
    )
sig_outer_product_2d = types.float32[:,:,:,:](
    types.float32[:,:,:], 
    types.float32[:,:,:]
    )

# Signatures for tensor transpose (single input argument)
sig_tensor_transpose_3d_32 = types.float32[:,:,:,:,:](
    types.float32[:,:,:,:,:]
    )
sig_tensor_transpose_3d_64 = types.float64[:,:,:,:,:](
    types.float64[:,:,:,:,:]
    )
