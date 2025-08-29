"""
Type signatures and constants for clustering operations.
Centralizes all Numba type definitions for friends-of-friends clustering.

Author: James R. Beattie
"""
from numba import types
import numpy as np

##############################################################################
# Global constants
##############################################################################

# Coordinate indices
X, Y, Z = 0, 1, 2

# Boundary condition types
PERIODIC = 0
NEUMANN = 1  
DIRICHLET = 2

# Default parameters
DEFAULT_LINKING_LENGTH = 0.2
DEFAULT_MIN_CLUSTER_SIZE = 1
HASH_FACTOR = 2.0  # spatial hash cell size factor relative to linking length

##############################################################################
# Type signatures for Numba functions
##############################################################################

# Distance calculations
sig_distance_2d_32 = types.float32(
    types.float32, types.float32,  # dx, dy
    types.float32, types.float32,  # box_size_x, box_size_y
    types.int32, types.int32       # bc_x, bc_y
)

sig_distance_2d_64 = types.float64(
    types.float64, types.float64,  # dx, dy
    types.float64, types.float64,  # box_size_x, box_size_y
    types.int32, types.int32       # bc_x, bc_y
)

sig_distance_3d_32 = types.float32(
    types.float32, types.float32, types.float32,  # dx, dy, dz
    types.float32, types.float32, types.float32,  # box_size_x, box_size_y, box_size_z
    types.int32, types.int32, types.int32         # bc_x, bc_y, bc_z
)

sig_distance_3d_64 = types.float64(
    types.float64, types.float64, types.float64,  # dx, dy, dz
    types.float64, types.float64, types.float64,  # box_size_x, box_size_y, box_size_z
    types.int32, types.int32, types.int32         # bc_x, bc_y, bc_z
)

# Spatial hashing
sig_hash_2d_32 = types.int64(
    types.float32, types.float32,  # x, y
    types.float32,                 # cell_size
    types.int64, types.int64       # nx_cells, ny_cells
)

sig_hash_2d_64 = types.int64(
    types.float64, types.float64,  # x, y
    types.float64,                 # cell_size
    types.int64, types.int64       # nx_cells, ny_cells
)

sig_hash_3d_32 = types.int64(
    types.float32, types.float32, types.float32,  # x, y, z
    types.float32,                                # cell_size
    types.int64, types.int64, types.int64         # nx_cells, ny_cells, nz_cells
)

sig_hash_3d_64 = types.int64(
    types.float64, types.float64, types.float64,  # x, y, z
    types.float64,                                # cell_size
    types.int64, types.int64, types.int64         # nx_cells, ny_cells, nz_cells
)

# Union-Find operations
sig_union_find_32 = types.int32(
    types.int32[:],  # parent array
    types.int32      # node index
)

sig_union_find_64 = types.int64(
    types.int64[:],  # parent array
    types.int64      # node index
)

sig_union_32 = types.void(
    types.int32[:],  # parent array
    types.int32,     # node1
    types.int32      # node2
)

sig_union_64 = types.void(
    types.int64[:],  # parent array
    types.int64,     # node1
    types.int64      # node2
)

# FOF main algorithm signatures
sig_fof_2d_32 = types.int32[:](
    types.float32[:, :],  # positions (N, 2)
    types.float32,        # linking_length
    types.float32[:],     # box_size (2,)
    types.int32[:],       # boundary_conditions (2,)
    types.int32           # min_cluster_size
)

sig_fof_2d_64 = types.int64[:](
    types.float64[:, :],  # positions (N, 2)
    types.float64,        # linking_length
    types.float64[:],     # box_size (2,)
    types.int32[:],       # boundary_conditions (2,)
    types.int64           # min_cluster_size
)

sig_fof_3d_32 = types.int32[:](
    types.float32[:, :],  # positions (N, 3)
    types.float32,        # linking_length
    types.float32[:],     # box_size (3,)
    types.int32[:],       # boundary_conditions (3,)
    types.int32           # min_cluster_size
)

sig_fof_3d_64 = types.int64[:](
    types.float64[:, :],  # positions (N, 3)
    types.float64,        # linking_length
    types.float64[:],     # box_size (3,)
    types.int32[:],       # boundary_conditions (3,)
    types.int64           # min_cluster_size
)

# Neighbor finding signatures
sig_find_neighbors_2d_32 = types.ListType(types.int32)(
    types.int32,          # particle index
    types.float32[:, :],  # positions (N, 2)
    types.float32,        # linking_length
    types.float32[:],     # box_size (2,)
    types.int32[:],       # boundary_conditions (2,)
)

sig_find_neighbors_2d_64 = types.ListType(types.int64)(
    types.int64,          # particle index
    types.float64[:, :],  # positions (N, 2)
    types.float64,        # linking_length
    types.float64[:],     # box_size (2,)
    types.int32[:],       # boundary_conditions (2,)
)

sig_find_neighbors_3d_32 = types.ListType(types.int32)(
    types.int32,          # particle index
    types.float32[:, :],  # positions (N, 3)
    types.float32,        # linking_length
    types.float32[:],     # box_size (3,)
    types.int32[:],       # boundary_conditions (3,)
)

sig_find_neighbors_3d_64 = types.ListType(types.int64)(
    types.int64,          # particle index
    types.float64[:, :],  # positions (N, 3)
    types.float64,        # linking_length
    types.float64[:],     # box_size (3,)
    types.int32[:],       # boundary_conditions (3,)
)