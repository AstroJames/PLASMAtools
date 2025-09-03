"""
PLASMAtools Clustering

High-performance Friends-of-Friends (FOF) and grid-connected clustering
with Numba-optimized kernels. Supports 2D/3D data and multiple boundary
conditions.

Author: James R. Beattie
"""

# Import main classes
from .operations import ClusteringOperations
from .morphology import ClusterMorphology
from .constants import PERIODIC, NEUMANN, DIRICHLET

# Import core functions for advanced users
from .core_functions import (
    distance_2d,
    distance_3d,
    hash_position_2d,
    hash_position_3d,
    find_root,
    union_nodes,
    fof_2d,
    fof_3d,
)

__version__ = "1.0.1"
__author__ = "James R. Beattie"

# Define public API
__all__ = [
    'ClusteringOperations',
    'ClusterMorphology',
    'PERIODIC', 'NEUMANN', 'DIRICHLET',
    # Core functions for advanced use
    'distance_2d',
    'distance_3d',
    'hash_position_2d',
    'hash_position_3d',
    'find_root',
    'union_nodes',
    'fof_2d',
    'fof_3d',
]
