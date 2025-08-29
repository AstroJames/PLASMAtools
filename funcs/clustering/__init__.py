"""
PLASMAtools Clustering Module

High-performance friends-of-friends clustering algorithms optimized with Numba JIT compilation.
Supports 2D and 3D spatial clustering with various boundary conditions for plasma simulation data.

Author: James R. Beattie
"""

# Import main classes
from .operations import ClusteringOperations
from .morphology import ClusterMorphology

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

# Version info
__version__ = "1.0.0"
__author__ = "James R. Beattie"

# Define public API
__all__ = [
    'ClusteringOperations',
    'ClusterMorphology',
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