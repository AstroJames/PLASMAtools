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