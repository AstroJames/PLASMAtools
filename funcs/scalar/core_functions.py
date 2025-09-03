import numpy as np
from .constants import *

##########################################################################################
# Core numpy functions for vector operations
##########################################################################################


def scalar_rms_np_core(
    scalar_field: np.ndarray) -> np.ndarray:
    """
    Compute the root-mean-squared of a scalar field.
    
    Args:
        scalar_field (np.ndarray): N,N,N array of scalar field,
                                   where N is the number of grid points in each direction  

    Returns:
        scalar_RMS (np.ndarray): scalar RMS of the scalar field
    """
    return np.sqrt(np.mean(scalar_field**2))