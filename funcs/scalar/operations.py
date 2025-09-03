import numpy as np
from .core_functions import *
from .constants import *

class ScalarOperations:
    """
    A class to perform operations on scalar fields.
    
    """
    
    def __init__(
        self) -> None:
        pass

    def scalar_rms(
        self,
        scalar_field : np.ndarray) -> np.ndarray:
        """
        Compute the root-mean-squared of a scalar field.
        Args:
            scalar_field (np.ndarray): N,N,N array of scalar field,
                                        where N is the number of grid points in each direction  
        Returns:
            scalar_RMS (np.ndarray): scalar RMS of the scalar field
        
        """
        return scalar_rms_np_core(scalar_field)