import numpy as np


class ScalarOperations:
    def __init__(self):
        pass

    def scalar_rms(self,
                    scalar_field : np.ndarray) -> np.ndarray:
        """
        Compute the root-mean-squared of a scalar field.
        
        Author: Neco Kriel
        
        Args:
            scalar_field (np.ndarray): N,N,N array of scalar field,
                                        where N is the number of grid points in each direction  

        Returns:
            scalar_RMS (np.ndarray): scalar RMS of the scalar field
        
        """
        return np.sqrt(np.mean(scalar_field**2))