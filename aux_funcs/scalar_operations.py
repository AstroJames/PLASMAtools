import numpy as np

class ScalarOperations:
    """
    A class to perform operations on scalar fields.
    
    Author: James Beattie
    
    """
    
    def __init__(self) -> None:
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