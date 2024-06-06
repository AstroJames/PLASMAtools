import numpy as np
from scipy.ndimage import uniform_filter

class TensorOperations:
    """
    A class to perform operations on tensor fields.
    
    """
    def __init__(self):
        pass

    def tensor_magnitude(self,
                         tensor_field : np.ndarray) -> np.ndarray:
        """
        Compute the tensor magnitude of a tensor field.
        
        Author: James Beattie
        
        Args:
            tensor_field (np.ndarray) : (i,j),N,N,N array of tensor field, where 
                                        (i,j) are the tensor components and N is the number of grid
                                        points in each direction
        Returns:
            tensor_mag (np.ndarray) : N,N,N array of tensor magnitude of the tensor field
        
        """
        
        return np.sqrt(self.tensor_double_contraction(tensor_field,
                                                      tensor_field))
    

    def tensor_double_contraction(self,
                           tensor_field_0 : np.ndarray,
                           tensor_field_1 : np.ndarray) -> np.ndarray:
        """
        Compute the A_ijA_ij scalar field from a tensor field.
        
        Author: James Beattie
        
        Args:
            tensor_field_0 (np.ndarray) : (i,j),N,N,N array of tensor field, where 
                                            (i,j) are the tensor components and N is the number of grid
                                            points in each direction
            tensor_field_1 (np.ndarray) : (i,j),N,N,N array of tensor field, where 
                                            (i,j) are the tensor components and N is the number of grid 
                                            points in each direction

        Returns:
            A_ijB_ij = a_11 b_11 + a_12 b_11 + ... a_nn b_nn: the contraction scalar field.
        
        """
            
        return np.einsum('ij...,ij...->...',
                         tensor_field_0,
                         tensor_field_1)
        

    def vector_dot_tensor(self,
                          vector_field : np.ndarray,
                          tensor_field : np.ndarray) -> np.ndarray:
        """
        Compute the A_iA_j tensor field from a vector field.
        
        Author: James Beattie
        
        Args:
            vector (np.ndarray)         : i,N,N,N array of vector field, where 
                                            i, are the vector components and N is the number of grid 
                                            points in each direction
            tensor_field_1 (np.ndarray) : (i,j),N,N,N array of tensor field, where 
                                            (i,j) are the tensor components and N is the number of grid
                                            points in each direction

        Returns:
            A_iB_ij : a_1b_1j + a_2b_2j + ... contraction vector field.
        
        """
            
        return np.einsum('i...,ij...->j...',
                         vector_field,
                         tensor_field)


    def tensor_transpose(self,
                         tensor_field : np.ndarray) -> np.ndarray:
        """
        Compute the transpose of tensor field.
        
        Author: James Beattie
        
        Args:
            tensor_field (np.ndarray): M,M,N,N,N array of tensor field, where 
            M is the tensor component and N is the number of grid points in each 
            direction

        Returns:
            the transpose A_ji of the A_ij tensor field
        
        """
            
        return np.einsum('ij... -> ji...',
                         tensor_field)
        
        
    def tensor_outer_product(self,
                             vector_field_0 : np.ndarray,
                             vector_field_1 : np.ndarray) -> np.ndarray:
        """
        Compute the A_iB_j tensor field from a vector field.
        
        Author: James Beattie
        
        Args:
            vector_field_0 (np.ndarray): 3,N,N,N array of vector field, where 
            3 is the vector component and N is the number of grid points in each 
            direction
            vector_field_1 (np.ndarray): 3,N,N,N array of vector field, where 
            3 is the vector component and N is the number of grid points in each 
            direction

        Returns:
            A_i_B_j: the A_iB_j tensor field
        
        """
            
        return np.einsum('i...,j...->ij...',
                         vector_field_0,
                         vector_field_1)
        
        
    def smooth_gradient_tensor(self,
                               gradient_tensor : np.ndarray, 
                               smoothing_size : int          = 10) -> np.ndarray:
        """
        Smooth a gradient tensor field by averaging over adjacent cells.

        Args:
            gradient_tensor (np.ndarray)    : The gradient tensor to smooth (shape: 3,3,N,N,N).
            smoothing_size (int)            : The size of the smoothing window (default: 10).

        Returns:
            smoothed tensor field (np.ndarray) : The smoothed gradient tensor.
        """

        return uniform_filter(gradient_tensor, 
                            size  = smoothing_size, 
                            axes  = (-3, -2, -1), 
                            mode  = 'nearest')
