import numpy as np

# indexes
X,Y,Z = 0,1,2

class VectorOperations:
    def __init__(self,
                 num_of_dims    : int      = 3):
        
        # Number of dimensions
        self.num_of_dims = num_of_dims

    def vector_magnitude(self,
                         vector_field : np.ndarray) -> np.ndarray:
        """
        Compute the vector magnitude of a vector.
        
        Author: Neco Kriel
        
        Args:
            vector_field (np.ndarray): 3,N,N,N array of vector field,
                                        where 3 is the vector component and N is the number of grid
                                        points in each direction

        Returns:
            vector mag (np.ndarray): N,N,N array of vector cross product of the two vectors
        
        """
        vector_field = np.array(vector_field)
        
        return np.sqrt(self.vector_dot_product(vector_field,
                                                vector_field))
        
        
    def vector_dot_product(self,
                           vector_field_1 : np.ndarray,
                           vector_field_2 : np.ndarray):
        """
        Compute the vector dot product of two vectors.
        
        Author: Neco Kriel, James Beattie
        
        Args:
            vector_field_1 (np.ndarray): 3,N,N,N array of vector field,
                                    where 3 is the vector component and N is the number of grid
                                    points in each direction    
            vector_field_2 (np.ndarray): 3,N,N,N array of vector field,
                                    where 3 is the vector component and N is the number of grid
                                    points in each direction 

        Returns:
            vector dot product (np.ndarray): N,N,N array of vector cross product of the two vectors
        
        """
        
        return np.einsum("i...,i...->...",
                         vector_field_1,
                         vector_field_2)


    def vector_cross_product(self,
                             vector_field_1 : np.ndarray,
                             vector_field_2 : np.ndarray):
        """
        Compute the vector cross product of two vectors.
        
        Author: Neco Kriel & James Beattie

        Args:
            vector_field_1 (np.ndarray): 3,N,N,N array of vector field,
                                    where 3 is the vector component and N is the number of grid
                                    points in each direction
            vector_field_2 (np.ndarray): 3,N,N,N array of vector field,
                                    where 3 is the vector component and N is the number of grid
                                    points in each direction

        Returns:
            vector_field_3 (np.ndarray): 3,N,N,N array of vector cross product of the two vectors
        
        """
        
        if self.num_of_dims == 1:
            ValueError("Vector cross product is not defined for 1D.")
        elif self.num_of_dims == 2:
            return np.array([
                0,
                0,
                vector_field_1[X] * vector_field_2[Y] - vector_field_1[Y] * vector_field_2[X]]
                        )
        elif self.num_of_dims == 3:
            return np.array([
                        vector_field_1[Y] * vector_field_2[Z] - vector_field_1[Z] * vector_field_2[Y],
                        vector_field_1[Z] * vector_field_2[X] - vector_field_1[X] * vector_field_2[Z],
                        vector_field_1[X] * vector_field_2[Y] - vector_field_1[Y] * vector_field_2[X]]
                            )