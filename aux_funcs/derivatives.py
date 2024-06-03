import numpy as np

# indexes
X, Y, Z = 0, 1, 2

# shifts for derivatives
F = -1  # shift forwards
B = +1  # shift backwards

################################################################
## Derivative class
################################################################

class Derivative:
    """
    Derivative class.
    
    This class provides methods for computing the first and second
    derivatives of a scalar field using finite differences.
    
    """
    
    def __init__(self, 
                 stencil : int = 2,
                 L       : float = 1.0) -> None:
        """
        Initialize the derivative class.

        Author: James Beattie

        Args:
            order (int, optional) : the size of the stencil. Defaults to 2 point.
            L (float, optional)   : the size of the domain. Defaults to 1.0.
        """
        
        # Stencil order
        if stencil not in [2, 4, 6, 8]:
            raise ValueError("Invalid stencil order")
        self.stencil = stencil
                
        # Number of ghost cells
        self.num_of_gcs = self.order // 2
        
        # Grid properties
        self.L = L
        
        
    def apply_boundary_conditions(self, 
                                  scalar_field          : np.ndarray,
                                  boundary_condition    : str,
                                  gradient_dir          : int) -> np.ndarray:
        """
        Apply boundary conditions to the scalar field.

        Author: James Beattie

        Args:
            scalar_field (np.ndarray): the scalar field to which boundary conditions are applied.
            boundary_condition (str): the type of boundary condition ('periodic', 'dirichlet', 'neumann').
            gradient_dir (int):        the direction of the gradient.

        Returns:
            np.ndarray: the scalar field with boundary conditions applied.
        """
        
        # Define pad width for ghost cells
        pad_width = [(0, 0)] * scalar_field.ndim
        pad_width[gradient_dir] = (self.num_of_gcs, self.num_of_gcs)
        
        # No changes needed for periodic boundaries
        if boundary_condition == 'periodic':
            return scalar_field  

        # For Dirichlet boundary conditions, set the boundary values to zero
        if boundary_condition == 'dirichlet':
            scalar_field = np.pad(scalar_field, 
                                  pad_width, 
                                  mode='constant', 
                                  constant_values=0)

        # For Neumann boundary conditions, replicate the edge values, 
        # setting derivative to zero
        if boundary_condition == 'neumann':
            scalar_field = np.pad(scalar_field, 
                                  pad_width, 
                                  mode='edge')

        return scalar_field
    
    
    def remove_ghost_cells(self, 
                           scalar_field: np.ndarray,
                           gradient_dir: int) -> np.ndarray:
        """
        Remove ghost cells from the scalar field.

        Author: James Beattie

        Args:
            scalar_field (np.ndarray): the scalar field with ghost cells.
            gradient_dir (int):        the direction of the gradient.

        Returns:
            np.ndarray: the scalar field with ghost cells removed.
        """
        
        slices               = [slice(None)] * scalar_field.ndim
        slices[gradient_dir] = slice(self.num_of_gcs, -self.num_of_gcs)
        
        return scalar_field[tuple(slices)]
    
    
    def compute_dr(self, shape, gradient_dir):
        """
        Compute the grid spacing `dr`.
        
        Author: James Beattie
        
        Args:
            shape (tuple): the shape of the scalar field.
            gradient_dir (int): the direction of the gradient.
            
        Returns:
            float: the grid spacing `dr`.
        """
        return self.L / (shape[gradient_dir] - 2 * self.num_of_gcs)


    def gradient(self,
                 scalar_field       : np.ndarray,
                 gradient_dir       : int,
                 derivative_order   : int = 1,
                 boundary_condition : str = "periodic") -> np.ndarray:
        """
        Compute the first derivative of a scalar field in one direction
        using finite differences.
        
        Author: James Beattie & Neco Kriel
        
        Args:
            scalar_field (np.ndarray)           : the scalar field to compute the gradient of.
            gradient_dir (int)                  : the direction of the gradient.
            derivative_order (int, optional)    : the order of the derivative. Defaults to 1.
                                                    (1 for first derivative, 2 for second derivative)
            boundary_condition (str, optional)  : the type of boundary condition 
                                                    ('periodic', 'dirichlet', 'neumann'). D
                                                    efaults to 'periodic'.

        Returns:
            np.ndarray: the gradient of the scalar field in the specified direction.
        
        """
        
        # Check for valid boundary conditions
        if boundary_condition not in ['periodic', 'dirichlet', 'neumann']:
            raise ValueError("Invalid boundary condition")
        # Check for valid gradient direction
        if gradient_dir < 0 or gradient_dir >= scalar_field.ndim:
            raise ValueError("Invalid gradient direction")
        # Check for valid derivative order
        if derivative_order not in [1, 2]:
            raise ValueError("Invalid derivative order")
        
        # Apply boundary conditions
        scalar_field = self.apply_boundary_conditions(scalar_field, 
                                                      boundary_condition, 
                                                      gradient_dir)
        
        # Compute the grid spacing `dr`
        dr = self.compute_dr(scalar_field.shape, gradient_dir)
        
        # Compute the first order derivative
        if derivative_order == 1:
            # two point stencil
            if self.stencil == 2:
                # df/dr = (f(r+dr) - f(r-dr))/2dr
                # and remove the ghost cells in 3D and return the gradient
                return self.remove_ghost_cells(( 
                        np.roll(scalar_field,   F, axis=gradient_dir) \
                        - np.roll(scalar_field, B, axis=gradient_dir)) / (2. * dr), 
                        gradient_dir)
            # four point stencil
            elif self.stencil == 4:
                # df/dr = (-f(r+2dr) + 8f(r+dr) - 8f(r-dr) + f(r-2dr))/12dr
                # and remove the ghost cells in 3D and return the gradient
                return self.remove_ghost_cells(( 
                        - np.roll(scalar_field,     2*F, axis=gradient_dir)  \
                        + 8*np.roll(scalar_field,     F, axis=gradient_dir)  \
                        - 8*np.roll(scalar_field,     B, axis=gradient_dir)  \
                        + np.roll(scalar_field,     2*B, axis=gradient_dir)) / (12. * dr), 
                        gradient_dir)
            # six point stencil
            elif self.stencil == 6:
                # df/dr = (-f(r+3dr) + 9f(r+2dr) - 45f(r+dr) + 45f(r-dr) - 9f(r-2dr) + f(r-3dr))/60dr
                # and remove the ghost cells in 3D and return the gradient
                return self.remove_ghost_cells(( 
                        - np.roll(scalar_field,     3*B, axis=gradient_dir)  \
                        + 9*np.roll(scalar_field,   2*B, axis=gradient_dir) \
                        - 45*np.roll(scalar_field,    B, axis=gradient_dir)  \
                        + 45*np.roll(scalar_field,    F, axis=gradient_dir)  \
                        - 9*np.roll(scalar_field,   2*F, axis=gradient_dir) \
                        + np.roll(scalar_field,     3*F, axis=gradient_dir)) / (60. * dr), 
                        gradient_dir)
            # eight point stencil
            elif self.stencil == 8:
                # df/dr = (-f(r+4dr) + 12f(r+3dr) - 66f(r+2dr) + 192f(r+dr) - 192f(r-dr) + 66f(r-2dr) - 12f(r-3dr) + f(r-4dr))/280dr
                # and remove the ghost cells in 3D and return the gradient
                return self.remove_ghost_cells((
                            -np.roll(scalar_field,      4*F, axis=gradient_dir)
                            + 4 * np.roll(scalar_field, 3*F, axis=gradient_dir)
                            - 5 * np.roll(scalar_field, 2*F, axis=gradient_dir)
                            + 16 * np.roll(scalar_field,  F, axis=gradient_dir)
                            - 16 * np.roll(scalar_field,  B, axis=gradient_dir)
                            + 5 * np.roll(scalar_field, 2*B, axis=gradient_dir)
                            - 4 * np.roll(scalar_field, 3*B, axis=gradient_dir)
                            + np.roll(scalar_field,     4*B, axis=gradient_dir)) / (280. * dr), 
                            gradient_dir)

        if derivative_order ==2:
            # two point stencil
            if self.stencil == 2:
                return self.remove_ghost_cells((
                    np.roll(scalar_field,   F, axis=gradient_dir) 
                    - 2 * scalar_field 
                    + np.roll(scalar_field, B, axis=gradient_dir)) / dr**2, 
                    gradient_dir)
            # four point stencil
            elif self.stencil == 4:
                return self.remove_ghost_cells((
                    - np.roll(scalar_field,    2*F, axis=gradient_dir)
                    + 16 * np.roll(scalar_field, F, axis=gradient_dir)
                    - 30 * scalar_field
                    + 16 * np.roll(scalar_field, B, axis=gradient_dir)
                    - np.roll(scalar_field,    2*B, axis=gradient_dir)) / (12. * dr**2), 
                    gradient_dir)
            # six point stencil
            elif self.stencil == 6:
                return self.remove_ghost_cells((
                    2 * np.roll(scalar_field, 3*F,  axis=gradient_dir)
                - 27 * np.roll(scalar_field,  2*F,  axis=gradient_dir)
                + 270 * np.roll(scalar_field,   F,  axis=gradient_dir)
                - 490 * scalar_field
                + 270 * np.roll(scalar_field,   B,  axis=gradient_dir)
                - 27 * np.roll(scalar_field,  2*B,  axis=gradient_dir)
                + 2 * np.roll(scalar_field,   3*B,  axis=gradient_dir)) / (180 * dr**2),
                gradient_dir) 
            # eight point stencil
            elif self.stencil == 8:
                return self.remove_ghost_cells((
                    - 9 * np.roll(scalar_field,     4*F, axis=gradient_dir)
                    + 128 * np.roll(scalar_field,   3*F, axis=gradient_dir)
                    - 1008 * np.roll(scalar_field,  2*F, axis=gradient_dir)
                    + 8064 * np.roll(scalar_field,    F, axis=gradient_dir)
                    - 14350 * scalar_field
                    + 8064 * np.roll(scalar_field,    B, axis=gradient_dir)
                    - 1008 * np.roll(scalar_field,  2*B, axis=gradient_dir)
                    + 128 * np.roll(scalar_field,   3*B, axis=gradient_dir)
                    - 9 * np.roll(scalar_field,     4*B, axis=gradient_dir)) / (5040. * dr**2),
                    gradient_dir)