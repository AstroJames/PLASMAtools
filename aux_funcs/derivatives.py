import numpy as np

# indexes
X,Y,Z = 0,1,2

# shifts for derivatives
F = -1 # shift forwards
B = +1 # shift backwards

################################################################
## Derivative stencil class 
################################################################

class derivative:
    """
    Derivative stencil class.
    
    """
    
    def __init__(self,
                 order        : int = 2,
                 L            : float = 1.0) -> None:
        """
        Initialize the derivative stencil class.

        Author: James Beattie

        Args:
            order (int, optional):  the order of the stencil. Defaults to 2.
            L (float, optional):    the size of the domain. Defaults to 1.0.
        """
        
        self.order          = order
        self.L              = L
        
    def gradient(self,
                 scalar_field : np.ndarray,
                 gradient_dir : int) -> np.ndarray:
        """
        Compute the gradient of a scalar field in one direction
        using a two point stencil (second order method).
        
        Author: James Beattie & Neco Kriel
        
        Args:
            scalar_field (np.ndarray): the scalar field to compute the gradient of.
            gradient_dir (int):        the direction of the gradient.

        Returns:
            np.ndarray: the gradient of the scalar field in the specified direction.
        
        """
        # two point stencil
        if self.order == 2:
            try:
                dr = 2. * self.L /scalar_field.shape[gradient_dir]
            except:
                return np.zeros_like(scalar_field)
            return ( np.roll(scalar_field, F, axis=gradient_dir) \
                   - np.roll(scalar_field, B, axis=gradient_dir) ) / dr

        # four point stencil
        elif self.order == 4:
            try:
                dr = 12. * self.L /scalar_field.shape[self.gradient_dir]
            except:
                return np.zeros_like(scalar_field)
            # df/dr = (-f(r+2dr) + 8f(r+dr) - 8f(r-dr) + f(r-2dr))/12dr
            return ( - np.roll(scalar_field,2*F,axis=gradient_dir)  \
                     + 8*np.roll(scalar_field,F,axis=gradient_dir)  \
                     - 8*np.roll(scalar_field,B,axis=gradient_dir)  \
                     + np.roll(scalar_field,2*B,axis=gradient_dir)) / dr

        # six point stencil
        elif self.order == 6:
            try:
                dr = 60. * self.L / scalar_field.shape[gradient_dir]
            except:
                return np.zeros_like(scalar_field)
            
            # df/dr = (-f(r+2dr) + 8f(r+dr) - 8f(r-dr) + f(r-2dr))/12dr
            return ( - np.roll(scalar_field,3*B,axis=gradient_dir)  \
                    + 9*np.roll(scalar_field,2*B,axis=gradient_dir) \
                    - 45*np.roll(scalar_field,B,axis=gradient_dir)  \
                    + 45*np.roll(scalar_field,F,axis=gradient_dir)  \
                    - 9*np.roll(scalar_field,2*F,axis=gradient_dir) \
                    + np.roll(scalar_field,3*F,axis=gradient_dir))  / dr