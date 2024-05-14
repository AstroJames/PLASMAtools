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
            args (_type_): 

        Returns:
            _type_: _description_
        
        """
        # two point stencil
        if self.order == 2:
            try:
                two_dr = 2. * self.L /scalar_field.shape[gradient_dir]
            except:
                return np.zeros_like(scalar_field)
            return ( np.roll(scalar_field, F, axis=gradient_dir) \
                   - np.roll(scalar_field, B, axis=gradient_dir) ) / two_dr

        # four point stencil
        elif self.order == 4:
            try:
                twelve_dr = 12. * self.L /scalar_field.shape[self.gradient_dir]
            except:
                return np.zeros_like(scalar_field)
            # df/dr = (-f(r+2dr) + 8f(r+dr) - 8f(r-dr) + f(r-2dr))/12dr
            return ( - np.roll(scalar_field,2*F,axis=gradient_dir)  \
                     + 8*np.roll(scalar_field,F,axis=gradient_dir)  \
                     - 8*np.roll(scalar_field,B,axis=gradient_dir)  \
                     + np.roll(scalar_field,2*B,axis=gradient_dir)) / twelve_dr

        # six point stencil
        elif self.order == 6:
            try:
                twelve_dr = 60. * self.L / scalar_field.shape[gradient_dir]
            except:
                return np.zeros_like(scalar_field)
            
            # df/dr = (-f(r+2dr) + 8f(r+dr) - 8f(r-dr) + f(r-2dr))/12dr
            return ( - np.roll(scalar_field,3*B,axis=gradient_dir)  \
                    + 9*np.roll(scalar_field,2*B,axis=gradient_dir) \
                    - 45*np.roll(scalar_field,B,axis=gradient_dir)  \
                    + 45*np.roll(scalar_field,F,axis=gradient_dir)  \
                    - 9*np.roll(scalar_field,2*F,axis=gradient_dir) \
                    + np.roll(scalar_field,3*F,axis=gradient_dir))  / twelve_dr