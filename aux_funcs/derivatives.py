import numpy as np

# indexes
X,Y,Z = 0,1,2

# shifts for derivatives
F = -1 # shift forwards
B = +1 # shift backwards

################################################################
## Derivative stencil functions 
################################################################


def gradient_order2(scalar_field : np.ndarray, 
                    gradient_dir : int, 
                    L            : float = 1.0 ):
    """
    Compute the gradient of a scalar field in one direction
    using a two point stencil (second order method).
    
    Author: Neco Kriel & James Beattie
    
    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    
    """
    
    # 2dr
    try:
        two_dr = 2. * L /scalar_field.shape[gradient_dir]
    except:
        return np.zeros_like(scalar_field)
    
    return (
        np.roll(scalar_field, F, axis=gradient_dir) - np.roll(scalar_field, B, axis=gradient_dir) ) / two_dr
    
    
def gradient_order4(scalar_field : np.ndarray, 
                    gradient_dir : int, 
                    L            : float = 1.0 ):
    """
    Compute the gradient of a scalar field  in one direction
    using a five point stencil (fourth order method).
    
    Author: James Beattie
    
    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    
    """
    
    # 12dr
    try:
        twelve_dr = 12. * L /scalar_field.shape[gradient_dir]
    except:
        return np.zeros_like(scalar_field)
    
    # df/dr = (-f(r+2dr) + 8f(r+dr) - 8f(r-dr) + f(r-2dr))/12dr
    return ( - np.roll(scalar_field,2*F,axis=gradient_dir) \
             + 8*np.roll(scalar_field,F,axis=gradient_dir) \
             - 8*np.roll(scalar_field,B,axis=gradient_dir) \
             + np.roll(scalar_field,2*B,axis=gradient_dir)) / twelve_dr


def gradient_order6(scalar_field : np.ndarray, 
                    gradient_dir : int, 
                    L            : float = 1.0 ):
    """
    Compute the gradient of a scalar field  in one direction
    using a seven point stencil (sixth order method).
    
    Author: James Beattie
    
    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    
    """
    
    # 60dr
    try:
        twelve_dr = 60. * L /scalar_field.shape[gradient_dir]
    except:
        return np.zeros_like(scalar_field)
    
    # df/dr = (-f(r+2dr) + 8f(r+dr) - 8f(r-dr) + f(r-2dr))/12dr
    return ( - np.roll(scalar_field,3*B,axis=gradient_dir)   \
             + 9*np.roll(scalar_field,2*B,axis=gradient_dir) \
             - 45*np.roll(scalar_field,B,axis=gradient_dir)  \
             + 45*np.roll(scalar_field,F,axis=gradient_dir)  \
             - 9*np.roll(scalar_field,2*F,axis=gradient_dir) \
             + np.roll(scalar_field,3*F,axis=gradient_dir)) / twelve_dr