"""
    Title: Derived Variable Functions
    Author: James R. Beattie
    Date: 09/07/2017
    Description: Functions for calculating derived variables in the read_flash classes
    
    Collaborators: Neco Kriel (all the curvature functions).

"""

## ###############################################################
## IMPORTS
## ###############################################################

import scipy.fft as fft
import numpy as np

## ###############################################################
## Derived Variable Functions
## ###############################################################

# indexes
X,Y,Z = 0,1,2

# shifts for derivatives
F = -1 # shift forwards
B = +1 # shift backwards

def helmholtz_decomposition(F: np.ndarray,
                            x: np.ndarray,
                            n_workers: int = 1):
    """
    Author: James Beattie (assumes periodic boundary conditions)
    """
    # F is a 4D array, with the last dimension being 3 (for the x, y, z components of the vector field)
    shape = F.shape[:-1]
    Fhat = fft.fftn(F, axes=(0, 1, 2),norm = 'forward',workers=n_workers)
    
    Fhat_irrot = np.zeros_like(Fhat, dtype=np.complex128)
    Fhat_solen = np.zeros_like(Fhat, dtype=np.complex128)
    norm       = np.zeros(shape, dtype=np.float64)
    
    # Compute wave numbers
    kx = fft.fftfreq(shape[X])* 2*np.pi * shape[X] / (x[-1] - x[0])
    ky = fft.fftfreq(shape[Y])* 2*np.pi * shape[Y] / (x[-1] - x[0])
    kz = fft.fftfreq(shape[Z])* 2*np.pi * shape[Z] / (x[-1] - x[0])
    kX, kY, kZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Avoid division by zero
    norm = kX**2 + kY**2 + kZ**2
    norm[0, 0, 0] = 1
    
    # Compute divergence and curl in Fourier space (note python doesn't seem to want to use i)
    divFhat = (kX * Fhat[..., X] + kY * Fhat[..., Y] + kZ * Fhat[..., Z])
    
    # Compute irrotational and solenoidal components in Fourier space
    Fhat_irrot = np.transpose(divFhat * np.array([kX, kY, kZ]) / norm[np.newaxis, ...],(1,2,3,0))
    Fhat_solen = Fhat - Fhat_irrot #curlFhat / norm[np.newaxis, ...]
    
    # Inverse Fourier transform to real space
    F_irrot = fft.ifftn(Fhat_irrot, axes=(X,Y,Z),workers=n_workers,norm = 'forward').real
    F_solen = fft.ifftn(Fhat_solen, axes=(X,Y,Z),workers=n_workers,norm = 'forward').real
    
    # Remove numerical noise
    threshold = 1e-16
    F_solen[np.abs(F_solen) < threshold] = 0
    F_irrot[np.abs(F_irrot) < threshold] = 0
    
    return F_irrot, F_solen


def vector_curl(field):
    """
    Compute the vector curl (assumes periodic boundary conditions) 
    using second order finite differences
    
    Author: James Beattie
    """
    
    # differentials
    two_dx = 2./field[0].shape[0]
    two_dy = 2./field[1].shape[0]
    two_dz = 2./field[2].shape[0]
    
    # x component of curl
    dFz_dy = (np.roll(field[Z],F,axis=Y) - np.roll(field[Z],B,axis=Y))/two_dy
    dFy_dz = (np.roll(field[Y],F,axis=Z) - np.roll(field[Y],B,axis=Z))/two_dz
    
    # y component of curl
    dFx_dz = (np.roll(field[X],F,axis=Z) - np.roll(field[X],B,axis=Z))/two_dz
    dFz_dx = (np.roll(field[Z],F,axis=X) - np.roll(field[Z],B,axis=X))/two_dx
    
    # z component of curl
    dFy_dx = (np.roll(field[Y],F,axis=X) - np.roll(field[Y],B,axis=X))/two_dx
    dFx_dy = (np.roll(field[X],F,axis=Y) - np.roll(field[X],B,axis=Y))/two_dy
    
    return np.array([dFz_dy - dFy_dz,
                     dFx_dz - dFz_dx,
                     dFy_dx - dFx_dy])
    
    
def scalar_laplacian(field):
    """
    Compute the scalar laplacian (assumes periodic boundary conditions)
    using second order finite differences
    
    Author: James Beattie
    """
        
    # differentials
    dx = 1./field[0].shape[0]
    dy = 1./field[1].shape[0]
    dz = 1./field[2].shape[0]
    
    d2Fx_dx2 = (np.roll(field[X],F,axis=X) - 2*field[X] + np.roll(field[X],B,axis=X))/dx**2
    d2Fy_dy2 = (np.roll(field[Y],F,axis=Y) - 2*field[Y] + np.roll(field[Y],B,axis=Y))/dy**2
    d2Fz_dz2 = (np.roll(field[Z],F,axis=Z) - 2*field[Z] + np.roll(field[Z],B,axis=Z))/dz**2
    
    return d2Fx_dx2 + d2Fy_dy2 + d2Fz_dz2
        
        
def vector_cross_product(vector1, vector2):
    """
    Compute the vector cross product of two vectors.
    
    Auxillary functions for computeTNBBasis
    
    Author: Neco Kriel
    """
    
    vector3 = np.array([
    vector1[Y] * vector2[Z] - vector1[Z] * vector2[Y],
    vector1[Z] * vector2[X] - vector1[X] * vector2[Y],
    vector1[X] * vector2[Y] - vector1[Y] * vector2[X]
    ])
    return vector3    


def vector_dot_product(vector1, vector2):
    """
    Compute the vector dot product of two vectors.
    
    Auxillary functions for computeTNBBasis
    
    Author: Neco Kriel
    """
    scalar = np.sum([
        v1_comp * v2_comp
        for v1_comp, v2_comp in zip(vector1, vector2)
    ], axis=0)
    return scalar


def field_magnitude(vector_field):
    """
    Compute the vector magnitude of a vector.
    
    Author: Neco Kriel
    """
    vector_field = np.array(vector_field)
    return np.sqrt(np.sum(vector_field**2, axis=0))


def gradient_2ocd(field, cell_width, gradient_dir):
    """
    Compute the gradient of a field in one direction.
    
    Auxillary functions for computeTNBBasis
    
    Author: Neco Kriel
    """
    F = -1 # shift forwards
    B = +1 # shift backwards
    return (
        np.roll(field, F, axis=gradient_dir) - np.roll(field, B, axis=gradient_dir)
    ) / (2*cell_width)


def field_RMS(scalar_field):
    """
    Compute the root-mean-squared of a field.
    
    Author: Neco Kriel
    """
    return np.sqrt(np.mean(scalar_field**2))


def field_gradient(scalar_field):
    """
    Compute the gradient of a scalar field.
    
    Author: Neco Kriel
    """
    ## format: (x, y, z)
    scalar_field = np.array(scalar_field)
    cell_width = 1 / scalar_field.shape[0]
    field_gradient = [
        gradient_2ocd(scalar_field, cell_width, gradient_dir)
        for gradient_dir in [0, 1, 2]
    ]
    return np.array(field_gradient)


def compute_TNB_basis(vector_field):
    """
    Compute the Fressnet frame of a vector field (TNB basis).
    
    Author: Neco Kriel
    """
    ## format: (component, x, y, z)
    vector_field = np.array(vector_field)
    field_magn = field_magnitude(vector_field)
    ## ---- COMPUTE TANGENT BASIS
    t_basis = vector_field / field_magn
    ## df_j/dx_i: (component-j, gradient-direction-i, x, y, z)
    gradient_tensor = np.array([
        field_gradient(field_component)
        for field_component in vector_field
    ])
    ## ---- COMPUTE NORMAL BASIS
    ## f_i df_j/dx_i
    n_basis_term1 = np.einsum("ixyz,jixyz->jxyz", vector_field, gradient_tensor)
    ## f_i f_j f_m df_m/dx_i
    n_basis_term2 = np.einsum("ixyz,jxyz,mxyz,mixyz->jxyz", vector_field, vector_field, vector_field, gradient_tensor)
    ## (f_i df_j/dx_i) / (f_k f_k) - (f_i f_j f_m df_m/dx_i) / (f_k f_k)^2
    n_basis = n_basis_term1 / field_magn**2 - n_basis_term2 / field_magn**4
    ## field curvature
    kappa = field_magnitude(n_basis)
    ## normal basis
    n_basis /= kappa
    ## ---- COMPUTE BINORMAL BASIS
    ## orthogonal to both t- and b-basis
    b_basis = vector_cross_product(t_basis, n_basis)
    return t_basis, n_basis, b_basis, kappa


## END OF LIBRARY