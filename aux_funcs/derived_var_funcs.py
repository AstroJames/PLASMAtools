"""
    Title: Derived Variable Functions
    Author: James R. Beattie
    Date: 09/07/2017
    Description: Functions for calculating derived variables in the read_flash classes
    
    Collaborators: Neco Kriel (curvature function).

"""

## ###############################################################
## IMPORTS
## ###############################################################

# python dependencies
import scipy.fft as fft
import numpy as np
from multiprocessing import Pool, shared_memory
from scipy.ndimage import uniform_filter

# import derivative stencils
from .derivatives import derivative

## ###############################################################
## Derived Variable Functions
## ###############################################################

# indexes
X,Y,Z = 0,1,2

# shifts for derivatives
F = -1 # shift forwards
B = +1 # shift backwards

def vector_potential(vector_field   : np.ndarray,
                     debug          : bool = False ) -> np.ndarray:
    """
    Create the underlying vector potential, a, of a vector field. For a magnetic field,
    assuming a Coulomb Gauage (div(a) = 0), this is the vector potential that satisfies the equation:
    
    \nabla x b = \nabla x \nabla x a = \nabla (\nabla \cdot a) -\nabla^2 a,

    \nabla \cdot a = 0,
    
    \nabla^2 a = -\nabla x b,
    
    where b is the magnetic field. In Fourier space:
    
    - k^2 \hat{a} = -i k \times \hat{b},
    
    \hat{a} = i \frac{k \times \hat{b}}{k^2},
    
    where k is the wavevector and \hat{a} is the Fourier transform of the vector potential, 
    \hat{b} is the Fourier transform of the magnetic field, and i is the imaginary i = \sqrt{-1}.
    
    Hence a can be found by taking the inverse Fourier transform of \hat{a}.
    
    
    Author: James Beattie

    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    
    """
    
    # Take FFT of vector field
    vector_field_FFT = fft.fftn(vector_field,
                                norm='forward',
                                axes=(1,2,3))

    # Assuming a cubic domain    
    N = vector_field.shape[1]  
    
    # The physical size of the domain
    L = 1.0
    
    # Space holder for the reconstructed magnetic field
    b_recon = 0.0
    
    # wave vectors
    kx = 2 * np.pi * fft.fftfreq(N, d=L/N) / L
    ky = 2 * np.pi * fft.fftfreq(N, d=L/N) / L
    kz = 2 * np.pi * fft.fftfreq(N, d=L/N) / L
    
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k = np.array([kx,ky,kz]) # This will be of shape (3, N, N, N)

    # Normalize k to get the unit wavevector
    k_norm = np.tile(np.linalg.norm(k, axis=0, keepdims=True), (3, 1, 1, 1)) # This will be of shape (1, N, N, N)

    # Replace zeros in k_norm with np.inf to avoid division by zero
    k_norm[k_norm == 0] = np.inf
    
    # Take the cross product of k and the vector field
    a_hat = 1j * vector_cross_product(k, vector_field_FFT) / k_norm**2
    
    # Take the inverse FFT to get the vector potential
    a = np.real(fft.ifftn(a_hat, 
                axes=(1, 2, 3),
                norm="forward"))
    
    # Take the curl of the vector potential to get the reconstructed magnetic field
    # for debuging
    if debug:
        # have to at least a fourth order derivative here 
        # to get a good reconstruction
        b_recon = vector_curl(a,order=4)  
    
    return a, b_recon


def magnetic_helicity(magnetic_vector_field : np.ndarray ) -> np.ndarray:
    """
    Compute the magnetic helicity of a vector field.
    
    Author: James Beattie

    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    """
    
    # compute the vector potential
    a, _ = vector_potential(magnetic_vector_field)
    
    # compute the magnetic helicity    
    return vector_dot_product(a,magnetic_vector_field)


def kinetic_helicity(velocity_vector_field : np.ndarray ) -> np.ndarray:
    """
    Compute the kinetic helicity of a vector field.
    
    Author: James Beattie
    
    Args:
        args (_type_):
        
    Returns:
        _type_: _description_
    
    """
    
    # compute the vorticity
    omega = vector_curl(velocity_vector_field)
    
    # compute the kinetic helicity
    return vector_dot_product(omega,velocity_vector_field)
    
    
def current_helicity(magnetic_vector_field : np.ndarray ) -> np.ndarray:
    """
    Compute the kinetic helicity of a vector field.
    
    Author: James Beattie
    
    Args:
        args (_type_):
        
    Returns:
        _type_: _description_
    
    """

    # compute the vorticity
    current = ( 1 / (4*np.pi) ) * vector_curl(magnetic_vector_field)
    
    # compute the kinetic helicity
    return vector_dot_product(current,magnetic_vector_field)
    

def gradient_tensor(vector_field    : np.ndarray,
                    order           : int = 2,
                    L               : float = 1.0) -> np.ndarray:
    """
    Compute the gradient tensor of a vector field 
    using finite differences.
    
    Author: James Beattie

    Args:
        vector_field (np.ndarray): 3,N,N,N array of vector field, 
        where 3 is the vector component and N is the number of grid 
        points in each direction
        order (int): order of finite difference used to compute the
        gradient tensor. Options are 2, 4 and 6. Default is 4.

    Returns:
        gradient_tensor: the gradient tensor of the vector field, 
        \partial_i f_j
    
    """
    
    # test if the vector field is 2D or 3D
    if len(vector_field[0].shape) == 2:
        two_D   = True 
        coords  = [X,Y]
    else:
        three_D = False
        coords  = [X,Y,Z]
    
    # set order and domain size for derivative
    d = derivative(order=order, L=L)
    
    # create the gradient tensor in 2D
    vel_grad = np.array([
        [d.gradient(vector_field[X],
                    gradient_dir=direction) for direction in coords],
        [d.gradient(vector_field[Y],
                    gradient_dir=direction) for direction in coords]])
    
    # and append 3d components if necessary
    if three_D:
        vel_grad = np.vstack([vel_grad,
                              [d.gradient(vector_field[Z],
                                          gradient_dir=direction) for direction in coords]])
    
    return np.einsum("ij...->ji...",vel_grad)
    
    
def smooth_gradient_tensor(gradient_tensor, smoothing_size=10) -> np.ndarray:
    """
    Smooth a gradient tensor field by averaging over adjacent cells.

    Args:
        gradient_tensor (np.ndarray): The gradient tensor to smooth (shape: 3,3,N,N,N).
        smoothing_size (int): The size of the smoothing window (default: 10).

    Returns:
        np.ndarray: The smoothed gradient tensor.
    """

    return uniform_filter(gradient_tensor, size=smoothing_size, axes=(-3, -2, -1), mode='nearest')
    

def orthogonal_tensor_decomposition(tensor_field : np.ndarray ) -> np.ndarray:
    """
    Compute the symmetric, anti-symmetric and bulk components of a tensor field.
    
    Author: James Beattie

    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    """
    
    # transpose
    tensor_transpose = np.einsum("ij... -> ji...",tensor_field)
    
    # bulk component
    tensor_bulk = (1./3.) * np.einsum('...,ij...->ij...',
                                       np.einsum("ii...",tensor_field),
                                       np.identity(3))
    
    # symmetric component
    tensor_sym = 0.5 * (tensor_field + tensor_transpose) - tensor_bulk
    
    # anti-symmetric component
    tensor_anti = 0.5 * (tensor_field - tensor_transpose)
    
    return tensor_sym, tensor_anti, tensor_bulk


def hermitian_eigenvalues(matrix : np.ndarray) -> np.ndarray:
    """
    Compute the eigenvalues and eigenvectors of a 
    hermitian tensor based on analyical relations in: https://hal.science/hal-01501221/document
    
    Author: James Beattie
    
    """
    
    def phi(x_2, x_1):
        if x_2 > 0:
            return np.arctan( np.sqrt( 4*x_1**3 - x_2**2 ) / x_2 )
        elif x_2 == 0.0:
            return np.pi/2
        elif x_2 < 0:
            return np.arctan( np.sqrt( 4*x_1**3 - x_2**2 ) / x_2 ) + np.pi
        
    # Read the matrix into variables
    a = matrix[0,0]; b = matrix[1,1]; c = matrix[2,2]
    d = matrix[1,0]; f = matrix[2,0]; e = matrix[2,1]
    d_star = matrix[0,1]; f_star = matrix[0,2]; e_star = matrix[1,2]
    
    # x_1 and x_2 coefficients for eigen values
    x_1 = a**2 + b**2 + c**2 - a*d - a*c - b*c + 3 * ( np.abs(d)**2 + np.abs(f)**2 + np.abs(e)**2 )
    x_2 = -( 2*a - b - c ) * ( 2*b - a - c ) * ( 2*c - a - b ) +\
            9 * (  (2*c - a - b)*np.abs(d)**2 + ( 2*b - a - c )*np.abs(f)**2 + ( 2*a - b - c )*np.abs(e)**2  ) -\
            54 * np.real(d_star*e_star*f_star)
                        
    # eigen values
    lambda_1 = ( a + b + c - 2 * np.sqrt(x_1) * np.cos( phi(x_2, x_1)/3.0 ) ) / 3.0
    lambda_2 = ( a + b + c + 2 * np.sqrt(x_1) * np.cos( (phi(x_2, x_1) - np.pi ) /3.0 ) ) / 3.0
    lambda_3 = ( a + b + c + 2 * np.sqrt(x_1) * np.cos( (phi(x_2, x_1) + np.pi ) /3.0 ) ) / 3.0        
            
    # m coefficients for eigen vectors
    m_1 = ( d * ( c - lambda_1 ) - e_star * f) / ( f * ( b - lambda_1) - d*e )
    m_2 = ( d * ( c - lambda_2 ) - e_star * f) / ( f * ( b - lambda_2) - d*e )
    m_3 = ( d * ( c - lambda_3 ) - e_star * f) / ( f * ( b - lambda_3) - d*e )
    
    # eigen vectors
    zeta_1 = np.array([( lambda_1 - c - e * m_1 )/f, m_1 , 1])
    zeta_2 = np.array([( lambda_2 - c - e * m_2 )/f, m_2 , 1])
    zeta_3 = np.array([( lambda_3 - c - e * m_3 )/f, m_3 , 1])
    
    return np.array([lambda_1, lambda_2, lambda_3]), zeta_1, zeta_2, zeta_3


def eigs_stretch_tensor(vector_field : np.ndarray,
                        tensor_field : np.ndarray,
                        n_processes  : int = 4,
                        parallel     : bool = False) -> np.ndarray:
    """
    Compute the stretch tensor of a tensor field along a given
    vector field (usually the magnetic field).
    
    I am assuming the tensor field is a symmetric tensor field
    to minimise the memory useage for the eigen value calculation.
    
    I am using a TNB basis for the local coordinate system of the vector
    field, which includes the curvature of the underlying field lines.
    
    If vector_field is the magnetic field, and tensor field is the velocity
    gradient tensor, this computes the local stretching tensor of the velocity, 
    through the eigen values of the velocity gradients in the coordinate system
    of the magnetic field. The eigen values are stored as a 3D array, ordered by
    the size of the eigen values. 
    
    (1) stretching eigen value (smallest) 
    (2) the null eigen value and 
    (3) the stretching eigen value (largest) 
    
    Author: James Beattie
    
    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    
    """
        
    # the number of grid elements
    N = vector_field.shape[1]       # assumes a cubic domain
    
    # symmetric component
    tensor_sym, _, _ = orthogonal_tensor_decomposition(tensor_field)
    
    # Compute TNB basis of vector field
    t_basis, n_basis, b_basis, _ = compute_TNB_basis(vector_field)
    X   = np.array([t_basis,n_basis,b_basis])
    X_T = np.einsum("ij... -> ji ...",X)
    
    # Put stretching tensor into the TNB basis
    trans_tensor_sym = np.einsum('ab...,bc...,cd... -> ad...', 
                                 X, 
                                 tensor_sym, 
                                 X_T)
    
    # Compute eigenvalues of the stretching tensor
    if parallel:
        print(f"stretch_tensor: computing eigen values of local " + 
            f"stretching tensor with {n_processes} processes")
        # Create shared memory for trans_tensor_sym
        shm = shared_memory.SharedMemory(create=True, size=trans_tensor_sym.nbytes)
        trans_tensor_sym_shared = np.ndarray(trans_tensor_sym.shape, dtype=trans_tensor_sym.dtype, buffer=shm.buf)
        np.copyto(trans_tensor_sym_shared, trans_tensor_sym)  # Copy data to shared memory

        # Assuming trans_tensor_sym has shape (3, 3, N, N, N)
        N = trans_tensor_sym.shape[2]
        indices = [(i, j, k) for i in range(N) for j in range(N) for k in range(N)]
        
        # Initialize an array to store the eigenvalues
        eigenvalues = np.zeros((3, N, N, N))

        # Prepare arguments for multiprocessing, including shared memory details
        args = [(index, trans_tensor_sym.shape, trans_tensor_sym.dtype, shm.name) for index in indices]

        # Use multiprocessing to compute eigenvalues in parallel
        with Pool(processes=n_processes) as pool:
            results = pool.map(compute_eigenvalues, args)

        # Store the results in the eigenvalues array
        for index, eigvals in results:
            eigenvalues[(slice(None),) + index] = np.sort(eigvals)

        # Clean up shared memory
        shm.close()
        shm.unlink()
    else:
        sorted_eigenvalues = np.zeros((3, N, N, N))
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    # Extract the 3x3 matrix for this position
                    matrix = trans_tensor_sym[:, :, x, y, z]

                    # Compute and sort the eigenvalues
                    eigenvalues = np.linalg.eigvalsh(matrix)
                    sorted_eigenvalues[:, x, y, z] = np.sort(eigenvalues)
    
    return sorted_eigenvalues


def tensor_outer_product(vector_field_0 : np.ndarray,
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
        
    return np.einsum('i...,j...->ij...',vector_field_0,vector_field_1)


def tensor_contraction(tensor_field_0 : np.ndarray,
                       tensor_field_1 : np.ndarray) -> np.ndarray:
    """
    Compute the A_iA_j tensor field from a vector field.
    
    Author: James Beattie
    
    Args:
        tensor_field_0 (np.ndarray): (i,j),N,N,N array of tensor field, where 
        (i,j) are the tensor components and N is the number of grid points in each 
        direction
        tensor_field_1 (np.ndarray): (i,j),N,N,N array of tensor field, where 
        (i,j) are the tensor components and N is the number of grid points in each 
        direction

    Returns:
        A_i_A_jB_i_B_j : the A_i_A_jB_i_B_j contraction scalar field.
    
    """
        
    return np.einsum('ij...,ij...->...',tensor_field_0, tensor_field_1)


def helmholtz_decomposition(vector_field : np.ndarray) -> np.ndarray:
    """
    Compute the irrotational and solenoidal components of a vector field.
    
    Author: James Beattie (assumes periodic boundary conditions)
    
    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    
    """
    # F is a 4D array, with the last dimension being 3 (for the x, y, z components of the vector field)
    # TODO: change the shape of F to be (3, N, N, N) instead of (N, N, N, 3) (consistent with other functions)
    
    shape = vector_field.shape[:-1]
    x     = np.linspace(-0.5,0.5,vector_field.shape[0]) # assuming a domian of [-L/2, L/2]
    
    # Fourier transform to Fourier space    
    Fhat = fft.fftn(vector_field, axes=(0, 1, 2),norm = 'forward')
    
    Fhat_irrot = np.zeros_like(Fhat, dtype=np.complex128)
    Fhat_solen = np.zeros_like(Fhat, dtype=np.complex128)
    norm       = np.zeros(shape, dtype=np.float64)
    
    # Compute wave numbers
    kx = 2*np.pi * np.fft.fftfreq(shape[X]) * shape[X] / (x[-1] - x[0])
    ky = 2*np.pi * np.fft.fftfreq(shape[Y]) * shape[Y] / (x[-1] - x[0])
    kz = 2*np.pi * np.fft.fftfreq(shape[Z]) * shape[Z] / (x[-1] - x[0])
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
    F_irrot = fft.ifftn(Fhat_irrot, axes=(X,Y,Z),norm = 'forward').real
    F_solen = fft.ifftn(Fhat_solen, axes=(X,Y,Z),norm = 'forward').real
    
    # Remove numerical noise
    # threshold = 1e-16
    # F_solen[np.abs(F_solen) < threshold] = 0
    # F_irrot[np.abs(F_irrot) < threshold] = 0
    
    return F_irrot, F_solen


def vector_curl(vector_field    : np.ndarray,
                order           : int = 2,
                L               : float = 1.0) -> np.ndarray:
    """
    Compute the vector curl (assumes periodic boundary conditions) 
    using either second or fourth order finite differences
    
    Author: James Beattie
    
    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    
    """
    
    # set order and domain size for derivative
    d = derivative(order=order, L=L)
        
    # x component of curl
    dFz_dy = d.gradient(vector_field[Z],gradient_dir=Y)
    dFy_dz = d.gradient(vector_field[Y],gradient_dir=Z)
    
    # y component of curl
    dFx_dz = d.gradient(vector_field[X],gradient_dir=Z)
    dFz_dx = d.gradient(vector_field[Z],gradient_dir=X)
    
    # z component of curl
    dFy_dx = d.gradient(vector_field[Y],gradient_dir=X)
    dFx_dy = d.gradient(vector_field[X],gradient_dir=Y)
        
    return np.array([dFz_dy - dFy_dz,
                     dFx_dz - dFz_dx,
                     dFy_dx - dFx_dy])
    

def vector_divergence(vector_field  : np.ndarray,
                        order       : int = 2,
                        L           : float = 1.0) -> np.ndarray:
    """
    Compute the vector divergence (assumes periodic boundary conditions)
    using either second or fourth order finite differences
    
    Author: James Beattie
    
    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    
    """
    
    # set order and domain size for derivative
    d = derivative(order=order, L=L)
    
    dFx_dx = d.gradient(vector_field[X],gradient_dir=X)
    dFy_dy = d.gradient(vector_field[Y],gradient_dir=Y)
    dFz_dz = d.gradient(vector_field[Z],gradient_dir=Z)
    
    return dFx_dx + dFy_dy + dFz_dz
    
    
def scalar_laplacian(scalar_field: np.ndarray) -> np.ndarray:
    """
    Compute the scalar laplacian (assumes periodic boundary conditions)
    using second order finite differences
    
    Author: James Beattie
    
    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    
    """
        
    # differentials
    dx = 1./scalar_field[X].shape[0]
    dy = 1./scalar_field[Y].shape[0]
    dz = 1./scalar_field[Z].shape[0]
    
    d2Fx_dx2 = (np.roll(scalar_field,F,axis=X) - 2*scalar_field + np.roll(scalar_field,B,axis=X))/dx**2
    d2Fy_dy2 = (np.roll(scalar_field,F,axis=Y) - 2*scalar_field + np.roll(scalar_field,B,axis=Y))/dy**2
    d2Fz_dz2 = (np.roll(scalar_field,F,axis=Z) - 2*scalar_field + np.roll(scalar_field,B,axis=Z))/dz**2
    
    return d2Fx_dx2 + d2Fy_dy2 + d2Fz_dz2
        
        
def vector_cross_product(vector1 : np.ndarray,
                         vector2 : np.ndarray):
    """
    Compute the vector cross product of two vectors.
    
    Auxillary functions for computeTNBBasis
    
    Author: Neco Kriel

    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    
    """
    
    vector3 = np.array([
    vector1[Y] * vector2[Z] - vector1[Z] * vector2[Y],
    vector1[Z] * vector2[X] - vector1[X] * vector2[Z],
    vector1[X] * vector2[Y] - vector1[Y] * vector2[X]
    ])
    return vector3    


def vector_dot_product(vector1 : np.ndarray,
                       vector2 : np.ndarray):
    """
    Compute the vector dot product of two vectors.
    
    Auxillary functions for computeTNBBasis
    
    Author: Neco Kriel
    
    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    
    """
    
    return np.einsum("i...,i...->...",vector1,vector2)


def field_magnitude(vector_field : np.ndarray) -> np.ndarray:
    """
    Compute the vector magnitude of a vector.
    
    Author: Neco Kriel
    
    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    
    """
    vector_field = np.array(vector_field)
    return np.sqrt(np.einsum("i...,i...->...",vector_field,vector_field))


def field_RMS(scalar_field : np.ndarray) -> np.ndarray:
    """
    Compute the root-mean-squared of a field.
    
    Author: Neco Kriel
    
    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    
    """
    return np.sqrt(np.mean(scalar_field**2))


def field_gradient(scalar_field : np.ndarray,
                   order        : int = 2,
                   L            : float = 1.0) -> np.ndarray:
    """
    Compute the gradient of a scalar field.
    
    Author: Neco Kriel & James Beattie
    
    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    
    """
    
    # set order and domain size for derivative
    d = derivative(order=order, L=L)
    
    ## format: (x, y, z)
    scalar_field = np.array(scalar_field)
    field_gradient = [
        d.gradient(scalar_field, gradient_dir)
        for gradient_dir in [X, Y, Z]
    ]
    return np.array(field_gradient)


def compute_TNB_basis(vector_field : np.ndarray) -> np.ndarray:
    """
    Compute the Fressnet frame of a vector field (TNB basis).
    
    Author: Neco Kriel & James Beattie
    
    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    
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
    n_basis_term1 = np.einsum("i...,ji...->j...", 
                              vector_field, 
                              gradient_tensor)
    ## f_i f_j f_m df_m/dx_i
    n_basis_term2 = np.einsum("i...,j...,m...,mi...->j...", 
                              vector_field, 
                              vector_field, 
                              vector_field, 
                              gradient_tensor)
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


def TNB_coordinate_transformation(vector_field : np.ndarray) -> np.ndarray:
    """
    Transform a vector field into the TNB coordinate system.
    
    Author: James Beattie
    
    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    
    """
    
    # compute the TNB basis
    t_basis, n_basis, b_basis, _ = compute_TNB_basis(vector_field)
    
    # transform vector field to TNB basis
    vector_field_TNB = np.array([
        vector_dot_product(vector_field, t_basis),
        vector_dot_product(vector_field, n_basis),
        vector_dot_product(vector_field, b_basis)
    ])
    
    return vector_field_TNB


def TNB_jacobian_stability_analysis(vector_field    : np.ndarray,
                                    traceless       : bool = True ) -> np.ndarray:
    """
    Compute the trace, determinant and eigenvalues of the Jacobian of a vector field in the 
    TNB coordinate system of an underlying vector field.
    
    See: https://arxiv.org/pdf/2312.15589.pdf
    
    Author: James Beattie
    
    Args:
        args (_type_): 

    Returns:
        _type_: _description_
    
    """
    
    def theta_eig(J_thresh  : np.ndarray,
                  J_3       : np.ndarray):
        """
        Compute the angle between the eigenvectors of the Jacobian.
        
        See: https://arxiv.org/pdf/2312.15589.pdf
        
        Author: James Beattie
        
        Args:
            args (_type_): 

        Returns:
            _type_: _description_
        
        """
        
        # Two conditions for O and X points
        condition = np.abs(J_3) < J_thresh
        ratio = np.where(condition, 
                         J_thresh / J_3, 
                         J_3 / J_thresh)
        
        return np.arctan( np.sqrt(ratio**2-1) )
    
    # Compute jacobian of B field
    jacobian = smooth_gradient_tensor(gradient_tensor(vector_field,
                               order=2))
    
    # Make jacobian traceless (numerical errors will result in some trace, which is
    # equivalent to div(B) modes)
    if traceless:   
        jacobian = jacobian - (1/3) * np.einsum("..., ij... -> ij...",
                                                np.einsum("ii...",
                                                        jacobian),
                                                np.eye(3))
    
    
    
    # # Compute TNB basis
    # t_basis, n_basis, b_basis, _ = compute_TNB_basis(vector_field)
    # X   = np.array([b_basis,n_basis,t_basis])
    # X_T = np.einsum("ij...->ji...",X)
    
    # # Put jacobian into the TNB basis
    # trans_jacobian = np.einsum('ab...,bc...,cd... -> ad...', 
    #                            X, 
    #                            jacobian, 
    #                            X_T)
    
    # Construct M, the 2D jacobian of the B_perp field
    M_11 = jacobian[0,0,...]
    M_12 = jacobian[0,1,...]
    M_21 = jacobian[1,0,...]
    M_22 = jacobian[1,1,...]
    M    = np.array([[M_11, M_12],
                     [M_21, M_22]])
    
    # Compute trace and determinant of M
    trace_M = np.einsum("ii...",M)
    det_M   = M_11 * M_22 - M_12 * M_21
    
    # Characteristic equation
    D = 4 * det_M - trace_M**2
    
    # J values for openning angles of X and O point
    # (these are physical currents)
    J_3         = M_21 - M_12
    J_thresh    = np.sqrt( (M_11 - M_22)**2 + (M_12 + M_21)**2 )
    
    # Eigen values of M from characteristic equation
    eig_1 = 0.5 * ( trace_M + np.sqrt( - (D + 0j)))
    eig_2 = 0.5 * ( trace_M - np.sqrt( - (D + 0j)))
    
    return trace_M, D, eig_1, eig_2, J_3, J_thresh, theta_eig(J_thresh,J_3)


def classification_of_critical_points(trace_M       : np.ndarray,
                                        D           : np.ndarray,
                                        J_3         : np.ndarray,
                                        J_thresh    : np.ndarray,
                                        eig_1       : np.ndarray,
                                        eig_2       : np.ndarray):
        """
        Classify the critical points of the 2D reduced Jacobian of the B field.
    
        See: https://arxiv.org/pdf/2312.15589.pdf
        
        Args:
            trace_M (np.ndarray):   trace of the 2D reduced Jacobian of the B field
            D (np.ndarray):         determinant of the 2D reduced Jacobian of the B field
            J_3 (np.ndarray):       J_3 value of the 2D reduced Jacobian of the B field
            J_thresh (np.ndarray):  J threshold value of the 2D reduced Jacobian of the B field
            eig_1 (np.ndarray):     eigen value 1 of the 2D reduced Jacobian of the B field
            eig_2 (np.ndarray):     eigen value 2 of the 2D reduced Jacobian of the B field
            
        Returns:
            classification_array (np.ndarray): 3D array of the critical point types
        """
        
        is_3D = np.abs(trace_M) > 0.0
        is_2D = np.isclose(trace_M,0.0)
        is_real_eig = np.abs(J_3) < J_thresh
        is_imag_eig = np.abs(J_3) > J_thresh
        is_parallel = np.isclose(np.abs(J_3),J_thresh,1e-3)
            
        # real and imaginary components of the eigenvalues
        eig1_real = np.real(eig_1)
        eig2_real = np.real(eig_2)
        
        # array initialisation to store each of the 9 critical point types
        classification_array = np.repeat(np.zeros_like(trace_M)[np.newaxis,...],
                                         9,
                                         axis=0)
        
        # 3D X point (trace > 0, determinant < 0, real eigenvalues less than 0)
        classification_array[0,...] = np.logical_and.reduce([is_3D, 
                                                             is_real_eig, 
                                                             eig1_real*eig2_real < 0.0])
        
        # 3D O point (repelling; trace > 0, determinant > 0, conjugate eigenvalues equal)
        classification_array[1,...] = np.logical_and.reduce([is_3D, 
                                                             is_imag_eig, 
                                                             trace_M > 0.0])
        
        # 3D O point (attracting; trace < 0, determinant > 0, conjugate eigenvalues equal)
        classification_array[2,...] = np.logical_and.reduce([is_3D, 
                                                              is_imag_eig, 
                                                              trace_M < 0.0])
        
        # 3D repelling (trace =/= 0, determinant <= 0, real eigenvalues greater than 0)
        classification_array[3,...] = np.logical_and.reduce([is_3D, 
                                                             is_real_eig, 
                                                             eig1_real > 0.0, 
                                                             eig2_real > 0.0])
                                                
        # 3D attracting (trace =/= 0, determinant <= 0, real eigenvalues less than 0)
        classification_array[4,...] = np.logical_and.reduce([is_3D, 
                                                             is_real_eig, 
                                                             eig1_real < 0.0, 
                                                             eig2_real < 0.0])
        
        # 3D antiparallel (trace =/= 0, determinant < 0, either real component of eigenvalues = 0)
        classification_array[5,...] = np.logical_and.reduce([is_3D, 
                                                             is_parallel])
        
        # 2D X point (trace = 0, determinant < 0, real component of eigenvalues equal in opposite sign)
        classification_array[6,...] = np.logical_and.reduce([is_2D, 
                                                             is_real_eig])
        
        # 2D O point (trace = 0, determinant > 0, imaginary component of eigenvalues equal in opposite sign)
        classification_array[7,...] = np.logical_and.reduce([is_2D, 
                                                             is_imag_eig])
        
        # 2D antiparallel (trace = 0, determinant = 0, all eigenvalues = 0)
        classification_array[8,...] = np.logical_and.reduce([is_2D, 
                                                             is_parallel])
        
        return classification_array
    
        
## END OF LIBRARY