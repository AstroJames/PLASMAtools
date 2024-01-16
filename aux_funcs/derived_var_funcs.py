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

#import scipy.fft as fft
import scipy.fft as fft
import numpy as np
from multiprocessing import Pool, shared_memory
import ctypes

## ###############################################################
## Derived Variable Functions
## ###############################################################

# indexes
X,Y,Z = 0,1,2

# shifts for derivatives
F = -1 # shift forwards
B = +1 # shift backwards

def vector_potential(vector_field,
                     debug = False):
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
    a = fft.ifftn(a_hat, 
                axes=(1, 2, 3),
                norm="forward").real
    
    # Take the curl of the vector potential to get the reconstructed magnetic field
    # for debuging
    if debug:
        # have to at least a fourth order derivative here 
        # to get a good reconstruction
        b_recon = vector_curl(a,order=4)  
    
    return a, b_recon


def magnetic_helicity(magnetic_vector_field):
    """
    Compute the magnetic helicity of a vector field.
    
    Author: James Beattie
    """
    
    # compute the vector potential
    a, _ = vector_potential(magnetic_vector_field)
    
    # compute the magnetic helicity
    helicity = np.einsum("i..., i... -> ...",a,magnetic_vector_field)
    
    return helicity


def gradient_tensor(vector_field,
                    order = 2):
    """
    Compute the gradient tensor of a vector field using 
    either second or fourth order differences.
    
    Author: James Beattie
    """
    
    # determine order of derivative
    if order == 2:
        grad_fun = gradient_order2
    elif order == 4:
        grad_fun = gradient_order4
    elif order == 6:
        grad_fun = gradient_order6
    
    return np.array([[grad_fun(vector_field[X], gradient_dir=direction) for direction in [X,Y,Z]],
                     [grad_fun(vector_field[Y], gradient_dir=direction) for direction in [X,Y,Z]],
                     [grad_fun(vector_field[Z], gradient_dir=direction) for direction in [X,Y,Z]]])
    

def orthogonal_tensor_decomposition(tensor_field):
    """
    Compute the symmetric, anti-symmetric and bulk components of a tensor field.
    
    Author: James Beattie
    """
    
    # transpose
    tensor_transpose = np.einsum("ij... -> ji...",tensor_field)
    
    # bulk component
    tensor_trace = (1./3.) * np.einsum("ii...",tensor_field)
    
    # symmetric component
    tensor_sym = 0.5 * (tensor_field + tensor_transpose) -  np.einsum('...,ij...->ij...',tensor_trace,np.identity(3))
    
    # anti-symmetric component
    tensor_anti = 0.5 * (tensor_field - tensor_transpose)
    
    return tensor_sym, tensor_anti, tensor_trace


# Top-level function for computing eigenvalues
def compute_eigenvalues(args):
    index, shape, dtype, shm_name = args
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    trans_tensor_sym_shared = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    tensor_at_point = trans_tensor_sym_shared[..., index[0], index[1], index[2]]
    eigvals = np.linalg.eigvalsh(tensor_at_point)
    existing_shm.close()
    return index, eigvals


def eigs_stretch_tensor(vector_field,
                        tensor_field,
                        n_processes = 4):
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
    the size of the eigen values. The first is the largest, stretching eigen
    value, followed by the null eigen value and then the compression eigen value.
    smallest. 
    
    Author: James Beattie
    """
    
    print(f"stretch_tensor: computing eigen values of local " + 
          f"stretching tensor with {n_processes} processes")
    
    # symmetric component
    tensor_sym, _, _ = orthogonal_tensor_decomposition(tensor_field)
    
    # Compute TNB basis of vector field
    t_basis, n_basis, b_basis, _ = compute_TNB_basis(vector_field)
    X   = np.array([t_basis,n_basis,b_basis])
    X_T = np.einsum("ij... -> ji ...",X)
    
    # Put stretching tensor into the TNB basis
    trans_tensor_sym = np.einsum('ij..., jk..., kl... -> il...', 
                                 X, 
                                 tensor_sym, 
                                 X_T)
    
    # Compute eigenvalues of the stretching tensor
    
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
        eigenvalues[(slice(None),) + index] = eigvals

    # Clean up shared memory
    shm.close()
    shm.unlink()
    
    return eigenvalues


def A_iA_j_tensor(vector_field):
    """
    Compute the A_iA_j tensor of a vector field.
    """
        
    return np.einsum('i...,j...->ij...',vector_field,vector_field)


def helmholtz_decomposition(vector_field: np.ndarray,
                            n_workers: int = 1):
    """
    Compute the irrotational and solenoidal components of a vector field.
    
    Author: James Beattie (assumes periodic boundary conditions)
    """
    # F is a 4D array, with the last dimension being 3 (for the x, y, z components of the vector field)
    
    shape = vector_field.shape[:-1]
    x     = np.linspace(-0.5,0.5,vector_field.shape[0]) # assuming a domian of [-L/2, L/2]
    
    # Fourier transform to Fourier space    
    Fhat = fft.fftn(vector_field, axes=(0, 1, 2),norm = 'forward',workers=n_workers)
    
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
    F_irrot = fft.ifftn(Fhat_irrot, axes=(X,Y,Z),workers=n_workers,norm = 'forward').real
    F_solen = fft.ifftn(Fhat_solen, axes=(X,Y,Z),workers=n_workers,norm = 'forward').real
    
    # Remove numerical noise
    # threshold = 1e-16
    # F_solen[np.abs(F_solen) < threshold] = 0
    # F_irrot[np.abs(F_irrot) < threshold] = 0
    
    return F_irrot, F_solen


def vector_curl(vector_field,
                order = 2):
    """
    Compute the vector curl (assumes periodic boundary conditions) 
    using either second or fourth order finite differences
    
    Author: James Beattie
    """
    
    if order == 2:
        grad_fun = gradient_order2
    elif order == 4:
        grad_fun = gradient_order4
    elif order == 6:
        grad_fun = gradient_order6
        
    # x component of curl
    dFz_dy = grad_fun(vector_field[Z],gradient_dir=Y)
    dFy_dz = grad_fun(vector_field[Y],gradient_dir=Z)
    
    # y component of curl
    dFx_dz = grad_fun(vector_field[X],gradient_dir=Z)
    dFz_dx = grad_fun(vector_field[Z],gradient_dir=X)
    
    # z component of curl
    dFy_dx = grad_fun(vector_field[Y],gradient_dir=X)
    dFx_dy = grad_fun(vector_field[X],gradient_dir=Y)
        
    return np.array([dFz_dy - dFy_dz,
                     dFx_dz - dFz_dx,
                     dFy_dx - dFx_dy])
    

def vector_divergence(vector_field,
                        order = 2):
    """
    Compute the vector divergence (assumes periodic boundary conditions)
    using either second or fourth order finite differences
    
    """
    
    if order == 2:
        grad_fun = gradient_order2
    elif order == 4:
        grad_fun = gradient_order4
    elif order == 6:
        grad_fun = gradient_order6
    
    # divergence
    dFx_dx = grad_fun(vector_field[X],gradient_dir=X)
    dFy_dy = grad_fun(vector_field[Y],gradient_dir=Y)
    dFz_dz = grad_fun(vector_field[Z],gradient_dir=Z)
    
    return dFx_dx + dFy_dy + dFz_dz
    
def scalar_laplacian(scalar_field):
    """
    Compute the scalar laplacian (assumes periodic boundary conditions)
    using second order finite differences
    
    Author: James Beattie
    """
        
    # differentials
    dx = 1./scalar_field[X].shape[0]
    dy = 1./scalar_field[Y].shape[0]
    dz = 1./scalar_field[Z].shape[0]
    
    d2Fx_dx2 = (np.roll(scalar_field,F,axis=X) - 2*scalar_field + np.roll(scalar_field,B,axis=X))/dx**2
    d2Fy_dy2 = (np.roll(scalar_field,F,axis=Y) - 2*scalar_field + np.roll(scalar_field,B,axis=Y))/dy**2
    d2Fz_dz2 = (np.roll(scalar_field,F,axis=Z) - 2*scalar_field + np.roll(scalar_field,B,axis=Z))/dz**2
    
    return d2Fx_dx2 + d2Fy_dy2 + d2Fz_dz2
        
        
def vector_cross_product(vector1, vector2):
    """
    Compute the vector cross product of two vectors.
    
    Auxillary functions for computeTNBBasis
    
    Author: Neco Kriel
    """
    
    vector3 = np.array([
    vector1[Y] * vector2[Z] - vector1[Z] * vector2[Y],
    vector1[Z] * vector2[X] - vector1[X] * vector2[Z],
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


def field_RMS(scalar_field):
    """
    Compute the root-mean-squared of a field.
    
    Author: Neco Kriel
    """
    return np.sqrt(np.mean(scalar_field**2))


def field_gradient(scalar_field):
    """
    Compute the gradient of a scalar field.
    
    Author: Neco Kriel & James Beattie
    """
    ## format: (x, y, z)
    scalar_field = np.array(scalar_field)
    field_gradient = [
        gradient_order2(scalar_field, gradient_dir)
        for gradient_dir in [X, Y, Z]
    ]
    return np.array(field_gradient)


def compute_TNB_basis(vector_field):
    """
    Compute the Fressnet frame of a vector field (TNB basis).
    
    Author: Neco Kriel + James Beattie
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


def TNB_coordinate_transformation(vector_field):
    """
    Transform a vector field into the TNB coordinate system.
    
    Author: James Beattie
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


def TNB_jacobian(vector_field):
    """
    Compute the Jacobian of a vector field in the TNB coordinate system.
    
    Note: the Jacobian is a 2D tensor which is only appropriate for 3D fields where
    there is no divergence component. This reduces the degress of freedom from 3 to 2.
    
    Author: James Beattie
    """
    
    # compute the vector field in the TNB basis
    vector_field_TNB = TNB_coordinate_transformation(vector_field)
    
    # indexes for the TNB basis
    n_basis = 1
    b_basis = 2
    
    # differentials
    two_dn = 2./vector_field_TNB[n_basis].shape[0]
    two_db = 2./vector_field_TNB[b_basis].shape[0]
    
    # now compute a 2D jacobian tensor for each (x, y, z) point
    # dn_basis
    dFn_basis_dn_basis = (np.roll(vector_field_TNB[n_basis],F,axis=n_basis) - np.roll(vector_field_TNB[n_basis],B,axis=n_basis))/two_dn
    dFb_basis_dn_basis = (np.roll(vector_field_TNB[b_basis],F,axis=n_basis) - np.roll(vector_field_TNB[b_basis],B,axis=n_basis))/two_dn
    
    # db_basis
    dFn_basis_db_basis = (np.roll(vector_field_TNB[n_basis],F,axis=b_basis) - np.roll(vector_field_TNB[n_basis],B,axis=b_basis))/two_db
    dFb_basis_db_basis = (np.roll(vector_field_TNB[b_basis],F,axis=b_basis) - np.roll(vector_field_TNB[b_basis],B,axis=b_basis))/two_db
    
    jacobian = np.array([[dFn_basis_dn_basis,dFn_basis_db_basis],
                         [dFb_basis_dn_basis,dFb_basis_db_basis]])
    
    
    return jacobian


def TNB_jacobian_stability_analysis(vector_field,
                                    traceless = True):
    """
    Compute the trace, determinant and eigenvalues of the Jacobian of a vector field in the TNB coordinate system.
    
    """
    
    def theta_eig(J_thresh,J_3):
        """
        Compute the angle between the eigenvectors of the Jacobian.
        """
        
        # Two conditions for O and X points
        condition = np.abs(J_3) < J_thresh
        ratio = np.where(condition, 
                         J_thresh / J_3, 
                         J_3 / J_thresh)
        
        return np.arctan( np.sqrt(ratio**2-1) )
    
    # Compute jacobian of B field
    jacobian = gradient_tensor(vector_field,
                               order=4)
    
    # Make jacobian traceless (numerical errors will result in some trace, which is
    # equivalent to div(B) modes)
    if traceless:   
        jacobian = jacobian - (1/3) * np.einsum("xyz, ...i",
                                                np.einsum("iixyz",
                                                        jacobian),
                                                np.eye(3))
    
    # Compute TNB basis
    t_basis, n_basis, b_basis, _ = compute_TNB_basis(vector_field)
    X   = np.array([b_basis,n_basis,t_basis])
    X_T = np.einsum("ijxyz->jixyz",X)
    
    # Put jacobian into the TNB basis
    trans_jacobian = np.einsum('abxyz,bcxyz,dcxyz->adxyz', 
                               X, 
                               jacobian, 
                               X_T)
    
    # Construct M, the 2D jacobian of the B_perp field
    M_11 = trans_jacobian[0,0,...]
    M_12 = trans_jacobian[0,1,...]
    M_21 = trans_jacobian[1,0,...]
    M_22 = trans_jacobian[1,1,...]
    M = np.array([[M_11, M_12],
                  [M_21, M_22]])
    
    # Compute trace and determinant of M
    trace_M = np.einsum("iixyz",M)
    det_M   = M_11 * M_22 - M_12 * M_21
    
    # Characteristic equation
    D = 4 * det_M - trace_M**2
    
    J_3         = M_21 - M_12
    J_thresh    = np.sqrt( (M_11 - M_22)**2 + (M_12 + M_21)**2 )
    
    # Eigen values of M from characteristic equation
    eig_1 = 0.5 * ( trace_M + np.sqrt( - (D + 0j)))
    eig_2 = 0.5 * ( trace_M - np.sqrt( - (D + 0j)))
    
    return trace_M, D, eig_1, eig_2, theta_eig(J_thresh,J_3)


################################################################
## Derivative stencil functions 
################################################################


def gradient_order2(scalar_field, gradient_dir, L=1.0):
    """
    Compute the gradient of a scalar field in one direction
    using a two point stencil (second order method).
    
    Author: Neco Kriel & James Beattie
    """
    
    # 2dr
    two_dr = 2. * L /scalar_field.shape[gradient_dir]
    
    return (
        np.roll(scalar_field, F, axis=gradient_dir) - np.roll(scalar_field, B, axis=gradient_dir) ) / two_dr
    
    
def gradient_order4(scalar_field, gradient_dir, L=1.0):
    """
    Compute the gradient of a scalar field  in one direction
    using a five point stencil (fourth order method).
    
    Author: James Beattie
    
    """
    
    # 12dr
    twelve_dr = 12. * L /scalar_field.shape[gradient_dir]
    
    # df/dr = (-f(r+2dr) + 8f(r+dr) - 8f(r-dr) + f(r-2dr))/12dr
    return ( - np.roll(scalar_field,2*F,axis=gradient_dir) \
             + 8*np.roll(scalar_field,F,axis=gradient_dir) \
             - 8*np.roll(scalar_field,B,axis=gradient_dir) \
             + np.roll(scalar_field,2*B,axis=gradient_dir)) / twelve_dr


def gradient_order6(scalar_field, gradient_dir, L=1.0):
    """
    Compute the gradient of a scalar field  in one direction
    using a seven point stencil (sixth order method).
    
    Author: James Beattie
    
    """
    
    # 60dr
    twelve_dr = 60. * L /scalar_field.shape[gradient_dir]
    
    # df/dr = (-f(r+2dr) + 8f(r+dr) - 8f(r-dr) + f(r-2dr))/12dr
    return ( - np.roll(scalar_field,3*B,axis=gradient_dir)   \
             + 9*np.roll(scalar_field,2*B,axis=gradient_dir) \
             - 45*np.roll(scalar_field,B,axis=gradient_dir)  \
             + 45*np.roll(scalar_field,F,axis=gradient_dir)  \
             - 9*np.roll(scalar_field,2*F,axis=gradient_dir) \
             + np.roll(scalar_field,3*F,axis=gradient_dir)) / twelve_dr


## END OF LIBRARY