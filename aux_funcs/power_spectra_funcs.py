"""
    Title: Power Spectra Funcs
    Author: James R. Beattie
    Date: 09/07/2017
    Description: Functions for calculating fourier variables in the read_flash classes


"""

## ###############################################################
## IMPORTS
## ###############################################################

import numpy.fft as fft
import numpy as np

## ###############################################################
## Spectral Functions
## ###############################################################

def compute_power_spectrum_3D(volume: np.ndarray) -> np.ndarray:
    """
    Computes the power spectrum of a 3D scalar field. Note the norm = "forward" 
    in the Fourier transform. This means that the power spectrum will be scaled by
    the number of grid points 1/N^3, and the inverse transform will be scaled by
    N^3.

    Args:
        volume (np.ndarray): 3D scalar field for which to compute the power spectrum

    Returns:
        power_spectrum_3D (np.ndarray): 3D power spectrum of the input volume
    """
    
    # Compute the 3D Fourier Transform
    ft = np.fft.fftn(volume, norm='forward')
    
    # Shift zero frequency component to center
    ft = np.fft.fftshift(ft)
    
    # Compute the 3D power spectrum
    power_spectrum_3D = np.abs(ft)**2
    
    return power_spectrum_3D

def spherical_integrate(data: np.ndarray, 
                        bins: int = None) -> tuple:
    """
    The spherical integrate function takes the 3D power spectrum and integrates
    over spherical shells of constant k. The result is a 1D power spectrum.
    
    It has been tested to reproduce the 1D power spectrum of an input 3D Gaussian
    random field.
    
    It has been tested to maintain the correct normalisation of the power spectrum
    i.e., the integral over the spectrum. For small grids (number of bins) the normalisation
    will be off by a small amount (roughly factor 2 for 128^3 with postive power-law indexes). 
    This is because the frequencies beyond the Nyquist limit are not included in the radial 
    integration. This is not a problem for grids of 256^3 or larger, or for k^-a style spectra,
    which are far more commonly encountered, and the normalisation is closer to 1/10,000 numerical
    error
    
    Args:
        data: The 3D power spectrum
        bins: The number of bins to use for the radial integration. 
              If not specified, the Nyquist limit is used (as should always be the case, anyway).

    Returns:
        k_modes: The k modes corresponding to the radial integration
        radial_sum: The radial integration of the 3D power spectrum (including k^2 correction)
    """
    z, y, x = np.indices(data.shape)
    center = np.array([(i - 1) / 2.0 for i in data.shape])
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    N = data.shape[0]
    if not bins:
        bins = N // 2

    bin_edges = np.linspace(0.5, bins, bins+1)

    # Use np.digitize to assign each element to a bin
    bin_indices = np.digitize(r, bin_edges)

    # Compute the radial profile
    radial_sum = np.zeros(bins)
    for i in range(1, bins+1):
        mask = bin_indices == i
        radial_sum[i-1] = np.sum(data[mask])

    # Generate the spatial frequencies with dk=1
    # Now k_modes represent the bin centers
    k_modes = np.ceil((bin_edges[:-1] + bin_edges[1:])/2)

    return k_modes, radial_sum

def cylindrical_integrate(data: np.ndarray, 
                          bins_perp: int = 0,
                          bins_para:  int = 0) -> tuple:
    """
    The cylindrical integrate function takes the 3D power spectrum and integrates
    over cylindrical shells of constant k in a plane, and then a 1D spectrum along
    the the remaining dimension. The result is a 2D power spectrum of k_perp and k_par
    modes.
    
    
   Args:
        data     : The 3D power spectrum
        bins_perp: Number of bins for k_perpendicular
        bins_para: Number of bins for k_parallel

    Returns:
        k_perp_modes    : k_perpendicular modes
        k_para_modes    : k_parallel modes
        cylindrical_sum : cylindrical integration of the 3D power spectrum
                          note that k_perp is on axis 0, k_para is on axis 1
    """
    
    z, y, x = np.indices(data.shape)
    center  = np.array([(i - 1) / 2.0 for i in data.shape])
    k_perp  = np.sqrt((x - center[0])**2 + (y - center[1])**2)  # Cylindrical radius
    k_para  = np.abs(z - center[2])                             # Distance from the plane

    N = data.shape[0]
    if bins_perp == 0:
        bins_perp = N // 2
    if bins_para == 0:
        bins_para = N // 2

    # initailize cylindrical sum
    cylindrical_sum = np.zeros((bins_perp, bins_para))

    # define bin edges (note starting at 0.5 to avoid binning the zero mode)
    bin_edges_perp = np.linspace(0, bins_perp, bins_perp+1)
    bin_edges_para = np.linspace(0, bins_para, bins_para+1)

    # Vectorized bin assignment
    bin_indices_perp = np.digitize(k_perp, bin_edges_perp) - 1
    bin_indices_para = np.digitize(k_para, bin_edges_para) - 1

    # Create 2D linear indices
    linear_indices = bin_indices_perp + bin_indices_para * bins_perp

    # Use np.bincount for efficient summation
    cylindrical_sum = np.bincount(linear_indices.ravel(), 
                                  weights=data.ravel(), 
                                  minlength=bins_perp * bins_para)
    
    # Ensure that the length matches the expected size
    cylindrical_sum = cylindrical_sum[:bins_perp * bins_para]
    cylindrical_sum = cylindrical_sum.reshape((bins_perp, bins_para))
    k_perp_modes = (bin_edges_perp[:-1] + bin_edges_perp[1:]) / 2
    k_para_modes = (bin_edges_para[:-1] + bin_edges_para[1:]) / 2

    return k_perp_modes, k_para_modes, cylindrical_sum


def helical_decomposition(vector_field):
    """
    Performs a helical decomposition of a vector field.

    Parameters:
    velocity_field (array-like): The velocity field corresponding to each k, an array of shape (N, 3).

    Returns:
    u_plus (array): The component of the vector field in the direction of the right-handed helical component.
    u_minus (array): The component of the vector field in the direction of the left-handed helical component.
    """
    # Convert inputs to numpy arrays
    vector_field = np.asarray(vector_field)
    
    # Take FFT of vector field
    vector_field_FFT = fft.fftn(vector_field,
                                norm='forward',
                                axes=(0,1,2))
    N = vector_field.shape[0]  # Assuming a cubic domain
    L = 1  # The physical size of the domain
    kx = fft.fftfreq(N, d=L/N)
    ky = fft.fftfreq(N, d=L/N)
    kz = fft.fftfreq(N, d=L/N)
    
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k = np.stack((kx, ky, kz), axis=-1)  # This will be of shape (N, N, N, 3)

    # Normalize k to get the unit wavevector
    k_norm = np.linalg.norm(k, axis=-1, keepdims=True)

    if np.any(np.isnan(k_norm)) or np.any(np.isinf(k_norm)):
        raise ValueError("NaN or Inf found in k_norm")    

    # Set the k_hat for zero wavevectors explicitly to zero (or some other appropriate value)
    k_hat = np.zeros_like(k)
    non_zero_indices = k_norm.squeeze() > 0  # Indices where the norm is non-zero
    k_hat[non_zero_indices] = k[non_zero_indices] / k_norm[non_zero_indices]
    
    if np.any(np.isnan(k_hat)) or np.any(np.isinf(k_hat)):
        raise ValueError("NaN or Inf found in k_hat")

    # Choose an arbitrary versor orthogonal to k
    z = np.array([0, 0, 1])
    e = np.cross(z, k_hat)
    e_norm = np.linalg.norm(e, axis=-1, keepdims=True)
    
    # Set the k_hat for zero wavevectors explicitly to zero (or some other appropriate value)
    e_hat = np.zeros_like(k)
    non_zero_indices = e_norm.squeeze() > 0  # Indices where the norm is non-zero
    e_hat[non_zero_indices] = e[non_zero_indices] / e_norm[non_zero_indices]

    # Ensure that e_hat is not a zero vector (which can happen if k is parallel to z)
    # In such a case, we can choose e_hat to be any vector orthogonal to k
    for i, e in enumerate(e_hat):
        if np.allclose(e, np.zeros_like(e)):
            # Choose a new e_hat that is not parallel to k
            if np.allclose(k_hat[i], np.array([1, 0, 0])) or np.allclose(k_hat[i], np.array([0, 0, 1])):
                e_hat[i] = np.array([0, 1, 0])
            else:
                e_hat[i] = np.array([1, 0, 0])

    # Calculate helical components    
    factor                 = 1/np.sqrt(2.0)
    e_cross_k              = np.cross(e_hat,k)
    e_cross_k_norm         = np.linalg.norm(e_cross_k,axis=-1,keepdims=True)
    k_cross_e_cross_k      = np.cross(k, e_cross_k)
    k_cross_e_cross_k_norm = np.linalg.norm(k_cross_e_cross_k,axis=-1,keepdims=True)
    
    h_plus =  factor * e_cross_k / e_cross_k_norm  + \
                 factor * 1j * k_cross_e_cross_k / k_cross_e_cross_k_norm
                 
    h_minus = factor * e_cross_k / e_cross_k_norm - \
                 factor * 1j * k_cross_e_cross_k / k_cross_e_cross_k_norm

    # test orthogonality 
    #print(np.abs(np.sum(h_plus * h_minus, axis=-1)))
    #print(np.sum(np.abs(h_minus * h_plus), axis=-1))

    # Project velocity field onto helical components
    u_plus = np.sum(vector_field_FFT * h_plus, axis=-1)
    u_minus = np.sum(vector_field_FFT * h_minus, axis=-1)

    # remove k = 0 mode   
    u_plus[np.isnan(u_plus)] = 0
    u_minus[np.isnan(u_minus)] = 0

    return u_plus, u_minus

def create_helical_field(N, k_index, A_plus=1, A_minus=0):
    """
    Creates a vector field with known helical components.

    Parameters:
    N (int): Size of the grid in each dimension.
    k_index (tuple): The index of the wavevector to be used.

    Returns:
    np.ndarray: The generated vector field.
    """
    # Create an empty field in Fourier space
    field_fft = np.zeros((N, N, N, 3), dtype=complex)

    # Generate the wavevector
    L = 1
    kx, ky, kz = np.meshgrid(np.fft.fftfreq(N, d=L/N), 
                             np.fft.fftfreq(N, d=L/N), 
                             np.fft.fftfreq(N, d=L/N), indexing='ij')
    k = np.stack((kx, ky, kz), axis=-1)

    # Calculate h_plus and h_minus for the selected wavevector
    k_vector = k[k_index]
    k_norm = np.linalg.norm(k_vector,axis=-1,keepdims=True)
    k_hat = k_vector / k_norm

    z = np.array([0, 0, 1])
    e = np.cross(z, k_hat)
    e_norm = np.linalg.norm(e,axis=-1)
    e_hat = e / e_norm

    factor = 1/np.sqrt(2.0)
    e_cross_k = np.cross(e_hat,k_vector)
    e_cross_k_norm = np.linalg.norm(e_cross_k,axis=-1,keepdims=True)
    k_cross_e_cross_k = np.cross(k_vector, e_cross_k)
    k_cross_e_cross_k_norm = np.linalg.norm(k_cross_e_cross_k,axis=-1,keepdims=True)
    
    h_plus =  factor * e_cross_k / e_cross_k_norm + \
                 factor * 1j * k_cross_e_cross_k / k_cross_e_cross_k_norm
                 
    h_minus = factor * e_cross_k / e_cross_k_norm - \
                 factor * 1j * k_cross_e_cross_k / k_cross_e_cross_k_norm

    # Assign coefficients in Fourier space
    field_fft[k_index] = A_plus * h_plus + A_minus * h_minus

    # Perform inverse FFT to get the field in physical space
    field = fft.ifftn(field_fft, axes=(0, 1, 2),norm="forward").real

    return field

def generate_isotropic_powerlaw_field(size:  int,
                                      alpha: float = 5./3.) -> np.ndarray:
    """
    This computes a random field with a power-law power spectrum. The power spectrum
    is P(k) = k^-alpha. The field is generated in Fourier space, and then inverse
    transformed to real space.

    Author: James Beattie (2023)

    Args:
        size (int): the linear dimension of the 3D field
        alpha (float): the negative 1D power-law exponent used in Fourier space. Defaults to 5/3.
                        Note that I use the negative exponent, because the power spectrum is
                        proportional to k^-alpha, and note that I make the transformations between
                        3D Fourier transform exponent and 1D power spectrum exponent in the code.

    Returns:
        ifft field (np.ndarray): the inverse fft of the random field with a power-law power spectrum
    """
    # Create a grid of frequencies
    kx = np.fft.fftfreq(size)
    ky = np.fft.fftfreq(size)
    kz = np.fft.fftfreq(size)
    
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Calculate the magnitude of k for each frequency component
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    k[0, 0, 0] = 1  # Avoid division by zero
    
    # Create a 3D grid of random complex numbers
    random_field = np.random.randn(size, size, size) + 1j * np.random.randn(size, size, size)
    
    # Adjust the amplitude of each frequency component to follow k^-5/3 (-11/3 in 3D)
    amplitude = np.where(k != 0, k**(-(alpha+2.0)/2.0), 0)
    adjusted_field = 10*random_field * amplitude

    return np.fft.ifftn(adjusted_field).real

def generate_anisotropic_powerlaw_field(size:  int,
                                        alpha: float = 5./3.,
                                        beta:  float = 5./3.) -> np.ndarray:
    """
    This computes a random field with a power-law power spectrum. The power spectrum
    is P(k) = k_perp^-alpha k_parallel^-beta. The field is generated in Fourier space, 
    and then inverse transformed to real space.
    
    Author: James Beattie

    Args:
        size (int): the linear dimension of the 3D field
        alpha (float): the negative 1D power-law exponent used in Fourier space for the 
                        perpendicular component. Defaults to 5/3.
                        Note that I make the transformations between 3D Fourier transform 
                        exponent and 1D power spectrum exponent in the code.
        beta (float): the negative 1D power-law exponent used in Fourier space for the 
                        parallel component. Defaults to 5/3.

    Returns:
        ifft field (np.ndarray): the inverse fft of the random field with a power-law power spectrum
    """
    # Create a grid of frequencies
    kx = np.fft.fftfreq(size)
    ky = np.fft.fftfreq(size)
    kz = np.fft.fftfreq(size)
    
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Calculate the magnitude of k for each frequency component
    k_perp              = np.sqrt(kx**2 + ky**2)
    k_par               = np.abs(kz)
    k_perp[k_perp==0]   = 1   # Avoid division by zero
    k_par[k_par==0]     = 1    # Avoid division by zero
    
    # Create a 3D grid of random complex numbers for phases
    random_field = np.random.randn(size, size, size) + 1j * np.random.randn(size, size, size)
    
    # Adjust the amplitude of each frequency component to follow k^-5/3 (-11/3 in 3D)
    amplitude = k_perp**(-(alpha)/2.0)*k_par**(-(beta)/2.0)

    return np.fft.ifftn(10*random_field * amplitude).real

def extract_isotropic_shell_X(vector_field: np.ndarray,
                              k_minus_dk:   float,
                              k_plus_dk:    float,
                              filter:       str     = 'tophat',
                              sigma:        float   = 10.0):
    """ 
    
    Extracts shell X-0.5 < k <X+0.5 of a vector field and 
    returns the inverse FFT of the shell. 
    
    Based on Philip Grete's transfer function code:
    https://github.com/pgrete/energy-transfer-analysis
     
    
    Author: James Beattie
        
    """
    
    def Gauss_filter(k,k0,sigma=sigma):
        filter = np.exp(-(k-k0)**2/(2*sigma**2))
        return filter / filter.max()    
    
    # The physical size of the domain
    L = 1.0
    
    # The wavenumebr shells (convert into units of 2pi/L)
    k_minus_dk = 2 * np.pi / L * k_minus_dk
    k_plus_dk  = 2 * np.pi / L * k_plus_dk
    
    # Take FFT of vector field
    vector_field_FFT = fft.fftn(vector_field,
                                norm='forward',
                                axes=(1,2,3))

    # Assuming a cubic domain    
    N = vector_field.shape[1]  
            
    # wave vectors
    kx = 2 * np.pi * fft.fftfreq(N, d=L/N) / L
    ky = 2 * np.pi * fft.fftfreq(N, d=L/N) / L
    kz = 2 * np.pi * fft.fftfreq(N, d=L/N) / L
    
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k = np.array([kx,ky,kz]) # This will be of shape (3, N, N, N)

    # Normalize k to get the unit wavevector
    # This will be of shape (3, N, N, N)
    k_norm = np.tile(np.linalg.norm(k, axis=0, keepdims=True), (3, 1, 1, 1)) 

    # Replace zeros in k_norm with np.inf to avoid division by zero
    k_norm[k_norm == 0] = np.inf
    
    # Set the k_hat for zero wavevectors explicitly to zero
    if filter == 'tophat':
        shell_k = np.where(np.logical_and(k_norm > k_minus_dk, k_norm <= k_plus_dk),vector_field_FFT,0.)
    elif filter == "gauss":
        shell_k = Gauss_filter(k_norm,(k_minus_dk+k_plus_dk)/2.0) * vector_field_FFT
    
    # Inverse FFT with just the wavenumbers from the shell 
    shell_real = np.fft.ifftn(shell_k,
                           axes=(1,2,3),
                           norm="forward").real
     
    return shell_real
