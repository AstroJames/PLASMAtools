"""
    Title: Power Spectra Funcs
    Author: James R. Beattie
    Date: 09/07/2019
    Description: Functions for calculating fourier variables in the read_flash classes

    Collaborators: Anne Noer Kolborg
"""

## ###############################################################
## IMPORTS
## ###############################################################

import numpy as np
import multiprocessing

pyfftw_import = False
try: 
    import pyfftw
    pyfftw_import = True
    pyfftw.interfaces.cache.enable()
    threads = multiprocessing.cpu_count()
    print(f"power_spectra_funcs: Using {threads} threads for FFTs with pyfftw.")
except ImportError:
    print("pyfftw not installed, using scipy's serial fft")

if pyfftw_import:
    # Use pyfftw for faster FFTs if available
    pyfftw.config.NUM_THREADS = threads
    fftn = pyfftw.interfaces.numpy_fft.fftn
    ifftn = pyfftw.interfaces.numpy_fft.ifftn
    fftfreq = pyfftw.interfaces.numpy_fft.fftfreq
    fftshift = pyfftw.interfaces.numpy_fft.fftshift
else:
    # Use numpy's FFT functions
    fftn = np.fft.fftn
    ifftn = np.fft.ifftn
    fftfreq = np.fft.fftfreq
    fftshift = np.fft.fftshift

## ###############################################################
## Spectral Functions
## ###############################################################

def compute_power_spectrum_3D(field: np.ndarray) -> np.ndarray:
    """
    Computes the power spectrum of a 3D vector field. Note the norm = "forward" 
    in the Fourier transform. This means that the power spectrum will be scaled by
    the number of grid points 1/N^3, and the inverse transform will be scaled by
    N^3.

    Args:
        volume (np.ndarray): 3D scalar field for which to compute the power spectrum

    Returns:
        power_spectrum_3D (np.ndarray): 3D power spectrum of the input volume
    """
    
    # the field should be i,N,N,N
    assert len(field.shape) == 4, "Field should be 3D"
        
    # Compute the 9 component Fourier Transform and shift zero frequency component to center,
    # then sum all the square components to get the power spectrum   
    return np.sum(
        np.abs(
            fftshift(
                fftn(field,
                     axes=(1, 2, 3),
                     norm='forward'),
                axes=(1, 2, 3)))**2,
        axis=0)

def compute_power_spectrum_2D(field: np.ndarray) -> np.ndarray:
    """
    Computes the power spectrum of a 2D vector field. Note the norm = "forward" 
    in the Fourier transform. This means that the power spectrum will be scaled by
    the number of grid points 1/N^3, and the inverse transform will be scaled by
    N^3.

    Args:
        volume (np.ndarray): 2D scalar field for which to compute the power spectrum

    Returns:
        power_spectrum_2D (np.ndarray): 2D power spectrum of the input volume
    """
    
    # the field should be i,N,N (i is the coordinate)
    assert len(field.shape) == 3, "Field should be 2D"
        
    # Compute the Fourier Transform and shift zero frequency component to center,
    # then sum all the square components to get the power spectrum   
    return np.sum(
        np.abs(
            fftshift(
                fftn(field,
                     axes=(1, 2),
                     norm='forward'),
                axes=(1, 2)))**2,
        axis=0)    

def compute_tensor_power_spectrum(field: np.ndarray) -> np.ndarray:
    """
    Computes the power spectrum of a 3D tensor field. Note the norm = "forward" 
    in the Fourier transform. This means that the power spectrum will be scaled by
    the number of grid points 1/N^3, and the inverse transform will be scaled by
    N^3.
    
    Author: James Beattie

    Args:
        tensor array (np.ndarray): 3D tensor field with 3,3,N,N,N components

    Returns:
        3d power spectrum (np.ndarray): 3D power spectrum of the input tensor volume with N,N,N grid array
        and shifted such that the zero frequency is in the center (to be used with spherical_integrate)
        
    """
    
    # the field should be 3,3,N,N,N
    assert len(field.shape) == 5, "Field should be a 3D tensor field, 3,3,N,N,N"
    assert field.shape[0] == 3, "Field should be a 3D tensor field, 3,3,N,N,N"
    assert field.shape[1] == 3, "Field should be a 3D tensor field, 3,3,N,N,N"
    
    # Compute the 9 component Fourier Transform and shift zero frequency component to center,
    # then sum all the square components to get the power spectrum   
    return np.sum(
        np.abs(
            fftshift(
                fftn(
                    field,
                    axes=(2,3,4),
                    norm='forward')))**2,
        axis=(0,1))


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


def spherical_integrate_2D(data: np.ndarray, 
                           bins: int = None) -> tuple:
    """
    The spherical integrate function takes the 2D power spectrum and integrates
    over spherical shells of constant k. The result is a 1D power spectrum.
    
    Needs to be tested in detail
    
    Args:
        data: The 2D power spectrum
        bins: The number of bins to use for the radial integration. 
              If not specified, the Nyquist limit is used (as should always be the case, anyway).

    Returns:
        k_modes: The k modes corresponding to the radial integration
        radial_sum: The radial integration of the 3D power spectrum (including k^2 correction)
    """
    y, x = np.indices(data.shape)
    center = np.array([(i - 1) / 2.0 for i in data.shape])
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)

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
    cylindrical_sum = cylindrical_sum.reshape((bins_perp, bins_para),
                                              order='F')
    # k_perp are in the first axis, k_par are in the second axis
    k_perp_modes    = (bin_edges_perp[:-1] + bin_edges_perp[1:]) / 2
    k_para_modes    = (bin_edges_para[:-1] + bin_edges_para[1:]) / 2

    return k_perp_modes, k_para_modes, cylindrical_sum


def helical_decomposition(vector_field):
    """
    Performs a helical decomposition of a vector field.

    Parameters:
    velocity_field (array-like): The velocity field corresponding to each k, an array of shape (N, 3).

    Returns:
    u_plus (array): The component of the vector field in the direction of the right-handed helical component.
    u_minus (array): The component of the vector field in the direction of the left-handed helical component.
    
    TODO: this whole function needs to be updated to conform to 3,N,N,N vecotr fields instead of
          N,N,N,3 vector fields
    """
    # Convert inputs to numpy arrays
    vector_field = np.asarray(vector_field)
    
    # Take FFT of vector field
    vector_field_FFT = fftn(vector_field,
                            norm='forward',
                            axes=(0,1,2))
    N = vector_field.shape[0]  # Assuming a cubic domain
    L = 1  # The physical size of the domain
    kx = fftfreq(N, d=L/N)
    ky = fftfreq(N, d=L/N)
    kz = fftfreq(N, d=L/N)
    
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
    kx, ky, kz = np.meshgrid(fftfreq(N, d=L/N), 
                             fftfreq(N, d=L/N), 
                             fftfreq(N, d=L/N), indexing='ij')
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
    field = ifftn(field_fft, axes=(0, 1, 2),norm="forward").real

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

def generate_anisotropic_powerlaw_field(N:      int,
                                        alpha:  float = 5./3.,
                                        beta:   float = 5./3.,
                                        L:      float = 1.0) -> np.ndarray:
    """
    This computes a random field with a power-law power spectrum. The power spectrum
    is P(k) = k_perp^-alpha k_parallel^-beta. The field is generated in Fourier space, 
    and then inverse transformed to real space.
    
    Author: James Beattie

    Args:
        N (int): the linear dimension of the 3D field
        alpha (float): the negative 1D power-law exponent used in Fourier space for the 
                        perpendicular component. Defaults to 5/3.
                        Note that I make the transformations between 3D Fourier transform 
                        exponent and 1D power spectrum exponent in the code.
        beta (float): the negative 1D power-law exponent used in Fourier space for the 
                        parallel component. Defaults to 5/3.
        L (float): the physical size of the domain. Defaults to 1.0.

    Returns:
        ifft field (np.ndarray): the inverse fft of the random field with a power-law power spectrum
    """
    
    # Create a grid of frequencies
    kx = np.fft.fftfreq(N)
    ky = np.fft.fftfreq(N)
    kz = np.fft.rfftfreq(N)   
    
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Calculate the magnitude of k for each frequency component
    k_perp              = np.sqrt(kx**2 + ky**2)
    k_par               = np.abs(kz)
    k_perp[k_perp==0]   = np.inf   # Avoid division by zero
    k_par[k_par==0]     = np.inf   # Avoid division by zero
    
    
    # Adjust the amplitude of each frequency component to follow k^-5/3 (-11/3 in 3D)
    amplitude = k_perp**(-(alpha+1.0)/2.0)*k_par**(-beta/2.0)

    # Create a 3D grid of random complex numbers for phases
    rphase = np.exp(2j*np.pi*np.random.rand(*amplitude.shape))
    
    Fhalf = 10*rphase * amplitude
    Fhalf[0,0,0] = 0.0  # Set the zero mode to zero to avoid DC component
    
    # build full cube via Hermitian symmetry
    F = np.zeros((N, N, N), dtype=complex)
    F[:, :, :N//2+1] = Fhalf
    F[:, :, N//2+1:] = np.conj(Fhalf[:, :, 1:N//2][..., ::-1])

    return np.fft.ifftn(F,norm="forward").real

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
    vector_field_FFT = fftn(vector_field,
                            norm='forward',
                            axes=(1,2,3))

    # Assuming a cubic domain    
    N = vector_field.shape[1]  
            
    # wave vectors
    kx = 2 * np.pi * fftfreq(N, d=L/N) / L
    ky = 2 * np.pi * fftfreq(N, d=L/N) / L
    kz = 2 * np.pi * fftfreq(N, d=L/N) / L
    
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
    return ifftn(shell_k,
                 axes=(1,2,3),
                 norm="forward").real


def extract_shell_X_2D(vector_field    : np.ndarray,
                       k_minus_dk      : float,
                       k_plus_dk       : float,
                       L               : list = [1.0,1.0] ) -> np.ndarray:
    """
    Extract and return the inverse FFT of a specific shell of a vector field.

    This method extracts the components of a vector field that fall within a 
    specified wavenumber shell, defined by the range `k_minus_dk < k < k_plus_dk`. 
    The shell is selected based on the direction specified in `self.direction` 
    ('parallel', 'perp', or 'iso'), and the inverse FFT of the filtered shell 
    is computed and returned.
    
    Author: James Beattie & Anne Noer Kolborg

    Args:
        vector_field (np.ndarray): The input vector field to be filtered and 
                                transformed. It is assumed to be a 3D field.
        k_minus_dk (float): The lower bound of the wavenumber shell.
        k_plus_dk (float): The upper bound of the wavenumber shell.

    Returns:
        np.ndarray: The inverse FFT of the filtered vector field, limited to 
                    the specified wavenumber shell.

    Raises:
        Assertion error: If the input vector field is not 3D.
        ValueError: If `self.direction` is not recognized. Valid options are: 
                    'parallel', 'perp', 'iso'.

    Workflow:
        1. The method first determines the type of filter to apply based on 
        `self.direction`. The filter type is either 'parallel', 'perp', 
        or 'iso', corresponding to different wavenumber components.
        2. It then creates a mask using the filter, selecting the wavenumbers 
        that fall within the specified range (`k_minus_dk` to `k_plus_dk`).
        3. The vector field is transformed into Fourier space using `fftn`.
        4. The mask is applied to isolate the desired wavenumber components.
        5. The inverse FFT (`ifftn`) of the masked field is computed and returned.

    Example:
        # Example usage to extract a shell and compute the inverse FFT:
        filtered_field = self.extract_shell_X(vector_field, 0.5, 1.5)

    Notes:
        - The method assumes that `self.kx`, `self.ky`, and `self.kz` have been 
        initialized and correspond to the wavenumbers of the grid.
        - The extracted shell is in the form of a 3D numpy array, and the output 
        is also a 3D numpy array representing the spatial domain.

    References:
        This method is based on the transfer function code by Philip Grete:
        https://github.com/pgrete/energy-transfer-analysis
    """
    
    #assert np.shape(vector_field)[0] == 2, "Error: Vector field must be 2D."
    
    N = vector_field.shape
    kx = 2 * np.pi * fftfreq(N, d=L[0]/N[0]) / L[0]
    ky = 2 * np.pi * fftfreq(N, d=L[1]/N[1]) / L[1]
    
    def filter(kmin, kmax, kx, ky, filter_type):
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        # Define filter types
        filters = {
            'twod': np.sqrt(kx**2 + ky**2),
        }
        # Calculate the filter
        k_filter = filters[filter_type]
        mask = np.logical_and(k_filter >= kmin, k_filter <= kmax)
        return np.array([mask.astype(float), mask.astype(float)])

    # Inverse FFT with just the wavenumbers from the shell 
    return np.real(
        ifftn(
            filter(k_minus_dk, 
                   k_plus_dk, 
                   kx, 
                   ky, 
                   'twod') * fftn(
                vector_field,
                norm='forward',
                axes=(1, 2)),
            axes=(1, 2),
            norm="forward"))
