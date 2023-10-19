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

def radial_integrate(data: np.ndarray,
                     bins: int = None) -> tuple:    
    """
    This radial integrate function takes the 3D power spectrum and integrates
    over spherical shells of constant k. The result is a 1D power spectrum.
    
    It has been tested to reproduce the 1D power spectrum of an input 3D Gaussian
    random field.
    
    It has been tested to maintain the correct normalisation of the power spectrum
    i.e., the integral over the spectrum. For small grids (number of bins) the normalisation
    will be off by a small amount (roughly factor 2 for 128^3 with postive power-law indexes). 
    This is because the frequencies beyond the Nyquist limit are not included in the radial 
    integration. This is not a problem for grids of 256^3 or larger, or for k^-a style spectra,
    which are far more commonly encountered.
    
    Args:
        data: The 3D power spectrum
        bins: The number of bins to use for the radial integration. 
              If not specified, the Nyquist limit is used (as should always be the case, anyway).

    Returns:
        k_modes: The k modes corresponding to the radial integration
        radial_sum: The radial integration of the 3D power spectrum (including k^2 correction)
    """
    
    # Compute the radial coordinate for every grid point
    z,y,x = np.indices(data.shape)
    center = np.array([(i-1)/2.0 for i in data.shape])
    r = np.sqrt((x - center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
    r = r.astype(int)

    # If bins aren't specified, choose the Nyquist limit (N//2)
    N = data.shape[0]
    if not bins:
        bins = N//2

    # Compute the radial profile (radial integrate)
    # note a subtly here, the k^2 correction is already in this sum simply
    # because the k mode density scales with k^2dk (dk=1 here)
    radial_sum = np.bincount(r.ravel(), data.ravel(), minlength=bins)

    # Generate the spatial frequencies with dk=1
    k_modes = np.arange(1, bins+1)  # This gives the center of each bin

    return k_modes, radial_sum[:bins]


def generate_powerlaw_field(size:  int,
                            alpha: float = 5./3.) -> np.ndarray:
    """
    This computes a random field with a power-law power spectrum. The power spectrum
    is P(k) = k^-alpha. The field is generated in Fourier space, and then inverse
    transformed to real space.

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