""""
    Title: Derived Variable Functions
    Author: James R. Beattie
    Date: 09/07/2017
    Description: Functions for calculating derived variables in the read_flash classes

""""

import numpy.fft as fft
import numpy as np

def helmholtz_decomposition(F,x):
    # F is a 4D array, with the last dimension being 3 (for the x, y, z components of the vector field)
    shape = F.shape[:-1]
    Fhat = fft.fftn(F, axes=(0, 1, 2),norm = 'forward')
    
    Fhat_irrot = np.zeros_like(Fhat, dtype=np.complex128)
    Fhat_solen = np.zeros_like(Fhat, dtype=np.complex128)
    norm       = np.zeros(shape, dtype=np.float64)
    
    # Compute wave numbers
    kx = fft.fftfreq(shape[0])* 2*np.pi * shape[0] / (x[-1] - x[0])
    ky = fft.fftfreq(shape[1])* 2*np.pi * shape[1] / (x[-1] - x[0])
    kz = fft.fftfreq(shape[2])* 2*np.pi * shape[1] / (x[-1] - x[0])
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Avoid division by zero
    norm = KX**2 + KY**2 + KZ**2
    norm[0, 0, 0] = 1
    
    # Compute divergence and curl in Fourier space (note python doesn't seem to want to use i)
    divFhat = (KX * Fhat[..., 0] + KY * Fhat[..., 1] + KZ * Fhat[..., 2])
    
    # Compute irrotational and solenoidal components in Fourier space
    Fhat_irrot = np.transpose(divFhat * np.array([KX, KY, KZ]) / norm[np.newaxis, ...],(1,2,3,0))
    Fhat_solen = Fhat - Fhat_irrot #curlFhat / norm[np.newaxis, ...]
    
    # Inverse Fourier transform to real space
    F_irrot = fft.ifftn(Fhat_irrot, axes=(0, 1, 2)).real
    F_solen = fft.ifftn(Fhat_solen, axes=(0, 1, 2)).real
    
    # Remove numerical noise
    threshold = 1e-16
    F_solen[np.abs(F_solen) < threshold] = 0
    F_irrot[np.abs(F_irrot) < threshold] = 0
    
    return F_irrot, F_solen
