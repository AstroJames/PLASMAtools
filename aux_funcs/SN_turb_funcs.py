import numpy as np 
from scipy.fft import fftn, ifftn, fftfreq
from FLASHtools.aux_funcs import derived_var_funcs as dvf

def bin_cylindrical(P_in, kx, ky, kz):
    # set up the kbins:
    kend = np.max(np.abs(kx)) # Highest k value is furtherst to the right after fftshifting
    kmax = np.sqrt(kend**2 + kend**2) # maximum radial kmode
        
    kmin = 0.0
    dk = np.diff(kx)[0]

    k_ortho_bins = np.arange(kmin, kmax + dk, dk)
       
    kmax = np.max(np.abs(kz)) # Maximum height kmode
    k_para_bins = np.arange(kmin, kmax + dk, dk)

    # bin along kz first - this can be a 1d operation 
    whichbin_para = np.digitize(np.abs(kz), k_para_bins, right = True)

    P_temp = np.zeros((len(kx), len(ky), len(k_para_bins)))
    for i in range(0, len(k_para_bins)):
        P_temp[:,:,i] = np.sum(P_in, where = (whichbin_para == i), axis =2)

    # binning in k_ortho has to be 3d, but we can do kz//2 for the 3rd dimension because of the previous binning 
    ## 3d meshgrid of wavenumbers: 
    kx3d, ky3d, _ = np.meshgrid(kx, ky, k_para_bins, indexing = 'ij')
    # calculate the magnitude of k at each point in the 3d domain
    k_ortho = np.sqrt(kx3d**2 + ky3d**2)

    whichbin_ortho = np.digitize(k_ortho, k_ortho_bins, right = False) -1.0
    
    P_binned = np.zeros((len(k_ortho_bins), len(k_para_bins)))

    ## Bin the spectrum
    for i in range(0, len(k_ortho_bins)):
        P_binned[i, :] = np.sum(P_temp, where = (whichbin_ortho == i), axis = (0,1))
    
    return P_binned, k_ortho_bins, k_para_bins
   
def bin_spherical(P_in, kx, ky, kz):
    # set up the kbins:
    kend = np.max(np.abs(kx))
    kmax = np.sqrt(kend**2 + kend**2 + kend**2)
        
    kmin = 0.0
    dk = np.diff(kx)[0]

    kbins = np.arange(kmin, kmax + dk, dk)

    P_binned = np.zeros((len(kbins)))

    ## 3d meshgrid of wavenumbers: 
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing = 'ij')
    # calculate the magnitude of k at each point in the 3d domain
    k = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    # bin in kspace
    whichbin = np.digitize(k, kbins, right = True) -1.0

    # bin the spectrum: 
    for n in range(0, len(kbins)):
        P_binned[n] = np.sum(P_in, where = (whichbin == n))

    return P_binned, kbins