"""
    Title: Shell transfer functions 
    Authors: James Beattie & Anne Noer Kolborg

    Notes: Only kinetic at the moment. 
"""


import numpy as np 
import os 
from scipy.fft import fftfreq, fftn, ifftn 
from FLASHtools.aux_funcs import derived_var_funcs as dvf
from joblib import Parallel, delayed

def logarithmic_bins(nx):
    resolution_exp = np.log(nx/8)/np.log(2) * 4 + 1
    
    K_bins = np.concatenate((np.array([1.]), 4.* 2** ((np.arange(0,resolution_exp + 1) - 1.) /4.)))
    Q_bins = K_bins
    
    return K_bins, Q_bins

# Function to extract the shells 
def extract_shell_X(vector_field: np.ndarray,
                              k_minus_dk:   float,
                              k_plus_dk:    float,
                              L: float = 1.0, 
                              filter_type : str = 'parallel'):    # The physical size of the domain

    """ 
    
    Extracts shell X-0.5 < k <X+0.5 of a vector field and 
    returns the inverse FFT of the shell. 
    
    Based on Philip Grete's transfer function code:
    https://github.com/pgrete/energy-transfer-analysis
    
    
    Author: James Beattie & Anne Noer Kolborg
        
    """

    def kparallel_filter(kx, ky, kz, kmin, kmax):
    
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing = 'ij')

        tophat = np.zeros(np.shape(kz))
        tophat[np.logical_and(np.abs(kz) >= kmin, np.abs(kz) <= kmax)] = 1.0

        return np.array([tophat, tophat, tophat])
    
    def kperp_filter(kx, ky, kz, kmin, kmax):
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing = 'ij')

        k_ortho = np.sqrt(kx**2 + ky**2)

        cylinder = np.zeros(np.shape(kx))
        cylinder[np.logical_and(k_ortho >= kmin, k_ortho <= kmax)] = 1.0

        return np.array([cylinder, cylinder, cylinder])

    # Take FFT of vector field
    vector_field_FFT = fftn(vector_field,
                            norm='forward',
                            axes=(1,2,3))

    nx, ny, nz = vector_field.shape[1:]
            
    # wave vectors
    kx = fftfreq(nx)*nx/L
    ky = fftfreq(ny)*ny/L
    kz = fftfreq(nz)*nz/L
        
    if filter_type == 'parallel':
        filt = kparallel_filter(kx, ky, kz, k_minus_dk, k_plus_dk)
    elif filter_type == 'orthogonal':
        filt = kperp_filter(kx, ky, kz, k_minus_dk, k_plus_dk)
    else:
        print('Error filter type not recognized, valid options are: parallel or orthogonal')
    shell_k = filt*vector_field_FFT
    
    # Inverse FFT with just the wavenumbers from the shell 
    shell_real = ifftn(shell_k,
                       axes=(1,2,3),
                       norm="forward").real
     
    return shell_real

def CalcTransfer(field, Kbins, Qbins, idx_K, idx_Q, order, direction, savepath):
    U_K = extract_shell_X(field, Kbins[idx_K], Kbins[idx_K + 1], filter_type = direction)
    U_Q = extract_shell_X(field, Qbins[idx_Q], Qbins[idx_Q + 1], filter_type = direction)

    # Compute the flux
    grad_U_Q        = dvf.gradient_tensor(U_Q, order)
    v_dot_grad_U_Q  = np.einsum("i...,ij...-> i...", field, grad_U_Q)
    incomp_term     = np.einsum("i...,i...-> ...", U_K, v_dot_grad_U_Q)

    divU            = dvf.vector_divergence(field, order)
    U_K_dot_U_Q     = np.einsum("i...,i...-> ...",U_K, U_Q)
    comp_term       = 0.5 * U_K_dot_U_Q * divU

    advection = -np.sum(incomp_term) # advection term
    compressive = -np.sum(comp_term) # compressive term

    # write every pair to disk 
    outfilename = savepath + direction + '_' + 'Kbin_' + str(idx_K) + '_Qbin_' + str(idx_Q) + '.txt'
    with open(outfilename, 'w') as file:
        file.write('{}, {}, {}, {}'.format(idx_K, idx_Q, advection, compressive))
    return 1.0 #advection, compressive, idx_K, idx_Q

def tjek_for_data(path, direction, Kbins):
    # check which files already exists and set up to skip those transfers 
    existing_files = os.listdir(path)
    existing_pairs = set()

    for filename in existing_files:
        if '.txt' in filename:
            parts = filename.split('_')
            # We only care about bin pairs with the appropriate filtering 
            if parts[0] == direction:
                idx_K = int(parts[2])
                idx_Q = int(parts[4].split('.')[0])
                existing_pairs.add((idx_K, idx_Q))

    n_bins = len(Kbins) -1
    all_pairs =[(idx_K, idx_Q) for idx_K in range(n_bins) for idx_Q in range(n_bins)]
    remaining_pairs = list(set(all_pairs) - existing_pairs)
    return remaining_pairs

def aggregate_results(path, direction, Kbins, save = True):
    # path to where preliminary data is stored 
    # direction of transfers (parallel or orthogonal)
    # array of bin edges 
    # save data to disk (True or False)

    n_bins = len(Kbins)-1

    UU_shape = (n_bins, n_bins)
    UU = np.zeros(UU_shape)
    UUc = np.zeros(UU_shape)

    for filename in os.listdir(path):
        if '.txt' in filename:
            parts = filename.split('_')
            if parts[0] == direction:
                with open(os.path.join(path, filename), 'r') as file:
                    line = file.readlines()
                    values = line[0].strip().split(',')
                    idx_K = int(values[0])
                    idx_Q = int(values[1])
                    advection_term = float(values[2])
                    compress_term = float(values[3])

                    UU[idx_K, idx_Q] = advection_term
                    UUc[idx_K, idx_Q] = compress_term

    if save:
        filename =  path + direction + '_shell_transfer.npz'
        np.savez(filename, UU=UU, UUc=UUc, Kbins = Kbins, Qbins = Kbins)
    return UU, UUc

def compute_transfers(field, direction, datapath = './', order = 2):
    # 3D field on which to compute the shell to shell transfers 
    # direction to compute transfers (parallel or orthogonal)
    # path to directory where data will be saved (ideally should be dedicated to only these files)
    # order to compute the gradients 

    nx, ny, nz = field.shape[1:]

    Kbins, Qbins = logarithmic_bins(nx)

    # Check which bin pairs may have already been calculated
    bin_pairs = tjek_for_data(datapath, direction, Kbins)

    if not bin_pairs: # there are no bin pairs left to compute so consolidate the results into a single file 
        UU, UUc = aggregate_results(datapath, direction, Kbins)
    else: #still bin pairs to compute, so do that
        Parallel(n_jobs=-1)(delayed(CalcTransfer)(field, Kbins, Qbins, idx_K, idx_Q, order, direction, datapath) for idx_K, idx_Q in bin_pairs)

    # Check for bin pairs again and aggregate the results 
    bin_pairs = tjek_for_data(datapath, direction, Kbins)
    if not bin_pairs: # there are no bin pairs left to compute so consolidate the results into a single file 
        UU, UUc = aggregate_results(datapath, direction, Kbins)

    return UU, UUc, Kbins, Qbins
