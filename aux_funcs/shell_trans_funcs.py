from PLASMAtools.read_funcs.read import Fields
import numpy as np 
import os 
from scipy.fft import fftfreq, fftn, ifftn 
from PLASMAtools.aux_funcs.derived_var_funcs import DerivedVars
from joblib import Parallel, delayed
import argparse

# Command line args
################################################################################################
ap = argparse.ArgumentParser(description='command line inputs')
ap.add_argument('-f',
                '--file_name',
                default='flash.par',
                help='the name of the flash parameter file',
                type=str)
ap.add_argument('-i',
                '--data_path',
                default='../',
                help='the data path for the files',
                type=str)
ap.add_argument('-t',
                '--spect_type',
                default='vel',
                help='the type of transfer function',
                type=str)
ap.add_argument('-dir',
                '--direction',
                default='perp',
                help='the direction of the transfer function',
                type=str)
ap.add_argument('-n',
                '--cores',
                default=1,
                help='the number of cores to use to parallelise the computation',
                type=int)
ap.add_argument('-nx',
                '--grid_size',
                default=288,
                help='the grid size of the simulation',
                type=int)
args = vars(ap.parse_args())
################################################################################################


# User defined functions
################################################################################################

def logarithmic_bins(nx : int) -> np.ndarray:
    """
    Description:
        This function computes the logarithmic bins for the shell to shell transfer functions
    
    Author:
        Anne Noer Kolborg
        
    Args:
        nx  : int, the grid size of the simulation
    
    Returns:
        K_bins: np.array, the wavenumber bins
        Q_bins: np.array, the wavenumber bins
        
    """
    
    resolution_exp = np.log(nx/8)/np.log(2) * 4 + 1
    
    K_bins = np.concatenate((np.array([1.]), 4.* 2** ((np.arange(0,resolution_exp + 1) - 1.) /4.)))
    Q_bins = K_bins
    
    return K_bins, Q_bins

# Function to extract the shells 
def extract_shell_X(vector_field: np.ndarray,
                    k_minus_dk  : float,
                    k_plus_dk   : float,
                    L           : float = 1.0, 
                    filter_type : str = 'perp') -> np.ndarray: 
    """ 
    Description:
        Extracts shell X-0.5 < k <X+0.5 of a vector field and 
        returns the inverse FFT of the shell. 
        
        Based on Philip Grete's transfer function code:
        https://github.com/pgrete/energy-transfer-analysis
    
    Author: 
        James Beattie & Anne Noer Kolborg
        
    Args:
        vector_field    : np.ndarray, the vector field
        k_minus_dk      : float, the lower bound of the shell
        k_plus_dk       : float, the upper bound of the shell
        L               : float, the physical size of the domain
        filter_type     : str, the type of filter to apply
        
    """

    def k_par_filter(kx, ky, kz, kmin, kmax):
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing = 'ij')
        tophat = np.zeros(np.shape(kz))
        tophat[np.logical_and(np.abs(kx) >= kmin, np.abs(kx) <= kmax)] = 1.0
        return np.array([tophat, tophat, tophat])
    
    def k_perp_filter(kx, ky, kz, kmin, kmax):
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing = 'ij')
        k_ortho     = np.sqrt(kx**2 + ky**2)
        cylinder    = np.zeros(np.shape(kx))
        cylinder[np.logical_and(k_ortho >= kmin, k_ortho <= kmax)] = 1.0
        return np.array([cylinder, cylinder, cylinder])
    
    def k_filter(kx, ky, kz, kmin, kmax):
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing = 'ij')
        k = np.sqrt(kx**2 + ky**2 + kz**2)
        tophat = np.zeros(np.shape(k))
        tophat[np.logical_and(k >= kmin, k <= kmax)] = 1.0
        return np.array([tophat, tophat, tophat])

    # Take FFT of vector field
    vector_field_FFT = fftn(vector_field,
                            norm='forward',
                            axes=(1,2,3))

    nx, ny, nz = vector_field.shape[1:]
            
    # wave vectors
    kx = fftfreq(nx)*nx/L
    ky = fftfreq(ny)*ny/L
    kz = fftfreq(nz)*nz/L
    
    # Apply the filter for the given direction
    if filter_type == 'par':
        filt = k_par_filter(kx, ky, kz, k_minus_dk, k_plus_dk)
    elif filter_type == 'perp':
        filt = k_perp_filter(kx, ky, kz, k_minus_dk, k_plus_dk)
    elif filter_type == "iso":
        filt = k_filter(kx, ky, kz, k_minus_dk, k_plus_dk)
    else:
        ValueError('Error filter type not recognized, valid options are: parallel or orthogonal')
    shell_k = filt*vector_field_FFT
    # Inverse FFT with just the wavenumbers from the shell 
    shell_real = ifftn(shell_k,
                       axes=(1,2,3),
                       norm="forward").real
    return shell_real


def check_for_data(path      : str, 
                   direction : str, 
                   Kbins     : np.array) -> list:
    """
    Description:
        This function checks which bin pairs have already been computed and 
        returns the remaining pairs to be computed.
    
    Author: 
        Anne Noer Kolborg & James Beattie
    
    Args: 
        path        : str, path to the directory where the files are saved
        direction   : str, the direction of the transfer function
        Kbins       : np.array, the wavenumber bins
        
    Returns:
        remaining_pairs: list, the bin pairs that have not been computed
    
    """
    
    # check which files already exists and set up to skip those transfers 
    existing_files = os.listdir(path)
    existing_pairs = set()
    for filename in existing_files:
        if '.txt' in filename:
            parts = filename.split('_')
            # We only care about bin pairs with the appropriate filtering 
            if parts[1] == direction:
                idx_K = int(parts[3])
                idx_Q = int(parts[5].split('.')[0])
                existing_pairs.add((idx_K, idx_Q))
    n_bins = len(Kbins) -1
    all_pairs = [(idx_K, idx_Q) for idx_K in range(n_bins) for idx_Q in range(n_bins)]
    remaining_pairs = list(set(all_pairs) - existing_pairs)
    return remaining_pairs


def aggregate_results(out_path  : str, 
                      direction : str, 
                      K_bins    : np.ndarray, 
                      save      : bool = True) -> np.ndarray:
    """
    Description:
        This function aggregates the results of the shell to shell transfers
    
    Author: 
        Anne Noer Kolborg & James Beattie
    
    Args:
        path        : str, path to the directory where the files are saved
        direction   : str, the direction of the transfer function
        K_bins      : np.array, the wavenumber bins
        save        : bool, whether to save the results to disk
        
    Returns:
        UU          : np.array, the advection term
        UUc         : np.array, the compressive term
    
    """

    n_bins   = len(K_bins)-1
    UU_shape = (n_bins, n_bins)
    UU       = np.zeros(UU_shape)
    UUc      = np.zeros(UU_shape)

    for filename in os.listdir(out_path):
        if '.txt' in filename:
            parts = filename.split('_')
            if parts[1] == direction:
                with open(os.path.join(out_path, filename), 'r') as file:
                    line                = file.readlines()
                    values              = line[0].strip().split(',')
                    idx_K               = int(values[0])
                    idx_Q               = int(values[1])
                    advection_term      = float(values[2])
                    compress_term       = float(values[3])
                    UU[idx_K, idx_Q]    = advection_term
                    UUc[idx_K, idx_Q]   = compress_term
    if save:
        filename =  out_path + args["spect_type"] + direction + '_shell_transfer.npz'
        np.savez(filename, UU=UU, UUc=UUc, Kbins = K_bins, Qbins = K_bins)
    return UU, UUc


def CalcTransfer(field_1        : np.ndarray,
                 field_2        : np.ndarray,
                 vel            : np.ndarray,
                 transfer_func  : str, 
                 Kbins          : np.ndarray, 
                 Qbins          : np.ndarray, 
                 idx_K          : np.ndarray, 
                 idx_Q          : np.ndarray, 
                 order          : int, 
                 direction      : str = "perp", 
                 savepath       : str = "../") -> None:
    """
    Description:
        This function computes the shell to shell transfer function for a given bin pair
        
    Author: 
        Anne Noer Kolborg & James Beattie
        
    Args:
        field_1         : np.array, the first field
        field_2         : np.array, the second field
        vel             : np.array, the velocity field
        transfer_func   : str, the type of transfer function
        Kbins           : np.array, the wavenumber bins
        Qbins           : np.array, the wavenumber bins
        idx_K           : int, the index of the first bin
        idx_Q           : int, the index of the second bin
        order           : int, the order of the gradient
        direction       : str, the direction of the transfer function
        savepath        : str, the path to save the results
        
    Returns:
        advection   : float, the advection term
        compressive : float, the compressive term
        idx_K       : int, the index of the first bin
        idx_Q       : int, the index of the second bin
    
    """
    
    
    dvf = DerivedVars(L,)
    
    print(f"Computing transfer for Kbin {idx_K} and Qbin {idx_Q}")
        
    # Compute the advective and compressive flux terms
    if transfer_func == "uu" or transfer_func == "bb":
        # field 1 and field 2 are the same field (either velocity uu or magnetic field bb)
        U_K = extract_shell_X(field_1, Kbins[idx_K], Kbins[idx_K + 1], filter_type = direction)
        U_Q = extract_shell_X(field_2, Qbins[idx_Q], Qbins[idx_Q + 1], filter_type = direction)
        
        
        grad_U_Q_s, grad_U_Q_a, grad_U_Q_b = dvf.orthogonal_tensor_decomposition(dvf.gradient_tensor(U_Q),
                                                                                 all=True)
  
        advection = -np.sum(np.einsum("i...,i...-> ...", U_K, 
                                      np.einsum("i...,ij...-> i...", 
                                                vel, 
                                                grad_U_Q_s + grad_U_Q_a)))  # advection term
        compressive = -np.sum(1.0/3.0 * np.einsum("i...,i...-> ...",
                                                  U_K,
                                                  vel) * grad_U_Q_b)        # compressive term
    # compute the cross term (magnetic tension)
    elif transfer_func == "ubt":
        # field 1 = 
        U_K = extract_shell_X(field_1, Kbins[idx_K], Kbins[idx_K + 1], filter_type = direction)
        U_Q = extract_shell_X(field_2, Qbins[idx_Q], Qbins[idx_Q + 1], filter_type = direction)
        
        # compute transfer functions here
        # TODO: implement the magnetic tension term here
        
    elif transfer_func == "ubp":
        
        U_K = extract_shell_X(field_1, Kbins[idx_K], Kbins[idx_K + 1], filter_type = direction)
        U_Q = extract_shell_X(field_2, Qbins[idx_Q], Qbins[idx_Q + 1], filter_type = direction)
        
        # compute transfer functions here
        # TODO: implement the magnetic pressure term here
        
    print("begining to write file")
    # write every pair to disk 
    outfilename = f"{savepath}{transfer_func}_{direction}_Kbin_{idx_K}_Qbin_{idx_Q}.txt"
    with open(outfilename, 'w') as file:
        file.write(f'{idx_K}, {idx_Q}, {advection}, {compressive}')
    return 1.0 #advection, compressive, idx_K, idx_Q


def compute_transfers(field_1       : np.ndarray,
                      field_2       : np.ndarray,
                      vel           : np.ndarray,
                      transfer_func : str,
                      nx            : int, 
                      direction     : str, 
                      n_cores       : int,
                      out_path      : str = "./", 
                      order         : int = 2) -> None :
    """
    Description: 
        This function is the top level function for computing the shell to shell transfer 
        for each bin pair, which it parallelizes using joblib.
        
    Author:
        Anne Noer Kolborg & James Beattie
        
    Args:
        field_1         : np.array, the first field
        field_2         : np.array, the second field
        vel             : np.array, the velocity field
        transfer_func   : str, the type of transfer function
        nx              : int, the grid size of the simulation
        direction       : str, the direction of the transfer function
        n_cores         : int, the number of cores to use
        out_path        : str, the path to save the results
        order           : int, the order of the gradient
        
    Returns:
        UU              : np.array, the advection term
        UUc             : np.array, the compressive term
        Kbins           : np.array, the wavenumber bins
        Qbins           : np.array, the wavenumber bins    
    """
    
    # 3D field on which to compute the shell to shell transfers 
    # direction to compute transfers (parallel or orthogonal)
    # path to directory where data will be saved (ideally should be dedicated to only these files)
    # order to compute the gradients 
    Kbins, Qbins = logarithmic_bins(nx)

    # Check which bin pairs may have already been calculated
    bin_pairs = check_for_data(out_path, direction, Kbins)

    if not bin_pairs: # there are no bin pairs left to compute so consolidate the results into a single file 
        UU, UUc = aggregate_results(out_path, direction, Kbins)
    else: #still bin pairs to compute, so do that
        print("Starting the transfer functions")
        Parallel(n_jobs=n_cores)(delayed(CalcTransfer)(field_1,
                                                       field_2,
                                                       vel,
                                                       transfer_func,
                                                       Kbins,
                                                       Qbins,
                                                       idx_K,
                                                       idx_Q,
                                                       order,
                                                       direction,
                                                       out_path) for idx_K, idx_Q in bin_pairs)

    # Check for bin pairs again and aggregate the results 
    bin_pairs = check_for_data(out_path, direction, Kbins)
    if not bin_pairs: # there are no bin pairs left to compute so consolidate the results into a single file 
        UU, UUc = aggregate_results(out_path, direction, Kbins)

    return UU, UUc, Kbins, Qbins

def read_fields(turb,transfer_func):
    # transform appriorate fields
    if transfer_func == "uu": 
        field_1     = turb.read("vel") 
        field_2     = field_1
        vel         = field_1
    elif transfer_func == "bb":
        field_1     = turb.read("mag")
        field_2     = field_1
        vel         = turb.read("vel")
    elif transfer_func == "ub":
        field_1     = turb.read("vel")
        field_2     = turb.read("mag")
        vel         = field_2/np.sqrt(4*np.pi)
        
    return field_1, field_2, vel


if __name__ == "__main__":
    
    # read in arguments 
    data_path       = args["data_path"]
    file_name       = args["file_name"]
    transfer_func   = args["spect_type"]
    n_grid          = args["grid_size"]
    n_cores         = args["cores"]
    direction       = args["direction"]
    
    # directory name
    dir_name = file_name.split("_")[0] + file_name.split("_")[-1]
    
    # read in the data
    turb = Fields(data_path + file_name,reformat=True)
    
    # read the appropriate fields
    field_1, field_2, vel = read_fields(turb,transfer_func)
    
    # check if there is a directory for the file_name in the current directory
    if not os.path.exists(f"./{dir_name}"):
        os.makedirs(f"{dir_name}")
        
    # compute transfers
    compute_transfers(field_1,
                      field_2,
                      vel,
                      args["spect_type"],
                      n_grid, 
                      direction, 
                      n_cores,
                      out_path = f"./{dir_name}/")
    
    
    """
    Plotting Example:
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import cmasher as cmr

    if __name__ == "__main__":
        direction = "par"
        x_label = {"perp": r"$K_{\perp}$",
                "par": r"$K_{\parallel}$",
                "iso": r"$K$"}
        y_label = {"perp": r"$Q_{\perp}$",
                "par": r"$Q_{\parallel}$",
                "iso": r"$Q$"}
        transfer_label = {"perp" : r"$\mathcal{T}_{bb}(K_{\perp},Q_{\perp})$",
                        "par": r"$\mathcal{T}_{bb}(K_{\parallel},Q_{\parallel})$",
                        "iso": r"$\mathcal{T}_{bb}(K,Q)$"}
        
        file_nums   = [1,5,100,200,300,400,600,700]
        time_labels = [f"$t/t_0 = $ {str(i/10)}" for i in file_nums]
        file_names  = [f"Turb{str(i).zfill(4)}" for i in file_nums]
        
        for file_name, time_label in zip(file_names,time_labels):
            print(f"Plotting {file_name}")
            data = np.load(f"{file_name}/bb{direction}_shell_transfer.npz")
            
            UU = data['UU'] #+ data['UUc']
            UUc = data['UUc']
            Kbins = data['Kbins']
            Qbins = data['Qbins']
            
            K,Q = np.meshgrid(Kbins, Qbins)
            
            min_val = np.min(UU[2:-1,2:-1])
            max_val = np.max(UU[2:-1,2:-1])
            normalized_data = -1 + 2 * (UU - min_val) / (max_val - min_val)
        
            f,ax = plt.subplots(1,1,dpi=150)
            plot = ax.pcolormesh(K,Q,normalized_data, 
                                cmap='cmr.prinsenvlag',
                                norm=colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
            ax.set_xlabel(x_label[direction],
                        fontsize=24)
            ax.set_ylabel(y_label[direction],
                        fontsize=24)
            ax.annotate(transfer_label[direction],
                        xy=(0.05,0.88),
                        xycoords='axes fraction',
                        fontsize=24)
            ax.annotate(time_label,
                        xy=(0.6,0.88),
                        xycoords='axes fraction',
                        fontsize=24)
            cb = f.colorbar(plot, ax=ax)
            cb.ax.set_ylabel(r"$\varepsilon/\varepsilon_0$",
                            fontsize=24)
            cb.ax.set_yscale('linear')
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim([2.85, Kbins[-1]])
            ax.set_ylim([2.85, Kbins[-1]])
            plt.savefig(f"./plots/{file_name}_mag_{direction}_shell_transfer.png")
            plt.close()
    
    """
