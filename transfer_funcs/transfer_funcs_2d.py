import numpy as np 
import os 
import gc
from PLASMAtools.aux_funcs.derived_var_funcs import DerivedVars
from PLASMAtools.aux_funcs.derivatives import Derivative
from PLASMAtools.read_funcs.read import Fields
from joblib import Parallel, delayed
import argparse
import h5py
import matplotlib.pyplot as plt
import cmasher as cmr

# Command line arguments 
################################################################################################

ap      = argparse.ArgumentParser(description = 'Input arguments')
ap.add_argument('-f',
                '--file_name',
                required=False,
                default="EXTREME_Turb_slice_xy_0005000",
                help='the file name',
                type=str)
ap.add_argument('-i',
                '--read_path',
                required=False,
                default=".",
                help='the path to the file',
                type=str)
ap.add_argument('-o',
                '--write_path',
                required=False,
                default="./shell_transfers",
                help='the path to write the results',
                type=str)
ap.add_argument('-nb',
                '--n_bins',
                default=-1,
                required=False,
                help='the number of bins for the transfer function',
                type=str)
ap.add_argument('-d',
                '--direction',
                required=False,
                default="iso",
                help='the direction of the transfer function relative to the potental',
                type=str)
ap.add_argument('-n',
                '--n_cores',
                required=False,
                default=12,
                help='the number of compute cores to parallelise the shell loop over',
                type=int)
ap.add_argument('-L',
                '--L',
                required=False,
                default=1.0,
                help='the characteristic length scale of the simulation',
                type=float)
ap.add_argument('-downsample',
                '--downsample',
                required=False,
                default=5,
                help='downsample by this factor',
                type=float)
ap.add_argument('-t',
                '--transfer',
                required=False,
                default="mag",
                help='the type of transfer to compute',
                type=str)
ap.add_argument('-debug',
                action="store_true",
                help="to debug or not to debug, that's the question.")
args    = vars(ap.parse_args())

################################################################################################

class TransferFunction():
    """
    The TransferFunction class is used to compute the transfer of energy 
    flux function for a set of two-dimensional fields.
    
    
    Author: James Beattie

    """
    def __init__(self, 
                file_name   : str = "Turb_plt_hdf5_0010",
                read_path   : str = ".",
                write_path  : str = "./shell_transfers",
                n_bins      : int = -1,
                n_cores     : int = 1,
                transfer    : str = "mag",
                L           : float = 1.0,
                block_size  : int = 5) -> None:
        """
        Initialize the transfer analysis class with the necessary parameters.

        This class handles the setup for calculating energy transfer between 
        different fields (e.g., magnetic, velocity) in a fluid simulation. It 
        reads the necessary data fields from files, sets up the wavenumber grid, 
        and prepares the bins for subsequent calculations.

        Args:
            file_name (str): The name of the file containing the simulation data 
                            to be read (default: "Turb_plt_hdf5_0010").
            read_path (str): The directory path where the simulation data file is 
                            located (default: ".").
            write_path (str): The directory path where the results of the energy 
                            transfer calculations will be saved 
                            (default: "./shell_transfers").
            n_bins (int): The number of bins to create for the energy transfer 
                        calculations. If set to -1, logarithmic bins will be 
                        created (default: -1).
            n_cores (int): The number of CPU cores to use for parallel processing 
                        (default: 1).
            transfer (str): The type of transfer to compute, such as 'mag' for 
                            magnetic or 'vel' for velocity (default: "mag").
            L (float): The characteristic length scale for the simulation domain 
                    (default: 1.0).

        Attributes:
            field_lookup (dict): A dictionary mapping transfer types to the fields 
                                required for those calculations.
            transfer_lookup (dict): A dictionary mapping transfer types to the 
                                    specific components of the energy transfer 
                                    that will be calculated.
            mag (np.ndarray): The magnetic field data, loaded from the file.
            vel (np.ndarray): The velocity field data, loaded from the file.
            dens (np.ndarray): The density field data, loaded from the file 
                            (only for 'vel' transfer).
            pressure (np.ndarray): The pressure field data, loaded from the file 
                                (only for 'vel' transfer).
            transfer (str): The type of transfer being computed.
            file_name (str): The name of the file containing the simulation data.
            read_path (str): The directory path where the simulation data file is 
                            located.
            write_path (str): The directory path where the results will be saved.
            n_bins (int): The number of bins to create for the transfer calculation.
            L (float): The characteristic length scale for the simulation domain.
            direction (str): The direction for binning ('iso', 'perp', 'parallel').
            n_cores (int): The number of CPU cores to use for parallel processing.
            coords (int): The number of spatial coordinates in the velocity field 
                        (assuming a 3D field).
            nx (int): The number of grid points in the x direction.
            ny (int): The number of grid points in the y direction.
            nz (int): The number of grid points in the z direction.
            kx (np.ndarray): The wavenumbers in the x direction, computed based 
                            on the grid size and length scale.
            ky (np.ndarray): The wavenumbers in the y direction, computed based 
                            on the grid size and length scale.
            kz (np.ndarray): The wavenumbers in the z direction, computed based 
                            on the grid size and length scale.
            dvf (DerivedVars): An instance of a helper class for vector and tensor 
                            operations, used in the transfer calculations.
            K_bins (list): The list of bin edges in the K direction, set up based 
                        on the input parameters.
            Q_bins (list): The list of bin edges in the Q direction, set up based 
                        on the input parameters.

        Methods:
            logarithmic_bins: Creates logarithmic bins if `n_bins` is set to -1.
            create_bins: Creates custom bins based on the direction and number of 
                        bins specified.

        Raises:
            AssertionError: If required fields (e.g., magnetic, velocity) are not 
                            loaded correctly.
            ValueError: If an invalid direction or transfer type is specified.
        """
        # Lookup dictionaries for fields and transfer components
        self.field_lookup = {
            "mag": ["mag", "vel"],
            "vel": ["vel", "dens"],
            "vel_helmholtz": ["vel"]
        }
        
        self.transfer_lookup = {
            "mag": ["Tbb_a", "Tbb_c"],#, "Tub_t", "Tub_p", "Tbu_t", "Tbu_p"],
            "vel": ["Tuu_a", "Tuu_c"]#, "Tuv_t", "Tuv_p", "Tvu_t", "Tvu_p"]
        }
        
        # Initialize fields to None; they will be loaded as needed
        self.mag        = None
        self.vel        = None
        self.dens       = None
        self.pressure   = None
        
        # Store parameters
        self.transfer   = transfer
        self.file_name  = file_name
        self.read_path  = read_path
        self.write_path = write_path
        self.n_bins     = n_bins
        self.L          = L
        self.n_cores    = n_cores
        self.direction  = "twod" 
        self.block_size = block_size
        
        # Initialize transfer counters
        self.num_of_transfers = 0
        self.transfer_counter = 0
        
        # Load the required fields based on the transfer type


        # Read velocity components (in quasi 3D)
        # vel_3d = np.array([self.load_component("mag","x"),
        #                    self.load_component("mag","y"),
        #                    self.load_component("mag","z")])
        
        # Read density slice (in 2D)
        magx = self.block_reduce(h5py.File(f"{self.read_path}{self.file_name}_magx.h5", "r")["magx_slice"][:, :])
        magy = self.block_reduce(h5py.File(f"{self.read_path}{self.file_name}_magy.h5", "r")["magy_slice"][:, :])
        #magz = self.block_reduce(h5py.File(f"{self.read_path}{self.file_name}_magz.h5", "r")["magz_slice"][:, :])
        velx = self.block_reduce(h5py.File(f"{self.read_path}{self.file_name}_velx.h5", "r")["velx_slice"][:, :])
        vely = self.block_reduce(h5py.File(f"{self.read_path}{self.file_name}_vely.h5", "r")["vely_slice"][:, :])
        dens_2d = self.block_reduce(h5py.File(f"{self.read_path}{self.file_name}_dens.h5", "r")["dens_slice"][:, :])
        
        vel_2d = np.sqrt(dens_2d)*np.array([velx,vely])
        mag_2d = np.array([magx,magy])

        setattr(self, "mag", mag_2d)
        setattr(self, "vel", vel_2d)
        setattr(self, "dens", dens_2d)
    
        # Set up the wavenumbers and the grid dimensions
        self.coords, self.nx, self.ny = self.vel.shape if self.vel is not None else (None, 0, 0, 0)
        self.kx = np.fft.fftfreq(self.nx) * self.nx / self.L[0]
        self.ky = np.fft.fftfreq(self.ny) * self.ny / self.L[1]
        
        self.dvf = DerivedVars(L=self.L,
                               num_of_dims=2,
                               bcs="00")
        
        # Set up the bins (logarithmic or custom)
        self.K_bins, self.Q_bins = [], []
        if self.n_bins == -1:
            self.logarithmic_bins()
        else:
            self.create_bins()
            
        print(f"TransferFunction: Initialized with {self.n_bins} logarithmic bins.")
            
    def load_component(self, 
                       field, 
                       component):
        # Helper function to load velocity slices for a given component
        file_suffixes = ["_minus_dx", "", "_plus_dx"]
        slices = [
            self.block_reduce(h5py.File(f"{self.read_path}{self.file_name}_{field}{component}{suffix}.h5", "r")[f"{field}{component}_slice"][:, :])
            for suffix in file_suffixes
        ]
        print("downsampled to: ", slices[0].shape)
        return np.transpose(np.array(slices), (1, 2, 0))
    
    def block_reduce(self,
                     field):
        return field.reshape(
            field.shape[0] // self.block_size, self.block_size,
            field.shape[1] // self.block_size, self.block_size
        ).mean(axis=(1, 3))
        
    def logarithmic_bins(self) -> None:
        """
        Create logarithmic bins for the transfer analysis based on grid resolution.

        This method calculates logarithmic bins to be used for binning the transfer 
        functions. The number of bins is determined by the resolution of the grid (`self.nx`) 
        and is based on an approach used in Philip Grete's transfer function code. The 
        calculated bins are stored in `self.K_bins` and `self.Q_bins`, and the number of bins 
        is updated in `self.n_bins`.

        Workflow:
            1. The method calculates an exponent (`resolution_exp`) based on the grid 
            resolution (`self.nx`) to determine the range and number of bins.
            2. Logarithmic bins are created using this exponent, starting from 1 and scaling 
            by powers of 2. These bins are stored in `self.K_bins` and `self.Q_bins`.
            3. The number of bins (`self.n_bins`) is updated to reflect the number of bins 
            created, which is the length of `self.K_bins` minus 1.

        Returns:
            None: The method does not return any value. Instead, it updates the attributes 
                `self.K_bins`, `self.Q_bins`, and `self.n_bins`.

        Example:
            # Example usage to create logarithmic bins:
            self.logarithmic_bins()

        Notes:
            - The method is based on the transfer function code by Philip Grete:
            https://github.com/pgrete/energy-transfer-analysis.
            - The calculated bins are suitable for logarithmic binning in energy transfer 
            analysis, particularly in simulations with grid-based data.

        Side Effects:
            - The bins are stored in `self.K_bins` and `self.Q_bins` as numpy arrays.
            - The number of bins is updated in `self.n_bins`.
        """
        
        resolution_exp  = np.log(self.nx/8)/np.log(2) * 4 + 1
        self.K_bins     = np.concatenate((np.array([1.]), 4.* 2** ((np.arange(0,resolution_exp + 1) - 1.) /4.)))
        self.Q_bins     = self.K_bins.copy() 
        self.n_bins     = len(self.K_bins) - 1
        
        
    def create_bins(self) -> None:
        """
        Create logarithmic bins for the transfer analysis based on the specified direction.

        This method computes the wavenumber magnitudes (`kmin` and `kmax`) for the specified 
        direction ('iso', 'perp', or 'parallel') and uses these values to create logarithmic 
        bins. The bins are stored in `self.K_bins` and `self.Q_bins`, which will be used 
        in the transfer function calculations.

        Raises:
            ValueError: If `self.direction` is not recognized. Valid options are: 
                        'iso', 'perp', 'parallel'.

        Workflow:
            1. The method defines a dictionary `k_functions` that maps the direction to 
            the corresponding wavenumber calculation:
            - 'iso': Computes the isotropic wavenumber magnitude (sqrt(kx^2 + ky^2 + kz^2)).
            - 'perp': Computes the perpendicular wavenumber magnitude (sqrt(kx^2 + ky^2)).
            - 'parallel': Computes the parallel wavenumber magnitude (abs(kz)).
            2. It checks if `self.direction` is valid. If not, a `ValueError` is raised.
            3. The wavenumber magnitudes (`kmin` and `kmax`) are computed based on the selected 
            direction.
            4. The method creates logarithmic bins using `np.logspace` with the calculated `kmin` 
            and `kmax` values. These bins are stored in `self.K_bins` and `self.Q_bins`.

        Example:
            # Example usage to create bins for the current direction:
            self.create_bins()

        Notes:
            - The method assumes that the wavenumber arrays (`self.kx`, `self.ky`, `self.kz`) 
            have been initialized.
            - The number of bins is determined by `self.n_bins`, and the bins are created using 
            the natural logarithm (base `np.e`).

        Side Effects:
            - The bins are stored in `self.K_bins` and `self.Q_bins` as numpy arrays.

        """

        # Dictionary to map direction to the appropriate kmin and kmax calculation
        k_functions = {
            'twod'      : lambda: np.sqrt(self.kx**2 + self.ky**2),
        }
        
        # Compute kmin and kmax based on the direction
        k_values = k_functions['twod']()
        kmin     = 1.0
        kmax     = np.max(k_values)

        # Create the bins
        self.K_bins = np.logspace(np.log(kmin), 
                                np.log(kmax), 
                                self.n_bins,  
                                base=np.e)
        self.Q_bins = self.K_bins.copy()

        
    def extract_shell_X(self,
                        vector_field : np.ndarray,
                        k_minus_dk   : float,
                        k_plus_dk    : float ) -> np.ndarray:
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

        def create_filter(kmin, kmax, filter_type):
            kx, ky = np.meshgrid(self.kx, self.ky, indexing='ij')
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
            np.fft.ifftn(
                create_filter(k_minus_dk, k_plus_dk, 'twod') * np.fft.fftn(
                    vector_field,
                    norm='forward',
                    axes=(1, 2)),
                axes=(1, 2),
                norm="forward"))


    def CalcVelTransfer(self, 
                        idx_K : int, 
                        idx_Q : int) -> None:
        """
        Calculate the kinetic energy transfer between different shells in the velocity 
        and magnetic fields for a given pair of indices (K, Q).

        This method extracts the relevant shells of the magnetic and velocity fields 
        based on the provided indices, computes the gradient tensors, and then 
        calculates various components of the magnetic energy transfer, including 
        advection, compression, tension, and pressure terms. The results are saved 
        to a file for further analysis.
        
        Author: James Beattie

        Args:
            idx_K (int): The index for the K bin.
            idx_Q (int): The index for the Q bin.

        Raises:
            AssertionError: If the magnetic or velocity fields are not loaded.
            ValueError: If an invalid transfer direction is provided.

        Notes:
            - The method assumes that the magnetic field (`self.mag`) and velocity 
            field (`self.vel`) are already loaded into the object.
            - The computation of transfer terms is based on the gradient tensor of the 
            extracted shells and the dot products and contractions defined in the 
            `dvf` (likely a helper class for vector and tensor field operations).
            - The method saves the computed transfer terms to a text file in a 
            directory specified by `self.write_path` and `self.file_name`.
            - This function is based on the energy transfer analysis framework and 
            can be used for analyzing magnetic energy transfer in simulations.

        Example:
            # Example call to the function:
            transfer_result = self.CalcMagTransfer(idx_K=0, idx_Q=1)
            
            # The result will be saved to a file named:
            # {self.write_path}/{self.file_name}/{self.transfer}_{self.direction}_Kbin_0_Qbin_1.txt
        """
        
        assert self.vel is not None, "Velocity field not loaded."
        
        print(f"CalcVelTransfer: computing transfer: {idx_K}, {idx_Q}")
        
        if args["debug"]:
            print("CalcVelTransfer: extracting shells.")
        # extract the shells
        u_K_2d       = self.extract_shell_X(self.vel, self.K_bins[idx_K], self.K_bins[idx_K + 1])
        #u_Q_minus_dx = self.extract_shell_X(self.vel[:,:,:,idx_2d-1], self.Q_bins[idx_Q], self.Q_bins[idx_Q + 1])        
        u_Q_2d       = self.extract_shell_X(self.vel, self.Q_bins[idx_Q], self.Q_bins[idx_Q + 1])
        #u_Q_plus_dx  = self.extract_shell_X(self.vel[:,:,:,idx_2d+1], self.Q_bins[idx_Q], self.Q_bins[idx_Q + 1])
        #u_Q_3d       = np.transpose(np.array([2*u_Q_plus_dx, u_Q_2d, 2*u_Q_minus_dx]), (1, 2, 3, 0))
            
        # Compute the flux
        if args["debug"]:
            print("CalcVelTransfer: computing gradient tensor.")

        d = Derivative()
        grad_f_Q = np.array([[d.gradient(u_Q_2d[0,:,:],1),d.gradient(u_Q_2d[1,:,:],1)],
                             [d.gradient(u_Q_2d[0,:,:],0),d.gradient(u_Q_2d[1,:,:],0)]])
        
        # f,ax = plt.subplots(2,2,dpi=150,figsize=(11.5,11.5))
        # ax[0,0].imshow(np.sum(u_K_2d,axis=0)/np.std(u_K_2d),cmap='cmr.gothic',vmax=2,vmin=0)
        # ax[0,1].imshow(np.sum(u_Q_2d,axis=0)/np.std(u_Q_2d),cmap='cmr.gothic',vmax=2,vmin=0)
        # ax[1,0].imshow(np.einsum("ij...,ij...->...",grad_f_Q,grad_f_Q)/np.std(np.einsum("ij...,ij...->...",grad_f_Q,grad_f_Q)),cmap='cmr.gothic',vmax=2,vmin=0)
        # ax[1,1].imshow(self.dvf.vector_dot_product(u_K_2d, 
        #                                                  self.dvf.vector_dot_tensor_i_ij(self.vel,
        #                                                                                  np.einsum("ij...->ji...",grad_f_Q))),
        #                cmap='cmr.gothic')
        # plt.savefig(f"./test_plots/fil_{idx_K}_{idx_Q}.png")
        # plt.close()

        # Compute the flux terms and store them in a dictionary
        transfer_terms = {
            'Tuu_a': -np.sum(self.dvf.vector_dot_product(u_K_2d, 
                                                         self.dvf.vector_dot_tensor_i_ij(self.vel,grad_f_Q 
                                                                                         )
                                                         )
                             ),
            'Tuu_c': -np.sum(0.5*self.dvf.vector_dot_product(u_K_2d, u_Q_2d) * (d.gradient(self.vel[0,:,:],1) + d.gradient(self.vel[1,:,:],0))),
        }
        
        # clean up
        del u_K_2d, u_Q_2d, d, grad_f_Q #u_Q_minus_dx, u_Q_plus_dx, u_Q_3d
        gc.collect()
        
        if args["debug"]:
            print("CalcVelTransfer: saving results.")
            
        # write every pair to disk 
        header = f"Kbin, Qbin, {', '.join(transfer_terms.keys())}"
        value_line = f"{idx_K}, {idx_Q}, {', '.join(f'{value}' for value in transfer_terms.values())}"
        outfilename = f"{self.write_path}/{self.file_name}/{self.transfer}_{self.direction}_Kbin_{str(idx_K)}_Qbin_{str(idx_Q)}.txt"
        with open(outfilename, 'w') as file:
            file.write(f"{header}\n")
            file.write(f'{value_line}')
            
        self.transfer_counter += 1
        if self.transfer_counter % 20 == 0:
            print(f"CalcVelTransfer: Number of bin pairs left to compute: {self.transfer_counter/self.num_of_transfers}")
        
                
    def CalcMagTransfer(self, 
                        idx_K : int, 
                        idx_Q : int) -> None:
        """
        Calculate the magnetic transfer between different shells in the velocity 
        and magnetic fields for a given pair of indices (K, Q).

        This method extracts the relevant shells of the magnetic and velocity fields 
        based on the provided indices, computes the gradient tensors, and then 
        calculates various components of the magnetic energy transfer, including 
        advection, compression, tension, and pressure terms. The results are saved 
        to a file for further analysis.
        
        Author: James Beattie

        Args:
            idx_K (int): The index for the K bin.
            idx_Q (int): The index for the Q bin.

        Raises:
            AssertionError: If the magnetic or velocity fields are not loaded.
            ValueError: If an invalid transfer direction is provided.

        Notes:
            - The method assumes that the magnetic field (`self.mag`) and velocity 
            field (`self.vel`) are already loaded into the object.
            - The computation of transfer terms is based on the gradient tensor of the 
            extracted shells and the dot products and contractions defined in the 
            `dvf` (likely a helper class for vector and tensor field operations).
            - The method saves the computed transfer terms to a text file in a 
            directory specified by `self.write_path` and `self.file_name`.
            - This function is based on the energy transfer analysis framework and 
            can be used for analyzing magnetic energy transfer in simulations.

        Example:
            # Example call to the function:
            transfer_result = self.CalcMagTransfer(idx_K=0, idx_Q=1)
            
            # The result will be saved to a file named:
            # {self.write_path}/{self.file_name}/{self.transfer}_{self.direction}_Kbin_0_Qbin_1.txt
        """
        
        assert self.mag is not None, "Magnetic field not loaded."
        
        print(f"CalcMagTransfer: computing transfer: {idx_K}, {idx_Q}")
        
        if args["debug"]:
            print("CalcMagTransfer: extracting shells.")
        # extract the shells
        b_K_2d = self.extract_shell_X(self.mag, self.K_bins[idx_K], self.K_bins[idx_K + 1])
        b_Q_2d = self.extract_shell_X(self.mag, self.Q_bins[idx_Q], self.Q_bins[idx_Q + 1])  
        
        d = Derivative()
        grad_f_Q = np.array([[d.gradient(b_Q_2d[0,:,:],1),d.gradient(b_Q_2d[1,:,:],1)],
                             [d.gradient(b_Q_2d[0,:,:],0),d.gradient(b_Q_2d[1,:,:],0)]])
        
        # Compute the flux
        if args["debug"]:
            print("CalcMagTransfer: computing gradient tensor.")
        #grad_b_Q = self.dvf.gradient_tensor(b_Q)

        # Compute the flux terms and store them in a dictionary
        transfer_terms = {
            'Tbb_a': -np.sum(self.dvf.vector_dot_product(b_K_2d, 
                                                         self.dvf.vector_dot_tensor_i_ij(self.vel,grad_f_Q 
                                                                                         )
                                                         )
                             ),
            'Tbb_c': -np.sum(0.5*self.dvf.vector_dot_product(b_K_2d, b_Q_2d) * (d.gradient(self.vel[0,:,:],1) + d.gradient(self.vel[1,:,:],0))),
        }
        
        # clean up
        del b_K_2d, b_Q_2d, grad_f_Q
        gc.collect()
        
        if args["debug"]:
            print("CalcMagTransfer: saving results.")
            
        # write every pair to disk 
        header = f"Kbin, Qbin, {', '.join(transfer_terms.keys())}"
        value_line = f"{idx_K}, {idx_Q}, {', '.join(f'{value}' for value in transfer_terms.values())}"
        outfilename = f"{self.write_path}/{self.file_name}/{self.transfer}_{self.direction}_Kbin_{str(idx_K)}_Qbin_{str(idx_Q)}.txt"
        with open(outfilename, 'w') as file:
            file.write(f"{header}\n")
            file.write(f'{value_line}')
            
        self.transfer_counter += 1
        if self.transfer_counter % 20 == 0:
            print(f"CalcMagTransfer: Number of bin pairs left to compute: {self.transfer_counter/self.num_of_transfers}")
    
    def read_and_save_results(self) -> None:
        """
        Aggregate individual transfer results from files and save the combined results as a .npy file.

        This method reads individual transfer function results from text files in the specified 
        results directory, aggregates them into a single dictionary of numpy arrays, and then saves 
        this aggregated data as a .npy file. Each numpy array in the dictionary corresponds to a 
        specific transfer component (e.g., Tbb_a, Tbb_c) and is indexed by the bin pairs (K, Q).

        Workflow:
            1. The method constructs the path to the results directory and the filename for saving 
            the aggregated results.
            2. It initializes a dictionary `txx_dict` where each key corresponds to a transfer 
            component (e.g., Tbb_a, Tbb_c), and the value is a numpy array with a shape based 
            on the number of bins (`self.n_bins`).
            3. It iterates over all files in the results directory, identifies the relevant files 
            based on the direction (`self.direction`), and reads the data from each file.
            4. For each relevant file, the method extracts the bin indices (idx_K, idx_Q) and 
            updates the corresponding positions in the numpy arrays within `txx_dict`.
            5. After all files have been processed, the method saves the aggregated data dictionary 
            as a .npy file with the specified filename.

        Returns:
            None

        Example:
            # Example usage to read and save the results:
            self.read_and_save_results()

        Notes:
            - The filenames in the results directory are expected to follow a specific pattern where 
            the first part of the filename indicates the direction (`self.direction`). Only files 
            matching this direction are processed.
            - The saved .npy file contains a dictionary where the keys correspond to transfer components 
            (e.g., Tbb_a) and the values are numpy arrays of shape `(self.n_bins, self.n_bins)`.

        Side Effects:
            - The method saves the aggregated transfer function results to a .npy file in the specified 
            `write_path` directory.

        Raises:
            OSError: If there is an issue reading from the directory or saving the .npy file.
        """
        
        results_dir = f"{self.write_path}/{self.file_name}"
        filename = f"{results_dir}/{self.transfer}_transfer_func_{self.direction}_{self.file_name}.npy"
        tf_shape = (self.n_bins, self.n_bins)
        txx_dict = {key: np.zeros(tf_shape) for key in self.transfer_lookup[self.transfer]}
        for file_name in os.listdir(results_dir):
            parts = file_name.split('_')
            if parts[1] == self.direction:
                with open(os.path.join(results_dir, file_name), 'r') as file:
                    lines = file.readlines()
                    values = lines[1].strip().split(',')
                    idx_K = int(values[0])
                    idx_Q = int(values[1])
                    # Update the corresponding TXX arrays from the file's values
                    for i, key in enumerate(self.transfer_lookup[self.transfer]):
                        txx_dict[key][idx_K, idx_Q] = float(values[i + 2])
        # Save the dictionary of TXX arrays
        np.save(filename,
                txx_dict,
                allow_pickle=True)

    def check_for_data(self) -> list:
        """
        Check which bin pairs have already been computed and identify the remaining pairs to process.

        This method scans the output directory for existing computed files and determines 
        which bin pairs (combinations of wavevector magnitudes) have already been processed. 
        It returns a list of bin pairs that still need to be computed.
        
        Author: James Beattie & Anne Noer Kolborg

        Returns:
            list of tuple: A list of tuples, where each tuple represents a bin pair 
                        (idx_K, idx_Q) that has not yet been computed.

        Workflow:
            1. The method checks if the directory specified by `self.write_path` and 
            `self.file_name` exists.
            2. If the directory exists, it lists all files in the directory and extracts 
            the bin pair indices (idx_K, idx_Q) from the filenames.
            3. It compares the set of existing bin pairs with the set of all possible 
            bin pairs (based on `self.n_bins`) to determine which pairs are left to compute.
            4. If the directory does not exist, it creates the directory and prepares 
            to compute all bin pairs.

        Example:
            # Example usage to check for remaining bin pairs:
            remaining_pairs = self.check_for_data()

        Side Effects:
            - If the directory specified by `self.write_path` and `self.file_name` does 
            not exist, the method creates it.

        Note:
            - The filenames in the directory are expected to follow a specific pattern 
            where the bin indices (idx_K, idx_Q) are separated by underscores. The 
            method assumes that the first part of the filename indicates the direction 
            (`self.direction`) and only processes files that match this direction.

        Raises:
            OSError: If there is an issue with creating the directory.
        """
        
        # Indices from the filenames
        # if filename changes, these indices need to be updated
        direction_idx = 1
        K_idx         = 3
        Q_idx         = 5
        
        if os.path.exists(f"{self.write_path}/{self.file_name}"):
            # check which files already exists and set up to skip those transfers 
            existing_files = os.listdir(f'{self.write_path}/{self.file_name}')
            existing_pairs = set()
            for filename in existing_files:
                parts = filename.split('_')
                # We only care about bin pairs with the appropriate filtering 
                if parts[direction_idx] == self.direction:
                    idx_K = int(parts[K_idx])
                    idx_Q = int(parts[Q_idx].split('.')[0])
                    existing_pairs.add((idx_K, idx_Q))
            all_pairs =[(idx_K, idx_Q) for idx_K in range(self.n_bins) for idx_Q in range(self.n_bins)]
            remaining_pairs = list(set(all_pairs) - existing_pairs)
            return remaining_pairs
        else:
            os.mkdir(f"{self.write_path}/{self.file_name}")
            # prepare to compute all transfers 
            return [(idx_K, idx_Q) for idx_K in range(self.n_bins) for idx_Q in range(self.n_bins)]

    def compute_transfers(self) -> None:   
        """
        Compute the energy or magnetic transfer between different shells in a simulation.

        This method determines which bin pairs (combinations of wavevector magnitudes) 
        need to be computed, selects the appropriate transfer function based on the 
        specified transfer type, and then computes the transfers using parallel processing. 
        If all bin pairs have already been computed, the method consolidates the results 
        into a single file and exits.
        
        Author: James Beattie & Anne Noer Kolborg

        Raises:
            NotImplementedError: If the specified transfer function is not implemented.

        Workflow:
            1. The method first checks which bin pairs have not been computed yet 
            by calling `self.check_for_data()`.
            2. It then selects the corresponding transfer function based on `self.transfer` 
            using a dictionary lookup.
            3. If no bin pairs are left to compute, it consolidates the existing results 
            by calling `self.read_and_save_results()` and exits.
            4. If there are bin pairs to compute, it processes them in parallel using 
            the `Parallel` class from the `joblib` library.
            5. After processing, the method consolidates the results into a single file.

        Example:
            # Example usage to compute transfers:
            self.compute_transfers()

        Note:
            - The transfer functions for "vel", "vel_helmholtz", and "mag_helicity" 
            are placeholders and have not been implemented yet.
            - Ensure that the corresponding transfer function is implemented before 
            calling this method with those transfer types.

        Side Effects:
            - The method will save the computed results to files in the specified 
            `write_path` directory.
        """
         
        # Check which bin pairs may have already been calculated
        bin_pairs = self.check_for_data()
        self.num_of_transfers = len(bin_pairs)
        print(f"compute_transfers: number of bin pairs left to compute: {self.num_of_transfers}")

        # Map transfer types to corresponding functions
        transfer_funcs = {
            "mag": self.CalcMagTransfer,
            "vel": self.CalcVelTransfer,            # Placeholder for velocity transfer function
            "vel_helmholtz": None,  # Placeholder for Helmholtz decomposition transfer function
            "mag_helicity": None    # Placeholder for magnetic helicity transfer function
        }

        print(f"compute_transfers: selecting {self.transfer} transfer functions")
        transfer_func = transfer_funcs.get(self.transfer)

        if transfer_func is None:
            raise NotImplementedError(f"Transfer function '{self.transfer}' is not implemented.")

        if not bin_pairs:
            # There are no bin pairs left to compute, consolidate the results and exit
            self.read_and_save_results()
            print("compute_transfers: There are no bin pairs left to compute. Exiting.")
            return

        # Compute the remaining bin pairs using parallel processing
        Parallel(n_jobs=self.n_cores)(delayed(transfer_func)(idx_K, idx_Q) for idx_K, idx_Q in bin_pairs)

        # Consolidate the results after computation
        self.read_and_save_results()

if __name__ == "__main__":
    
    TransferFunction(file_name  = args["file_name"], 
                     read_path  = args["read_path"],
                     write_path = args["write_path"],
                     n_bins     = args["n_bins"],
                     n_cores    = args["n_cores"],
                     transfer   = args["transfer"],
                     L          = args["L"],
                     block_size = args["downsample"]  
                     ).compute_transfers()
    
    

