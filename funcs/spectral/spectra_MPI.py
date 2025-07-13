"""
MPI-Parallel Power Spectrum Computation using mpi4py-fft

This module provides high-performance parallel computation of power spectra for
3D fields using MPI (Message Passing Interface) and distributed FFTs.

Author:     James R. Beattie
License:    Same as parent project

Overview:
-------------------------------------------------------------------------------------------
For large 3D fields (e.g., 1024³ or larger), computing power spectra becomes
memory and computationally intensive. This module solves this by:
    1. Distributing data across multiple MPI processes
    2. Using parallel FFTs with optimized communication
    3. Computing spectra without gathering full fields

Key Features:
-------------------------------------------------------------------------------------------
    - Distributed memory parallel computation using MPI
    - Efficient pencil decomposition for 3D FFTs
    - Real-to-complex transforms for 2x memory savings
    - Support for multi-component fields (vectors, tensors)
    - Both 3D and 1D isotropic power spectra
    - Optimized memory usage with pre-allocated buffers

Algorithm:
-------------------------------------------------------------------------------------------
    1. Input field is distributed from root to all MPI processes
    2. Each process computes FFT of its local data portion
    3. Power spectrum |FFT|² is computed locally
    4. For 3D output: Results gathered to root process
    5. For 1D output: Spherical binning and reduction to root

Usage:
-------------------------------------------------------------------------------------------
Basic example for computing 1D power spectrum:

    from mpi4py import MPI
    import numpy as np
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Only root process needs the field
    if rank == 0:
        field = np.random.randn(3, 256, 256, 256)  # 3-component field
    else:
        field = None
    
    # All processes participate
    k, Pk = compute_power_spectrum_1D_mpi(field)
    
    # Results only on root
    if rank == 0:
        import matplotlib.pyplot as plt
        plt.loglog(k, Pk)
        plt.show()

Run with: mpirun -n 4 python script.py

Performance Tips:
-------------------------------------------------------------------------------------------
    1. Use power-of-2 grid sizes for optimal FFT performance
    2. Use float32 instead of float64 when precision allows
    3. Increase MPI processes for larger grids (typically N_procs ≤ N/4)
    4. Ensure input arrays are C-contiguous for MPI efficiency

Dependencies:
-------------------------------------------------------------------------------------------
    - numpy: Numerical arrays and operations
    - mpi4py: Python bindings for MPI
    - mpi4py-fft: Distributed FFT implementation
    - pyfftw (optional): Faster FFT backend

"""

import numpy as np
import multiprocessing
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
from PLASMAtools.aux_funcs import power_spectra_funcs as psf

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Still use pyfftw for any local FFT operations
pyfftw_import = False
try:
    import pyfftw
    pyfftw_import = True
    pyfftw.interfaces.cache.enable()
    threads = multiprocessing.cpu_count()
    if rank == 0:
        print(f"power_spectra_funcs: Using {threads} threads for local FFTs with pyfftw on {size} MPI processes.")
except ImportError:
    if rank == 0:
        print(f"pyfftw not installed, using numpy's fft for local operations on {size} MPI processes")

if pyfftw_import:
    # Calculate threads per MPI process to avoid oversubscription
    total_cores = multiprocessing.cpu_count()
    threads_per_process = max(1, total_cores // size)
    pyfftw.config.NUM_THREADS = threads_per_process
    if rank == 0:
        print(f"power_spectra_funcs: Using {threads_per_process} threads per MPI process")
        print(f"(Total cores: {total_cores}, MPI processes: {size})")

## ###############################################################
## Spectral Functions with mpi4py-fft
## ###############################################################

class PowerSpectrum:
    """
    A class to handle distributed 3D power spectrum computations using mpi4py-fft
    with pencil decomposition.
    
    This class implements a distributed algorithm for computing power spectra of large
    3D fields that may not fit in a single machine's memory. It uses MPI to distribute
    the computation across multiple processes, with each process handling a portion
    of the data.
    
    The algorithm:
    1. Distributes the input field from the root process to all MPI processes
    2. Each process computes the FFT of its local portion
    3. The power spectrum |FFT|² is computed locally
    4. Results are gathered or reduced back to the root process
    
    Key features:
    - Uses pencil decomposition for optimal parallel FFT performance
    - Handles real-to-complex (r2c) transforms for efficiency
    - Supports multiple field components (e.g., vector fields)
    - Computes both 3D and 1D isotropic power spectra
    """
    
    def __init__(self, shape, dtype=np.float64, comm=None):
        """
        Initialize the distributed FFT objects and allocate necessary arrays.
        
        This sets up the parallel FFT infrastructure using mpi4py-fft's PFFT class,
        which automatically handles the domain decomposition and communication patterns
        for distributed FFTs.
        
        Args:
            shape (tuple): Shape of the 3D grid (N1, N2, N3). This is the global shape
                          of the full field, not the local portion each process handles.
            dtype (np.dtype): Data type for the arrays (np.float32 or np.float64).
                             Using float32 can significantly reduce memory usage and
                             improve performance for large fields.
            comm (MPI.Comm): MPI communicator (default: MPI.COMM_WORLD). This defines
                            which processes participate in the computation.
        
        Attributes created:
            self.fft: PFFT object that handles distributed FFTs
            self.field_r: Distributed array for real-space field (local portion)
            self.field_c: Distributed array for complex k-space field (local portion)
            self.shape_r: Shape of the local real-space portion on this process
            self.shape_c: Shape of the local complex-space portion on this process
            self.scatter_buffer: Pre-allocated buffer for scatter operations
        """
        
        self.comm = comm or MPI.COMM_WORLD
        self.shape = shape
        self.dtype = dtype
        
        # Create PFFT object for 3D real-to-complex transform
        # This automatically handles pencil decomposition
        self.fft = PFFT(self.comm, 
                        shape, 
                        axes=(0, 1, 2), 
                        dtype=dtype, 
                        collapse=True, 
                        backend='fftw' if pyfftw_import else 'numpy')
        
        # Get local shapes for distributed arrays
        self.shape_r = self.fft.shape(forward_output=False)  # Real space local shape
        self.shape_c = self.fft.shape(forward_output=True)   # Complex space local shape
        
        # Create distributed arrays
        self.field_r = newDistArray(self.fft, forward_output=False)  # Real space
        self.field_c = newDistArray(self.fft, forward_output=True)   # Complex space
        
        # Pre-allocate buffers for scatter operation
        if self.comm.size > 1:
            self.scatter_buffer = np.empty(self.shape_r, dtype=dtype)
        
        if rank == 0:
            print(f"Initialized PFFT with global shape {shape}")
            print(f"Local real shape: {self.shape_r}")
            print(f"Local complex shape: {self.shape_c}")
    
    def compute_power_spectrum_3D(self, field_components):
        """
        Compute power spectrum of a 3D vector field using distributed FFT.
        
        This is the main computational routine that:
        1. Distributes each field component from root to all processes
        2. Computes the FFT of each component in parallel
        3. Calculates |FFT|² and accumulates across components
        4. Applies corrections for Hermitian symmetry
        
        The power spectrum is defined as:
            P(k) = Σᵢ |FFT(field_i)|²
        
        where the sum is over all field components. The result is kept distributed
        across processes for memory efficiency.
        
        Args:
            field_components: Input field data (only needs to be valid on root process).
                Can be one of:
                - np.ndarray with shape (n_components, N1, N2, N3) for vector fields
                - np.ndarray with shape (N1, N2, N3) for scalar fields
                - List of np.ndarray, each with shape (N1, N2, N3)
                - None on non-root processes
        
        Returns:
            power_spectrum (DistArray): 3D power spectrum distributed across processes.
                Shape is (N1, N2, N3//2+1) due to real-to-complex transform symmetry.
                Each process holds a portion of this array.
        
        Algorithm details:
            - Uses real-to-complex (r2c) FFT for efficiency, exploiting Hermitian symmetry
            - The output is multiplied by 2 to account for negative frequencies
            - DC (k=0) and Nyquist components are corrected to avoid double-counting
            - In-place operations are used where possible to minimize memory usage
        """
        
        # Handle input format - only on root process
        if self.comm.rank == 0:
            if field_components is None:
                raise ValueError("Field components cannot be None on root process")
            if isinstance(field_components, np.ndarray):
                if len(field_components.shape) == 4:
                    n_components = field_components.shape[0]
                    # Don't convert to list - work directly with array slices
                else:
                    # Single 3D component
                    n_components = 1
            else:
                n_components = len(field_components)
        else:
            n_components = 0
        
        # Broadcast the number of components
        n_components = self.comm.bcast(n_components, root=0)
        
        # Initialize power spectrum accumulator as a distributed array
        power_spectrum_k = newDistArray(self.fft, forward_output=True)
        power_spectrum_k[:] = 0.0
        
        # Process each field component
        for i in range(n_components):
            # Get component from root process efficiently
            if self.comm.rank == 0:
                if isinstance(field_components, np.ndarray) and len(field_components.shape) == 4:
                    component = field_components[i]  # Direct slice, no copy
                elif isinstance(field_components, np.ndarray):
                    component = field_components
                else:
                    component = field_components[i]
            else:
                component = None
            
            # Distribute the component directly from root to all processes
            self._scatter_field(component, self.field_r)
            
            # Forward FFT: real space -> k-space
            self.field_c = self.fft.forward(self.field_r, self.field_c, normalize=True)
            
            # Add to power spectrum (|F(k)|^2)
            # Use in-place operations for efficiency
            np.abs(self.field_c, out=self.field_c.real)
            np.square(self.field_c.real, out=self.field_c.real)
            power_spectrum_k += self.field_c.real
        
        # Account for Hermitian symmetry to match full k-space
        power_spectrum_k *= 2.0
        
        # Don't double-count DC component (k=0)
        # In distributed arrays, we need to check if we have the DC component
        try:
            # Try to access the DC component
            power_spectrum_k[0, 0, 0] /= 2.0
        except:
            # This process doesn't have the DC component
            pass
        
        # Don't double-count Nyquist frequencies if they exist
        if self.shape[2] % 2 == 0:  # Even size in last dimension
            try:
                # The Nyquist frequency is at the last index in the third dimension
                power_spectrum_k[:, :, -1] /= 2.0
            except:
                # This process doesn't have the Nyquist frequency
                pass
          
        return power_spectrum_k

    def _scatter_field(self, global_field, local_field):
        """
        Scatter a global field from root to distributed array.
        
        This is a critical function that distributes data from the root process
        (which has the full field) to all MPI processes (which each need their
        local portion). It's optimized to minimize memory copies and communication.
        
        The scatter pattern depends on the domain decomposition chosen by mpi4py-fft,
        which typically uses pencil decomposition for 3D FFTs. Each process receives
        a contiguous "pencil" or "slab" of the data.
        
        Args:
            global_field (np.ndarray): Global field array (only valid on root).
                Shape must match self.shape. Can be None on non-root processes.
            local_field (DistArray): Local distributed array to fill.
                Each process will receive its portion of the global field.
        
        Communication pattern:
            Root (rank 0): Has full data → Sends slices to each process
            Other ranks: Send slice info → Receive their data portion
        
        Memory optimization:
            - Uses pre-allocated buffers to avoid repeated allocations
            - Ensures data is contiguous before MPI Send operations
            - Direct assignment for root's local portion (no copy)
        """
        if self.comm.size == 1:
            # Single process - just copy
            if global_field is not None:
                if global_field.shape != self.shape:
                    raise ValueError(f"Global field shape {global_field.shape} doesn't match expected {self.shape}")
                local_field[:] = global_field[:]
            else:
                local_field[:] = 0.0
        else:
            # Multi-process case
            # First, broadcast whether we have valid data
            has_data = global_field is not None if self.comm.rank == 0 else None
            has_data = self.comm.bcast(has_data, root=0)
            
            if not has_data:
                local_field[:] = 0.0
                return
            
            # Get local slice for this process
            local_slice = local_field.local_slice()
            
            # Use pre-allocated buffer
            local_buffer = self.scatter_buffer
            
            # Each process needs to receive its portion
            if self.comm.rank == 0:
                # Validate input
                if global_field.shape != self.shape:
                    raise ValueError(f"Global field shape {global_field.shape} doesn't match expected {self.shape}")
                
                # Send each process its portion
                for dest in range(self.comm.size):
                    if dest == 0:
                        # Local copy for root - direct assignment
                        local_field[:] = global_field[local_slice]
                    else:
                        # Need to figure out what slice to send to each rank
                        # We'll gather slice information first
                        remote_slice = self.comm.recv(source=dest, tag=dest)
                        # Send the appropriate data - use contiguous array
                        data_to_send = np.ascontiguousarray(global_field[remote_slice])
                        self.comm.Send(data_to_send, dest=dest, tag=dest+1000)
            else:
                # Send local slice info to root
                self.comm.send(local_slice, dest=0, tag=self.comm.rank)
                # Receive data from root into pre-allocated buffer
                self.comm.Recv(local_buffer, source=0, tag=self.comm.rank+1000)
                # Copy to distributed array
                local_field[:] = local_buffer
    
    def gather_result(self, local_result):
        """
        Gather distributed result to root process.
        
        Args:
            local_result (DistArray): Local portion of the result
        
        Returns:
            global_result (np.ndarray): Full result (only valid on root)
        """
        # For single process, just return a copy
        if self.comm.size == 1:
            return local_result.copy()
        
        # Use get() method which is the proper way to gather in mpi4py-fft
        # get() requires a global slice argument
        if hasattr(local_result, "get"):
            # Create a slice for the entire array
            full_slice = tuple(slice(None) for _ in range(local_result.ndim))
            
            # Get the full array - this gathers from all processes
            global_array = local_result.get(full_slice)
            
            # Only return on rank 0
            if self.comm.rank == 0:
                return global_array
            else:
                return None
        
        # Fallback for plain ndarray
        return local_result if self.comm.rank == 0 else None
    
    def _get_local_wavenumbers(self, dist_array):
        """
        Get wavenumber arrays for the local portion of distributed array.
        
        In Fourier space, each grid point corresponds to a specific wavenumber k.
        This function computes the k-values for the local portion of data that
        this MPI process handles.
        
        For a grid with N points and spacing dx=1, the wavenumbers are:
            k = 2π * n / N  where n = 0, 1, ..., N/2 for positive frequencies
        
        Since we use real-to-complex transforms, we only store positive kz values.
        The kx and ky can be negative (they span the full range).
        
        Args:
            dist_array (DistArray): Distributed array to get wavenumbers for
        
        Returns:
            tuple: (kx, ky, kz) arrays with shape matching local array
                - kx: Wavenumbers in x-direction (can be negative)
                - ky: Wavenumbers in y-direction (can be negative)  
                - kz: Wavenumbers in z-direction (only positive due to r2c)
        
        Note:
            Results are cached after first computation for efficiency.
            The cache is cleared if the shape changes.
        """
        # Cache wavenumbers if not already done
        if not hasattr(self, '_cached_local_k'):
            # Get local slice information
            local_slice = dist_array.local_slice()
            
            # Global wavenumber arrays
            N1, N2, N3 = self.shape
            
            # Wavenumbers for r2c transform (no fftshift needed)
            kx_global = np.fft.fftfreq(N1, d=1.0) * N1
            ky_global = np.fft.fftfreq(N2, d=1.0) * N2
            kz_global = np.arange(N3//2 + 1)  # Only positive frequencies for r2c
            
            # Extract local portions
            kx_local = kx_global[local_slice[0]]
            ky_local = ky_global[local_slice[1]]
            kz_local = kz_global[local_slice[2]]
            
            # Create 3D arrays - use float32 if input is float32
            k_dtype = np.float32 if self.dtype == np.float32 else np.float64
            kx = kx_local[:, np.newaxis, np.newaxis].astype(k_dtype)
            ky = ky_local[np.newaxis, :, np.newaxis].astype(k_dtype)
            kz = kz_local[np.newaxis, np.newaxis, :].astype(k_dtype)
            
            self._cached_local_k = (kx, ky, kz)
        
        return self._cached_local_k
    
    def compute_isotropic_spectrum_distributed(self, power_spectrum_k):
        """
        Compute 1D isotropic power spectrum from distributed 3D spectrum.
        
        This function performs a spherical average of the 3D power spectrum to obtain
        the 1D isotropic spectrum P(k), where k = |k| is the magnitude of the wavevector.
        
        The algorithm:
        1. Compute |k| for each mode in the local portion
        2. Bin the power into spherical shells using histogram
        3. Reduce (sum) across all MPI processes
        4. Return the binned power spectrum on root
        
        The binning follows the convention:
            - Bins edges from 0.5 to N/2 in steps of 1
            - Each mode is assigned to the nearest bin
            - Excludes k=0 mode from the first bin
        
        Args:
            power_spectrum_k (DistArray): Distributed 3D power spectrum from r2c FFT.
                Each process has a local portion of the full spectrum.
        
        Returns:
            On root process (rank 0):
                k_bins (np.ndarray): Wavenumber bin centers, shape (N//2,)
                power_1d (np.ndarray): 1D isotropic power spectrum, shape (N//2,)
            On other processes:
                (None, None)
        
        Performance notes:
            - Uses numpy.histogram for efficient binning (vectorized)
            - Pre-computes and caches wavenumber arrays
            - Single MPI reduction instead of gathering full 3D array
        """
        # Get wavenumber arrays for this process's local data
        local_k = self._get_local_wavenumbers(power_spectrum_k)
        
        # Compute magnitude of k for each mode - use pre-allocated array
        k_mag = np.sqrt(local_k[0]**2 + local_k[1]**2 + local_k[2]**2)
        
        # Define bins to match spherical_integrate exactly
        N = self.shape[0]
        bins = N // 2
        bin_edges = np.linspace(0.5, bins, bins + 1)
        
        # Local binning
        local_power_sum = np.zeros(bins, dtype=power_spectrum_k.dtype)
        
        # Flatten arrays for more efficient iteration
        k_mag_flat = k_mag.ravel()
        power_flat = power_spectrum_k.ravel()
        
        # Vectorized binning using histogram
        # This is much faster than the triple loop for large arrays
        hist, _ = np.histogram(k_mag_flat, bins=bin_edges, weights=power_flat)
        local_power_sum[:] = hist
        
        # Reduce across all processes
        if self.comm.size > 1:
            global_power_sum = np.zeros_like(local_power_sum)
            self.comm.Reduce(local_power_sum, global_power_sum, op=MPI.SUM, root=0)
        else:
            global_power_sum = local_power_sum
        
        # Compute k_modes and return on rank 0
        if self.comm.rank == 0:
            # Generate k_modes exactly as in spherical_integrate
            k_modes = np.ceil((bin_edges[:-1] + bin_edges[1:]) / 2)
            
            return k_modes, global_power_sum
        else:
            return None, None


def compute_power_spectrum_3D_mpi_pencil(field, shape=None, dtype=np.float64):
    """
    Compute 3D power spectrum using MPI parallel pencil decomposition.
    
    This is the main user-facing function for computing 3D power spectra in parallel.
    It handles the full workflow: initialization, distribution, computation, and gathering.
    
    The function is designed to be called by all MPI processes simultaneously (SPMD model).
    Only the root process needs to provide the input field; other processes can pass None.
    
    Args:
        field (np.ndarray or None): Input field data.
            - On root process: Must be provided with shape (n_components, N1, N2, N3)
              for vector fields or (N1, N2, N3) for scalar fields
            - On other processes: Can be None
            - Must be C-contiguous for optimal MPI performance
        
        shape (tuple, optional): Shape of the 3D grid (N1, N2, N3).
            - If None, inferred from field on root and broadcast to all processes
            - Useful when you want to specify shape explicitly
        
        dtype (np.dtype): Data type for computations (default: np.float64).
            - np.float32: Less memory, faster, sufficient for many applications
            - np.float64: More precision, needed for high-accuracy calculations
    
    Returns:
        power_spectrum (np.ndarray or None): 
            - On root process: 3D power spectrum with shape (N1, N2, N3//2+1)
            - On other processes: None
            - The reduced size in the last dimension is due to Hermitian symmetry
    
    Example:
        >>> # Run with: mpirun -n 4 python script.py
        >>> import numpy as np
        >>> from mpi4py import MPI
        >>> 
        >>> comm = MPI.COMM_WORLD
        >>> rank = comm.Get_rank()
        >>> 
        >>> # Only root creates the field
        >>> if rank == 0:
        >>>     field = np.random.randn(3, 128, 128, 128)  # 3-component field
        >>> else:
        >>>     field = None
        >>> 
        >>> # All processes call the function
        >>> power_spec = compute_power_spectrum_3D_mpi_pencil(field)
        >>> 
        >>> # Only root has the result
        >>> if rank == 0:
        >>>     print(f"Power spectrum shape: {power_spec.shape}")
    """
    
    # Get shape from field if not provided
    if rank == 0:
        if shape is None:
            if field is not None:
                if len(field.shape) == 4:
                    shape = field.shape[1:]
                else:
                    shape = field.shape
            else:
                raise ValueError("Field cannot be None on root process")
    else:
        # Non-root processes don't know the shape yet
        shape = None
    
    # Broadcast shape from rank 0 to all other ranks
    shape = comm.bcast(shape, root=0)

    # Initialize the power spectrum calculator
    ps_calc = PowerSpectrum(shape, dtype=dtype, comm=comm)
    
    # Compute power spectrum
    local_power = ps_calc.compute_power_spectrum_3D(field if rank == 0 else None)
    
    # Gather result to root
    global_power = ps_calc.gather_result(local_power)
    
    return global_power

def compute_power_spectrum_1D_mpi(field, shape=None, dtype=np.float64):
    """
    Compute 1D isotropic power spectrum using MPI parallel processing.
    
    This function computes the spherically averaged (isotropic) power spectrum P(k)
    from a 3D field. It combines the 3D FFT computation with spherical binning to
    produce a 1D spectrum that depends only on the magnitude |k| of the wavevector.
    
    The complete workflow:
    1. Distribute the field across MPI processes
    2. Compute 3D power spectrum in parallel
    3. Bin the 3D spectrum into spherical shells
    4. Reduce the binned results to root process
    
    This is more memory-efficient than computing the full 3D spectrum and then
    binning, as the 3D spectrum never needs to be gathered to a single process.
    
    Args:
        field (np.ndarray or None): Input field data.
            - On root: Shape (n_components, N1, N2, N3) or (N1, N2, N3)
            - On other processes: Can be None
            - Typically represents velocity, magnetic field, or other vector fields
        
        shape (tuple, optional): Shape of 3D grid (N1, N2, N3).
            If None, inferred from field shape on root process.
        
        dtype (np.dtype): Data type for computations (default: np.float64).
            Consider np.float32 for large fields to save memory.
    
    Returns:
        On root process (rank 0):
            k (np.ndarray): Wavenumber bin centers, shape (N//2,).
                Units depend on your grid spacing (typically 2π/L).
            P(k) (np.ndarray): 1D power spectrum, shape (N//2,).
                Units are [field units]² × [volume units].
        
        On other processes:
            (None, None)
    
    Physical interpretation:
        The 1D spectrum P(k) represents the power at wavenumber k, integrated
        over all directions. For turbulence, P(k) ∝ k^(-5/3) in the inertial range.
    
    Example:
        >>> # Generate a turbulent field and compute its spectrum
        >>> if rank == 0:
        >>>     # Create field with k^(-5/3) spectrum
        >>>     field = generate_turbulent_field(256)
        >>> else:
        >>>     field = None
        >>> 
        >>> k, Pk = compute_power_spectrum_1D_mpi(field)
        >>> 
        >>> if rank == 0:
        >>>     plt.loglog(k, Pk)
        >>>     plt.xlabel('k')
        >>>     plt.ylabel('P(k)')
    """
    # Get shape from field if not provided
    if rank == 0:
        if shape is None:
            if field is not None:
                if len(field.shape) == 4:
                    shape = field.shape[1:]
                else:
                    shape = field.shape
            else:
                raise ValueError("Field cannot be None on root process")
    else:
        shape = None
    
    # Broadcast shape
    shape = comm.bcast(shape, root=0)
    
    # Initialize power spectrum calculator
    ps_calc = PowerSpectrum(shape, dtype=dtype, comm=comm)
    
    # Compute distributed 3D power spectrum
    power_3d_dist = ps_calc.compute_power_spectrum_3D(field if rank == 0 else None)
    
    # Compute 1D isotropic spectrum (distributed computation)
    k_bins, power_1d = ps_calc.compute_isotropic_spectrum_distributed(power_3d_dist)
    
    return k_bins, power_1d


# Example usage and testing functions
def example_usage():
    """Example of how to use the pencil decomposition power spectrum"""
    
    # Create test data (only on root)
    if rank == 0:
        N = 64  # Grid size
        n_components = 9  # Number of field components
        
        # Create a test field with some structure
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        y = np.linspace(0, 2*np.pi, N, endpoint=False)
        z = np.linspace(0, 2*np.pi, N, endpoint=False)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        field = np.zeros((n_components, N, N, N))
        # Add some wave modes
        for i in range(n_components):
            field[i] = np.sin(X + i) * np.cos(Y + i) * np.sin(Z + i)
        
        print(f"Created test field with shape {field.shape}")
        print(f"Field statistics: min={np.min(field):.3f}, max={np.max(field):.3f}, mean={np.mean(field):.3f}")
        shape = field.shape[1:]
    else:
        field = None
        shape = None
    
    
    # Compute power spectrum using pencil decomposition
    if rank == 0:
        print("\nComputing power spectrum with class-based pencil decomposition...")
    
    power_spectrum = compute_power_spectrum_3D_mpi_pencil(field, shape=shape)
    
    if rank == 0:
        print(f"Class-based approach - Power spectrum computed successfully!")
        print(f"Class-based approach - Power spectrum shape: {power_spectrum.shape}")
        print(f"Class-based approach - Power spectrum sum: {np.sum(power_spectrum):.6e}")
        print(f"Class-based approach - Power spectrum max: {np.max(power_spectrum):.6e}")
        print(f"Class-based approach - Power spectrum min: {np.min(power_spectrum):.6e}")
    
    # Compare with serial computation for validation
    if rank == 0:
        print("\nComparing with serial computation...")
        
        # Serial computation using your original approach
        def compute_serial_power_spectrum(field):
            if pyfftw_import:
                fftn_func = pyfftw.interfaces.numpy_fft.fftn
                fftshift_func = pyfftw.interfaces.numpy_fft.fftshift
            else:
                fftn_func = np.fft.fftn
                fftshift_func = np.fft.fftshift
            
            return np.sum(
                np.abs(
                    fftshift_func(
                        fftn_func(field, axes=(1, 2, 3), norm='forward'),
                        axes=(1, 2, 3)))**2,
                axis=0)
        
        serial_power = compute_serial_power_spectrum(field)
        print(f"Serial computation - Power spectrum sum: {np.sum(serial_power):.6e}")
        print(f"Serial computation - Power spectrum max: {np.max(serial_power):.6e}")
        
        # Check if they match (accounting for different k-space layouts)
        if power_spectrum is not None:
            print(f"\nClass MPI vs Serial ratio: {np.sum(power_spectrum) / np.sum(serial_power):.6f}")
        
        print(f"\nNote: The MPI power spectrum has shape {power_spectrum.shape}")
        print(f"while the serial has shape {serial_power.shape}")
        print("This is because MPI uses real-to-complex FFT which exploits Hermitian symmetry")
        print("to store only half the coefficients.")
        
        # Check max values
        print(f"\nMax value comparison:")
        if power_spectrum is not None:
            print(f"MPI class max: {np.max(power_spectrum):.6e}")
        print(f"Serial max: {np.max(serial_power):.6e}")
        print("Note: These differ by factor of 2 due to different spectral representations")


def benchmark_comparison():
    """Compare performance with and without MPI"""
    
    if rank == 0:
        N = 576  # Smaller size for quick testing        
        # Create test field
        field = np.array([psf.generate_isotropic_powerlaw_field(N),
                          psf.generate_isotropic_powerlaw_field(N),
                          psf.generate_isotropic_powerlaw_field(N)]).astype(np.float32)
        print(f"Benchmarking with field shape {field.shape}")
        shape = field.shape[1:]
    else:
        field = None
        shape = None
        
    # Time the MPI version
    comm.Barrier()
    start_time = MPI.Wtime()
    
    power_spectrum = compute_power_spectrum_3D_mpi_pencil(field, shape=shape, dtype=np.float32)
    
    if rank == 0:
        print(f"MPI pencil decomposition power spectrum shape: {power_spectrum.shape}")
        print(f"MPI pencil decomposition power spectrum sum: {np.sum(power_spectrum):.6e}")
    
    comm.Barrier()
    end_time = MPI.Wtime()
    
    if rank == 0:
        print(f"MPI pencil decomposition time: {end_time - start_time:.4f} seconds")
        print(f"Using {size} MPI processes")
        
    # After computing 3D spectrum, also compute 1D
    if rank == 0:
        print("\nComputing 1D isotropic power spectrum...")
    
    k_bins, power_1d = compute_power_spectrum_1D_mpi(field, shape=shape, dtype=np.float32)
    
    if rank == 0 and k_bins is not None:
        print(f"1D spectrum computed: {len(k_bins)} k-bins")
        print(f"k range: {k_bins[0]:.1f} to {k_bins[-1]:.1f}")
        print(f"Total power in 1D spectrum: {np.sum(power_1d):.6e}")
        
        # Plot if matplotlib available
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            plt.loglog(k_bins, power_1d, 'b-', linewidth=2)
            plt.xlabel('k')
            plt.ylabel('P(k)')
            plt.title('1D Isotropic Power Spectrum')
            plt.grid(True, alpha=0.3)
            plt.savefig('power_spectrum_1d.png')
            print("Saved plot to power_spectrum_1d.png")
        except ImportError:
            print("Matplotlib not available for plotting")


if __name__ == "__main__":
    if rank == 0:
        print("MPI Power Spectrum with Pencil Decomposition")
        print("=" * 50)
    
    # Run example
    # example_usage()
    
    if rank == 0:
        print("\nBenchmark:")
        print("-" * 20)
    
    benchmark_comparison()
    
    if rank == 0:
        print("\nDone!")