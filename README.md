# Parallelised Python tools for reading and manipulating plasma fluid simulations

Utilises MP parallelisation, lots of vectorisation through the `numpy` library, JIT (mostly AOT) compiling for I/O and almost all post-processing functions through `numba`. There are a number of FFTs the library can use, depending upon what is available. The fastest (by a factor of a few) is `pyfftw`. This works quite well up to 2k^3, but the `spectra_mpi.py` tool in the repository can be used for larger jobs. All spectral manipulations, like integrating, binning, etc., are written in `numba` and are incredibly efficient.

Currently the reading of FLASH data is handled by the `read.py` code, which has classes for particles and fields, and can read `FLASH`, `RAMSES` and `BHAC` simulation data. All of the tools really only have been tested thoroughly on uniform grids. For `BHAC`, where the simulation has been run in AMR, the read also does an efficient interpolation onto a uniform mesh using `numba` to linearly interpolate and `pykdtree` for constructing a KD tree for a NN search.

The post-processing functions are contained within `funcs/derived_vars/operations.py`, `funcs/spectral/operations.py`, `transfer_funcs/shell_trans_funcs.py` for derived variable functions, spectral variable functions and transfer functions respectively. The derivative class is in `funcs/derivative/derivatives.py`.

All scalar/vector operations can be used in 1D, 2D or 3D, for a variety of boundary conditions on each boundary. 

At the moment, the post-processing functions can be used as stand-alone functions that are directly applied to data, but for quite a few of them (look at `read.py`) they can be called as a method for the data object, which adds new derived fields directly to the data object. 

The functions are:

## Module Architecture and Conventions
- Subpackage layout: each topical area lives under `PLASMAtools/funcs/<topic>/` (e.g., `vector/`, `tensor/`, `spectral/`, `clustering/`). I/O is under `PLASMAtools/io/`, plotting under `PLASMAtools/plot_funcs/`, and transfer functions under `PLASMAtools/transfer_funcs/`.
- Public API: each subpackage exposes user-facing methods in `operations.py`. Import from these rather than calling kernels directly.
- Core kernels: heavy numerical routines live in `core_functions.py` and are Numba-compiled (fastmath, cache, nogil where appropriate). These expect contiguous arrays and minimal copying.
- Constants and signatures: each subpackage has a `constants.py` defining enums (e.g., `PERIODIC`, `NEUMANN`, `DIRICHLET`), axis indices (`X, Y, Z = 0,1,2`), defaults, and AOT Numba type signatures used by the kernels.
- Init modules: `__init__.py` re-exports the primary classes, key constants, and advanced kernels for power users.

Data conventions
- Scalars: shape `(Nx, Ny, Nz)`; vectors: shape `(3, Nx, Ny, Nz)`; tensors: shape `(3,3,Nx,Ny,Nz)`.
- Domain sizes: many APIs take `L` or `grid_spacing`. Use a scalar for cubic domains or a 3-array `[Lx, Ly, Lz]` for anisotropic boxes.
- Boundary conditions: use constants from the relevant `constants.py`. Periodic distances use the minimal-image convention.
- Dtypes: float32 by default for memory efficiency; some routines accept float64.
- Threading: control Numba threads with `NUMBA_NUM_THREADS` env var.

## Read I/O:
* BHAC (tested in 2.5D)
* FLASH (any D)
* RAMSES (only tested in 3D)
* AthenaK (planned).

API:
- `from PLASMAtools.io.read import Fields, Particles`
- `fld = Fields('plotfile.h5', reformat=True, sim_data_type='flash'); fld.read('vel'); vel = fld.vel`
- `pts = Particles('particles.h5'); pts.read('vel'); pts.sort_particles()`

## Derivatives:
* a range of derivative stencils for first-order, central difference spatial derivatives, from two to eight-point with periodic, Neumann, and Dirichlet BCs.
* limited stencils for second-order, central spatial derivatives.
* all derivatives are parallelised and AOT compiled using numba

API:
- `from PLASMAtools.funcs.derivative.derivatives import Derivative`
- `d = Derivative(stencil=4)`
- `gx = d.gradient(scalar, gradient_dir=0, L=1.0)`; `curl = d.vector_curl_fast(vel, L=[Lx, Ly, Lz])`; `div = d.vector_divergence_fast(vel, L=[Lx, Ly, Lz])`

## Scalar operations:
* scalar gradient
* scalar laplacian
* scalar rms

API:
- `from PLASMAtools.funcs.scalar import ScalarOperations`
- `ScalarOperations().scalar_rms(field)`
- For gradients/laplacians, use the `Derivative` API above.

## Vector operations:
* vector cross product
* vector dot product
* vector field magnitude
* vector field RMS
* vector field projection onto another vector field
* angle between vectors
* vector triple product
* vector curl
* vector divergence
* construction of TNB basis (Frenet–Serret coordinates of the vector field)
* coordinate transformation into TNB basis
* field line curvature
* magnetic helicity (in Coulomb gauge, \partial_i a_i = 0; a · b)
* current helicity ( j.b )
* kinetic helicity ( vort . v )
* \dot{Q} = <P div(v)> heating
* vector laplacian

API:
- `from PLASMAtools.funcs.vector import VectorOperations`
- `vo = VectorOperations(num_of_dims=3)`
- `mag = vo.vector_magnitude(v)`; `dot = vo.vector_dot_product(v, w)`; `crs = vo.vector_cross_product(v, w)`

## Tensor operations:
* tensor outer product
* tensor double contraction (A_ij A_ij or A_ij A_ji)
* vector dot tensor (u_i A_ij or u_j A_ij)
* tensor magnitude
* tensor field transpose
* tensor invariants 
* gradient tensor
* gradient tensor decomposition into trace, symmetric, antisymmetric tensors
* eigen values of Hermitian tensors ( for, e.g., symmetric stretching tensor; used for dynamo growth rate modelling )

API:
- `from PLASMAtools.funcs.tensor import TensorOperations`
- `to = TensorOperations()`
- `I1, I2, I3 = to.tensor_invariants(T)`; `sym, asym, bulk = to.orthogonal_tensor_decomposition(T, all=True)`; `AijBij = to.tensor_double_contraction_ij_ij(A, B)`

## Decompositions:
* orthogonal gradient tensor decomposition (symmetric, antisymmetric and trace tensor)
* Helmholtz decomposition (incompressible and compressible modes) of a vector field
* vorticity decomposition (stretching, compression, baroclinicity)
* decomposition into left and right helical eigenmodes of a vector field

API:
- Tensor orthogonal decomposition: see `TensorOperations.orthogonal_tensor_decomposition(...)` above.
- Helmholtz decomposition: `from PLASMAtools.funcs.spectral import SpectralOperations as SO`; `u_c, u_s = SO(L=[Lx,Ly,Lz]).helmholtz_decomposition(vel)`

## Spectral operations:
All FFTs multithreaded with `pyfftw` backend, but fall back to `numpy` if `pyfftw` does not exist.
* vector potential (in Coulomb gauge)
* scalar power spectrum
* vector field power spectrum
* tensor field power spectrum
* vector field power spectrum using MPI
* mixed/cross power spectrum Re{a(k) · b†(k)} (e.g., helicity spectrum)
* shear-coordinate power spectra via phase-correction remap on one coordinate (shearing-box transform)
* spherical shell integration
* cylindrical shell integration
* k-space filtering through isotropic k-shells (for transfer function analysis)
* k-space filtering through cylindrical k-shells (for transfer function analysis)

API:
- `from PLASMAtools.funcs.spectral import SpectralOperations as SO`
- Power spectrum (with optional shear): `S = SO(L=[Lx,Ly,Lz]); P = S.compute_power_spectrum_3D(field, shear=True, S=qOmega, t=t0)`
- Mixed spectrum: `Pab = S.compute_mixed_spectrum_3D(a, b)` which computes `Re{a(k) · b†(k)}`
- Isotropic 1D spectrum: `k, Pk = S.spherical_integrate_3D(P, coords='physical', S=qOmega, t=t0)`

## Generating synthetic stochastic fields:
* generating chiral random field (e.g., net j·b on system scale)
* generating power-law random field

API:
- `from PLASMAtools.funcs.spectral import SpectralOperations as SO`
- Isotropic: `f = SO.generate_isotropic_powerlaw_field(size=256, alpha=5/3)`
- Anisotropic: `f = SO.generate_anisotropic_powerlaw_field(N=256, alpha=5/3, beta=2)`

## Energy flux transfer functions:
* kinetic energy transfer functions
* magnetic energy transfer functions
* kinetic - magnetic energy interaction transfer functions
* helmholtz decomposed transfer functions

API:
- `from PLASMAtools.transfer_funcs.transfer_funcs_MPI import compute_transfer_functions_mpi`
- Run with MPI: `mpirun -n 8 python -c "from PLASMAtools.transfer_funcs.transfer_funcs_MPI import compute_transfer_functions_mpi; compute_transfer_functions_mpi(mag_field=mag, vel_field=vel, transfer_type='mag', n_bins=20, L=1.0)"`

## Transfer functions:
* 2D transfer functions for Newtonian kinetic (momentum) and magnetic (induction) field, specifically for cascade transfers (u' to u'' or b' to b''). Scalable to 10k^2 grids.
* 3D transfer functions for Newtonian ideal MHD equations. Scalable to 1k^3 grids.

API:
- 2D serial class: `from PLASMAtools.transfer_funcs.transfer_funcs_2d import TransferFunction`
- Initialize: `tf = TransferFunction(vector_field=vel, write_path='./shell_transfers', direction='iso', n_bins=20)`; then `tf.CalcVelTransfer(idx_K, idx_Q)` / `tf.CalcMagTransfer(idx_K, idx_Q)`
- 3D large grids: use the MPI function above.

## Plotting functions:
* vectorised implementation of line integral convolution algorithm (implemented by Neco Kriel). Works very fast, even up to (10k)^2 grids.
* An interpolation function that allows for smooth interpolation between different time realisations.

API:
- `from PLASMAtools.plot_funcs.LIC import computeLIC`
- `lic = computeLIC(vfield=(vx, vy), sfield_in=None, streamline_length=20)`; then `plt.imshow(lic)`

## Critical point analysis
* o and x point detector based on vector potential / stream function. Only works in 2D. 

API:
- `from PLASMAtools.funcs.derived_vars import DerivedVars as DV`
- Use DV utilities to compute the reduced 2D Jacobian metrics and then `DV().classification_of_critical_points(trace_M, det_M, J3, J_thresh, eig1, eig2)` to label X/O/antiparallel regions.

## Clustering and morphology
Friends-of-Friends (FOF) and grid-connected clustering for identifying coherent structures such as supernova bubbles or hot regions. The implementation uses efficient sparse connected-components for grid data and Numba-optimized kernels for particle FOF.

- Features: 2D/3D support, periodic/Neumann/Dirichlet BCs, morphology filtering to remove thin shells, optional splitting of merged components, and robust cluster radii via circular means on periodic axes.
- API: `PLASMAtools.funcs.clustering.ClusteringOperations`
- Quick start (grid mode):
  - `from PLASMAtools.funcs.clustering import ClusteringOperations, PERIODIC, NEUMANN`
  - `ops = ClusteringOperations(num_of_dims=3, precision='float32')`
  - `res = ops.cluster_3d_field(field=temperature, threshold=-1.5, threshold_type='greater', grid_spacing=1.0, morphological_filter=True, min_thickness=2, min_cluster_size=500, method='grid', connectivity=26, boundary_conditions=np.array([PERIODIC, PERIODIC, NEUMANN], dtype=np.int32), return_field=True)`
  - Properties are in `res['properties']` with `cluster_sizes`, `cluster_centers`, and `cluster_radii`.
- Quick start (FOF on positions):
  - `labels = ops.friends_of_friends(positions, linking_length=0.2, box_size=[Lx, Ly, Lz], boundary_conditions=[PERIODIC, PERIODIC, NEUMANN], min_cluster_size=10)`
  - `props = ops.get_cluster_properties(positions, labels, box_size=[Lx, Ly, Lz], boundary_conditions=[PERIODIC, PERIODIC, NEUMANN])`

See `PLASMAtools/funcs/clustering/examples/main_grid_demo.py` for an end-to-end example with caching and visualization.
