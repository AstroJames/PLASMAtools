# Parallelised Python tools for reading and manipulating plasma fluid simulations

Utilises MP parallelisation, lots of vectorisation through the `numpy` library, JIT (mostly AOT) compiling for I/O and almost all post-processing functions through `numba`. There are a number of FFTs the library can use, depending upon what is available. The fastest (by a factor of a few) is `pyfftw`. This works quite well up to 2k^3, but the `spectra_mpi.py` tool in the repository can be used for larger jobs. All spectral manipulations, like integrating, binning, etc., are written in `numba` and are incredibly efficient.

Currently the reading of FLASH data is handled by the `read.py` code, which has classes for particles and fields, and can read `FLASH`, `RAMSES` and `BHAC` simulation data. All of the tools really only have been tested thoroughly on uniform grids. For `BHAC`, where the simulation has been run in AMR, the read also does an efficient interpolation onto a uniform mesh using `numba` to linearly interpolate and `pykdtree` for constructing a KD tree for a NN search.

The post-processing functions are contained within `funcs/derived_vars/operations.py`, `funcs/spectral/operations.py`, `transfer_funcs/shell_trans_funcs.py` for derived variable functions, spectral variable functions and transfer functions respectively. The derivative class is in `funcs/derivative/derivatives.py`.

All scalar/vector operations can be used in 1D, 2D or 3D, for a variety of boundary conditions on each boundary. 

At the moment, the post-processing functions can be used as stand-alone functions that are directly applied to data, but for quite a few of them (look at `read.py`) they can be called as a method for the data object, which adds new derived fields directly to the data object. 

The functions are:

## Read I/O:
* BHAC (tested in 2.5D)
* FLASH (any D)
* RAMSES (only tested in 3D)
* AthenaK coming soon.

## Derivatives:
* a range of derivative stencils for first-order, central difference spatial derivatives, from two to eight-point with periodic, Neumann, and Dirichlet BCs.
* limited stencils for second-order, central spatial derivatives.
* all derivatives are parallelised and AOT compiled using numba

## Scalar operations:
* scalar gradient
* scalar laplacian
* scalar rms

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
* construction of TNB basis ( Frenet-Serret coordinates of the vector field ) 
* coordinate transformation into TNB basis
* field line curvature
* magnetic helicity ( in Coloumb gauge \partial_ia_i = 0; a . b )
* current helicity ( j.b )
* kinetic helicity ( vort . v )
* \dot{Q} = <P div(v)> heating
* vector laplacian

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

## Decompositions:
* orthogonal gradient tensor decomposition ( symmetric, antisymmetric and trace tensor )
* helmholtz decomposition (incompressible and compressible modes) of a vector field
* vorticity decomposition (stretching, compression, baroclinicity)
* decomposition into left and right helical eigen modes of a vector field

## Spectral operations:
All FFTS multithreaded with `pyfftw` backend, but fall back to `numpy` if `pyfftw` does not exist..
* vector potential ( in Coloumb gauge )
* scalar power spectrum
* vector field power spectrum
* tensor field power spectrum
* vector field power spectrum using mpi
* mixed vector field spectrum (for e.g., helicity spectrum)
* spherical shell integration
* cylindrical shell integration
* k space filtering through isotropic k shells ( for transfer function analysis )
* k space filtering through cylindrical k shells ( for transfer function analysis )

## Generating synthetic stochastic fields:
* generating chiral random field ( e.g. net j.b on system scale )
* generating power-law random field

## Energy flux transfer functions:
* kinetic energy transfer functions
* magnetic energy transfer functions
* kinetic - magnetic energy interaction transfer functions
* helmholtz decomposed transfer functions

## Transfer functions (all use parallelisation over shells, but are not good for larger mem. jobs, e.g., >= 2k^3 cells):
* 2D transfer functions for Newtonian kinetic (momentum) and magnetic (induction) field, specifically for casacade transfers (u' to u'' or b' to b''). Scalable to 10k^2 grids.
* 3D transfer functions for Newtonian ideal MHD equations.

## Plotting functions:
* vectorised implementation of line integral convolution algorithm (implemented by Neco Kriel). Works very fast, even up to (10k)^2 grids.
* An interpolation function that allows for smoothe interpolation between different time realisations.

## Critial point analysis
* o and x point detector based on vector potential / stream function. Only works in 2D. 
