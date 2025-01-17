# Python tools for reading and manipulating plasma fluid simulations

Utilises MP parallelisation, lots of vectorisation through the `numpy` library, JIT compiling for I/O through `numba`, and `joblib` for any trivial parallelisation. There are a number of FFTs the library can use, depending upon what is available. The fastest (by a factor of a few) is `pyfftw`. Currently I am working on MPI implemtnations for both FFT and derivative class, which is invaluable for large mem. jobs with more than (1k^3) cells.

Currently the reading of FLASH data is handled by the `read.py` code, which has classes for particles and fields, and can read `FLASH`, `RAMSES` and `BHAC` simulation data. 

The post-processing functions are contained within `aux_funcs/derived_var_funcs.py`, `aux_funcs/spectral_var_funcs.py`, `aux_funcs/shell_trans_funcs.py` for derived variable functions, spectral variable functions and transfer functions respectively. The derivative class is in `derivative.py`.

All scalar/vector operations can be used in 1D, 2D or 3D. 

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
* all derivatives are vectorised, but currently working towards MPI implementation for large mem. problems.

## Scalar operations:
* scalar gradient
* scalar laplacian

## Vector operations:
* vector cross product
* vector dot product
* vector field magnitude
* vectof field RMS
* vector curl
* vector divergence
* construction of TNB basis ( Frenet-Serret coordinates of the vector field ) 
* coordinate transformation into TNB basis
* field line curvature
* magnetic helicity ( in Coloumb gauge \partial_ia_i = 0; a . b )
* current helicity ( j.b )
* kinetic helicity ( vort . v )

## Tensor operations:
* tensor outer product
* tensor contraction
* vector dot tensor
* gradient tensor
* eigen values of Hermitian tensors ( for, e.g., symmetric stretching tensor; used for dynamo growth rate modelling )

## Decompositions:
* orthogonal gradient tensor decomposition ( symmetric, antisymmetric and trace tensor )
* helmholtz decomposition (incompressible and compressible modes) of a vector field
* vorticity decomposition (stretching, compression, baroclinicity)
* decomposition into left and right helical eigen modes of a vector field
* decomposition into vorticity sources ( compressive, stretching, baroclinic, tension )

## Spectral operations:
* vector potential ( via fft in Coloumb gauge )
* 3D vector sclar power spectrum
* 3D vector field power spectrum
* 3D tensor field power spectrum
* spherical shell binning
* cylindrical shell binning
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
* vectorised implementation of line integral convolution algorithm. Works very fast, even up to (10k)^2 grids.

## Critial point analysis
* 

## Functionality coming to the repo.
* faster, mpi parallelised ffts for large mem. jobs, specifically important for transfer functions.
