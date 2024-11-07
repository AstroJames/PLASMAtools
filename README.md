# Python tools for reading and manipulating plasma fluid simulations

Utilises MP parallelisation, lots of vectorisation through the `numpy` library, and JIT compiling for I/O through `numba`.

Currently the reading of FLASH data is handled by the `read.py` code, which has classes for particles and fields, and can read `FLASH`, `RAMSES` and `BHAC` simulation data. 


The post-processing functions are contained within `aux_funcs/derived_var_funcs.py`, `aux_funcs/spectral_var_funcs.py`, `aux_funcs/shell_trans_funcs.py` for derived variable functions, spectral variable functions and transfer functions respectively. The derivative class is in `derivative.py`.

All functions can be used in 1D, 2D or 3D. 

At the moment, the post-processing functions can be used as stand-alone functions that are directly applied to data, but for quite a few of them (look at `read.py`) they can be called as a method for the data object, which adds new derived fields directly to the data object. 

The functions are:

## Derivatives:
* a range of derivative stencils for first-order, central difference spatial derivatives, from two to eight-point with periodic, Neumann, and Dirichlet BCs.

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
* curvature
* magnetic helicity ( in Coloumb gauge; a . b )
* current helicity ( j.b )
* kinetic helicity ( vort . v )

## Tensor operations:
* tensor outer product
* tensor contraction
* vector dot tensor
* gradient tensor
* eigen values of Hermitian tensors ( for, e.g., symmetric stretching tensor )
* gradient tensor stability analysis in the TNB basis ( classification of critical points in a vector field )

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

## Functionality coming to the repo.
* transfer functions are currently utilising functionality from repo, but not properly incorporated in repo.
* add all of the derived functions to the actual data object, so any derived variable can just be called as a method in the data class.
* faster, parallelised ffts
