# Python tools for reading and manipulating FLASH data

Utilises some MP parallelisation, and lots of vectorisation through the numpy library. 

Currently the reading of FLASH data is handled by the `read_flash.py` code, which has classes for particles and fields. 

Most of the post-processing functions are contained within `aux_funcs/derived_var_funcs.py` and `aux_funcs/spectral_var_funcs.py`, for derived variable functions and spectral variable functions, respectively. 

Some of the functions are:
. vector potential
. magnetic helicity
> gradient tensor
> orthogonal tensor decomposition
> eigen values of the stretching tensor
> tensor outer product
> tensor contraction
> helmholtz decomposition (incompressible and compressible modes) of a vector field
> vector curl
> vector divergence
> scalar laplacian
> vector cross product
> vector dot product
> vector field magnitude
> vectof field RMS
> scalar gradient
> TNB basis
> coordinate transformation into TNB basis
> jacobian stability analysis in the TNB basis
> classification of critical points
> a range of derivative stencils from second to sixth order (assuming periodicity)
> 3D scalar power spectrum
> spherical shell binning
> cylcindrical shell binning
> decomposition into left and right helical eigen modes of a vector field
> generating helical fields
> generating power-law random field
> k space filtering through isotropic k shells (for transfer function analysis)
