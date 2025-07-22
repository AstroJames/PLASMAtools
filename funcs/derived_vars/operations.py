"""
Title:          Derived Variable Functions
Author:         James R. Beattie
Description:    Functions for calculating derived variables in the read class


Collaborators:  Neco Kriel, Tanisha Ghosal, Anne Noer Kolborg, 
                Shashvat Varma, 

"""

## ###############################################################
## IMPORTS
## ###############################################################

# python dependencies
import numpy as np

# import derivative stencils
from ..derivative.derivatives import Derivative
from .constants import *
from .core_functions import *
from .utils import *

# Keep consistent throughout library:
# scalar fields : 1,N,N,N   (i, x, y, z)
# vector fields : 3,N,N,N   (i, x, y, z)
# tensor fields : M,M,N,N,N (i, j, x, y, z)

# Note: for power spectra we are using the SpetraOperations class.
# self._do_fft( is used to compute the FFT of the vector field.
        # self, 
        # data, 
        # axes, 
        # forward=True, 
        # real=False, 
        # norm='forward')
# is the signature of the function.

## ###############################################################
## Derived Variable Functions
## ###############################################################

class DerivedVars(ScalarOperations,
                  VectorOperations,
                  TensorOperations,
                  SpectralOperations):
    """
    Class for calculating derived variables from the magnetic field, velocity field, and density field.
    """
    
    def __init__(
        self, 
        L              : float    = [1.0,1.0,1.0],
        stencil        : int      = 2,
        bcs            : str      = "000",
        mu0            : float    = 4*np.pi,
        num_of_dims    : int      = 3,
        debug          : bool     = False):
        """
        Initialize the derived variable class

        Args:        
            L (float)           : the size of the domain (needed for computing dx,dy,dz). Default is 1.0.
            stencil (int)       : order of finite difference stencil used to compute the derivatives.
                                    Options are 2, 4, 6, 8. Default is 2.
            bcs (str)           : boundary conditions for the vector field. Default is "000" (triply periodic),
                                    where the string is ordered as XYZ.
                                    idx 0: periodic boundary
                                    idx 1: Neumann boundary
                                    idx 2: Dirichlet boundary
            mu0 (float)         : magnetic permeability. Default is 4*np.pi.
            num_of_dims (int)   : number of dimensions. Default is 3.
            debug (bool)        : debug flag for debugging the vector potential calculation. Default is False.
            
        """
        
        # Type check and convert L to proper format
        if isinstance(L, np.ndarray):
            # Convert numpy array to list of floats
            L = L.astype(float).tolist()
            if debug:
                print(f"Warning: L was provided as numpy array, converted to list: {L}")
        elif isinstance(L, (int, float)):
            # Single scalar - keep as float
            L = float(L)
        elif isinstance(L, (list, tuple)):
            # Convert all elements to floats
            try:
                L = [float(x) for x in L]
            except (ValueError, TypeError):
                raise TypeError("All elements in L must be convertible to float")
        else:
            raise TypeError(f"L must be a float, int, list, tuple, or numpy array. Got {type(L)}")
        
        # Set the class variables
        self.L           = L # domain size
        self.bcs         = [boundary_lookup[int(list(bcs)[i])] 
                            for i in range(len(list(bcs)))] # boundary conditions
        self.mu0         = mu0         # magnetic permeability
        self.stencil     = stencil     # finite difference stencil size
        self.num_of_dims = num_of_dims # number of dimensions in the domain
        
        # create a derivative object to be shared globally
        self.d = Derivative(self.stencil)
        
        # add the inherited classes
        ScalarOperations.__init__(
            self)
        VectorOperations.__init__(
            self,
            self.num_of_dims)
        TensorOperations.__init__(
            self)
        SpectralOperations.__init__(
            self,
            self.L,
            cache_plans=False)
        
        # Set the number of dimensions
        if self.num_of_dims == 3:
            self.coords = [X,Y,Z] # 3D
            assert len(self.bcs) == 3, "Boundary conditions must be specified for all three dimensions."
        elif self.num_of_dims == 2:
            self.coords = [X,Y]   # 2D
            assert len(self.bcs) == 2, "Boundary conditions must be specified for all two dimensions."
        elif self.num_of_dims == 1:
            self.coords = [X]     # 1D 
            assert len(self.bcs) == 1, "Boundary conditions must be specified for all one dimension."
        else:
            raise ValueError("num_of_dims must be 1, 2, or 3.")
        
        self.debug = debug
        if self.debug:
            print(f"DerivedVariables: L={L}, stencil={stencil}, bcs={bcs}, mu0={mu0}, num_of_dims={num_of_dims}, debug={debug}")
        
        
    def _set_stencil(
        self,
        stencil : int) -> None:
        """
        Set the stencil for the finite difference derivative.
        
        This function is used to change the stencil order of the 
        finite difference derivative.
        """
        self.stencil = stencil  
        self._set_derivative(
            self.stencil)
        
        
    def _set_derivative(
        self,
        stencil : int) -> None:
        """
        Set the stencil for the finite difference derivative.
        
        This function is used to change the stencil order of the 
        finite difference derivative.
        """
        self.d = Derivative(stencil)
    
    
    def magnetic_helicity(
        self,
        magnetic_vector_field : np.ndarray ) -> np.ndarray:
        """
        Compute the magnetic helicity in the Coloumb gauge (gauge fixed).
        
        Author: James Beattie
        
        Args:
            magnetic_vector_field (np.ndarray): 3,N,N,N array of vector field,
            
        Returns:
            magnetic helicity (np.ndarray): 1,N,N,N array of magnetic helicity (a . b).
        
        """
        
        out = np.zeros_like(
            magnetic_vector_field[X])
        
        #TODO: need to fix up z axis to accomodate for only read
        # FFT (saves factor > 2 in memory but produces Nx, Ny, Nz/2 outputs)
        
        # compute the magnetic helicity   
        out = self.vector_dot_product(
            self.vector_potential(
                magnetic_vector_field),
            magnetic_vector_field)
         
        return out


    def kinetic_helicity(
        self,
        velocity_vector_field : np.ndarray ) -> np.ndarray:
        """
        Compute the kinetic helicity of a velocity field.
        
        Author: James Beattie
        
        Args:
            velocity_vector_field (np.ndarray): 3,N,N,N array of vector field,
            
        Returns:
            kinetic helicity (np.ndarray): 1,N,N,N array of kinetic helicity (omega . v).
        
        """
        
        out = np.zeros_like(
            velocity_vector_field[X])
        
        # compute the kinetic helicity
        out = self.vector_dot_product(
            self.vector_curl(
                velocity_vector_field),
            velocity_vector_field)
        
        return out
    
    
    def current_helicity(
        self,
        magnetic_vector_field : np.ndarray ) -> np.ndarray:
        """
        Compute the current helicity of a magnetic field.
        
        Author: James Beattie
        
        Args:
            magnetic_vector_field (np.ndarray): 3,N,N,N array of vector field,
            
        Returns:
            current helicity (np.ndarray): 1,N,N,N array of current helicity (j . b).
        
        """
        
        out = np.zeros_like(
            magnetic_vector_field[X])

        # compute the current helicity
        out = self.vector_dot_product(
            self.vector_curl(
                magnetic_vector_field) / self.mu0,
            magnetic_vector_field)

        return out
   
       
    def gradient_tensor(
        self,
        vector_field: np.ndarray) -> np.ndarray:
        """
        Compute the gradient tensor of a vector field, fast :).
        """
        
        out = np.zeros((self.num_of_dims, 
                        self.num_of_dims, 
                        *vector_field[X].shape),
                       dtype=vector_field.dtype)
        
        out = self.d.gradient_tensor_fast(
            vector_field,
            self.L[X])
        
        return out


    def vector_dot_gradient_tensor(
        self,
        vector_field_1 : np.ndarray,
        vector_field_2 : np.ndarray) -> np.ndarray:
        """
        u_j partial_j u_i
        
        Compute the dot product of a vector field with the gradient tensor 
        of another vector field. This is used to compute transfers functions 
        of the nonlinear terms in the Navier-Stokes equations.
        
        """
        
        out = np.zeros((self.num_of_dims, 
                        *vector_field_1[X].shape),
                       dtype=vector_field_1.dtype)
        
        out = np.array([vector_field_1[X] * self.d.gradient(
            vector_field_2[coord], 
            gradient_dir=X, 
            L=self.L[X],
            boundary_condition=self.bcs[X]) + 
                        vector_field_1[Y] * self.d.gradient(
                            vector_field_2[coord], 
                            gradient_dir=Y, 
                            L=self.L[Y],
                            boundary_condition=self.bcs[Y]) +
                         vector_field_1[Z] * self.d.gradient(
                            vector_field_2[coord], 
                            gradient_dir=Z, 
                            L=self.L[Z],
                            boundary_condition=self.bcs[Z]) for coord in self.coords])
        
        return out
                                      

    def vorticity_decomp(
        self,
        velocity_vector_field    : np.ndarray,
        magnetic_vector_field    : np.ndarray    = None,
        density_scalar_field     : np.ndarray    = None,
        pressure_scalar_field    : np.ndarray    = None) -> np.ndarray:
        """
        Compute the terms in the vorticity equation, including the compressive, stretching, baroclinic,
        baroclinic magnetic, and tension terms. The magnetised (ideal) vorticity equation is given by:
        
        frac{D omega}{Dt} = - (nabla . u) omega + omega . nabla u +
            frac{1}{rho^2} nabla p times nabla rho + 
            frac{1}{rho^2} nabla rho times nabla b^2 / 2mu_0 + 
            nabla times (1/rho) b . nabla b / mu_0,
            
        where omega = nabla times u is the vorticity, u is the velocity field, p is the pressure field,
        rho is the density field, and b is the magnetic field and D/Dt is the Lagrangian derivative. 
        
        The terms are:
        
        compression term            : - (nabla . u) omega,
        stretching term             : omega . nabla u,
        baroclinic term             : 1/rho^2 nabla p times nabla rho,
        baroclinic magnetic term    : 1/rho^2 nabla rho times nabla b^2 / 2mu_0,
        tension term                : 1/mu_0 nabla times (1/rho) b . nabla b.
        
        hence the vorticity equation can be written as:
        
        frac{D omega}{Dt} = compression + stretching + baroclinic + baroclinic_magnetic + tension.

        Author:
            James Beattie

        Args:
            velocity_vector_field (np.ndarray)              : velocity vector field (3,N,N,N).
            magnetic_vector_field (np.ndarray, optional)    : magnetic vector field (3,N,N,N). Defaults to None.
            density_scalar_field (np.ndarray, optional)     : gas density scalar field (1,N,N,N). Defaults to None.
            pressure_scalar_field (np.ndarray, optional)    : pressure scalar field (1,N,N,N). Defaults to None.

        Returns:
            omega (np.ndarray)                                                      : vorticity field.
            compress (np.ndarray)                                                   : compressive term in the vorticity equation.
            stretch (np.ndarray)                                                    : vortex stretching term in the vorticity equation. 
            baroclinic (np.ndarray, None if density_scalar_field is None)           : gas baroclinic term in the vorticity equation.
            baroclinic_magnetic (np.ndarray, None if magnetic_vector_field is None) : magnetic baroclinic term in the vorticity equation. 
            tension (np.ndarray, None if magnetic_vector_field is None)             : magnetic tension term in the vorticity equation.
        """
        
        # initialise terms that may or may not be computed
        baroclinic           = None
        baroclinic_magnetic  = None
        tension              = None
        
        # now construct the terms in the vorticity equation
        
        # compute the vorticity, omega = nabla \times u
        omega = self.vector_curl(
            velocity_vector_field)
        
        # vorticity compression term, - (2/3) (nabla . u) omega
        compress = - 2.0 * omega/3.0 * self.vector_divergence(
            velocity_vector_field)[X]   
        
        # vortex stretching term, omega . nabla u
        grad_u =  self.gradient_tensor(velocity_vector_field)
        
        stretch_tensor = self.orthogonal_tensor_decomposition(
            grad_u,
            sym=True)
        
        stretch = self.vector_dot_tensor_i_ij(
            omega, 
            stretch_tensor)
        
        # if the magnetic and density is not None, compute the magnetic terms
        if ( magnetic_vector_field is not None ) and ( density_scalar_field is not None ):
                    
            # magnetic baroclinic term, 1/rho^2 nabla rho \times nabla b^2 / 2mu_0
            baroclinic_magnetic = (1./density_scalar_field[X]**2) * self.vector_cross_product(
                self.scalar_gradient(
                    self.vector_dot_product(
                        magnetic_vector_field,
                        magnetic_vector_field) / (2 * self.mu0)),
                self.scalar_gradient(
                    density_scalar_field))
            
            # magnetic tension term 1/\mu_0 \nabla \times (1/\rho) b . \nabla b)
            tension = self.vector_curl(
                (1./density_scalar_field[X]) * self.vector_dot_tensor_i_ij(
                    magnetic_vector_field,
                    self.gradient_tensor(
                        magnetic_vector_field)) / self.mu0
                )
            
        # if density and pressure is not None, compute the baroclinic term
        if ( density_scalar_field is not None ) and ( pressure_scalar_field is not None ):
            
            # baroclinic term, 1/\rho^2 \nabla p \times \nabla \rho
            baroclinic = (1/ density_scalar_field[X]**2) * self.vector_cross_product(
                self.scalar_gradient(
                    pressure_scalar_field),
                self.scalar_gradient(
                    density_scalar_field)
                )
            
        return omega, compress, stretch, baroclinic, baroclinic_magnetic, tension
        
        
    def tension_force(
        self,
        magnetic_vector_field : np.ndarray) -> np.ndarray:
        """
        Compute the tension force from the magnetic field.
        
        Args:
            magnetic_vector_field (np.ndarray): magnetic vector field (3,N,N,N). Defaults to None.

        Returns:
            the tension force field (3,N,N,N).
            
        """
        
        out = np.zeros_like(magnetic_vector_field)
        out =self.vector_dot_tensor_i_ij(
            magnetic_vector_field,
            self.gradient_tensor(
                magnetic_vector_field)) / self.mu0 
        return out
        
        
    def symmetric_eigvals(
        self, 
        tensor_field : np.ndarray, 
        find_vectors : bool = False) -> np.ndarray:
        """
        
        Finds the eigenvalues of a symmetric 3x3 matrix from https://hal.science/hal-01501221/document

        Authors: Shashvat Varma, James Beattie

        Parameters
        ----------
        matrix       : numpy ndarray shape (3,3,Nx,Ny,Nz)
                        must be a real symmetric 3x3 matrix defined pointwise in an arbitrary grid.
        find_vectors : bool, optional
                        If True, the eigenvectors will be computed as well. 
                        Default is False to save computing time.

        Returns
        -------
        eigenvalues : numpy ndarray shape (3,Nx, Ny,Nz)
                        array containing the three eigenvalues of the matrix, which will always be real for symmetric matrices. Is
                        sorted from smallest to largest.
        eigenvectors : numpy ndarray shape (3,3, Nx, Ny, Nz)
                        array containing the eigenvectors of the matrix. Organized as rows in the matrix, first row corresponds to the
                        first eigenvalue and so on. Only returned if find_vectors is True.

        """
        #define the values of the matrix to be used in computations (NOTE: Assumes all values are real)
        assert tensor_field.shape[N_COORDS_TENS] == 3 and tensor_field.shape[M_COORDS_TENS] == 3, "Matrix must be 3x3"
        #make sure tensor is symmetric
        assert np.allclose(tensor_field, self.tensor_transpose(tensor_field)), "Matrix must be symmetric"
        
        a = tensor_field[X,X,:,:,:]
        b = tensor_field[Y,Y,:,:,:]
        c = tensor_field[Z,Z,:,:,:]
        d = tensor_field[X,Y,:,:,:]
        e = tensor_field[Y,Z,:,:,:]
        f = tensor_field[X,Z,:,:,:]

        #begin the computations
        x1 = a**2 + b**2 + c**2 - a*b - a*c - b*c+3*(d**2 + f**2 + e**2)
        x2 = (-1)*(2*a-b-c)*(2*b-a-c)*(2*c-a-b) + \
                9*((2*c-a-b)*d**2 + (2*b-a-c)*f**2 + (2*a-b-c)*e**2) - 54*d*e*f

        #define what phi is conditional to previous variables
        condition_list  = [x2>0.0, x2==0.0, x2<0.0]
        choice_list     = [np.arctan((np.sqrt(4.0*x1**3.0-x2**2.0))/(x2)), 
                           np.pi/2.0, 
                           np.arctan((np.sqrt(4.0*x1**3.0-x2**2.0))/(x2))+np.pi]
        phi             = np.select(condition_list, choice_list)

        #calculate the eigenvalues
        sqrt_x1     = np.sqrt(x1)
        lambda1     = (a+b+c-2.0*sqrt_x1*np.cos(phi/3.0))/3.0
        lambda2     = (a+b+c+2.0*sqrt_x1*np.cos((phi-np.pi)/3.0))/3.0
        lambda3     = (a+b+c+2.0*sqrt_x1*np.cos((phi+np.pi)/3.0))/3.0
        eig_array   = np.array([lambda1, lambda2, lambda3])

        #perform the sort, saving indices of sort to use on eigenvectors later
        idx = np.argsort(
            eig_array,
            axis=N_COORDS_VEC)
        eig_array = np.take_along_axis(
            eig_array, 
            idx, 
            axis=N_COORDS_VEC)
        
        if find_vectors:
            #compute the eigenvectors
            m1 = (d*(c-lambda1) - e*f) / (f*(b-lambda1) - d*e)
            m2 = (d*(c-lambda2) - e*f) / (f*(b-lambda2) - d*e)
            m3 = (d*(c-lambda3) - e*f) / (f*(b-lambda3) - d*e)

            vec1 = [(lambda1 - c - e * m1) / f, m1, np.ones(np.shape(m1))]
            vec2 = [(lambda2 - c - e * m2) / f, m2, np.ones(np.shape(m2))]
            vec3 = [(lambda3 - c - e * m3) / f, m3, np.ones(np.shape(m3))]
            vec_array = np.array([vec1, vec2, vec3])

            del m1, m2, m3, vec1, vec2, vec3

            #do the corresponding sort on the vec array
            vec_array[:,X,:,:,:] = np.take_along_axis(
                vec_array[:,X,:,:,:], 
                idx, 
                axis=N_COORDS_VEC)
            vec_array[:,Y,:,:,:] = np.take_along_axis(
                vec_array[:,Y,:,:,:], 
                idx, 
                axis=N_COORDS_VEC)
            vec_array[:,Z,:,:,:] = np.take_along_axis(
                vec_array[:,Z,:,:,:], 
                idx, 
                axis=N_COORDS_VEC)

            return eig_array, vec_array
        else:
            return eig_array
        

    def vector_curl(
        self, 
        vector_field: np.ndarray) -> np.ndarray:
        """
        Optimized curl computation using fused kernel
        """
        if self.num_of_dims == 1:
            raise ValueError("Vector curl is not defined for 1D.")
        elif self.num_of_dims == 2:
            out = np.zeros_like(vector_field[X])
            # 2D curl still uses separate calls (could be optimized similarly)
            return self.d.gradient(
                vector_field[Y_GRID_VEC], 
                gradient_dir=X,
                L=self.L[X],
                boundary_condition=self.bcs[X]) - \
                   self.d.gradient(
                        vector_field[X_GRID_VEC],
                        gradient_dir=Y, 
                        L=self.L[Y],
                        boundary_condition=self.bcs[Y])
        elif self.num_of_dims == 3:
            out = np.zeros_like(vector_field)
            out =  self.d.vector_curl_fast(
                vector_field, 
                self.L, 
                self.bcs[X])
            return out
    
    def vector_divergence(
        self,
        vector_field: np.ndarray) -> np.ndarray:
        """
        Optimized divergence computation using fused kernel
        """
        
        vector_field = ensure_float32(
            vector_field,
            "vector_field")
        
        out = np.zeros_like(
            vector_field[N_COORDS_VEC])
        
        if self.num_of_dims == 3:
            # Use the optimized fused kernel
            #TODO: implement the boundary conditions
            out = self.d.vector_divergence_fast(
                vector_field, 
                self.L, 
                self.bcs[X])
            return out[np.newaxis,...]
        else:
            # Fallback for 1D/2D
            out = np.sum(
                np.array([self.d.gradient(vector_field[coord],
                                        gradient_dir=coord,
                                        L=self.L[coord],
                                        boundary_condition=self.bcs[coord]) 
                         for coord in range(self.num_of_dims)]),
                axis=0)
            return out[np.newaxis,...]
    
    
    def heating_rate(
        self,
        pressure_scalar_field : np.ndarray,
        velocity_vector_field : np.ndarray) -> np.ndarray:
        """
        Compute the heating rate of a given species:
        Heating = - <P nabla . u>
        
        Author: James Beattie
        
        Args:
            pressure_scalar_field (np.ndarray)   : 1,N,N,N array of scalar field,
                                                    where N is the number of grid points in each direction
            velocity_vector_field (np.ndarray)   : 3,N,N,N array of vector field,
                                                    where 3 is the vector component and N is the number of grid points in each direction    
                                                    
                                                    
        Returns:
        
            heating rate (float) 
        
        """
        
        out = np.zeros_like(
            pressure_scalar_field[N_COORDS_VEC])
        # compute the heating rate
        out =  - np.mean(
            pressure_scalar_field[X] *
            self.vector_divergence(
                velocity_vector_field)[X])
        
        return out
        
        
    def vector_laplacian(
        self,
        vector_field : np.ndarray) -> np.ndarray:
        """
        Compute the scalar laplacian using finite differences.
        
        Author: James Beattie
        
        Args:
            vector_field (np.ndarray)   : 3,N,N,N array of scalar field,
                                            where N is the number of grid points in each direction 

        Returns:
            laplacian vector field (np.ndarray) : 3,N,N,N array of laplacian of the vector field
        
        """
        
        out = np.zeros_like(
            vector_field)
        out = np.array(
            [self.scalar_laplacian(
                vector_field[coord]) 
             for coord in self.coords])
        return out

        
    def scalar_laplacian(
        self,
        scalar_field : np.ndarray) -> np.ndarray:
        """
        Compute the scalar laplacian using finite differences.
        
        Author: James Beattie
        
        Args:
            scalar_field (np.ndarray)   : N,N,N array of scalar field,
                                            where N is the number of grid points in each direction
                                            
        Returns:
            laplacian scalar field (np.ndarray) : N,N,N array of laplacian of the scalar field
        
        """
        
        # compute and return the laplacian with arbitrary boundary conditions
        
        out = np.zeros_like(scalar_field)
        out = np.sum(
            np.array([self.d.gradient(
                scalar_field[N_COORDS_VEC],
                gradient_dir       = coord,
                L                  = self.L[coord],
                derivative_order   = 2, 
                boundary_condition = self.bcs[coord]) 
                      for coord in self.coords]),
            axis=N_COORDS_VEC)
        return out


    def scalar_gradient(
        self,
        scalar_field : np.ndarray) -> np.ndarray:
        """
        Compute the gradient of a scalar field, grad(phi). 
        
        Author: Neco Kriel & James Beattie
        
        Args:
            scalar_field (np.ndarray)   : N,N,N array of vector field,
                                            where 3 is the vector component and N is the number of grid
                                            points in each direction

        Returns:
            grad_scalar_field (np.ndarray) : 3,N,N,N array of gradient of the scalar field
        
        """

        out = np.zeros((
            self.num_of_dims, 
            *scalar_field.shape))
        
        out = np.array([
            self.d.gradient(
                scalar_field[N_COORDS_VEC], 
                gradient_dir = coord,
                L            = self.L[coord]) 
            for coord in self.coords])
        return out


    def compute_TNB_basis(
        self,
        vector_field : np.ndarray) -> np.ndarray:
        """
        Compute the Fressnet frame of a vector field (TNB basis).
        
        Author: Neco Kriel & James Beattie
        
        Args:
            vector_field (np.ndarray): 3,N,N,N array of vector field,
                                        where 3 is the vector component and N is the number of grid
                                        points in each direction

        Returns:
            t_basis (np.ndarray) : 3,N,N,N array of tangent basis of the vector field
            n_basis (np.ndarray) : 3,N,N,N array of normal basis of the vector field
            b_basis (np.ndarray) : 3,N,N,N array of binormal basis of the vector field
            kappa (np.ndarray)   : N,N,N array of curvature of the vector field
        
        """
        ## format: (component, x, y, z)
        field_magn      = self.vector_magnitude(vector_field)
        ## ---- COMPUTE TANGENT BASIS
        t_basis = vector_field / field_magn
        ## df_j/dx_i: (component-j, gradient-direction-i, x, y, z)
        gradient_tensor = self.gradient_tensor(vector_field)
        ## ---- COMPUTE NORMAL BASIS
        ## f_i df_j/dx_i
        n_basis_term1 = np.einsum("i...,ji...->j...", 
                                vector_field, 
                                gradient_tensor)
        ## f_i f_j f_m df_m/dx_i
        n_basis_term2 = np.einsum("i...,j...,m...,mi...->j...", 
                                vector_field, 
                                vector_field, 
                                vector_field, 
                                gradient_tensor)
        ## (f_i df_j/dx_i) / (f_k f_k) - (f_i f_j f_m df_m/dx_i) / (f_k f_k)^2
        n_basis = n_basis_term1 / field_magn**2 - n_basis_term2 / field_magn**4
        ## field curvature
        kappa = self.vector_magnitude(n_basis)
        ## normal basis
        n_basis /= kappa
        ## ---- COMPUTE BINORMAL BASIS
        ## orthogonal to both t- and b-basis
        b_basis = self.vector_cross_product(t_basis, n_basis)
        return t_basis, n_basis, b_basis, kappa


    def TNB_coordinate_transformation(
        self,
        vector_field : np.ndarray) -> np.ndarray:
        """
        Transform a vector field into the Fressnet frame of a vector field (TNB basis).
        
        Author: James Beattie
        
        Args:
            vector_field (np.ndarray): 3,N,N,N array of vector field,
                                        where 3 is the vector component and N is the number of grid
                                        points in each direction

        Returns:
            vector_field_TNB (np.ndarray) : 3,N,N,N array of vector field in the TNB basis
        
        """
        
        # compute the TNB basis
        t_basis, n_basis, b_basis, _ = self.compute_TNB_basis(vector_field)
        
        # transform vector field to TNB basis
        return np.array([
            self.vector_dot_product(vector_field, t_basis),
            self.vector_dot_product(vector_field, n_basis),
            self.vector_dot_product(vector_field, b_basis)])
    
     
    def TNB_jacobian_stability_analysis(
        self,
        vector_field    : np.ndarray,
        traceless       : bool = True ) -> np.ndarray:
        """
        Compute the trace, determinant and eigenvalues of the Jacobian of a vector field in the 
        TNB coordinate system of an underlying vector field.        
        See: https://arxiv.org/pdf/2312.15589.pdf
        """
        
        def theta_eig(
            J_thresh : np.ndarray,
            J_3 : np.ndarray) -> np.ndarray:
            """
            Compute the angle between the eigenvectors of the Jacobian.
            See: https://arxiv.org/pdf/2312.15589.pdf            
            """
            # Two conditions for O and X points
            condition = np.abs(J_3) < J_thresh
            ratio = np.where(condition, 
                            J_thresh / J_3, 
                            J_3 / J_thresh)
            
            return np.arctan( np.sqrt(ratio**2-1) )
        
        # compute vector potential
        a = self.vector_potential(vector_field)
        
        # Compute jacobian of B field
        self._set_stencil(6)
        jacobian = self.gradient_tensor(a)
        self._set_stencil(2)
        
        # Make jacobian traceless (numerical errors will result in some trace, which is
        # equivalent to div(B) modes)
        if traceless:   
            jacobian = jacobian - (1.0/self.num_of_dims) * np.einsum("..., ij... -> ij...",
                                                                   np.einsum("ii...",
                                                                             jacobian),
                                                                   np.eye(self.num_of_dims))
        
        
        # # Compute TNB basis
        t_basis, n_basis, b_basis, _ = self.compute_TNB_basis(vector_field)
        X   = np.array([b_basis,n_basis,t_basis])
        X_T = np.einsum("ij...->ji...",X)
        
        # Put jacobian into the TNB basis
        trans_jacobian = np.einsum('ab...,bc...,cd... -> ad...', 
                                X, 
                                jacobian, 
                                X_T)
        
        # Construct M, the 2D jacobian of the B_perp field
        M_11 = trans_jacobian[0,0,...]
        M_12 = trans_jacobian[0,1,...]
        M_21 = trans_jacobian[1,0,...]
        M_22 = trans_jacobian[1,1,...]
        
        # Compute trace and determinant of M
        trace_M = M_11 + M_22
        det_M   = M_11 * M_22 - M_12 * M_21
        
        # Characteristic equation
        D = 4 * det_M - trace_M**2
        
        # J values for openning angles of X and O point
        # (these are physical currents)
        J_3         = M_21 - M_12
        J_thresh    = np.sqrt( (M_11 - M_22)**2 + (M_12 + M_21)**2 )
        
        # Eigen values of M from characteristic equation
        eig_1 = 0.5 * ( trace_M + np.sqrt( - (D + 0j)))
        eig_2 = 0.5 * ( trace_M - np.sqrt( - (D + 0j)))
        
        return trace_M, D, eig_1, eig_2, J_3, J_thresh, theta_eig(J_thresh,J_3)


    def classification_of_critical_points(
        self,
        trace_M  : np.ndarray,
        det_M    : np.ndarray,
        J_3      : np.ndarray,
        J_thresh : np.ndarray,
        eig_1    : np.ndarray,
        eig_2    : np.ndarray) -> np.ndarray:
        """
        Classify the critical points of the 2D reduced Jacobian of the B field.
    
        See: https://arxiv.org/pdf/2312.15589.pdf
        
        Args:
            trace_M (np.ndarray):   trace of the 2D reduced Jacobian of the B field
            D (np.ndarray):         determinant of the 2D reduced Jacobian of the B field
            J_3 (np.ndarray):       J_3 value of the 2D reduced Jacobian of the B field
            J_thresh (np.ndarray):  J threshold value of the 2D reduced Jacobian of the B field
            eig_1 (np.ndarray):     eigen value 1 of the 2D reduced Jacobian of the B field
            eig_2 (np.ndarray):     eigen value 2 of the 2D reduced Jacobian of the B field
            
        Returns:
            classification_array (np.ndarray): 3D array of the critical point types
        """
        # real and imaginary components of the eigenvalues
        eig1_real = np.real(eig_1)
        eig2_real = np.real(eig_2)
        
        eig1_imag = np.imag(eig_1)
        eig2_imag = np.imag(eig_2)
        
        is_2D = np.isclose(trace_M,0.0,1e-3)
        is_3D = (is_2D == False)
        is_real_eig = np.abs(J_3) < J_thresh
        is_imag_eig = np.abs(J_3) > J_thresh
        # is_real_eig = np.isclose(np.real(eig_1),0.0,atol=1e-1) & np.isclose(np.real(eig_2),0.0,atol=1e-1)
        # is_imag_eig = (is_real_eig == False)
        is_parallel = np.isclose(np.abs(J_3),J_thresh,1e-3)
        
        # array initialisation to store each of the 9 critical point types
        classification_array = np.repeat(np.zeros_like(trace_M)[np.newaxis,...],
                                        9,
                                        axis=0)
        
        # 3D X point (trace > 0, determinant < 0, real eigenvalues less than 0)
        classification_array[0,...] = np.logical_and.reduce([is_3D, 
                                                            is_real_eig, 
                                                            eig1_real*eig2_real < 0.0])
        
        # 3D O point (repelling; trace > 0, determinant > 0, conjugate eigenvalues equal)
        classification_array[1,...] = np.logical_and.reduce([is_3D, 
                                                            is_imag_eig, 
                                                            trace_M > 0.0])
        
        # 3D O point (attracting; trace < 0, determinant > 0, conjugate eigenvalues equal)
        classification_array[2,...] = np.logical_and.reduce([is_3D, 
                                                            is_imag_eig, 
                                                            trace_M < 0.0])
        
        # 3D repelling (trace =/= 0, determinant <= 0, real eigenvalues greater than 0)
        classification_array[3,...] = np.logical_and.reduce([is_3D, 
                                                            is_real_eig, 
                                                            eig1_real > 0.0, 
                                                            eig2_real > 0.0])
                                                
        # 3D attracting (trace =/= 0, determinant <= 0, real eigenvalues less than 0)
        classification_array[4,...] = np.logical_and.reduce([is_3D, 
                                                            is_real_eig, 
                                                            eig1_real < 0.0, 
                                                            eig2_real < 0.0])
        
        # 3D antiparallel (trace =/= 0, determinant < 0, either real component of eigenvalues = 0)
        classification_array[5,...] = np.logical_and.reduce([is_3D, 
                                                            is_parallel])
        
        # 2D X point (trace = 0, determinant < 0, real component of eigenvalues equal in opposite sign)
        classification_array[6,...] = np.logical_and.reduce([is_2D, 
                                                            is_real_eig])
        
        # 2D O point (trace = 0, determinant > 0, imaginary component of eigenvalues equal in opposite sign)
        classification_array[7,...] = np.logical_and.reduce([is_2D, 
                                                            is_imag_eig])
        
        # 2D antiparallel (trace = 0, determinant = 0, all eigenvalues = 0)
        classification_array[8,...] = np.logical_and.reduce([is_2D, 
                                                            is_parallel])
        return classification_array
        
    def vector_potential(
        self,
        vector_field: np.ndarray,
        debug: bool = False,
        field_name: str = "vector potential") -> tuple:
        """
        Calculate the vector potential of a vector field in both 2D and 3D.
        Now with optional debug mode that uses the Derivative class for verification.
        
        In 3D: Computes A such that B = ∇ x A (Coulomb gauge: ∇·A = 0)
        In 2D: Computes stream function ψ such that F = ∇ x ψẑ
        
        Author:
            James Beattie (2024)

        Args:
            vector_field (np.ndarray): The input vector field. 
                                    For 3D: shape (3, nx, ny, nz)
                                    For 2D: shape (2, nx, ny)
            debug (bool): If True, returns the reconstructed vector field for debugging.
            field_name (str): Name of the field for error messages.

        Returns:
            For 3D:
                a (np.ndarray): The vector potential of shape (3, nx, ny, nz).
                b_recon (np.ndarray, optional): The reconstructed vector field if debug is True.
            For 2D:
                psi (np.ndarray): The stream function of shape (nx, ny).
                F_recon (np.ndarray, optional): The reconstructed vector field if debug is True.
        """
        # Ensure data is float32 for memory efficiency
        vector_field = ensure_float32(
            vector_field, 
            field_name=field_name)
        
        if self.num_of_dims == 3:
            nx, ny, nz = vector_field.shape[X_GRID_VEC:]
            # 3D case
            # Create wave vectors
            kx = 2 * np.pi * fftfreq(nx, d=self.L[X]/nx)
            ky = 2 * np.pi * fftfreq(ny, d=self.L[Y]/ny)
            kz = 2 * np.pi * fftfreq(nz, d=self.L[Z]/nz)
            # Create meshgrid
            kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz,
                                                    indexing='ij')
            k = np.array([kx_grid, ky_grid, kz_grid],
                         dtype=np.float32)
            # FFT of vector field
            field_fft = self._do_fft(
                vector_field,
                axes=(X_GRID_VEC, Y_GRID_VEC, Z_GRID_VEC),
                forward=True,
                real=False,#np.isrealobj(vector_field),
                norm='forward'
            )
            # Compute vector potential in Fourier space using JIT function
            a_hat = compute_vector_potential_3D_core(
                k, 
                field_fft, 
                kx_grid.astype(np.float32), 
                ky_grid.astype(np.float32), 
                kz_grid.astype(np.float32)
            )
            # Inverse FFT to get vector potential in real space
            a = self._do_fft(
                a_hat,
                axes=(X_GRID_VEC, Y_GRID_VEC, Z_GRID_VEC),
                forward=False,
                real=False,#np.isrealobj(vector_field),
                norm='forward'
            )
            a = np.real(a).astype(np.float32)
            if debug:
                # Reconstruct the vector field using the optimized curl function
                b_recon = self.vector_curl(a)
                return a, b_recon
            
            return a
            
        else:
            # 2D case
            nx, ny = vector_field.shape[1:]
            # Create wave vectors
            kx = 2 * np.pi * fftfreq(nx, d=self.L[0]/nx)
            ky = 2 * np.pi * fftfreq(ny, d=self.L[1]/ny)
            
            # Create meshgrid
            kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
            k = np.array([kx_grid, ky_grid], dtype=np.float32)
            
            # FFT of vector field
            field_fft = self._do_fft(
                vector_field,
                axes=(X_GRID_VEC, Y_GRID_VEC),
                forward=True,
                real=np.isrealobj(vector_field),
                norm='forward'
            )
            
            # Compute stream function in Fourier space using JIT function
            psi_hat = compute_vector_potential_2D_core(
                k, field_fft, kx_grid, ky_grid
            )
            
            # Inverse FFT to get stream function in real space
            psi = self._do_fft(
                psi_hat,
                axes=(X_GRID_VEC, Y_GRID_VEC),
                forward=False,
                real=False,
                norm='forward'
            )
            psi = np.real(psi).astype(np.float32)
            
            if debug:
                # Reconstruct using gradient operations
                # F = ∇ × ψẑ = (∂ψ/∂y, -∂ψ/∂x)
                grad_psi = self.scalar_gradient(psi)
                F_recon = np.array([grad_psi[1], -grad_psi[0]])
                return psi, F_recon
            return psi