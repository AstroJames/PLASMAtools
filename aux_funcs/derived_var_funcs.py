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
import scipy.fft as fft
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# import derivative stencils
from .derivatives import Derivative
from .tensor_operations import TensorOperations
from .vector_operations import VectorOperations
from .scalar_operations import ScalarOperations

## ###############################################################
## Global variables
## ###############################################################

# indexes
X,Y,Z = 0,1,2

# TODO: make consistent throughout library:
# scalar fields : 1,N,N,N   (i, x, y, z)
# vector fields : 3,N,N,N   (i, x, y, z)
# tensor fields : M,M,N,N,N (i, j, x, y, z)

# shifts for derivatives
F = -1 # shift forwards
B = +1 # shift backwards

boundary_lookup = {0: 'periodic', 
                   1: 'neumann', 
                   2: 'dirichlet'}

## ###############################################################
## Derived Variable Functions
## ###############################################################

class DerivedVars(ScalarOperations,
                  VectorOperations,
                  TensorOperations):
    """
    Class for calculating derived variables from the magnetic field, velocity field, and density field.
    """
    
    def __init__(self, 
                 L              : float    = 1.0,
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
        
        # Set the class variables
        self.L              = L                                     # domain size
        self.bcs            = [boundary_lookup[int(list(bcs)[i])] 
                                for i in range(len(list(bcs)))]     # boundary conditions
        self.mu0            = mu0                                   # magnetic permeability
        self.stencil        = stencil                               # finite difference stencil size
        self.num_of_dims    = num_of_dims                           # number of dimensions in the domain
        
        # create a derivative object to be shared globally
        self.d = Derivative(self.stencil,
                            self.L)
        
        # add the inherited classes
        ScalarOperations.__init__(self)
        VectorOperations.__init__(self,
                                  self.num_of_dims)
        TensorOperations.__init__(self)
        
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
        
        
    def set_stencil(self,
                    stencil : int) -> None:
        """
        Set the stencil for the finite difference derivative.
        
        This function is used to change the stencil order of the 
        finite difference derivative.
        """
        self.stencil = stencil  
        self.set_derivative(self.stencil,
                            self.L)
        
        
    def set_derivative(self,
                       stencil : int,
                       L       : float) -> None:
        """
        Set the stencil for the finite difference derivative.
        
        This function is used to change the stencil order of the 
        finite difference derivative.
        """
        self.d = Derivative(stencil,
                            L)
    
    
    def vector_potential(self,
                         vector_field   : np.ndarray,
                         debug          : bool          = False) -> np.ndarray:
        """
        Create the underlying vector potential, a, of a vector field. For a magnetic field,
        assuming a Coulomb Gauage (div(a) = 0), this is the vector potential that satisfies the equation:
        
        \nabla x b = \nabla x \nabla x a = \nabla (\nabla \cdot a) -\nabla^2 a,

        \nabla \cdot a = 0,
        
        \nabla^2 a = -\nabla x b,
        
        where b is the magnetic field. In Fourier space:
        
        - k^2 \hat{a} = -i k \times \hat{b},
        
        \hat{a} = i \frac{k \times \hat{b}}{k^2},
        
        where k is the wavevector and \hat{a} is the Fourier transform of the vector potential, 
        \hat{b} is the Fourier transform of the magnetic field, and i is the imaginary i = \sqrt{-1}.
        
        Hence a can be found by taking the inverse Fourier transform of \hat{a}.
        
        Author: James Beattie

        Args:
            vector_field (np.ndarray)   : 3,N,N,N array of vector field, where 3 is the vector 
                                            component and N is the number of grid points in each direction
            debug (bool)                : debug flag for debugging the vector potential calculation. 
                                            Default is False.

        Returns:
            a (np.ndarray)       : 3,N,N,N array of vector potential, where 3 is the vector component and 
                                    N is the number of grid points in each direction
            b_recon (np.ndarray) : the original vector field reconstructed from the vector potential for debugging.
        
        """
        
        # Take FFT of vector field
        vector_field_FFT = fft.fftn(vector_field,
                                    norm='forward',
                                    axes=(1,2,3))

        # Assuming a cubic domain    
        N = vector_field.shape[1]  
                        
        # wave vectors
        kx = 2 * np.pi * fft.fftfreq(N, d=self.L/N) / self.L
        ky = 2 * np.pi * fft.fftfreq(N, d=self.L/N) / self.L
        kz = 2 * np.pi * fft.fftfreq(N, d=self.L/N) / self.L
        
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
        k = np.array([kx,ky,kz]) # This will be of shape (3, N, N, N)

        # Normalize k to get the unit wavevector
        k_norm = np.tile(np.linalg.norm(k, 
                                        axis=0, 
                                        keepdims=True), 
                         (3, 1, 1, 1)) # This will be of shape (1, N, N, N)

        # Replace zeros in k_norm with np.inf to avoid division by zero
        k_norm[k_norm == 0] = np.inf
        
        # Take the cross product of k and the vector field
        # Take the inverse FFT to get the vector potential
        a = np.real(fft.ifftn(1j * self.vector_cross_product(k,
                                                             vector_field_FFT) / k_norm**2, 
                              axes=(1, 2, 3),
                              norm="forward"))
        
        # Take the curl of the vector potential to get the reconstructed magnetic field
        # for debuging
        if debug:
            # Space holder for the reconstructed magnetic field
            b_recon = np.zeros_like(vector_field)
            # have to at least a fourth order derivative here 
            # to get a good reconstruction
            self.set_stencil(4)
            b_recon = self.vector_curl(a)  
            self.set_stencil(2)
            return a, b_recon
        
        return a


    def magnetic_helicity(self,
                          magnetic_vector_field : np.ndarray ) -> np.ndarray:
        """
        Compute the magnetic helicity in the Coloumb gauge (gauge fixed).
        
        Author: James Beattie
        
        Args:
            magnetic_vector_field (np.ndarray): 3,N,N,N array of vector field,
            
        Returns:
            magnetic helicity (np.ndarray): N,N,N array of magnetic helicity (a . b).
        
        """

        # compute the magnetic helicity    
        return np.array([self.vector_dot_product(
            self.vector_potential(magnetic_vector_field),
            magnetic_vector_field)])


    def kinetic_helicity(self,
                        velocity_vector_field : np.ndarray ) -> np.ndarray:
        """
        Compute the kinetic helicity of a velocity field.
        
        Author: James Beattie
        
        Args:
            velocity_vector_field (np.ndarray): 3,N,N,N array of vector field,
            
        Returns:
            kinetic helicity (np.ndarray): N,N,N array of kinetic helicity (\omega . v).
        
        """
        
        # compute the kinetic helicity
        return np.array([self.vector_dot_product(
            self.vector_curl(velocity_vector_field),
            velocity_vector_field)])
    
    
    def current_helicity(self,
                         magnetic_vector_field : np.ndarray ) -> np.ndarray:
        """
        Compute the current helicity of a magnetic field.
        
        Author: James Beattie
        
        Args:
            magnetic_vector_field (np.ndarray): 3,N,N,N array of vector field,
            
        Returns:
            current helicity (np.ndarray): N,N,N array of current helicity (j . b).
        
        """

        # compute the current helicity
        return np.array([self.vector_dot_product(
            self.vector_curl(magnetic_vector_field) / self.mu0,
            magnetic_vector_field)])
    

    def gradient_tensor(self, 
                        vector_field: np.ndarray) -> np.ndarray:
        """
        Compute the gradient tensor of a vector field using finite differences and
        multiple threads.
        
        Author: James Beattie

        Args:
            vector_field (np.ndarray): 3,N,N,N array of vector field, 
                                        where 3 is the vector component and N is the number of grid 
                                        points in each direction

        Returns:
            gradient_tensor: the gradient tensor of the vector field, 
            \partial_i f_j
        """
        
        # initialise the gradient tensor
        grad_tensors = np.empty((self.num_of_dims, self.num_of_dims) + vector_field.shape[1:])
        
        # a single component of the gradient tensor
        def compute_gradient(component_idx, coord):
            return self.d.gradient(vector_field[component_idx], 
                                   gradient_dir=coord, 
                                   boundary_condition=self.bcs[component_idx])

        # compute the gradient tensor in parallel
        with ThreadPoolExecutor() as executor:
            futures = {}
            for i in range(self.num_of_dims):
                for j, coord in enumerate(self.coords):
                    futures[(i, j)] = executor.submit(compute_gradient, i, coord)

            for (i, j), future in futures.items():
                grad_tensors[i, j] = future.result()

        return self.tensor_transpose(grad_tensors)           

        
    def orthogonal_tensor_decomposition(self,
                                        tensor_field : np.ndarray ) -> np.ndarray:
        """
        Compute the symmetric, anti-symmetric and bulk components of a rank 2 tensor field.
        
        Consider a gradient tensor of the form:
        
        \partial_i f_j,
        
        then is has a symmetric component:
        
        \partial_i f_j = 0.5 ( \partial_i f_j + \partial_j f_i ) - 1/3 \delta_{ij} \partial_k f_k,
        
        an anti-symmetric component:
        
        \partial_i f_j = 0.5 ( \partial_i f_j - \partial_j f_i ),
        
        and a trace component:
        
        \partial_i f_j = 1/3 \delta_{ij} \partial_k f_k.
        
        Author: James Beattie

        Args:
            tensor_field (np.ndarray): M,M,N,N,N array of tensor field, where M,M are the tensor components 

        Returns:
            sym_tensor (np.ndarray) : M,M,N,N,N array of symmetric tensor field.
            tensor_anti (np.ndarray): M,M,N,N,N array of anti-symmetric tensor field.
            tensor_bulk (np.ndarray): M,M,N,N,N array of trace tensor field.
        """
        
        # transpose
        tensor_transpose = self.tensor_transpose(tensor_field)
        
        # bulk component
        tensor_bulk = (1./self.num_of_dims) * np.einsum('...,ij...->ij...',
                                                        np.einsum("ii...",tensor_field),
                                                        np.identity(self.num_of_dims))
        
        # symmetric component
        tensor_sym = 0.5 * (tensor_field + tensor_transpose) - tensor_bulk
        
        # anti-symmetric component
        tensor_anti = 0.5 * (tensor_field - tensor_transpose)
        
        return tensor_sym, tensor_anti, tensor_bulk


    def vorticity_decomp(self,
                         velocity_vector_field    : np.ndarray,
                         magnetic_vector_field    : np.ndarray    = None,
                         density_scalar_field     : np.ndarray    = None,
                         pressure_scalar_field    : np.ndarray    = None) -> np.ndarray:
        """
        Compute the terms in the vorticity equation, including the compressive, stretching, baroclinic,
        baroclinic magnetic, and tension terms. The magnetised (ideal) vorticity equation is given by:
        
        \frac{D \omega}{Dt} = - (\nabla . u) \omega + \omega . \nabla u +
            \frac{1}{\rho^2} \nabla p \times \nabla \rho + 
            \frac{1}{\rho^2} \nabla \rho \times \nabla b^2 / 2\mu_0 + 
            \nabla \times (1/\rho) b . \nabla b / \mu_0,
            
        where \omega = \nabla \times u is the vorticity, u is the velocity field, p is the pressure field,
        \rho is the density field, and b is the magnetic field and D/Dt is the Lagrangian derivative. 
        
        The terms are:
        
        compression term            : - (\nabla . u) \omega,
        stretching term             : \omega . \nabla u,
        baroclinic term             : 1/\rho^2 \nabla p \times \nabla \rho,
        baroclinic magnetic term    : 1/\rho^2 \nabla \rho \times \nabla b^2 / 2\mu_0,
        tension term                : 1/\mu_0 \nabla \times (1/\rho) b . \nabla b.
        
        hence the vorticity equation can be written as:
        
        \frac{D \omega}{Dt} = compression + stretching + baroclinic + baroclinic_magnetic + tension.

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
        
        # compute the vorticity, \omega = \nabla \times u
        omega = self.vector_curl(velocity_vector_field)
        
        # vorticity compression term, - (\nabla . u) \omega
        compress = - omega * self.vector_divergence(velocity_vector_field)   
        
        # vortex stretching term, \omega . \nabla u
        stretch = self.vector_dot_tensor(omega,
                                         self.gradient_tensor(velocity_vector_field))
        
        # if the magnetic and density is not None, compute the magnetic terms
        if ( magnetic_vector_field is not None ) and ( density_scalar_field is not None ):
                    
            # magnetic baroclinic term, 1/\rho^2 \nabla \rho \times \nabla b^2 / 2\mu_0
            baroclinic_magnetic = (1./density_scalar_field[X]**2) * self.vector_cross_product(
                self.scalar_gradient(self.vector_dot_product(magnetic_vector_field,
                                                             magnetic_vector_field) / (2 * self.mu0)),
                self.scalar_gradient(density_scalar_field[X]))
            
            # magnetic tension term 1/\mu_0 \nabla \times (1/\rho) b . \nabla b)
            tension = self.vector_curl(
                (1./density_scalar_field[X]) * self.vector_dot_tensor(
                    magnetic_vector_field,
                    self.gradient_tensor(magnetic_vector_field)) / self.mu0
                )
            
        # if density and pressure is not None, compute the baroclinic term
        if ( density_scalar_field is not None ) and ( pressure_scalar_field is not None ):
            
            # baroclinic term, 1/\rho^2 \nabla p \times \nabla \rho
            baroclinic = (1/ density_scalar_field[X]**2) * self.vector_cross_product(
                self.scalar_gradient(pressure_scalar_field[X]),
                self.scalar_gradient(density_scalar_field[X])
                )
            
        return omega, compress, stretch, baroclinic, baroclinic_magnetic, tension
        
        
    def symmetric_eigvals(self, matrix, find_vectors = False):
        """Finds the eigenvalues of a symmetric 3x3 matrix from https://hal.science/hal-01501221/document

        Parameters
        ----------
        matrix : numpy ndarray shape (3,3,Nx,Ny,Nz)
            must be a real symmetric 3x3 matrix defined pointwise in an arbitrary grid.
        find_vectors : bool, optional
            If True, the eigenvectors will be computed as well. Default is False to save computing time.

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
        a = matrix[0,0,:,:,:]
        b = matrix[1,1,:,:,:]
        c = matrix[2,2,:,:,:]
        d = matrix[0,1,:,:,:]
        e = matrix[1,2,:,:,:]
        f = matrix[0,2,:,:,:]

        #begin the computations
        x1 = a**2 + b**2 + c**2 - a*b - a*c - b*c+3*(d**2 + f**2 + e**2)
        x2 = (-1)*(2*a-b-c)*(2*b-a-c)*(2*c-a-b) + 9*((2*c-a-b)*d**2 + (2*b-a-c)*f**2 + (2*a-b-c)*e**2) - 54*d*e*f

        #define what phi is conditional to previous variables
        condition_list = [x2>0, x2==0, x2<0]
        choice_list = [np.arctan((np.sqrt(4*x1**3-x2**2))/(x2)), 
                       np.pi/2, 
                       np.arctan((np.sqrt(4*x1**3-x2**2))/(x2))+np.pi]
        phi = np.select(condition_list, choice_list)

        #calculate the eigenvalues
        sqrt_x1 = np.sqrt(x1)
        lambda1 = (a+b+c-2*sqrt_x1*np.cos(phi/3))/3
        lambda2 = (a+b+c+2*sqrt_x1*np.cos((phi-np.pi)/3))/3
        lambda3 = (a+b+c+2*sqrt_x1*np.cos((phi+np.pi)/3))/3
        eig_array = np.array([lambda1, lambda2, lambda3])

        #perform the sort, saving indices of sort to use on eigenvectors later
        idx = np.argsort(eig_array, axis=0)
        eig_array = np.take_along_axis(eig_array, idx, axis=0)
        
        if find_vectors:
            #compute the eigenvectors
            m1 = (d*(c-lambda1) - e*f) / (f*(b-lambda1) - d*e)
            m2 = (d*(c-lambda2) - e*f) / (f*(b-lambda2) - d*e)
            m3 = (d*(c-lambda3) - e*f) / (f*(b-lambda3) - d*e)

            vec1 = [(lambda1 - c - e * m1)/f, m1, np.ones(np.shape(m1))]
            vec2 = [(lambda2 - c - e * m2)/f, m2, np.ones(np.shape(m2))]
            vec3 = [(lambda3 - c - e * m3)/f, m3, np.ones(np.shape(m3))]
            vec_array = np.array([vec1, vec2, vec3])

            #do the corresponding sort on the vec array
            vec_array[:,0,:,:,:] = np.take_along_axis(vec_array[:,0,:,:,:], idx, axis=0)
            vec_array[:,1,:,:,:] = np.take_along_axis(vec_array[:,1,:,:,:], idx, axis=0)
            vec_array[:,2,:,:,:] = np.take_along_axis(vec_array[:,2,:,:,:], idx, axis=0)

            return eig_array, vec_array
        else:
            return eig_array
        

    def helmholtz_decomposition(self,
                                vector_field : np.ndarray) -> np.ndarray:
        """
        Compute the irrotational and solenoidal components of a vector field using the Helmholtz decomposition
        in Fourier space (assumes periodic boundary conditions).
        
        Author: James Beattie 
        
        Args:
            vector_field (np.ndarray): 3,N,N,N array of vector field, where 
                                        3 is the vector component and N is the number of grid points in each 
                                        direction

        Returns:
            F_irrot (np.ndarray) : 3,N,N,N array of irrotational component of the vector field (curl free)
            F_solen (np.ndarray) : 3,N,N,N array of solenoidal component of the vector field (divergence free)
        
        """
        # F is a 4D array, with the last dimension being 3 (for the x, y, z components of the vector field)
        # TODO: change the shape of F to be (3, N, N, N) instead of (N, N, N, 3) (consistent with other functions)
        
        shape = vector_field.shape[:-1]
        x     = np.linspace(-self.L/2.0,
                            self.L/2.0,
                            vector_field.shape[0]) # assuming a domian of [-L/2, L/2]
        
        # Fourier transform to Fourier space    
        Fhat = fft.fftn(vector_field, axes=(0, 1, 2),norm = 'forward')
        
        Fhat_irrot = np.zeros_like(Fhat, dtype=np.complex128)
        Fhat_solen = np.zeros_like(Fhat, dtype=np.complex128)
        norm       = np.zeros(shape, dtype=np.float64)
        
        # Compute wave numbers
        kx = 2*np.pi * np.fft.fftfreq(shape[X]) * shape[X] / (x[-1] - x[0])
        ky = 2*np.pi * np.fft.fftfreq(shape[Y]) * shape[Y] / (x[-1] - x[0])
        kz = 2*np.pi * np.fft.fftfreq(shape[Z]) * shape[Z] / (x[-1] - x[0])
        kX, kY, kZ = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Avoid division by zero
        norm = kX**2 + kY**2 + kZ**2
        norm[0, 0, 0] = 1
        
        # Compute divergence and curl in Fourier space (note python doesn't seem to want to use i)
        divFhat = (kX * Fhat[..., X] + kY * Fhat[..., Y] + kZ * Fhat[..., Z])
        
        # Compute irrotational and solenoidal components in Fourier space
        Fhat_irrot = np.transpose(divFhat * np.array([kX, kY, kZ]) / norm[np.newaxis, ...],(1,2,3,0))
        Fhat_solen = Fhat - Fhat_irrot #curlFhat / norm[np.newaxis, ...]
        
        # Inverse Fourier transform to real space
        F_irrot = fft.ifftn(Fhat_irrot, axes=(X,Y,Z),norm = 'forward').real
        F_solen = fft.ifftn(Fhat_solen, axes=(X,Y,Z),norm = 'forward').real
        
        # Remove numerical noise
        # threshold = 1e-16
        # F_solen[np.abs(F_solen) < threshold] = 0
        # F_irrot[np.abs(F_irrot) < threshold] = 0
        
        return F_irrot, F_solen


    def vector_curl(self,
                    vector_field : np.ndarray) -> np.ndarray:
        """
        Compute the curl of a vector field (assumes periodic boundary conditions).
        
        Author: James Beattie
        
        Args:
            vector_field (np.ndarray)   : 3,N,N,N array of vector field,
                                            where 3 is the vector component and N is the number of grid
                                            points in each direction
            
        Returns:
            curl vector field (np.ndarray) : 3,N,N,N array of curl of the vector field
        
        """
        
        if self.num_of_dims == 1:
            ValueError("Vector curl is not defined for 1D.")
        elif self.num_of_dims == 2:
            return np.array([0,
                             0,
                             self.d.gradient(vector_field[Y],
                                gradient_dir       = X,
                                boundary_condition = self.bcs[X]) - 
                             self.d.gradient(vector_field[X],
                                gradient_dir       = Y,
                                boundary_condition = self.bcs[Y])])
        elif self.num_of_dims == 3: 
            return np.array([self.d.gradient(vector_field[Z],
                                gradient_dir        = Y,
                                boundary_condition  = self.bcs[Y]) - 
                            self.d.gradient(vector_field[Y],
                                gradient_dir        = Z,
                                boundary_condition  = self.bcs[Z]),
                            self.d.gradient(vector_field[X],
                                gradient_dir        = Z,
                                boundary_condition  = self.bcs[Z]) -
                            self.d.gradient(vector_field[Z],
                                gradient_dir        = X,
                                boundary_condition  = self.bcs[X]),
                            self.d.gradient(vector_field[Y],
                                gradient_dir        = X,
                                boundary_condition  = self.bcs[X]) - 
                            self.d.gradient(vector_field[X],
                                gradient_dir        = Y,
                                boundary_condition  = self.bcs[Y])])
    

    def vector_divergence(self,
                          vector_field : np.ndarray) -> np.ndarray:
        """
        Compute the vector divergence (assumes periodic boundary conditions).
        
        Author: James Beattie
        
        Args:
            vector_field (np.ndarray)   : 3,N,N,N array of vector field,
                                            where 3 is the vector component and N is the number of grid
                                            points in each direction
            
        Returns:
            divergence vector field (np.ndarray) : 1,N,N,N array of divergence of the vector field
        
        """
        
        return np.sum(
            np.array([self.d.gradient(vector_field[coord],
                                gradient_dir       = coord,
                                boundary_condition = self.bcs[coord]) for coord in self.coords]),
            axis=0)
    

    def vector_laplacian(self,
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
                    
        return np.array([self.scalar_laplacian(vector_field[coord]) 
                         for coord in self.coords])

        
    def scalar_laplacian(self,
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
        return np.sum(
            np.array([self.d.gradient(scalar_field,
                                gradient_dir       = coord,
                                derivative_order   = 2, 
                                boundary_condition = self.bcs[coord]) for coord in self.coords]),
            axis=0)


    def scalar_gradient(self,
                        scalar_field : np.ndarray) -> np.ndarray:
        """
        Compute the gradient of a scalar field, grad(\phi). 
        
        Author: Neco Kriel & James Beattie
        
        Args:
            scalar_field (np.ndarray)   : N,N,N array of vector field,
                                            where 3 is the vector component and N is the number of grid
                                            points in each direction

        Returns:
            grad_scalar_field (np.ndarray) : 3,N,N,N array of gradient of the scalar field
        
        """

        return np.array([self.d.gradient(scalar_field, 
                                         gradient_dir = coord) 
                         for coord in self.coords])


    def compute_TNB_basis(self,
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
        vector_field    = np.array(vector_field)
        field_magn      = self.vector_magnitude(vector_field)
        ## ---- COMPUTE TANGENT BASIS
        t_basis = vector_field / field_magn
        ## df_j/dx_i: (component-j, gradient-direction-i, x, y, z)
        gradient_tensor = np.array([
            self.scalar_gradient(field_component)
            for field_component in vector_field
        ])
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


    def TNB_coordinate_transformation(self,
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
    
     
    def TNB_jacobian_stability_analysis(self,
                                        vector_field    : np.ndarray,
                                        traceless       : bool          = True ) -> np.ndarray:
        """
        Compute the trace, determinant and eigenvalues of the Jacobian of a vector field in the 
        TNB coordinate system of an underlying vector field.
        
        See: https://arxiv.org/pdf/2312.15589.pdf
        
        Author: James Beattie
        
        Args:
            args (_type_): 

        Returns:
            _type_: _description_
        
        """
        
        def theta_eig(J_thresh  : np.ndarray,
                    J_3         : np.ndarray) -> np.ndarray:
            """
            Compute the angle between the eigenvectors of the Jacobian.
            
            See: https://arxiv.org/pdf/2312.15589.pdf
            
            Author: James Beattie
            
            Args:
                args (_type_): 

            Returns:
                _type_: _description_
            
            """
            
            # Two conditions for O and X points
            condition = np.abs(J_3) < J_thresh
            ratio = np.where(condition, 
                            J_thresh / J_3, 
                            J_3 / J_thresh)
            
            return np.arctan( np.sqrt(ratio**2-1) )
        
        # Compute jacobian of B field
        self.set_stencil(6)
        jacobian = self.gradient_tensor(vector_field)
        self.set_stencil(2)
        
        # Make jacobian traceless (numerical errors will result in some trace, which is
        # equivalent to div(B) modes)
        if traceless:   
            jacobian = jacobian - (1/self.num_of_dims) * np.einsum("..., ij... -> ij...",
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
        M    = np.array([[M_11, M_12],
                        [M_21, M_22]])
        
        # Compute trace and determinant of M
        trace_M = np.einsum("ii...",M)
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


    def classification_of_critical_points(self,
                                          trace_M  : np.ndarray,
                                          D        : np.ndarray,
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
            
            is_3D = np.abs(trace_M) > 0.0
            is_2D = np.isclose(trace_M,0.0)
            is_real_eig = np.abs(J_3) < J_thresh
            is_imag_eig = np.abs(J_3) > J_thresh
            is_parallel = np.isclose(np.abs(J_3),J_thresh,1e-3)
                
            # real and imaginary components of the eigenvalues
            eig1_real = np.real(eig_1)
            eig2_real = np.real(eig_2)
            
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
            classification_array[1,...] += np.logical_and.reduce([is_3D, 
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
        
        
## END OF LIBRARY