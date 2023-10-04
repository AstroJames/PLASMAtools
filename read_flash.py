## ###############################################################
## IMPORTS
## ###############################################################

from h5py import File
import numpy as np
import numpy.polynomial.polynomial as poly
import timeit
from .aux_funcs import derived_var_funcs as dvf
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

## ###############################################################
## Global variabes
## ###############################################################

field_lookup_type = {
    "dens": "scalar",
    "dvvl": "scalar",
    "alpha": "scalar",
    "vel" : "vector",
    "mag" : "vector",
    "tens": "vector",
    "vort": "vector",
    "mpr" : "vector",
    "tprs": "vector",
    "cur" : "vector",
    "vxb" : "vector",
    "mXc" : "vector",
    "mXc_mag": "scalar"
}

## ###############################################################
## Classes
## ###############################################################

class Particles:

    def __init__(self,filename) -> None:
        # file attributes
        self.filename       = filename
        # particle attribues 
        self.n_particles    = 0
        self.__set_number_of_particles()
        self.blocks         = np.zeros(self.n_particles, dtype=np.int32)
        self.id             = np.zeros(self.n_particles, dtype=np.int32)
        # particle physics attribues 
        self.dens           = np.zeros(self.n_particles, dtype=np.float32)
        self.posx           = np.zeros(self.n_particles, dtype=np.float32)
        self.posy           = np.zeros(self.n_particles, dtype=np.float32)
        self.posz           = np.zeros(self.n_particles, dtype=np.float32)
        self.velx           = np.zeros(self.n_particles, dtype=np.float32)
        self.vely           = np.zeros(self.n_particles, dtype=np.float32)
        self.velz           = np.zeros(self.n_particles, dtype=np.float32)
        self.magx           = np.zeros(self.n_particles, dtype=np.float32)
        self.magy           = np.zeros(self.n_particles, dtype=np.float32)
        self.magz           = np.zeros(self.n_particles, dtype=np.float32)

    def __set_number_of_particles(self) -> None:
        """
        This function reads the number of particles in the FLASH particle file
        """
        g = File(self.filename, "r")
        self.n_particles = len(g['tracer particles'][:,0])
        print(f"Numer of particles: {self.n_particles}")
        g.close()

    def __reformat_part_str(self,
                            part_str,
                            idx: int) -> str:
        """
        This function reformats the particle string to be in the correct format
        """
        return str(part_str[idx][0]).split("'")[1]

    def read(self,
             part_str) -> None:
        """
        This function reads in the FLASH particle block data,
        i.e., the block membership of each particle
        """

        block_idx_dic = {"block":0,
                         "id"   :6,
                         "dens" :1,
                         "posx" :2,
                         "posy" :3,
                         "posz" :4,
                         "velx" :7,
                         "vely" :8,
                         "velz" :9,
                         "magx" :10,
                         "magy" :11,
                         "magz" :12
        }

        field_lookup_type = {
            "dens"  : "scalar",
            "id"    : "scalar",
            "block" : "scalar",            
            "vel"   : "vector",
            "mag"   : "vector",
            "pos"   : "vector",
            "vort"  : "vector",
            "mpr"   : "vector",
            "tprs"  : "vector"
        }

        if field_lookup_type[part_str] == "scalar":
            # read single component particle data
            g = File(self.filename, 'r')
            print(f"Reading in particle attribute: {self.__reformat_part_str(g['particle names'],block_idx_dic[part_str])}")
            if part_str == "id":
                setattr(self,
                        part_str,
                        g['tracer particles'][:,block_idx_dic[part_str]].astype(int))
            else:
                setattr(self,
                        part_str,
                        g['tracer particles'][:,block_idx_dic[part_str]])
            g.close() 

        elif field_lookup_type[part_str] == "vector":
            # read vector component particle data
            g = File(self.filename, 'r')
            for coord in ["x","y","z"]:
                data_set_str = f"{part_str}{coord}"
                print(f"Reading in particle attribute: {self.__reformat_part_str(g['particle names'],block_idx_dic[data_set_str])}")
                setattr(self,
                        data_set_str,
                        g['tracer particles'][:,block_idx_dic[data_set_str]])
            g.close()      

    def sort_particles(self):
        # sort quick sort particles by id (O n log n)
        print("sort_particles: beginning to sort particles by id.")
        idx = np.argsort(self.id)
        print("sort_particles: finished sorting particles by id.")
        # loop through all of the attributes and sort them
        for attr in self.__dict__:
            if (attr != "filename") and (attr != "n_particles"):
                setattr(self,attr,getattr(self,attr)[idx])



class Fields():

    def __init__(self,
                 filename: str,
                 reformat: bool = False) -> None:
        """
        Initialize a FLASHGridData object by reading in the data from the specified file.

        Parameters
        ----------
        filename : str 
            The name of the file containing the FLASH grid data.
        reformat : bool, optional
            Whether to reformat the data in the file into 3D arrays (True) or keep it in 1D arrays (False).
            Default is False.

        Attributes
        ----------
        filename : str
            The name of the file containing the FLASH grid data.
        reformat : bool
            Whether to reformat the data in the file into 3D arrays (True) or keep it in 1D arrays (False).
        n_cores : int
            The number of cores used in the simulation.
        nxb : int
            The number of blocks in the x direction.
        nyb : int
            The number of blocks in the y direction.
        nzb : int
            The number of blocks in the z direction.
        n_cells : int
            The total number of cells in the simulation.
        int_properties : dict
            A dictionary containing integer simulation properties.
        str_properties : dict
            A dictionary containing string simulation properties.
        logic_properties : dict
            A dictionary containing logical simulation properties.
        dens : numpy array
            The density field.
        velx : numpy array
            The x velocity field.
        vely : numpy array
            The y velocity field.
        velz : numpy array
            The z velocity field.
        magx : numpy array
            The x magnetic field.
        magy : numpy array
            The y magnetic field.
        magz : numpy array
            The z magnetic field.
        tensx : numpy array
            The x magnetic tension field.
        tensy : numpy array
            The y magnetic tension field.
        tensz : numpy array
            The z magnetic tension field.
        vortx : numpy array
            The x vorticity field.
        vorty : numpy array
            The y vorticity field.
        vortz : numpy array
            The z vorticity field.
        curx: numpy array
            The x current field.
        cury: numpy array
            The y current field.
        curz: numpy array
            The z current field.

        """

        # simulation attributes
        self.filename           = filename
        self.reformat           = reformat
        self.n_cores            = 0
        self.nxb                = 0
        self.nyb                = 0
        self.nzb                = 0
        self.n_cells            = 0
        self.int_properties     = {}
        self.str_properties     = {}
        self.logic_properties   = {} 

        # read in the simulation properties
        self.__read_sim_properties()  
        self.__read_sim_cells() 

        # grid data attributes
        # if the data is going to be reformated, preallocate the 3D
        # arrays for the grid data
        if self.reformat:
            init_field = np.zeros((self.nyb*self.int_properties["jprocs"],
                                   self.nxb*self.int_properties["iprocs"],
                                   self.nzb*self.int_properties["kprocs"]), dtype=np.float32)
        else:
            # otherwise, preallocate the 1D arrays for the grid data
            init_field = np.zeros(self.n_cells, dtype=np.float32)

        self.dens               = init_field
        self.velx               = init_field
        self.vely               = init_field
        self.velz               = init_field
        self.magx               = init_field
        self.magy               = init_field
        self.magz               = init_field
        self.tensx              = init_field
        self.tensy              = init_field
        self.tensz              = init_field
        self.vortx              = init_field
        self.vorty              = init_field
        self.vortz              = init_field   
        self.curx               = init_field
        self.cury               = init_field
        self.curz               = init_field 
        self.vxbx               = init_field
        self.vxby               = init_field
        self.vxbz               = init_field
        self.alpha              = init_field
        self.mXcx               = init_field
        self.mXcy               = init_field
        self.mXcz               = init_field
        self.mXc_mag            = init_field

    def __read_sim_cells(self) -> None:
        """
        This function reads in the number of cores from the FLASH file.
        """
        g = File(self.filename, 'r')
        self.n_cores    = g['dens'].shape[0]
        self.nxb        = g['dens'].shape[3]
        self.nyb        = g['dens'].shape[1]
        self.nzb        = g['dens'].shape[2]
        self.n_cells    = self.n_cores*self.nxb*self.nyb*self.nzb
        #print(f"Number of cells: {self.n_cells}")
        g.close()

    def __read_sim_properties(self) -> None:
        """
        This function reads in the FLASH field properties.
        """
        g = File(self.filename, 'r')
        self.int_properties     = {str(key).split("'")[1].strip(): value for key, value in g["integer runtime parameters"]}
        self.str_properties     = {str(key).split("'")[1].strip(): str(value).split("'")[1].strip() for key, value in g["string runtime parameters"]}
        self.logic_properties   = {str(key).split("'")[1].strip(): value for key, value in g["logical runtime parameters"]}
        g.close()

    def set_reformat(self,
                     reformat: bool) -> None:
        """
        This function sets the reformat flag for the FLASH field data.
        """
        self.reformat = reformat

    def read(self,
             field_str: str,
             vector_magnitude: bool = False,
             debug: bool = False) -> None:
        """
        This function reads in the FLASH grid data
        """

        if field_lookup_type[field_str] == "scalar":
            g = File(self.filename, 'r')
            print(f"Reading in grid attribute: {field_str}")
            if self.reformat:
                print(f"Reading in reformatted grid attribute: {field_str}")
                setattr(self, field_str, 
                        reformat_FLASH_field(g[field_str][:,:,:,:],
                                            self.nxb,
                                            self.nyb,
                                            self.nzb,
                                            self.int_properties["iprocs"],
                                            self.int_properties["jprocs"],
                                            self.int_properties["kprocs"],
                                            debug))
            else:
                setattr(self, field_str, g[field_str][:,:,:,:])
            g.close()
        
        elif field_lookup_type[field_str] == "vector":
            g = File(self.filename, 'r')
            for coord in ["x","y","z"]:
                print(f"Reading in grid attribute: {field_str}{coord}")
                if self.reformat:
                    #time1 = timeit.default_timer()
                    field_var = reformat_FLASH_field(g[f"{field_str}{coord}"][:,:,:,:],
                                                    self.nxb,
                                                    self.nyb,
                                                    self.nzb,
                                                    self.int_properties["iprocs"],
                                                    self.int_properties["jprocs"],
                                                    self.int_properties["kprocs"],
                                                    debug)
                    print(f"Reading in reformatted grid attribute: {field_str}{coord}")
                    #time2 = timeit.default_timer()
                    #print(f"The total time it took is: {time2-time1}")
                    if vector_magnitude: # if the mag is required, add and accumulate the square of the components
                        if coord == "x":
                            field_mag = field_var**2
                        else:
                            field_mag += field_var**2
                    else:
                        setattr(self, f"{field_str}{coord}",field_var)
                else:
                    if vector_magnitude:
                        if coord == "x":
                            field_mag = g[f"{field_str}{coord}"][:,:,:,:]**2
                        else:
                            field_mag += g[f"{field_str}{coord}"][:,:,:,:]**2
                    else:
                        setattr(self, f"{field_str}{coord}", g[f"{field_str}{coord}"][:,:,:,:])
            if vector_magnitude:
                setattr(self, f"{field_str}_mag", np.sqrt(field_mag))
            
            g.close()
            
    def derived_var(self,
                    field_str: str,
                    eta: float = 0.0,
                    nu: float = 0.0 ) -> None:
        
        var_lookup_table = {
            "E": ["Ex","Ey","Ez"],
            "ExB": ["ExBx","ExBy","ExBz"],
            "VxB": ["VxBx","VxBy","VxBz"],
            "jacobian_mag": ["eig_1","eig_2","eig_3"],
            "helmholtz": ["vel_comx","vel_comy","vel_comz","vel_solx","vel_soly","vel_solz"]
        }
        
        # grid data attributes
        # if the data is going to be reformated, preallocate the 3D
        # arrays for the grid data
        if self.reformat:
            init_field = np.zeros((self.nyb*self.int_properties["jprocs"],
                                self.nxb*self.int_properties["iprocs"],
                                self.nzb*self.int_properties["kprocs"]), dtype=np.float32)
        else:
            # otherwise, preallocate the 1D arrays for the grid data
            init_field = np.zeros(self.n_cells, dtype=np.float32)
        
        if field_str not in var_lookup_table:
            raise Exception(f"derived_var: {field_str} not in new_var_lookup_table. Add the variable defn. first.")
        
        for field in var_lookup_table[field_str]:    
            setattr(self, field, init_field)
            
        # Variable definitions
        if field_str == "E" or field_str == "ExB":
            print(f"derived_var: Calculating E field.")
            
            if eta == 0.0:
                Warning(f"derived_var: eta is 0.0. E fields will not include current.")
            
            # read in the grid data
            self.read("vel")
            self.read("mag")
            if eta != 0.0:
                self.read("cur")
            
            # calculate the new variable
            Ex = eta*self.curx/(4*np.pi) - (self.vely*self.magz - self.velz*self.magy) #+eta*self.curx/(4*np.pi) 
            Ey = eta*self.cury/(4*np.pi) - (self.velz*self.magx - self.velx*self.magz) #+eta*self.cury/(4*np.pi) 
            Ez = eta*self.curz/(4*np.pi) - (self.velx*self.magy - self.vely*self.magx) #+eta*self.curz/(4*np.pi) 
            
            # write the new variable to the object
            setattr(self, "Ex", Ex)
            setattr(self, "Ey", Ey)
            setattr(self, "Ez", Ez)
            
        if field_str == "VxB":
            """
            the velocity field cross the magnetic field.
            
            """
            
            self.read("vel")
            self.read("mag")
        
            VxBx = self.vely*self.magz - self.velz*self.magy
            VxBy = self.velz*self.magx - self.velx*self.magz
            VxBz = self.velx*self.magy - self.vely*self.magx
            
            # write the new variable to the object
            setattr(self, "VxBx", VxBx)
            setattr(self, "VxBy", VxBy)
            setattr(self, "VxBz", VxBz)
            
        if field_str == "ExB":
            """
            the electric field cross magnetic field.
            
            """
            
            ExBx = self.Ey*self.magz - self.Ez*self.magy
            ExBy = self.Ez*self.magx - self.Ex*self.magz
            ExBz = self.Ex*self.magy - self.Ey*self.magx
            
            # write the new variable to the object
            setattr(self, "ExBx", ExBx)
            setattr(self, "ExBy", ExBy)
            setattr(self, "ExBz", ExBz)
            
        if field_str == "jacobian_mag":
            """
            Compute the Jacobian of the magnetic field and its eigenvalues.
            
            """
            
            if self.reformat:            
                self.read("mag")
                
                # assume that the box is cubic with the same dx, dy, dz
                two_dX = 2.0/self.magx.shape[0]
                two_dY = two_dX
                two_dZ = two_dX
                
                print("derived_var: Calculating the Jacobian of the magnetic field.")
                
                # x components
                dBx_dx = (np.roll(self.magx, -1, axis=0) - np.roll(self.magx, 1, axis=0))/two_dX
                dBx_dy = (np.roll(self.magx, -1, axis=1) - np.roll(self.magx, 1, axis=1))/two_dY
                dBx_dz = (np.roll(self.magx, -1, axis=2) - np.roll(self.magx, 1, axis=2))/two_dZ
                
                # y components
                dBy_dx = (np.roll(self.magy, -1, axis=0) - np.roll(self.magy, 1, axis=0))/two_dX
                dBy_dy = (np.roll(self.magy, -1, axis=1) - np.roll(self.magy, 1, axis=1))/two_dY
                dBy_dz = (np.roll(self.magy, -1, axis=2) - np.roll(self.magy, 1, axis=2))/two_dZ
                
                # z components
                dBz_dx = (np.roll(self.magz, -1, axis=0) - np.roll(self.magz, 1, axis=0))/two_dX
                dBz_dy = (np.roll(self.magz, -1, axis=1) - np.roll(self.magz, 1, axis=1))/two_dY
                dBz_dz = (np.roll(self.magz, -1, axis=2) - np.roll(self.magz, 1, axis=2))/two_dZ
                
                # Jacobian
                Jacobian = np.zeros((self.magx.shape[0], 
                                    self.magx.shape[1], 
                                    self.magx.shape[2], 3, 3))
                
                Jacobian[..., 0, 0] = dBx_dx
                Jacobian[..., 0, 1] = dBx_dy
                Jacobian[..., 0, 2] = dBx_dz
                Jacobian[..., 1, 0] = dBy_dx
                Jacobian[..., 1, 1] = dBy_dy
                Jacobian[..., 1, 2] = dBy_dz
                Jacobian[..., 2, 0] = dBz_dx
                Jacobian[..., 2, 1] = dBz_dy
                Jacobian[..., 2, 2] = dBz_dz
                
                print("derived_var: Jacobian of the magnetic field calculated.")
            
                # Calculate the eigenvalues of the Jacobian without looping

                # Compute the coefficients for the characteristic equation
                """
                lamba^3 - Tr(J)lambda^2 + ( Tr(J^2) /2 - Tr(J)^2/2 )lamba - Det(J) = 0
                
                """
                
                print("derived_var: Calculating the eigenvalues of the Jacobian via characteristic equation.")
                
                trace_A = np.trace(Jacobian, axis1=-2, axis2=-1)
                det_A = np.linalg.det(Jacobian)
                trace_A_sq = trace_A ** 2
                trace_A2 = np.einsum('...ii->...',np.matmul(Jacobian, Jacobian))

                # Cubic coefficients
                a = -trace_A
                b = (trace_A2 - trace_A_sq) / 2
                c = -det_A

                print("derived_var: Characteristic equation calculated.")

                # Calculate eigenvalues using numpy's roots function for a cubic equation
                coefficients = np.stack([np.ones(np.shape(a)), a, b, c], axis=-1)
                
                # Reshape coefficients for a flattened spatial dimension
                coefficients_reshaped = coefficients.reshape(-1, 4)
                
                print("derived_var: Calculating the eigenvalues of the Jacobian.")
                
                # Progress bar setup
                #pbar = tqdm(total=coefficients_reshaped.shape[0], desc="Computing roots")
                
                # define progress bar update
                def roots_with_progress(i):
                    return poly.polyroots(coefficients_reshaped[i])
                
                #eigenvalues = np.apply_along_axis(roots_with_progress, 1, coefficients_reshaped)
                #eigenvalues = Parallel(n_jobs=64)(delayed(roots_with_progress)(i) for i in range(coefficients_reshaped.shape[0]))
                #eigenvalues = np.array(eigenvalues)
                
                # Close the progress bar
                #pbar.close()
                
                print("derived_var: Eigenvalues of the Jacobian calculated.")
            
                #eigenvalues = eigenvalues.reshape(*np.shape(a), 3)
            
                # eigenvalues
                setattr(self, "eig_1", a)#eigenvalues[...,0])
                setattr(self, "eig_2", b)#eigenvalues[...,1])
                setattr(self, "eig_3", c)#eigenvalues[...,2])
        else:
            print("Cannot compute the gradient of the magnetic field without reformating the data.")
                
        if field_str == "helmholtz":
            self.read("vel")
            
            F_irrot, F_solen = dvf.helmholtz_decomposition(np.array([self.velx, self.vely, self.velz]))
            
            for coords, idx in enumerate(["x", "y", "z"]):
                setattr(self, f"vel_comp{coords}", F_irrot[idx])
                setattr(self, f"vel_sol{coords}", F_solen[idx])
    

class PowerSpectra():
    
    def __init__(self,
                 filename: str) -> None:
        self.filename: str              = filename
        self.names: list                = []
        self.wavenumber: list           = []
        self.power:list                 = [] 
        self.correlation_scale: float   = 0.0
        self.microscale:float           = 0.0
        self.peakscale: float           = 0.0
        
    def set_power_spectra_header(self,
                                 field_str: str) -> None:
        if field_lookup_type[field_str] == "vector":
            self.names = ["#00_BinIndex",
                     "#01_KStag",
                     "#02_K",
                     "#03_DK",
                     "#04_NCells",
                     "#05_SpectDensLgt",
                     "#06_SpectDensLgtSigma",
                     "#07_SpectDensTrv",
                     "#08_SpectDensTrvSigma",
                     "#09_SpectDensTot",
                     "#10_SpectDensTotSigma",
                     "#11_SpectFunctLgt",
                     "#12_SpectFunctLgtSigma",
                     "#13_SpectFunctTrv",
                     "#14_SpectFunctTrvSigma",
                     "#15_SpectFunctTot",
                     "#16_SpectFunctTotSigma",
                     "#17_CompSpectFunctLgt",
                     "#18_CompSpectFunctLgtSigma",
                     "#19_CompSpectFunctTrv",
                     "#20_CompSpectFunctTrvSigma",
                     "#21_DissSpectFunct",
                     "#22_DissSpectFunctSigma"]
    
    def read(self,
             filename: str,
             field_str: str) -> None:
        
        skip = 6
        p_spec = pd.read_table(self.filename,
                      names=self.names,
                      skiprows=skip,
                      sep=r"\s+")
        self.power      = p_spec["#15_SpectFunctTot"].to_numpy()
        self.power_trv  = p_spec["#13_SpectFunctTrv"].to_numpy()
        self.power_lng  = p_spec["#11_SpectFunctLgt"].to_numpy()
        self.power_norm = np.sum(self.power)
        self.wavenumber = p_spec["#01_KStag"].to_numpy()
        
        self.__compute_correlation_scale()
        self.__compute_micro_scale()
        self.__compute_peaks_cale()
        
    def __compute_correlation_scale(self):
        self.correlation_scale = np.sum( self.power_norm /(self.power * self.wavenumber**-1) )

    def __compute_micro_scale(self):
        self.microscale = np.sqrt(np.sum( self.power * self.wavenumber**2 / self.power_norm ))
    
    def __compute_energy_containing_scale(self):
        self.microscale = np.sqrt(np.sum( self.power * self.wavenumber**2 / self.power_norm ))
    
    def __compute_peak_scale(self):
        pass
    
        
         
## ###############################################################
## Auxillary reading functions (can be jit compiled)
## ###############################################################            
    
def reformat_FLASH_field(field  : np.ndarray,
                         nxb    : int,
                         nyb    : int,
                         nzb    : int,
                         iprocs : int,
                         jprocs : int,
                         kprocs : int,
                         debug) -> np.ndarray:
    """
    This function reformats the FLASH block / core format into
    (x,y,z) format for processing in real-space coordinates utilising
    numba's jit compiler, producing roughly a two-orders of magnitude
    speedup compared to the pure python version.

    INPUTS:
    field   - the FLASH field in (core,block_x,block_y,block_z) coordinates
    iprocs  - the number of cores in the x-direction
    jprocs  - the number of cores in the y-direction
    kprocs  - the number of cores in the z-direction
    debug   - flag to print debug information


    OUTPUTs:
    field_sorted - the organised 3D field in (x,y,z) coordinates

    """

    # The block counter for looping through blocks
    block_counter: int = 0

    if debug:
        print(f"reformat_FLASH_field: nxb = {nxb}")
        print(f"reformat_FLASH_field: nyb = {nyb}")
        print(f"reformat_FLASH_field: nzb = {nzb}")
        print(f"reformat_FLASH_field: iprocs = {iprocs}")
        print(f"reformat_FLASH_field: jprocs = {jprocs}")
        print(f"reformat_FLASH_field: kprocs = {kprocs}")

    # Initialise an empty (x,y,z) field
    # has to be the same dtype as input field (single precision)
    field_sorted = np.zeros((nyb*jprocs,
                             nzb*kprocs,
                             nxb*iprocs),
                             dtype=np.float32)


    #time1 = timeit.default_timer()
    
    # Sort the unsorted field
    if debug:
        print("reformat_FLASH_field: Beginning to sort field.")
    for j in range(jprocs):
        for k in range(kprocs):
            for i in range(iprocs):
                field_sorted[j*nyb:(j+1)*nyb,
                             k*nzb:(k+1)*nzb,
                             i*nxb:(i+1)*nxb, 
                             ] = field[block_counter, :, :, :]
                block_counter += 1
    
    #time2 = timeit.default_timer()     
    #print(f"The total time it took is: {time2-time1}")

    if debug:
        print("reformat_FLASH_field: Sorting complete.")
    return field_sorted
