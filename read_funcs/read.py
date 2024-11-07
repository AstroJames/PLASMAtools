## ###############################################################
## IMPORTS
## ###############################################################

from uu import Error
from h5py import File
import numpy as np
import numpy.polynomial.polynomial as poly
import timeit
from ..aux_funcs.derived_var_funcs import DerivedVars as DV
# import pandas as pd
from joblib import Parallel, delayed

## ###############################################################
## Auxillary reading functions (can be jit compiled)
## ###############################################################

from .read_FLASH        import  reformat_FLASH_field, unsort_flash_field
from .read_BHAC_fast    import  reformat_BHAC_field
from .read_RAMSES       import  reformat_RAMSES_field

## ###############################################################
## Global variabes
## ###############################################################

field_lookup_type = {
    "dens"      : "scalar", # mass density
    "dvvl"      : "scalar", # 
    "alpha"     : "scalar", # force free parameter
    "vel"       : "vector", # velocity field
    "mag"       : "vector", # magnetic field
    "tens"      : "vector", # magnetic tension
    "vort"      : "vector", # vorticity
    "mpr"       : "vector", # magnetic pressure
    "tprs"      : "vector", # thermal pressure
    "cur"       : "vector", # electric current
    "vxb"       : "vector", # ideal electric field (induction)
    "jxb"       : "vector", # Lorentz force
    "jxb_mag"   : "scalar", # mag of Lorentz force
    "jdotb"     : "scalar", # current helicity
    "pres"      : "scalar"
}

bhac_lookup_dict = {"vel"  : ["u1", "u2", "u3"],
                    "mag"  : ["b1", "b2", "b3"],
                    "dens" : ["rho"],
                    "press": ["p"],
                    "elec" : ["e1", "e2", "e3"]}

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
                 filename         : str,
                 reformat         : bool = False,
                 sim_data_type    : str = "flash") -> None:
        """
        Initialize a FLASHGridData object by reading in the data from the specified file.

        Parameters
        ----------
        filename : str 
            The name of the file containing the FLASH grid data.
        reformat : bool, optional
            Whether to reformat the data in the file into 3D arrays (True) or keep it in 1D arrays (False).
            Default is False.
        sim_data_type : str, optional
            The type of simulation data to read in. Default is "flash".

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

        """

        # simulation attributes
        self.filename           = filename
        self.reformat           = reformat
        self.sim_data_type      = sim_data_type 
        self.n_cores            = 0
        self.nxb                = 0
        self.nyb                = 0
        self.nzb                = 0
        self.n_cells            = 0
        self.int_properties     = {}
        self.str_properties     = {}
        self.logic_properties   = {} 

        # read in the simulation properties
        if self.sim_data_type == "flash":
            self.__read_sim_properties()  

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

        # add a new method here for initialisation 
        # (could speed up IO a bit but haven't properly timed it)
        
        # read in state (read in true or false if the reader
        # has actually been called -- this is all to save time 
        # with derived vars. Note that the way that it is 
        # written means that it will create new read_... state variables
        # even if one hasn't been initialised here.)
        self.read_dens          = False
        self.read_vel           = False
        self.read_mag           = False  
        self.derived_cur        = False
        
        # use a CGS unit system
        self.mu0                = 4 * np.pi 


    def __read_sim_properties(self) -> None:
        """
        This function reads in the FLASH field properties directly from
        the hdf5 file metadata. 
        
        Author: James Beattie
        
        """
        g = File(self.filename, 'r')
        self.int_scalars        = {str(key).split("'")[1].strip(): value for key, value in g["integer scalars"]}
        self.int_properties     = {str(key).split("'")[1].strip(): value for key, value in g["integer runtime parameters"]}
        self.str_properties     = {str(key).split("'")[1].strip(): str(value).split("'")[1].strip() for key, value in g["string runtime parameters"]}
        self.logic_properties   = {str(key).split("'")[1].strip(): value for key, value in g["logical runtime parameters"]}
        g.close()
        
        # read out properties of the grid
        self.n_cores  = self.int_scalars["globalnumblocks"]
        self.nxb      = self.int_scalars["nxb"]
        self.nyb      = self.int_scalars["nyb"]
        self.nzb      = self.int_scalars["nzb"]
        self.iprocs   = self.int_properties["iprocs"]
        self.jprocs   = self.int_properties["jprocs"]
        self.kprocs   = self.int_properties["kprocs"]
        self.n_cells  = self.n_cores*self.nxb*self.nyb*self.nzb
        self.plot_file_num = self.int_scalars["plotfilenumber"]


    def set_reformat(self,
                     reformat: bool) -> None:
        """
        This function sets the reformat flag for the FLASH field data.
        """
        self.reformat = reformat
        
        
    def reformat_error(self,
                       var: str) -> None:
        Error(f"Cannot compute {var} without reformating the data.")
    
    
    def read(self,
             field_str          : str,
             vector_magnitude   : bool = False,
             debug              : bool = False,
             interpolate        : bool = True,
             N_grid_x           : int  = 256,
             N_grid_y           : int  = 256,
             N_grid_z           : int  = 256) -> None:
        """
        This function reads in grid data.
        
        Args:
            field_str (str):            The field to read in.
            vector_magnitude (bool):    Whether to read in the magnitude of the vector field.
            debug (bool):               Whether to print debug information.
            N_grid_x (int):             The number of grid points to interpolate onto in the x direction.
            N_grid_y (int):             The number of grid points to interpolate onto in the y direction.
            N_grid_z (int):             The number of grid points to interpolate onto in the z direction.
        
        """
        
        setattr(self,f"read_{field_str}",True)
        
        if self.sim_data_type == "flash":     
            # setting the read in states here (this adds an attribute)
            if field_lookup_type[field_str] == "scalar":
                g = File(self.filename, 'r')
                print(f"Reading in grid attribute: {field_str}")
                if self.reformat:
                    print(f"Reading in reformatted grid attribute: {field_str}")
                    setattr(self, field_str, 
                            np.array([reformat_FLASH_field(g[field_str][:,:,:,:],
                                                 self.nxb,
                                                 self.nyb,
                                                 self.nzb,
                                                 self.iprocs,
                                                 self.jprocs,
                                                 self.kprocs,
                                                 debug)]))
                else:
                    setattr(self, field_str, np.array([g[field_str][:,:,:,:]]))
                g.close()
            
            elif field_lookup_type[field_str] == "vector":
                g = File(self.filename, 'r')
                for coord in ["x","y","z"]:
                    print(f"Reading in grid attribute: {field_str}{coord}")
                    if self.reformat: # if reformatting is required
                        #time1 = timeit.default_timer()
                        field_var = reformat_FLASH_field(g[f"{field_str}{coord}"][:,:,:,:],
                                                         self.nxb,
                                                         self.nyb,
                                                         self.nzb,
                                                         self.iprocs,
                                                         self.jprocs,
                                                         self.kprocs,
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
                            if coord == "x":
                                field = np.array([field_var])
                            else:
                                field = np.concatenate([field,[field_var]])
                                
                    else: # vector mag without reformatting
                        if vector_magnitude:
                            if coord == "x":
                                field_mag = g[f"{field_str}{coord}"][:,:,:,:]**2
                            else:
                                field_mag += g[f"{field_str}{coord}"][:,:,:,:]**2   
                        else:
                            if coord == "x":
                                field = np.array(g[f"{field_str}{coord}"][:,:,:,:])[np.newaxis, ...]
                            else:
                                field = np.concatenate([field,
                                                        np.array(g[f"{field_str}{coord}"][:,:,:,:])[np.newaxis, ...]],axis=0)                  
                            
                # now read in the fields       
                if vector_magnitude:
                    setattr(self, f"{field_str}_mag", np.sqrt(field_mag))
                    del field_mag
                else:
                    setattr(self, f"{field_str}",field)    
                    del field
                if self.reformat:
                    del field_var
                
                g.close()
                
        elif self.sim_data_type == "bhac":
            
            # now instantiate the bhac object
            d = reformat_BHAC_field(file_name=self.filename)
            
            # read in coordinates
            if field_lookup_type[field_str] == "scalar":
                 for var in bhac_lookup_dict[field_str]:
                     if interpolate:
                        setattr(self, field_str, d.interpolate_uniform_grid(var_name=var,
                                                            n_grid_x=N_grid_x,
                                                            n_grid_y=N_grid_y)) 
                    # should add a method here for uniform grid          
            elif field_lookup_type[field_str] == "vector": 
                for var, coord in zip(bhac_lookup_dict[field_str],["x","y"]):
                    t1 = timeit.default_timer()
                    if interpolate:
                        setattr(self, field_str + coord, d.interpolate_uniform_grid(var_name=var,
                                                            n_grid_x=N_grid_x,
                                                            n_grid_y=N_grid_y))
                        # should add a method here for uniform grid  
                    t2 = timeit.default_timer()
                    print(f"The total time it took is: {t2-t1}")
               
            # clean up after creating the object (not sure if this is required) 
            del d
            
            
    def write(self,
              field_str : str,
              new_field : np.ndarray) -> None:
        """
        
        This function writes a new field to the FLASH file.

        Args:
            field_str (str)         : The field to write to the FLASH file.
            new_field (np.ndarray)  : The new field to write to the FLASH file.
        """
        
        assert self.sim_data_type == "flash", "write: Only FLASH data can be written."
        
        # dont accept underscores in the field string (not sure why this doesn't work)
        if "_" in field_str:
            raise Exception("write: Field string cannot contain underscores.")
        
        # now write this to the hdf5 file
        f = File(self.filename, 'a')
        if field_str in f.keys():
            print("write: deleting the old field.")
            del f[field_str]
        
        if self.reformat:
            print("write: adding the new reformated field.")
            f.create_dataset(field_str, data=unsort_flash_field(new_field,
                                                                self.nxb,
                                                                self.nyb,
                                                                self.nzb,
                                                                self.iprocs,
                                                                self.jprocs,
                                                                self.kprocs))
        else:
            print("write: adding the new field.")
            f.create_dataset(field_str, data=new_field)
            
        # check if the unknown names are in the file
        name_added      = False
        unknown_names   = f["unknown names"]
        for i in unknown_names:
            if field_str.encode("utf-8") in i[0]:
                name_added = True
                print(f"write: Field {field_str} already in file. Skipping.")
        if not name_added:
            print(f"write: Adding {field_str} to unknown names.")
            del f["unknown names"]
            f.create_dataset("unknown names",
                                data=np.vstack([unknown_names,
                                                np.array([field_str],
                                                         dtype="S4")]))
        
        f.close()

            
    def derived_var(self,
                    field_str: str,
                    eta: float      = 0.0,
                    nu: float       = 0.0,
                    n_workers: int  = 1) -> None:
        """
        General function for adding derived variables to the data object,
        rather than having to derive them in the aux funcs script.
        
        """
        
        # this is a pre-defined table for common derived vars
        # any new commonly used variables should be added here 
        # so you can just call them with .derived_var("E"), etc.
        var_lookup_table = {
            "E"            : ["Ex","Ey","Ez"],                   # electric field
            "Exb"          : ["Exbx","Exby","Exbz"],             # reconnection inflow velocity
            "uxb"          : ["uxbx","uxby","uxbz"],             # induction
            "helmholtz"    : ["vel_comx","vel_comy","vel_comz",  # helmholtz decomp.
                              "vel_solx","vel_soly","vel_solz"], # 
            "cur"          : ["curx","cury","curz"],             # current
            "jxb"          : ["jxbx","jxby","jxbz"],             # lorentz force
            "jdotb"        : ["jdotb"],                          # current dot magnetic field 
            "tens"         : ["tensx","tensy","tensz"],          # magnetic tension
            "vort"         : ["vortx","vorty","vortz"]           # vorticity
        }
        
        # Initialise derived var functions
        dvf = DV()

        # throw an error if the derived var doesn't exist
        if field_str not in var_lookup_table:
            raise Exception(f"derived_var: {field_str} not in new_var_lookup_table. Add the variable defn. first.")

        
        print(f"derived_var: Beginning to calculate derived variables with n_workers = {n_workers}")
        # grid data attributes
        # if the data is going to be reformated, preallocate the 3D
        # arrays for the grid data
        if self.reformat:
            init_field = np.zeros((self.nyb*self.jprocs,
                                   self.nxb*self.iprocs,
                                   self.nzb*self.kprocs), dtype=np.float32)
        else:
            # otherwise, preallocate the 1D arrays for the grid data
            init_field = np.zeros(self.n_cells, dtype=np.float32)
        
        # add the init field to the data object
        for field in var_lookup_table[field_str]:    
            setattr(self, field, init_field)
        
        ######################################################  
        # Derived vars 
        ######################################################
            
        # Defn: the current density
        # assuming \mu_0 = 4\pi for current
        if field_str == "cur":
            print(f"derived_var: Calculating current density.")
            
            if not self.read_mag:
                self.read("mag")
                
            # write the new variable to the object
            setattr(self, field_str, dvf.vector_curl(self.mag) / (self.mu0))
            
        # Defn: the current density
        # assuming \mu_0 = 4\pi for current
        if field_str == "vort":
            print(f"derived_var: Calculating vorticity density.")
            
            if not self.read_vel:
                self.read("vel")
                            
            # write the new variable to the object
            setattr(self, field_str, dvf.vector_curl(self.vel))
                
        # add more implementations here.
            
            

    

class PowerSpectra():
    
    def __init__(self,
                 filename: str) -> None:
        self.basefilename: str          = filename
        self.filename: str              = ""
        self.names: list                = []
        self.skip_lines: int            = []
        self.wavenumber: list           = []
        self.power: list                = [] 
        self.correlation_scale: float   = 0.0
        self.microscale: float          = 0.0
        self.peakscale: float           = 0.0
            
    def read(self,
             field_str: str) -> None:
        
        skip = 6
        if field_str == "vel":
            self.filename = self.basefilename + "_spect_vels.dat" 
        elif field_str == "mag":
            self.filename = self.basefilename + "_spect_mags.dat"       

        # set self.names
        self.__set_power_spectra_header(field_str)

        # p_spec = pd.read_table(self.filename,
        #               names=self.names,
        #               skiprows=skip,
        #               sep=r"\s+")
        # self.power      = p_spec["#15_SpectFunctTot"].to_numpy()
        # self.power_trv  = p_spec["#13_SpectFunctTrv"].to_numpy()
        # self.power_lng  = p_spec["#11_SpectFunctLgt"].to_numpy()
        # self.power_norm = np.sum(self.power)
        # self.wavenumber = p_spec["#01_KStag"].to_numpy()
        
        self.__compute_correlation_scale()
        self.__compute_micro_scale()
        self.__compute_energy_containing_scale()
        self.__compute_peak_scale()
        
    def __set_power_spectra_header(self,
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
        
    def __compute_correlation_scale(self):
        self.correlation_scale = np.sum( self.power_norm /(self.power * self.wavenumber**-1) )

    def __compute_micro_scale(self):
        self.microscale = np.sqrt(np.sum( self.power * self.wavenumber**2 / self.power_norm ))
    
    def __compute_energy_containing_scale(self):
        self.microscale = np.sum( self.power * self.wavenumber / self.power_norm )
    
    def __compute_peak_scale(self):
        pass
