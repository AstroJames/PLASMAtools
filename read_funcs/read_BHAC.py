''' BHAC data class for reading in BHAC data into a uniform grid.'''
#=============================================================================
import numpy as np
import vtk as v
from vtk.util.numpy_support import vtk_to_numpy as v2n
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from joblib import Parallel, delayed
from scipy.interpolate import griddata, NearestNDInterpolator
from scipy.ndimage import distance_transform_edt

default_timer = time.time

# def convert_to_parallel_vtk(file_name):
#     # Create or load your unstructured grid data
#     unstructuredGrid = v.vtkUnstructuredGrid()

#     # Assume `unstructuredGrid` is your VTK data
#     # Partition your data into N pieces
#     N = 4  # Number of partitions
#     partitioner = v.vtkUnstructuredGridPartitioner()
#     partitioner.SetInputData(unstructuredGrid)
#     partitioner.SetNumberOfPartitions(N)
#     partitioner.Update()

#     # Write each partition to a .vtu file
#     writers = []
#     for i in range(N):
#         part = partitioner.GetOutput().GetBlock(i)
#         writer = v.vtkXMLUnstructuredGridWriter()
#         writer.SetFileName(f"{file_name}.vtu")
#         writer.SetInputData(part)
#         writer.Write()
#         writers.append(writer)

#     # Write the .pvtu file
#     pwriter = v.vtkXMLPUnstructuredGridWriter()
#     pwriter.SetFileName(f"{file_name}.vtu")
#     pwriter.SetInputData(unstructuredGrid)
#     pwriter.SetNumberOfPieces(N)
#     pwriter.Write()

class reformat_BHAC_field:
    """
    Loader class for vtu and pvtu files.
    """
    def __init__(self           : object,
                 get            : int = 1,
                 file           : str ='data',
                 type           : str ='vtu',
                 mirrorPlane    : str = None,
                 rotateX        : float = 0,
                 rotateY        : float = 0,
                 rotateZ        : float = 0,
                 scaleX         : float = 1,
                 scaleY         : float = 1,
                 scaleZ         : float = 1,
                 silent         : bool = False,
                 parallel       : bool = False):
        
        self.filename       = file
        self.filenameout    = file
        self.type           = type
        self.parallel       = parallel    
        self.isLoaded       = False
        self.mirrorPlane    = mirrorPlane
        self.silent         = silent
        
        self.rotateX = rotateX
        self.rotateY = rotateY
        self.rotateZ = rotateZ

        self.scaleX = scaleX
        self.scaleY = scaleY
        self.scaleZ = scaleZ

        
        self.datareader = v.vtkXMLUnstructuredGridReader()
        if (not self.silent): print('========================================')
        if (not self.silent): print('loading file %s' % (self.filename))

        if get != None:
            self.getAll()

    def getAll(self):
        """
        
        Populate all of the attributes in the BHAC data class.
        
        Args:
        
        Returns:

        
        """
        
        # get data
        t0 = default_timer()
        #convert_to_parallel_vtk(self.filename.split(".")[0])
        
        if self.parallel:
            pass
        else:
            self.getData()
            
        tdata = default_timer()
        
        if (not self.silent): 
            print('Reading data time= %f sec' % (tdata-t0))
        
        if self.mirrorPlane != None:
            if (not self.silent): 
                print('========== Mirror about plane ',self.mirrorPlane,' ... ============')
            self.mirror()
        
        if (not self.silent): 
            print('========== Initializing ... ============')
        
        if self.parallel:
            self.get_vars_parallel()
        else:
            self.getVars()
        tvars = default_timer()
        
        if (not self.silent): 
            print('Getting vars time= %f sec' % (tvars-tdata))
        
        self.getPoints()
        tpoints = default_timer()
        
        if (not self.silent): 
            print('Getting points time= %f sec' % (tpoints-tvars))
        
        self.getTime()
        tend = default_timer()
        
        if (not self.silent): 
            print('========== Finished loading %d cells in %f sec! ===========' % (self.ncells, (tend-t0) ))
        
    def get_vars_parallel(self):
        """
        Parallel method to get variables from VTK data using ProcessPoolExecutor.
        """
        nvars = self.data.GetCellData().GetNumberOfArrays()
        varnames = [self.data.GetCellData().GetArrayName(i) for i in range(nvars)]

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(read_var_from_file, 
                                       self.filename, 
                                       varnames[i]): varnames[i] for i in range(nvars)}

            for future in as_completed(futures):
                varname, values = future.result()
                setattr(self, varname, values)


    def getTime(self):
        try:
            self.time=v2n(self.data.GetFieldData().GetArray(0))[0]
        except AttributeError:
            self.time = np.nan
        return self.time


    def getVar(self,varname):
        try:
            exec("tmp=self.%s" % (varname))
            return tmp
        except AttributeError:
            print("Unknown variable", varname)
    
    
    def getData(self):
        self.datareader.SetFileName(self.filename)
        self.datareader.Update()
        self.data       = self.datareader.GetOutput()
        self.ncells     = self.data.GetNumberOfCells()
        self.isLoaded   = True
        if (self.rotateX != 0 or 
            self.rotateY != 0 or 
            self.rotateZ != 0 or 
            self.scaleX != 1 or 
            self.scaleY != 1 or 
            self.scaleZ != 1):
            transform = v.vtkTransform()
            transform.RotateX(self.rotateX)
            transform.RotateY(self.rotateY)
            transform.RotateZ(self.rotateZ)
            transform.Scale(self.scaleX, self.scaleY, self.scaleZ)
            transfilter = v.vtkTransformFilter()
            transfilter.SetTransform(transform)
            transfilter.SetInputData(self.data)
            self.data = transfilter.GetOutput()
            transfilter.Update()

    def getData_parallel(self):
        self.datareader.SetFileName(self.filename)
        self.datareader.Update()
        return self.datareader.GetOutput()


    def getVars(self):
        nvars= self.data.GetCellData().GetNumberOfArrays()
        for i in range(nvars):
            varname = self.data.GetCellData().GetArrayName(i)
            if (not self.silent): print("Assigning variable:", varname)
            vtk_values = self.data.GetCellData().GetArray(varname)
            exec("self.%s = v2n(vtk_values)[0:self.ncells]" % (varname))


    def getVarnames(self):
        nvars= self.data.GetCellData().GetNumberOfArrays()
        varnames=[]
        for i in range(nvars):
            varnames.append(self.data.GetCellData().GetArrayName(i))
        return varnames


    def getBounds(self):
        return self.data.GetBounds()


    def getVert(self,icell):
        if self.data.GetCell(icell).GetCellType() == 8 :
            pts=v2n(self.data.GetCell(icell).GetPoints().GetData())
            return np.array((pts[0][0:2],pts[1][0:2],pts[3][0:2],pts[2][0:2]))
        if self.data.GetCell(icell).GetCellType() == 3 :
            pts=v2n(self.data.GetCell(icell).GetPoints().GetData())
            return np.array((pts[0][0],pts[1][0]))
        else: 
            if (not self.silent): print("Can handle only type 3 or type 8")
        

    def getPointList(self):
        tstart = default_timer()
        try:
            self.data
        except AttributeError:
            self.getData()
        try:
            [self.xlist,self.ylist]
        except AttributeError:
            if self.data.GetCell(0).GetCellType() != 8 :
                if (not self.silent): print("Can handle pixel types only")
                pass
            self.xlist = []
            self.ylist = []
            for icell in range(self.ncells):
                pts=v2n(self.data.GetCell(icell).GetPoints().GetData())
                # here is the 5th idx none
                self.xlist.extend((pts[0][0],pts[1][0],pts[3][0],pts[2][0],None))
                self.ylist.extend((pts[0][1],pts[1][1],pts[3][1],pts[2][1],None))
        tend = default_timer()
        if (not self.silent): print('Getting formatted pointlist time=%f sec' % (tend-tstart))
        return [self.xlist,self.ylist]        


    def getPointList3D(self):
        tstart = default_timer()
        try:
            self.data
        except AttributeError:
            self.getData()
        try:
            [self.xlist,self.ylist,self.zlist]
        except AttributeError:
            self.xlist = []
            self.ylist = []
            self.zlist = []
            for icell in range(self.ncells):
                pts=v2n(self.data.GetCell(icell).GetPoints().GetData())
                self.xlist.extend((pts[0][0],pts[1][0],pts[3][0],pts[2][0],pts[4][0],pts[5][0],pts[7][0],pts[6][0],None))
                self.ylist.extend((pts[0][1],pts[1][1],pts[3][1],pts[2][1],pts[4][1],pts[5][1],pts[7][1],pts[6][1],None))
                self.zlist.extend((pts[0][2],pts[1][2],pts[3][2],pts[2][2],pts[4][2],pts[5][2],pts[7][2],pts[6][2],None))
        tend = default_timer()
        if (not self.silent): print('Getting formatted pointlist time=%f sec' % (tend-tstart))
        return [self.xlist,self.ylist,self.zlist]


    def getCenterPoints(self):
        tstart = default_timer()
        firstget = False
        try:
            self.data
        except AttributeError:
            self.getData()
            firstget = True
        try:
            self.centerpoints
        except AttributeError:
            if self.getNdim() == 2 or self.getNdim() == 3:
                self.centerpoints=np.empty((self.ncells,2))
                for icell in range(self.ncells):
                    vert=self.getVert(icell)
                    self.centerpoints[icell,0]=vert[:,0].mean()
                    self.centerpoints[icell,1]=vert[:,1].mean()
            if self.getNdim() == 1 :
                self.centerpoints=np.empty((self.ncells))
                for icell in range(self.ncells):
                    vert=self.getVert(icell)
                    self.centerpoints[icell]=vert.mean()
            tend = default_timer()
        if firstget:
            if (not self.silent): print('Getting cell center coordiantes time=%f sec' % (tend-tstart))
        return self.centerpoints


    def showValues(self,icell):
        if (not self.silent): print('=======================================================')
        if (not self.silent): print('icell= %d; x=%e; y=%e' % (icell,self.getCenterPoints()[icell,0],self.getCenterPoints()[icell,1]))
        for varname in self.getVarnames():
            exec("if (not self.silent): print  '%s =', self.%s[icell]" % (varname,varname))


    def getPoints(self):
        try:
            self.data
        except AttributeError:
            self.getData()
        try:
            self.points
        except AttributeError:
            vtk_points=self.data.GetPoints().GetData()
            self.points=v2n(vtk_points)
        return self.points


    def mirror(self):
        """
        Called when mirrorPlane != None
        The reflection plane is labeled as follows: From the vtk documentation: 
        ReflectionPlane {
        USE_X_MIN = 0, USE_Y_MIN = 1, USE_Z_MIN = 2, USE_X_MAX = 3,
        USE_Y_MAX = 4, USE_Z_MAX = 5, USE_X = 6, USE_Y = 7,
        USE_Z = 8
        }
        """

        vr=v.vtkReflectionFilter()
        vr.SetInputData(self.data)
        vr.SetPlane(self.mirrorPlane)
        self.data=vr.GetOutput()
        vr.Update()
        #self.data.Update()
        self.ncells = self.data.GetNumberOfCells()


    def getNdim(self):
        smalldouble = 1e-10
        self.ndim = 3
        if abs(self.data.GetBounds()[1] - self.data.GetBounds()[0]) <= smalldouble:
            self.ndim=self.ndim - 1
        if abs(self.data.GetBounds()[3] - self.data.GetBounds()[2]) <= smalldouble:
            self.ndim=self.ndim - 1
        if abs(self.data.GetBounds()[5] - self.data.GetBounds()[4]) <= smalldouble:
            self.ndim=self.ndim - 1
        return self.ndim
    
            
    def fill_nan(self, grid):
        nan_mask = np.isnan(grid)
        not_nan_mask = ~nan_mask
        grid[nan_mask] = griddata((np.where(not_nan_mask)[0], 
                                   np.where(not_nan_mask)[1]),
                                  grid[not_nan_mask],
                                  (np.where(nan_mask)[0], np.where(nan_mask)[1]),
                                  method='nearest')
        return grid


    def interpolate_uniform_grid(self, 
                                 varname  : str = "b2", 
                                 N_grid_x : int = 256,
                                 N_grid_y : int = 256):  
        """
        description: interpolates the variable onto a uniform grid, and uses nearest 
        neighbour interpolation for the bounds

        inputs: varname: the variable to plot. 
        N_grid_x: the x dimension of the interpolated grid
        N_grid_y: the y dimension of the interpolated grid

        outputs: the interpolated grid
        
        """  

        # compute the cell centers for BHAC
        centerpoints = self.getCenterPoints()
        
        # compute boundaries for the grid
        bounds = self.getBounds()

        # Generate a regular grid
        x_grid = np.linspace(bounds[0], bounds[1], N_grid_x)
        y_grid = np.linspace(bounds[2], bounds[3], N_grid_y)
        
        # Create meshgrid for the regular grid
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        
        # Interpolate data onto the regular grid
        interp_grid = (griddata((centerpoints[:, 1], 
                                centerpoints[:, 0]), 
                                self.data.GetCellData().GetArray(varname), 
                                (grid_x, grid_y), 
                                method='linear')).T
        
        # Fill nans with nearest neighbour interpolation, and then check for nans
        interp_grid = self.fill_nan(interp_grid)
        
        if (not self.silent): print(f"Number of NaNs: {np.sum(np.isnan(interp_grid))}")

        return interp_grid
    
    
    def uniform_grid(self,
                     varname: str = "b2"):
        """
        This function reads directly from the data and reorganizes it into a uniform grid.
        It assumes that the data has the same number of cells in each direction.
        """
        
        num_of_cells_in_each_dim = int(np.sqrt(self.ncells))

        # Compute the cell centers for BHAC
        centerpoints = self.getCenterPoints()
        
        # Generate a regular grid
        x_coords = centerpoints[:, 1]
        y_coords = centerpoints[:, 0]
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        # Create an empty grid
        grid = np.zeros((num_of_cells_in_each_dim,
                         num_of_cells_in_each_dim))
        
        # Normalize coordinates to grid indices
        x_indices = np.floor((x_coords - x_min) / (x_max - x_min) * (num_of_cells_in_each_dim - 1)).astype(int)
        y_indices = np.floor((y_coords - y_min) / (y_max - y_min) * (num_of_cells_in_each_dim - 1)).astype(int)

        # Read the data and place it into the regular grid
        grid[x_indices, y_indices] = np.array(self.data.GetCellData().GetArray(varname))
        
        return grid
        
        
        