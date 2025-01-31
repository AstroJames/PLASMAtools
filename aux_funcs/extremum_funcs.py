import numpy as np
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from scipy.signal import find_peaks
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp
from PLASMAtools.aux_funcs import derived_var_funcs as dv
from PLASMAtools.read_funcs.read import Fields
import pandas as pd
import cmasher as cmr

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Ellipse

# indexes
X,Y,Z = 0,1,2

class Extremum():
    
    def __init__(self,
                 field        : np.ndarray,
                 output_label : str,
                 phase_plot   : bool    = False,
                 L            : float   = 1.0,
                 debug        : bool    = False,
                 sigma        : float   = 10.0,
                 min_distance : int     = 1,
                 num_of_peaks : int     = 3,
                 rtol         : float   = 0.05,
                 radius_min   : float   = 0.0,
                 radius_max   : float   = 0.05,
                 n_steps_in_integration : int = 500,
                 n_steps_in_limit_cycle : int = 10):#1500) -> None:
        
        # General class attributes
        self.debug        = debug
        self.output_label = output_label
        self.phase_plot   = phase_plot
        
        # Smoothing sigma for kernel
        self.sigma = sigma 
        
        # Attributes for grid
        num_of_dims, N_grid, _ = field.shape
        self.num_of_dims       = num_of_dims        
        self.N_grid            = N_grid
        self.field             = field
        self.L                 = L
        
        assert self.num_of_dims == 2, "This method only works for two dimensions."
        
        # For grid for interpolation (assuming domain is [-L/2,L/2]U[-L/2,L/2])
        self.x = np.linspace(-self.L/2.0, self.L/2.0, self.N_grid)
        self.y = np.linspace(-self.L/2.0, self.L/2.0, self.N_grid)
        self.X, self.Y = np.meshgrid(self.x,self.y,
                          indexing="ij")
        self.cs_coords = {"L_x":0.1*self.L,"L_y":0.5*self.L}
        # Created interpolated vector field for IVPs (use this for determining size)
        # of plasmoids
        self.interp_u = RegularGridInterpolator((self.x, self.y),
                                                gaussian(self.field[X],
                                                         sigma=self.sigma))
        self.interp_v = RegularGridInterpolator((self.x, self.y),
                                                gaussian(self.field[Y],
                                                         sigma=self.sigma))
                
        # vector potential (assumes periodic boundaries)
        self.dvf = dv.DerivedVars(bcs="00",
                                  num_of_dims=self.num_of_dims)
        
        # Compute vector potential immediately upon init
        if self.debug:
            print("__init__: computing vec. pot.")
        self.a_z = self.dvf.vector_potential(self.field)
        self.mag = self.dvf.vector_magnitude(self.field)
        if self.debug:
            print("__init__: finished vec pot.")
            
        # Compute hessian tensor immediately upon init
        if self.debug:
            print("__init__: computing Hessian tensor")
        self.hessian = self.dvf.gradient_tensor(self.dvf.scalar_gradient(self.a_z))
        if self.debug:
            print("__init__: finished Hessian tensor")
        
        # Attributes for extremum
        self.minimum_coords = []
        self.min_distance   = min_distance
        self.o_points       = []
        self.x_points       = []
        self.o_point_stats  = []
        
        # Attributes for peak finding for the limit cycle detection
        self.num_of_peaks = num_of_peaks
        self.rtol         = rtol
        self.radius_min   = radius_min
        self.radius_max   = radius_max
        self.n_steps_in_integration = n_steps_in_integration
        self.n_steps_in_limit_cycle = n_steps_in_limit_cycle
        
        # Compute local extremum and classigy into X and O
        print("__init__: computing extremum")
        self.__count_local_extrema()
        self.__classify_critical_point()
        print(f"__init__: found {len(self.o_points)} O points and {len(self.x_points)} X points")
        print("__init__: finished extremum")
        
        # Determine outer extent of O points for plasmoids
        print("__init__: computing stats for O region")
        self.__compute_region_statistics()
        print("__init__: finished stats for O region")    
        
    def __count_local_extrema(self):
        """
        Create the coordinates of the local extrema in the field.
        
        Author: James R. Beattie
        
        """
        self.minimum_coords = peak_local_max(-gaussian(self.a_z,sigma=self.sigma),
                                             min_distance=self.min_distance)
        self.maximum_coords = peak_local_max(gaussian(self.a_z,sigma=self.sigma),
                                             min_distance=self.min_distance)
    
    
    def __classify_critical_point(self):
        """
        Classify the critical points as either O or X points 
        based on eigen values of the Hessian matrix.
        
        Author: James R. Beattie
        
        """
        for coord in np.vstack([self.minimum_coords,self.maximum_coords]):
            eigs = np.linalg.eigvalsh(self.hessian[:,:,coord[0],coord[1]])
            if eigs[0]*eigs[1] > 0:
                if self.debug:
                    print("classify_critical_point: O point detected")
                self.o_points.append([coord[0],coord[1]])
            else:
                if self.debug:
                    print("classify_critical_point: X point detected")
                self.x_points.append([coord[0],coord[1]])
           
        # cast into numpy arrays and then into sim coordinates  
        self.o_points = np.array(self.o_points) / self.N_grid - self.L/2.0
        self.x_points = np.array(self.x_points) / self.N_grid - self.L/2.0
        
    
    def __interp_vector_field(self, 
                              t, 
                              z):
        """
        Defines the vector field function for integration, using 
        the interpolated vector field.

        Author: James R. Beattie

        Args:
            t (float): the time parameter used as a dummy variable for the integration
            z (float,float): the current position in the 2D domain

        Returns:
            [float,float]: the integral curve at the current position
        """
        
        # Unpack the current position
        x, y = z
        
        # define the domain that contains the current sheet
        if (-self.cs_coords["L_x"] <= x <= self.cs_coords["L_x"]) and \
            (-self.cs_coords["L_y"] <= y <= self.cs_coords["L_y"]):
            u = self.interp_u((x, y))
            v = self.interp_v((x, y))
            return [u, v]
        else:
            # Return zero if out of bounds, to keep solver in the domain
            return [0.0, 0.0]


    def __solve_trajectory(self,
                           x0     : float, 
                           y0     : float, 
                           t_span : list = [0, 2]):
        """
        Function to solve the trajectory for a given (x0,y0) 
        initial condition and time span. The function uses the
        solve_ivp function from scipy to solve the trajectory.
        
        Author: James R. Beattie

        Args:
            x0 (float)              : initial x position
            y0 (float)              : initial y position
            t_span (list, optional) : the domain to integrate the dummy time 
                                        variable over Defaults to [0, 2].

        Returns:
            [np.ndarray, np.ndarray, np.ndarray]: the time parameter and 
                the trajectory (x,y)
        """
        
        sol = solve_ivp(self.__interp_vector_field,
                        t_span,
                        [x0, y0],
                        t_eval=np.linspace(*t_span, 
                                           self.n_steps_in_integration))
        
        return sol.t, sol.y[0], sol.y[1]


    def __check_for_limit_cycle(self,
                                x : np.ndarray):
        """
        This function checks if a limit cycle is present in the
        trajectory. It does this by checking if the trajectory
        returns to the same point after a certain number of 
        steps. The function returns a boolean and the peaks
        in the trajectory.
        
        Author: James R. Beattie

        Args:
            x (np.ndarray): the trajectory to check for a limit cycle in

        Returns:
            [bool, np.ndarray]: boolean indicating if a limit cycle is present
                                and the peaks in the trajectory
        """
        
        # Find peaks in the array
        peaks, _ = find_peaks(x)
        
        if len(peaks) < self.num_of_peaks:
            return False, None  # No limit cycle detected

        # Calculate intervals between peaks and check if they are close
        # e.g., returning on the same peak each cycle.
        peak_intervals  = np.diff(peaks)
        is_periodic     = np.allclose(peak_intervals,
                                      peak_intervals[0],
                                      rtol = self.rtol)  # Allow small variation
        
        return is_periodic, peaks

    def __compute_region_statistics(self):
        """
        This function loops through the O points and works out 
        the extent of the region in terms of the largest closed
        integral.
        
        Author: James R. Beattie
        
        Returns:
            x0    : x coordinate of O point
            y0    : y coordinate of O point
            a_max : the maximum radius 
            a_min : the minimum radius
            t     : the time parameter used for the integration
            x     : the x coordinate of the largest limit cycle
            y     : the y xoordinate of the largest limit cycle
        
        """
        
        o_point_list     = []
        local_stats_list = [] 
        
        for idx, (x0, y0) in enumerate(self.o_points):
            print(f"compute_O_region: computing O point ({x0},{y0})")
            print(f"compute_O_region: computing O point {idx}") 
            largest_cycle_radius = 0
            a_max = 0
            a_min = 0
            periodic = False
            if self.phase_plot:
                f,ax = plt.subplots(figsize=(5,5),dpi=150)
            for shift in np.linspace(self.radius_min,
                                     self.radius_max,
                                     self.n_steps_in_limit_cycle):
                t, x, y = self.__solve_trajectory(x0+shift,
                                                  y0+shift)
                is_periodic_x, _ = self.__check_for_limit_cycle(x)
                is_periodic_y, _ = self.__check_for_limit_cycle(y)
                if self.debug:
                    print(f"compute_O_region: {is_periodic_x} {is_periodic_y}")
                if is_periodic_y and is_periodic_x:
                    periodic = True
                    if self.phase_plot:
                        ax.plot(y0,x0,c='r',marker='o',ls='None',markersize=4)
                        ax.plot(y,x,linewidth=1)
                    # Measure the size of the limit cycle
                    # max radii
                    a_max = np.max(np.sqrt((x-x0)**2 + (y-y0)**2))  
                    # min radii
                    a_min = np.min(np.sqrt((x-x0)**2 + (y-y0)**2))
                    # Check if the limit cycle is the largest
                    if a_max > largest_cycle_radius:
                        if self.debug:
                            print("compute_O_region: limit cycle detected")
                        largest_cycle_radius = a_max
                        limit_cycle_info = {"x0"    : x0, 
                                            "y0"    : y0,
                                            "a_max" : a_max,
                                            "a_min" : a_min,
                                            "t"     : t[self.n_steps_in_integration//2:],
                                            "x"     : x[self.n_steps_in_integration//2:],
                                            "y"     : y[self.n_steps_in_integration//2:]}            
            if periodic:
                # append the entire limit cycle information
                print("__compute_region_statistics: periodic")
                o_point_list.append(limit_cycle_info)

                # append only the statistics that will be used in the csv
                local_stats_list.append({
                    "x0"    : limit_cycle_info["x0"],
                    "y0"    : limit_cycle_info["y0"],
                    "a_max" : limit_cycle_info["a_max"],
                    "a_min" : limit_cycle_info["a_min"],
                    "type"  : "o_point"})
                
            if self.phase_plot:
                ax.set_xlabel(r"$x/L$")
                ax.set_ylabel(r"$y/L$")
                plt.savefig(f"/Users/beattijr/Documents/Research/2024/test_BHAC/plots/phase_space/{idx}_new.png")
                plt.close()
                
            if self.debug:
                print(f"compute_O_region: saving limitcycle info ({x0},{y0})")
    
        # save the local o point list
        np.save("o_points.npy",o_point_list)
    
        for idx, (x0, y0) in enumerate(self.x_points):
            local_stats_list.append({
            "x0"    : limit_cycle_info["x0"],
            "y0"    : limit_cycle_info["y0"],
            "a_max" : 0.0,
            "a_min" : 0.0,
            "type"  : "x_point"})
                
        # Create DataFrame from the list of dictionaries
        print(local_stats_list)
        df = pd.DataFrame(local_stats_list)

        # Save the DataFrame to a CSV file
        df.to_csv(f"{self.output_label}_local_statistics.csv",
                  index=False)
        
                
def main():
    data_path   = "/Users/beattijr/Documents/Research/2024/test_BHAC/data/data0075.vtu"
    read        = False
    N_grid      = 2048 
    sigma       = 10.0
    sim         = Fields(data_path,
                         sim_data_type="bhac")
    L = 1.0
    
    x   = np.linspace(-L/2.0, L/2.0, N_grid)
    y   = np.linspace(-L/2.0, L/2.0, N_grid)
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    if read:
        sim.read("mag", N_grid_x=N_grid,N_grid_y=N_grid)
        b_field = np.array([sim.magy,sim.magx])
        np.save("b_field.npy",b_field)
    else:
        print("Reading")
        b_field = np.load("/Users/beattijr/Documents/Research/2024/test_BHAC/b_field.npy")
        print("Reading done.")

    extremum    = Extremum(field=b_field,
                           output_label="data0075",
                           sigma=sigma,
                           phase_plot=True)  
    
    data = pd.read_csv("data0075_local_statistics.csv")    
    dvf = dv.DerivedVars(bcs="00",
                         num_of_dims=2)
    
    mag = dvf.vector_magnitude(b_field)
    
    f,ax = plt.subplots(figsize=(1.7*N_grid/150,N_grid/2.0/150),dpi=150)
    p = ax.imshow(gaussian(mag,sigma=sigma)**2/np.mean(mag**2),
              cmap="cmr.ghostlight",
              origin="lower",
              norm=colors.LogNorm(vmin=1e-1,vmax=1e1),
              extent=[-0.5,0.5,-0.5,0.5])
    ax.streamplot(Y,
                  X,
                  gaussian(b_field[1],sigma=sigma),
                  gaussian(b_field[0],sigma=sigma),
                  color="w",
                  linewidth=0.05,
                  density=10,
                  broken_streamlines=False,
                  arrowsize=0.0,
                  integration_direction="both")
    ax.plot(data["y0"],data["x0"],c='r',marker='o',ls='None',markersize=4)
    # Create the ellipse patch
    for idx, row in data.iterrows():
        print((row["y0"],row["x0"]))
        ellipse = Ellipse(xy=(row["y0"],row["x0"]), 
                          width  = 2.0*row["a_max"], 
                          height = 2.0*row["a_min"], 
                          angle=0.0, 
                          edgecolor='yellow',
                          facecolor='none')
        ax.add_patch(ellipse)
    ax.legend(loc="upper right",frameon=True,fontsize=24)
    cb = f.colorbar(p,pad=0.01,aspect=20,shrink=0.3)
    cb.set_label(r"$b^2/\left\langle b^2 \right\rangle$",fontsize=24)
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.1,0.1)
    plt.savefig("/Users/beattijr/Documents/Research/2024/test_BHAC/plots/mag_energy_new_test.pdf")
    plt.close()
 
    
if __name__ == "__main__":
    main()