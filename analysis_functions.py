from FLASHtools.read_flash import Fields
import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from mpl_toolkits.axes_grid1 import make_axes_locatable


def energy_evolution(filename_structure, file_identifier_digits, max_timestep, directory):
    """Plots average ratio of mag to kin energy as a function of the timestep. Returns a plot and saves it
    as a pdf in the working directory.

    Parameters
    ----------
    filename_structure : str
        Name of the file without the numerical identifier (e.g. "data_0000" should simply be written as "data_")
    file_identifier_digits : int
        Number of digits in the file identifier (e.g. "0000" should be written as 4)
    max_timestep : int
        Maximum timestep value of data
    directory : str
        Location of Data. Please separate folders of the filepath with 2 slashes \\ (e.g. C:\\Users\\ etc.)
    """   
    #define arrays
    energy_ratio = []
    time_step = []

    #extract values from each datacube. 
    for i in np.arange(0,max_timestep+1):
        filename = 	str(filename_structure) + str(str("%0" + file_identifier_digits +"d") % i)

        #read data
        turb = Fields(str(directory) + filename, reformat = True)
        turb.read('dens')
        turb.read('vel')
        turb.read('mag')

        #compute ave mag energy
        mag_energy = (turb.magx**2 + turb.magy**2 + turb.magz**2)/(8*np.pi)
        ave_mag_energy = np.mean(mag_energy)

        #compute ave kin energy
        kin_energy = (1/2)*(turb.velx**2 + turb.vely**2 + turb.velz**2)*turb.dens
        ave_kin_energy = np.mean(kin_energy)

        #determine ratio        
        energy_ratio.append(ave_mag_energy/ave_kin_energy)
        time_step.append(i)
    #produce plot
    plt.figure()
    plt.scatter(time_step, energy_ratio)
    plt.xlabel('Time Step')
    plt.ylabel('Energy Ratio')
    plt.yscale('log')
    plt.title('Energy Ratio Evolution')
    plt.savefig('Energy_Ratio_Evolution.pdf')
    plt.show()

def Animate_Data2D(filename_structure, file_identifier_digits, max_timestep, directory, slice_axis, slice_value, quantity, fps_value):
    """Creates animation of the evolution of the data cube through time along a particular slice. Animation is saved
    as .mp4 in the working directory.

    Parameters
    ----------
    filename_structure : str
        Name of the file without the numerical identifier (e.g. "data_0000" should simply be written as "data_")
    file_identifier_digits : int
        Number of digits in the file identifier (e.g. "0000" should be written as 4)
    max_timestep : int
        Maximum timestep value of data
    directory : str
        Location of Data. Please separate folders of the filepath with 2 slashes \\ (e.g. C:\\Users\\ etc.)
    slice_axis : int
        0,1,2 for x,y,z respectively
    slice_value : int
        Index value to slice array at along specified axis. Must be in the bounds of the array
    quantity : str
        Indicates which quantity is to be animated. 'dens', 'vel', 'mag' for now, calculating energy evolution by default as of now. 
    fps_value : int
        fps to render the animation videoa as.
    """    
    #initialize definitions
    fig,ax = plt.subplots()
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    camera = Camera(fig)

    #array for possible quantities
    possible_quantities = ['dens', 'vel', 'mag']
    
    #compute for each timestep
    for i in np.arange(0,max_timestep+1):
        #read data
        filename = 	str(filename_structure) + str(str("%0" + file_identifier_digits +"d") % i)
        #initialize class
        turb = Fields(str(directory) + filename, reformat = True)
        if quantity in possible_quantities:
            #turb.read(quantity)
            turb.read('dens')                         #TEMP, edit once Fields class is edited to contain these values by default
            turb.read('vel')
            turb.read('mag')
            #compute mag energy
            mag_energy = (turb.magx**2 + turb.magy**2 + turb.magz**2)/(8*np.pi)
            
            #compute kin energy
            kin_energy = (1/2)*(turb.velx**2 + turb.vely**2 + turb.velz**2)*turb.dens
            
            #determine ratio
            energy_density = mag_energy/kin_energy

            #create slice                       
            if slice_axis == '0':
                slice = energy_density[slice_value,:,:]
            elif slice_axis == '1':
                slice = energy_density[:,slice_value,:]
            elif slice_axis == '2':
                slice = energy_density[:,:,slice_value]
            else:
                raise ValueError('Invalid axis. Please choose from 0,1,2 for x,y,z respectively.')
            
            #take snapshot for animation
            image = ax.imshow(np.log(slice), cmap = 'inferno', origin = 'lower', vmin=-20, vmax=8)
            fig.colorbar(image, cax = cax)
            camera.snap()

        else:
            raise ValueError('Invalid quantity. Please choose from dens, vel, mag')
    #create animation
    animation = camera.animate()
    animation.save('animation.mp4', fps=fps_value)

def compute_derivative(axis, data):
    """Will compute the derivative along the specified coordinate axis to the given data array.
        This function assumes the length scale of entire box is normalized to 1. 

    Parameters
    ----------
    axis : int
        0,1,2, corresponding to x,y,z axes respectively
    data : array
        3D spacial array of data to be differentiated
    """
    derivative  = (np.roll(data, +1, axis) - np.roll(data, -1, axis))/(1/np.shape(data)[0])
    return derivative


def compute_current_density(B_vector):
    """Will compute the current density given the magnetic field components. Returns the current vector

    Parameters
    ----------
    B_vector : array
        the components of B, given in a spacial 3D arrangement as [Bx,By,Bz]
    """
    #compute the derivative of the magnetic field components
    dBx_dx = compute_derivative(0, B_vector[0,:,:,:])
    dBx_dy = compute_derivative(1, B_vector[0,:,:,:])
    dBx_dz = compute_derivative(2, B_vector[0,:,:,:])
    dBy_dx = compute_derivative(0, B_vector[1,:,:,:])
    dBy_dy = compute_derivative(1, B_vector[1,:,:,:])
    dBy_dz = compute_derivative(2, B_vector[1,:,:,:])
    dBz_dx = compute_derivative(0, B_vector[2,:,:,:])
    dBz_dy = compute_derivative(1, B_vector[2,:,:,:])
    dBz_dz = compute_derivative(2, B_vector[2,:,:,:])

    #compute the current density components
    Jx = (1/(4*np.pi))*(dBz_dy - dBy_dz)
    Jy = (1/(4*np.pi))*(dBx_dz - dBz_dx)
    Jz = (1/(4*np.pi))*(dBy_dx - dBx_dy)

    return ([Jx,Jy,Jz])


def dot_product(vector_a, vector_b):
    """Will compute the dot product of two vectors

    Parameters
    ----------
    vector_a : array
        3D spacial array of vector components
    vector_b : array
        3D spacial array of vector components
    """
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    return (np.sum(vector_a*vector_b, axis = 0))

def vector_angle(vector_a, vector_b):
    """Will compute the angle between two vectors

    Parameters
    ----------
    vector_a : array
        3D spacial array of vector components
    vector_b : array
        3D spacial array of vector components
    """
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    return np.arccos(dot_product(vector_a, vector_b)/(np.sqrt(dot_product(vector_a, vector_a))*np.sqrt(dot_product(vector_b, vector_b))))

def pi_axis_formatter(val, pos, denomlim=10, pi=r'\pi'):
    """ REQUIRED BY angle_pdf() FUNCTION
    format label properly with pi values
    for example: 0.6666 pi --> 2π/3
               : 0      pi --> 0
               : 0.50   pi --> π/2  
    """
    minus = "-" if val < 0 else ""
    val = abs(val)
    ratio = frac(val/np.pi).limit_denominator(denomlim)
    n, d = ratio.numerator, ratio.denominator
    
    fmt2 = "%s" % d 
    if n == 0:
        fmt1 = "0"
    elif n == 1:
        fmt1 = pi
    else:
        fmt1 = r"%s%s" % (n,pi)
        
    fmtstring = "$" + minus + (fmt1 if d == 1 else r"{%s}/{%s}" % (fmt1, fmt2)) + "$"
    
    return fmtstring



def angle_pdf(filepath, timesteps, bins, save=False, animate = False):
    """Creates the probability distribution function of the angle between J and B for a list of timesteps

        Returns either a plot of all specified timesteps, or an animation that shows the evolution through the
        timesteps.

    Parameters
    ----------
    filepath : str
        string of filepath to data folder
    timesteps : array
        array containing the timesteps wished to be plotted
    bins : int
        number of bins to be used in the histogram
    save : bool, optional
        If True, will save the animation/plot as a .mp4//pdf file in the working directory. The default is False.
    animate : bool, optional
        If True, the output of the function will be an animated plot instead of a static plot. The default is False.
    """
    fig = plt.figure(figsize=(7,5))  
    ax = plt.gca()
    camera = Camera(fig)
    plt.xlim(0,np.pi)
    plt.ylim(0,0.75)
    ticklen = np.pi/3

    # setting ticks labels
    ax.xaxis.set_major_formatter(FuncFormatter(pi_axis_formatter))
    # setting ticks at proper numbers
    ax.xaxis.set_major_locator(MultipleLocator(base=ticklen))

    for i in timesteps:
        filename = 	str('Turb_hdf5_plt_cnt_') + str(str("%0" + str(4) +"d") % i)

        #read data
        turb = Fields(str(filepath) + filename, reformat = True)
        turb.derived_var("bj_angle")
        angle = turb.bj_angle

        #flatten the angle array
        angle = angle.flatten()

        #create histogram
        plt.title("Probability Density of $J$, $B$ Angle")
        plot = plt.hist(angle, bins = bins, density = True, histtype = 'step', color='black', label = "Timestep " + str(i))        
        plt.xlabel('Angle between $J$ and $B$ (radians)')
        plt.ylabel('Probability')
        if animate == True:
            plt.legend(plot[2], [f'Timestep {i}'])  #need to index plot object by 2 in order for matplotlib to properly display labels
            camera.snap()
        else:
            plt.legend()

    if animate == True:   
        animation = camera.animate()
        plt.show()
        if save == True:
            animation.save('current_anglepdf.mp4', writer = 'ffmpeg', fps=3) 
    else:
        plt.show()
        if save == True:
            plt.savefig('current_anglepdf.pdf')


