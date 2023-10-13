from FLASHtools.read_flash import Fields
import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera

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
    fig = plt.figure()
    camera = Camera(fig)

    #array for possible
    possible_quantities = ['dens', 'vel', 'mag']
    

    for i in np.arange(0,max_timestep+1):
        filename = 	str(filename_structure) + str(str("%0" + file_identifier_digits +"d") % i)
        #read data
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
            
            plt.imshow(slice)
            camera.snap()

        else:
            raise ValueError('Invalid quantity. Please choose from dens, vel, mag')
    animation = camera.animate()
    animation.save('animation.mp4', fps=fps_value)





