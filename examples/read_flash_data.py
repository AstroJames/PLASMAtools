"""
    Example script to read flash data using PLASMAtools
    Author: James R. Beattie

"""

from PLASMAtools.io.read import Fields
from PLASMAtools.funcs.derived_vars import DerivedVars as DV

# Constants for field indices
X, Y, Z = 0, 1, 2

if __name__ == "__main__":
    # Specify the path to your flash data file
    file_path = "/Users/beattijr/Documents/Research/2024/vortex_dynamics/data/Turb_hdf5_plt_cnt_0050_M2MA01"  # Replace with your actual file path
    
    # Initialize Fields class to read data
    fields = Fields(file_path,
                    reformat = True)
    
    # Initialize derived variable class
    dvf = DV(
        L=[1.0, 1.0, 1.0])  # Length of the domain in each direction)
    
    # Read specific density field
    fields.read("dens")
    
    # Shape of scalar fields (1, N, N, N)
    print(f"Shape of density field: {fields.dens.shape}")
    
    # Compute density gradient
    grad_dens = dvf.scalar_gradient(fields.dens)
    
    # Shape of vector fields (3, N, N, N)
    print(f"Shape of density gradient: {grad_dens.shape}")
    
    # There are many vector operations available, such as:
    # Compute the magnitude of the density gradient
    grad_dens_mag = dvf.vector_magnitude(grad_dens)
    
    
    
    
    
    