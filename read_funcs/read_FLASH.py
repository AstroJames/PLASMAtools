"""
    read_flash files

    this function reformats 4D FLASH files into
    3D grid coordinates

"""

import numpy as np

# Try to import numba
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    print("Utilising numba for I/O.")
except ImportError:
    NUMBA_AVAILABLE = False
    
## ###############################################################
## Auxillary reading functions (can be jit compiled)
## ###############################################################            


def sort_flash_field(field    : np.ndarray,
                     nxb      : int,
                     nyb      : int,
                     nzb      : int, 
                     iprocs   : int,
                     jprocs   : int,
                     kprocs   : int) -> np.ndarray:
    
    # Initialise an empty (x,y,z) field
    field_sorted = np.zeros((nyb*jprocs,
                             nzb*kprocs,
                             nxb*iprocs),
                            dtype=np.float32)

    # The block counter for looping through blocks
    block_counter = 0

    # Sort the unsorted field
    for j in range(jprocs):
        for k in range(kprocs):
            for i in range(iprocs):
                field_sorted[j*nyb:(j+1)*nyb, k*nzb:(k+1)*nzb, i*nxb:(i+1)*nxb] = field[block_counter, :, :, :]
                block_counter += 1
    return field_sorted


def unsort_flash_field(field_sorted : np.ndarray,
                       nxb          : int,
                       nyb          : int,
                       nzb          : int,
                       iprocs       : int,
                       jprocs       : int,
                       kprocs       : int) -> np.ndarray:

    # Swap the axes to get the correct orientation
    field_sorted = np.transpose(field_sorted,(2,1,0))
    
    # Calculate the total number of blocks
    total_blocks = iprocs * jprocs * kprocs
        
    # Initialise an empty (core, block_x, block_y, block_z) field
    field_unsorted = np.zeros((total_blocks, 
                               nyb, 
                               nzb, 
                               nxb), 
                              dtype=np.float32)

    # The block counter for looping through blocks
    block_counter = 0

    # Unsort the sorted field
    for j in range(jprocs):
        for k in range(kprocs):
            for i in range(iprocs):
                field_unsorted[block_counter, :, :, :] = field_sorted[j*nyb:(j+1)*nyb, k*nzb:(k+1)*nzb, i*nxb:(i+1)*nxb]
                block_counter += 1
                
    return field_unsorted


if NUMBA_AVAILABLE:
    sort_flash_field = njit('float32[:,:,:](float32[:,:,:,:], int32, int32, int32, int32, int32, int32)', 
                             parallel  = True, 
                             fastmath  = True, 
                             nogil     = True)(sort_flash_field)
    
    unsort_flash_field = njit('float32[:,:,:,:](float32[:,:,:], int64, int64, int64, int64, int64, int64)',
                              parallel  = True, 
                              fastmath  = True,
                              nogil     = True)(unsort_flash_field)


def reformat_FLASH_field(field  : np.ndarray,
                         nxb    : int,
                         nyb    : int,
                         nzb    : int,
                         iprocs : int,
                         jprocs : int,
                         kprocs : int,
                         debug  : bool) -> np.ndarray:
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

    if debug:
        print(f"reformat_FLASH_field: nxb = {nxb}")
        print(f"reformat_FLASH_field: nyb = {nyb}")
        print(f"reformat_FLASH_field: nzb = {nzb}")
        print(f"reformat_FLASH_field: iprocs = {iprocs}")
        print(f"reformat_FLASH_field: jprocs = {jprocs}")
        print(f"reformat_FLASH_field: kprocs = {kprocs}")

    # Initialise an empty (x,y,z) field
    # has to be the same dtype as input field (single precision)
    # swap axes to get the correct orientation
    # x = 0, y = 1, z = 2
    return np.transpose(sort_flash_field(field, 
                                         nxb, 
                                         nyb, 
                                         nzb, 
                                         iprocs, 
                                         jprocs, 
                                         kprocs),
                        (2,1,0))
    