"""
    read_FLASH

    this function reformats 4D FLASH files into 3D grid coordinates 
    (x,y,z) for processing in real-space coordinates utilising
    numba's jit compiler.
    
    Author: James R. Beattie


"""

import numpy as np
from numba import njit, prange, types
       
##############################################################################
# Type signatures for Numba functions
##############################################################################


sig32_sort = types.float32[:,:,:](
    types.float32[:,:,:,:], 
    types.int32, 
    types.int32, 
    types.int32, 
    types.int32, 
    types.int32, 
    types.int32
    )

sig32_usort = types.float32[:,:,:,:](
    types.float32[:,:,:], 
    types.int32, 
    types.int32, 
    types.int32, 
    types.int32, 
    types.int32, 
    types.int32)

@njit([sig32_sort], parallel=True, fastmath=True, nogil=True, cache=True)
def sort_flash_field_opt(
    field    : np.ndarray,
    nxb      : int,
    nyb      : int,
    nzb      : int, 
    iprocs   : int,
    jprocs   : int,
    kprocs   : int) -> np.ndarray:
    """
    Optimized version with several performance improvements:
    1. Pre-calculate base indices
    2. Use contiguous memory access patterns
    3. Minimize index calculations in inner loops
    """
    
    # Initialise an empty (x,y,z) field
    field_sorted = np.zeros((nyb*jprocs,
                             nzb*kprocs,
                             nxb*iprocs),
                            dtype=np.float32)

    # Pre-calculate some constants
    jprocs_iprocs = jprocs * iprocs
    total_blocks = kprocs * jprocs_iprocs
    
    for block_idx in prange(total_blocks):
        # Reconstruct k, j, i from block index
        k = block_idx // jprocs_iprocs
        temp = block_idx % jprocs_iprocs
        j = temp // iprocs
        i = temp % iprocs
        
        # Pre-calculate base indices
        base_z = k * nzb
        base_y = j * nyb
        base_x = i * nxb
        
        # Access pattern optimized for cache efficiency
        # Since field is stored as (block, nyb, nzb, nxb), we iterate in that order
        for idx0 in range(nyb):
            z_idx = base_z + idx0
            for idx1 in range(nzb):
                y_idx = base_y + idx1
                for idx2 in range(nxb):
                    field_sorted[z_idx, y_idx, base_x + idx2] = field[block_idx, idx0, idx1, idx2]
                    
    return field_sorted


@njit([sig32_sort], parallel=True, fastmath=True, nogil=True, cache=True)
def sort_flash_field_vec(
    field    : np.ndarray,
    nxb      : int,
    nyb      : int,
    nzb      : int, 
    iprocs   : int,
    jprocs   : int,
    kprocs   : int) -> np.ndarray:
    """
    Vectorized version that copies entire rows at once
    """
    
    # Initialise an empty (x,y,z) field
    field_sorted = np.zeros((nyb*jprocs,
                             nzb*kprocs,
                             nxb*iprocs),
                            dtype=np.float32)

    # Pre-calculate some constants
    jprocs_iprocs = jprocs * iprocs
    total_blocks = kprocs * jprocs_iprocs
    
    for block_idx in prange(total_blocks):
        # Reconstruct k, j, i from block index
        k = block_idx // jprocs_iprocs
        temp = block_idx % jprocs_iprocs
        j = temp // iprocs
        i = temp % iprocs
        
        # Pre-calculate base indices
        base_z = k * nzb
        base_y = j * nyb
        base_x = i * nxb
        
        # Copy entire rows at once (vectorized operations)
        for idx0 in range(nyb):
            for idx1 in range(nzb):
                field_sorted[base_z + idx0, base_y + idx1, base_x:base_x + nxb] = field[block_idx, idx0, idx1, :]
                    
    return field_sorted


@njit([sig32_usort], parallel=True, fastmath=True, nogil=True)
def unsort_FLASH_field(
    field_sorted : np.ndarray,
    nxb          : int,
    nyb          : int,
    nzb          : int,
    iprocs       : int,
    jprocs       : int,
    kprocs       : int) -> np.ndarray:

    # Note: This function expects field_sorted to already be transposed to (x,y,z)
    # from the original (y,z,x) format
    
    # Calculate the total number of blocks
    total_blocks = iprocs * jprocs * kprocs
        
    # Initialise an empty (core, block_x, block_y, block_z) field
    field_unsorted = np.zeros((total_blocks, 
                               nyb, 
                               nzb, 
                               nxb), 
                              dtype=np.float32)

    # Parallelize over blocks
    for block_idx in prange(total_blocks):
        # Reconstruct k, j, i from block index
        k = block_idx // (jprocs * iprocs)
        j = (block_idx % (jprocs * iprocs)) // iprocs
        i = block_idx % iprocs
        
        # Copy from transposed field_sorted (now in x,y,z format) back to blocks
        for idx0 in range(nyb):
            for idx1 in range(nzb):
                for idx2 in range(nxb):
                    field_unsorted[block_idx, idx0, idx1, idx2] = field_sorted[i*nxb + idx2, j*nyb + idx0, k*nzb + idx1]
                
    return field_unsorted


def reformat_FLASH_field(
    field  : np.ndarray,
    nxb    : int,
    nyb    : int,
    nzb    : int,
    iprocs : int,
    jprocs : int,
    kprocs : int,
    debug  : bool,
    use_version : str = 'optimized') -> np.ndarray:
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
    use_version - which version to use: 'basic', 'optimized', or 'vectorized'


    OUTPUTs:
    field_sorted - the organised 3D field in (x,y,z) coordinates

    """
    
    # Enforce single precision 
    field = field.astype(np.float32)

    out = np.zeros(
        (nxb*iprocs,
         nyb*jprocs,
         nzb*kprocs),
        dtype=field.dtype)

    if debug:
        print(f"reformat_FLASH_field: nxb = {nxb}")
        print(f"reformat_FLASH_field: nyb = {nyb}")
        print(f"reformat_FLASH_field: nzb = {nzb}")
        print(f"reformat_FLASH_field: iprocs = {iprocs}")
        print(f"reformat_FLASH_field: jprocs = {jprocs}")
        print(f"reformat_FLASH_field: kprocs = {kprocs}")
        print(f"reformat_FLASH_field: using version = {use_version}")

    # Choose which version to use
    if use_version == 'vectorized':
        sort_func = sort_flash_field_vec
    else:  # default to optimized
        sort_func = sort_flash_field_opt
    
    # Initialise an empty (x,y,z) field
    # has to be the same dtype as input field (single precision)
    # swap axes to get the correct orientation
    # x = 0, y = 1, z = 2
        
    out = np.transpose(sort_func(field, 
                                  nxb, 
                                  nyb, 
                                  nzb, 
                                  iprocs, 
                                  jprocs, 
                                  kprocs),
                        (2,1,0))
    
    return out


def benchmark_versions(field, nxb, nyb, nzb, iprocs, jprocs, kprocs):
    """
    Benchmark different versions of the sorting function
    """
    import time

    
    # Warm up JIT compilation
    print("Warming up JIT compilation...")
    for version in ['optimized', 'vectorized']:
        _ = reformat_FLASH_field(field, nxb, nyb, nzb, iprocs, jprocs, kprocs, 
                                debug=False, use_version=version)
    
    # Benchmark each version
    n_runs = 5
    for version in ['optimized', 'vectorized']:
        times = []
        for _ in range(n_runs):
            start = time.time()
            _ = reformat_FLASH_field(field, nxb, nyb, nzb, iprocs, jprocs, kprocs, 
                                    debug=False, use_version=version)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"{version:12s}: {avg_time:.3f} ± {std_time:.3f} seconds")

