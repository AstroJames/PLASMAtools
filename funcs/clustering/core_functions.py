"""
Core Numba JIT compiled functions for friends-of-friends clustering.
These are the performance-critical numerical kernels optimized for speed.

Author: James R. Beattie
"""
import numpy as np
from numba import njit, types
from .constants import *

##########################################################################################
# Distance calculations with boundary conditions
##########################################################################################

@njit([sig_distance_2d_32, sig_distance_2d_64])
def distance_2d(dx, dy, box_size_x, box_size_y, bc_x, bc_y):
    """
    Calculate distance with boundary conditions in 2D.
    Supports both float32 and float64 precisions via numba signatures.
    
    Args:
        dx, dy: coordinate differences
        box_size_x, box_size_y: domain sizes
        bc_x, bc_y: boundary condition types (PERIODIC, NEUMANN, DIRICHLET)
    
    Returns:
        Euclidean distance accounting for boundary conditions
    """
    # Apply periodic boundary conditions if needed
    if bc_x == PERIODIC:
        if dx > 0.5 * box_size_x:
            dx -= box_size_x
        elif dx < -0.5 * box_size_x:
            dx += box_size_x
    
    if bc_y == PERIODIC:
        if dy > 0.5 * box_size_y:
            dy -= box_size_y
        elif dy < -0.5 * box_size_y:
            dy += box_size_y
    
    return np.sqrt(dx*dx + dy*dy)

@njit([sig_distance_3d_32, sig_distance_3d_64])
def distance_3d(dx, dy, dz, box_size_x, box_size_y, box_size_z, bc_x, bc_y, bc_z):
    """
    Calculate distance with boundary conditions in 3D.
    Supports both float32 and float64 precisions via numba signatures.
    
    Args:
        dx, dy, dz: coordinate differences
        box_size_x, box_size_y, box_size_z: domain sizes
        bc_x, bc_y, bc_z: boundary condition types (PERIODIC, NEUMANN, DIRICHLET)
    
    Returns:
        Euclidean distance accounting for boundary conditions
    """
    # Apply periodic boundary conditions if needed
    if bc_x == PERIODIC:
        if dx > 0.5 * box_size_x:
            dx -= box_size_x
        elif dx < -0.5 * box_size_x:
            dx += box_size_x
    
    if bc_y == PERIODIC:
        if dy > 0.5 * box_size_y:
            dy -= box_size_y
        elif dy < -0.5 * box_size_y:
            dy += box_size_y
            
    if bc_z == PERIODIC:
        if dz > 0.5 * box_size_z:
            dz -= box_size_z
        elif dz < -0.5 * box_size_z:
            dz += box_size_z
    
    return np.sqrt(dx*dx + dy*dy + dz*dz)

##########################################################################################
# Spatial hashing for efficient neighbor finding
##########################################################################################

@njit([sig_hash_2d_32, sig_hash_2d_64])
def hash_position_2d(x, y, cell_size, nx_cells, ny_cells):
    """
    Hash 2D position to spatial grid cell.
    Supports both float32 and float64 precisions via numba signatures.
    
    Args:
        x, y: spatial coordinates
        cell_size: size of hash grid cells
        nx_cells, ny_cells: number of cells in each dimension
    
    Returns:
        Linear hash index for the grid cell
    """
    ix = int(x / cell_size)
    iy = int(y / cell_size)
    
    # Clamp to valid range
    ix = max(0, min(ix, nx_cells - 1))
    iy = max(0, min(iy, ny_cells - 1))
    
    return ix * ny_cells + iy

@njit([sig_hash_3d_32, sig_hash_3d_64])
def hash_position_3d(x, y, z, cell_size, nx_cells, ny_cells, nz_cells):
    """
    Hash 3D position to spatial grid cell.
    Supports both float32 and float64 precisions via numba signatures.
    
    Args:
        x, y, z: spatial coordinates
        cell_size: size of hash grid cells
        nx_cells, ny_cells, nz_cells: number of cells in each dimension
    
    Returns:
        Linear hash index for the grid cell
    """
    ix = int(x / cell_size)
    iy = int(y / cell_size)
    iz = int(z / cell_size)
    
    # Clamp to valid range
    ix = max(0, min(ix, nx_cells - 1))
    iy = max(0, min(iy, ny_cells - 1))
    iz = max(0, min(iz, nz_cells - 1))
    
    return (ix * ny_cells + iy) * nz_cells + iz

##########################################################################################
# Union-Find data structure with path compression
##########################################################################################

@njit([sig_union_find_32, sig_union_find_64])
def find_root(parent, node):
    """
    Find root of node with iterative path compression.
    Supports both int32 and int64 precisions.
    Iterative implementation avoids recursion issues that can cause hanging.
    """
    # Find root without path compression first
    root = node
    while parent[root] != root:
        root = parent[root]
    
    # Apply path compression in second pass
    current = node
    while parent[current] != current:
        next_node = parent[current]
        parent[current] = root
        current = next_node
    
    return root

@njit([sig_union_32, sig_union_64])
def union_nodes(parent, node1, node2):
    """
    Union two nodes.
    Supports both int32 and int64 precisions.
    """
    root1 = find_root(parent, node1)
    root2 = find_root(parent, node2)
    
    if root1 != root2:
        parent[root2] = root1

##########################################################################################
# Union-find by size to reduce tree height (placed before kernels that use it)
##########################################################################################

@njit([types.void(types.int32[:], types.int32[:], types.int32, types.int32),
       types.void(types.int64[:], types.int64[:], types.int64, types.int64)], cache=True, nogil=True)
def union_nodes_by_size(parent, sizes, node1, node2):
    r1 = find_root(parent, node1)
    r2 = find_root(parent, node2)
    if r1 == r2:
        return
    # Attach smaller to larger
    if sizes[r1] < sizes[r2]:
        r1, r2 = r2, r1
    parent[r2] = r1
    sizes[r1] += sizes[r2]

##########################################################################################
# Cell-list neighbor finding with on-the-fly union
##########################################################################################

@njit(fastmath=True, cache=True, nogil=True)
def find_neighbors_2d_celllist_union(positions, linking_length, box_size, boundary_conditions, parent, sizes):
    """
    Cell-list neighbor search with on-the-fly union operations (2D).
    Assumes positions lie in [0, box) when periodic; operations.py shifts when inferring box.
    """
    n_particles = positions.shape[0]
    if n_particles <= 1:
        return

    # Grid configuration
    cell_size = linking_length * HASH_FACTOR
    if cell_size <= 0:
        cell_size = linking_length

    box_x = box_size[0]
    box_y = box_size[1]
    nx = max(1, int(box_x / cell_size))
    ny = max(1, int(box_y / cell_size))
    is_periodic_x = boundary_conditions[0] == PERIODIC
    is_periodic_y = boundary_conditions[1] == PERIODIC

    # Build cell ids and counting-based binning
    n_cells = nx * ny
    cell_ids = np.empty(n_particles, dtype=np.int64)
    offsets = np.zeros(n_cells + 1, dtype=np.int64)
    inv_cell = 1.0 / cell_size
    for i in range(n_particles):
        ix = int(positions[i, 0] * inv_cell)
        iy = int(positions[i, 1] * inv_cell)
        if ix < 0:
            ix = 0
        elif ix >= nx:
            ix = nx - 1
        if iy < 0:
            iy = 0
        elif iy >= ny:
            iy = ny - 1
        cid = ix * ny + iy
        cell_ids[i] = cid
        offsets[cid + 1] += 1
    for c in range(n_cells):
        offsets[c + 1] += offsets[c]
    # Fill indices array per cell
    write_ptr = offsets.copy()
    order = np.empty(n_particles, dtype=np.int64)
    for i in range(n_particles):
        cid = cell_ids[i]
        idx = write_ptr[cid]
        order[idx] = i
        write_ptr[cid] = idx + 1

    half_x = box_x * 0.5
    half_y = box_y * 0.5
    ll2 = linking_length * linking_length

    # Iterate cells and neighbors (cover each pair once)
    for cx in range(nx):
        for cy in range(ny):
            cell_id = cx * ny + cy
            a0 = offsets[cell_id]
            a1 = offsets[cell_id + 1]

            # Within-cell pairs
            for ia in range(a0, a1):
                i = order[ia]
                for ja in range(ia + 1, a1):
                    j = order[ja]
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    if is_periodic_x:
                        if dx > half_x:
                            dx -= box_x
                        elif dx < -half_x:
                            dx += box_x
                    if is_periodic_y:
                        if dy > half_y:
                            dy -= box_y
                        elif dy < -half_y:
                            dy += box_y
                    if dx * dx + dy * dy <= ll2:
                        union_nodes_by_size(parent, sizes, i, j)

            # Fixed forward neighbor offsets: (1,0), (0,1), (1,1), (-1,1)
            for t in range(4):
                if t == 0:
                    dx_off = 1; dy_off = 0
                elif t == 1:
                    dx_off = 0; dy_off = 1
                elif t == 2:
                    dx_off = 1; dy_off = 1
                else:
                    dx_off = -1; dy_off = 1

                nx_idx = cx + dx_off
                ny_idx = cy + dy_off
                if is_periodic_x:
                    if nx_idx < 0:
                        nx_idx += nx
                    elif nx_idx >= nx:
                        nx_idx -= nx
                else:
                    if nx_idx < 0 or nx_idx >= nx:
                        continue
                if is_periodic_y:
                    if ny_idx < 0:
                        ny_idx += ny
                    elif ny_idx >= ny:
                        ny_idx -= ny
                else:
                    if ny_idx < 0 or ny_idx >= ny:
                        continue
                nb_id = nx_idx * ny + ny_idx
                b0 = offsets[nb_id]
                b1 = offsets[nb_id + 1]
                for ia in range(a0, a1):
                    i = order[ia]
                    for jb in range(b0, b1):
                        j = order[jb]
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        if is_periodic_x:
                            if dx > half_x:
                                dx -= box_x
                            elif dx < -half_x:
                                dx += box_x
                        if is_periodic_y:
                            if dy > half_y:
                                dy -= box_y
                            elif dy < -half_y:
                                dy += box_y
                        if dx * dx + dy * dy <= ll2:
                            union_nodes_by_size(parent, sizes, i, j)


@njit(fastmath=True, cache=True, nogil=True)
def find_neighbors_3d_celllist_union(positions, linking_length, box_size, boundary_conditions, parent, sizes):
    """
    Cell-list neighbor search with on-the-fly union operations (3D).
    Assumes positions lie in [0, box) when periodic; operations.py shifts when inferring box.
    """
    n_particles = positions.shape[0]
    if n_particles <= 1:
        return

    cell_size = linking_length * HASH_FACTOR
    if cell_size <= 0:
        cell_size = linking_length

    box_x = box_size[0]
    box_y = box_size[1]
    box_z = box_size[2]
    nx = max(1, int(box_x / cell_size))
    ny = max(1, int(box_y / cell_size))
    nz = max(1, int(box_z / cell_size))
    is_periodic_x = boundary_conditions[0] == PERIODIC
    is_periodic_y = boundary_conditions[1] == PERIODIC
    is_periodic_z = boundary_conditions[2] == PERIODIC

    # Build cell ids and counting-based binning
    n_cells = nx * ny * nz
    cell_ids = np.empty(n_particles, dtype=np.int64)
    offsets = np.zeros(n_cells + 1, dtype=np.int64)
    inv_cell = 1.0 / cell_size
    for i in range(n_particles):
        ix = int(positions[i, 0] * inv_cell)
        iy = int(positions[i, 1] * inv_cell)
        iz = int(positions[i, 2] * inv_cell)
        if ix < 0:
            ix = 0
        elif ix >= nx:
            ix = nx - 1
        if iy < 0:
            iy = 0
        elif iy >= ny:
            iy = ny - 1
        if iz < 0:
            iz = 0
        elif iz >= nz:
            iz = nz - 1
        cid = (ix * ny + iy) * nz + iz
        cell_ids[i] = cid
        offsets[cid + 1] += 1
    for c in range(n_cells):
        offsets[c + 1] += offsets[c]
    # Fill indices array per cell
    write_ptr = offsets.copy()
    order = np.empty(n_particles, dtype=np.int64)
    for i in range(n_particles):
        cid = cell_ids[i]
        idx = write_ptr[cid]
        order[idx] = i
        write_ptr[cid] = idx + 1

    half_x = box_x * 0.5
    half_y = box_y * 0.5
    half_z = box_z * 0.5
    ll2 = linking_length * linking_length

    # Iterate cells and 13-forward neighbors to avoid duplicates
    for cx in range(nx):
        for cy in range(ny):
            for cz in range(nz):
                cell_id = (cx * ny + cy) * nz + cz
                a0 = offsets[cell_id]
                a1 = offsets[cell_id + 1]

                # Within-cell pairs
                for ia in range(a0, a1):
                    i = order[ia]
                    for ja in range(ia + 1, a1):
                        j = order[ja]
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        dz = positions[i, 2] - positions[j, 2]
                        if is_periodic_x:
                            if dx > half_x:
                                dx -= box_x
                            elif dx < -half_x:
                                dx += box_x
                        if is_periodic_y:
                            if dy > half_y:
                                dy -= box_y
                            elif dy < -half_y:
                                dy += box_y
                        if is_periodic_z:
                            if dz > half_z:
                                dz -= box_z
                            elif dz < -half_z:
                                dz += box_z
                        if dx * dx + dy * dy + dz * dz <= ll2:
                            union_nodes_by_size(parent, sizes, i, j)

                # Neighbor offsets set (13 forward neighbors), enumerated explicitly
                for t in range(13):
                    if t == 0:
                        dx_off, dy_off, dz_off = 1, 0, 0
                    elif t == 1:
                        dx_off, dy_off, dz_off = 0, 1, 0
                    elif t == 2:
                        dx_off, dy_off, dz_off = 1, 1, 0
                    elif t == 3:
                        dx_off, dy_off, dz_off = -1, 1, 0
                    elif t == 4:
                        dx_off, dy_off, dz_off = -1, -1, 1
                    elif t == 5:
                        dx_off, dy_off, dz_off = 0, -1, 1
                    elif t == 6:
                        dx_off, dy_off, dz_off = 1, -1, 1
                    elif t == 7:
                        dx_off, dy_off, dz_off = -1, 0, 1
                    elif t == 8:
                        dx_off, dy_off, dz_off = 0, 0, 1
                    elif t == 9:
                        dx_off, dy_off, dz_off = 1, 0, 1
                    elif t == 10:
                        dx_off, dy_off, dz_off = -1, 1, 1
                    elif t == 11:
                        dx_off, dy_off, dz_off = 0, 1, 1
                    else:
                        dx_off, dy_off, dz_off = 1, 1, 1

                    nx_idx = cx + dx_off
                    ny_idx = cy + dy_off
                    nz_idx = cz + dz_off
                    if is_periodic_x:
                        if nx_idx < 0:
                            nx_idx += nx
                        elif nx_idx >= nx:
                            nx_idx -= nx
                    else:
                        if nx_idx < 0 or nx_idx >= nx:
                            continue
                    if is_periodic_y:
                        if ny_idx < 0:
                            ny_idx += ny
                        elif ny_idx >= ny:
                            ny_idx -= ny
                    else:
                        if ny_idx < 0 or ny_idx >= ny:
                            continue
                    if is_periodic_z:
                        if nz_idx < 0:
                            nz_idx += nz
                        elif nz_idx >= nz:
                            nz_idx -= nz
                    else:
                        if nz_idx < 0 or nz_idx >= nz:
                            continue

                    nb_id = (nx_idx * ny + ny_idx) * nz + nz_idx
                    b0 = offsets[nb_id]
                    b1 = offsets[nb_id + 1]
                    for ia in range(a0, a1):
                        i = order[ia]
                        for jb in range(b0, b1):
                            j = order[jb]
                            dx = positions[i, 0] - positions[j, 0]
                            dy = positions[i, 1] - positions[j, 1]
                            dz = positions[i, 2] - positions[j, 2]
                            if is_periodic_x:
                                if dx > half_x:
                                    dx -= box_x
                                elif dx < -half_x:
                                    dx += box_x
                            if is_periodic_y:
                                if dy > half_y:
                                    dy -= box_y
                                elif dy < -half_y:
                                    dy += box_y
                            if is_periodic_z:
                                if dz > half_z:
                                    dz -= box_z
                                elif dz < -half_z:
                                    dz += box_z
                            if dx * dx + dy * dy + dz * dz <= ll2:
                                union_nodes_by_size(parent, sizes, i, j)


##########################################################################################
# Core FOF algorithm implementations
##########################################################################################

@njit([sig_fof_2d_32, sig_fof_2d_64], fastmath=True, cache=True, nogil=True)
def fof_2d(positions, linking_length, box_size, boundary_conditions, min_cluster_size):
    """
    Optimized Friends-of-friends clustering in 2D using Phase 5 memory optimizations.
    
    Args:
        positions: (N, 2) array of particle positions
        linking_length: maximum distance for friendship
        box_size: (2,) array of domain sizes
        boundary_conditions: (2,) array of BC types per dimension
        min_cluster_size: minimum particles per cluster
    
    Returns:
        cluster_labels: (N,) array of cluster IDs (-1 for noise)
    """
    n_particles = positions.shape[0]
    
    if n_particles == 0:
        return np.empty(0, dtype=type(min_cluster_size))
    
    # Build clusters using Union-Find (with sizes for union-by-size)
    parent = np.arange(n_particles, dtype=type(min_cluster_size))
    sizes = np.ones(n_particles, dtype=type(min_cluster_size))

    # Cell-list neighbor search with unions on the fly
    find_neighbors_2d_celllist_union(positions, linking_length, box_size, boundary_conditions, parent, sizes)
    
    # Assign final cluster labels
    cluster_labels = np.full(n_particles, -1, dtype=type(min_cluster_size))
    cluster_sizes = np.zeros(n_particles, dtype=type(min_cluster_size))
    
    # Count cluster sizes
    for i in range(n_particles):
        root = find_root(parent, i)
        cluster_sizes[root] += 1
    
    # Assign labels only to clusters meeting minimum size
    cluster_id = 0
    cluster_map = np.full(n_particles, -1, dtype=type(min_cluster_size))
    
    for i in range(n_particles):
        root = find_root(parent, i)
        if cluster_sizes[root] >= min_cluster_size:
            if cluster_map[root] == -1:
                cluster_map[root] = cluster_id
                cluster_id += 1
            cluster_labels[i] = cluster_map[root]
    
    return cluster_labels


@njit([sig_fof_3d_32, sig_fof_3d_64], fastmath=True, cache=True, nogil=True)
def fof_3d(positions, linking_length, box_size, boundary_conditions, min_cluster_size):
    """
    Optimized Friends-of-friends clustering in 3D using Phase 5 memory optimizations.
    Supports both float32 and float64 precisions.
    """
    n_particles = positions.shape[0]
    
    if n_particles == 0:
        return np.empty(0, dtype=type(min_cluster_size))
    
    # Build clusters using Union-Find (with sizes for union-by-size)
    parent = np.arange(n_particles, dtype=type(min_cluster_size))
    sizes = np.ones(n_particles, dtype=type(min_cluster_size))

    # Cell-list neighbor search with unions on the fly
    find_neighbors_3d_celllist_union(positions, linking_length, box_size, boundary_conditions, parent, sizes)
    
    # Assign final cluster labels
    cluster_labels = np.full(n_particles, -1, dtype=type(min_cluster_size))
    cluster_sizes = np.zeros(n_particles, dtype=type(min_cluster_size))
    
    # Count cluster sizes
    for i in range(n_particles):
        root = find_root(parent, i)
        cluster_sizes[root] += 1
    
    # Assign labels only to clusters meeting minimum size
    cluster_id = 0
    cluster_map = np.full(n_particles, -1, dtype=type(min_cluster_size))
    
    for i in range(n_particles):
        root = find_root(parent, i)
        if cluster_sizes[root] >= min_cluster_size:
            if cluster_map[root] == -1:
                cluster_map[root] = cluster_id
                cluster_id += 1
            cluster_labels[i] = cluster_map[root]
    
    return cluster_labels

##########################################################################################
# Sparse voxel connected components for friends_of_friends_grid
##########################################################################################

@njit(cache=True, nogil=True)
def _next_pow2(n):
    m = 1
    while m < n:
        m <<= 1
    return m

@njit(cache=True, nogil=True)
def _build_hash_int64(keys):
    """
    Build an open-addressing hash for int64 keys -> index (0..N-1).
    Returns (hkeys, hvals, capacity).
    """
    n = keys.size
    cap = _next_pow2(max(8, n * 2))
    hkeys = -np.ones(cap, dtype=np.int64)
    hvals = -np.ones(cap, dtype=np.int64)
    for i in range(n):
        k = keys[i]
        # Simple multiplicative hash then modulo power-of-two capacity
        idx = int((k * 11400714819323198485) & (cap - 1))
        # Linear probing
        while True:
            if hkeys[idx] == -1:
                hkeys[idx] = k
                hvals[idx] = i
                break
            elif hkeys[idx] == k:
                # Already inserted (keys are unique); nothing to do
                break
            idx = (idx + 1) & (cap - 1)
    return hkeys, hvals, cap

@njit(cache=True, nogil=True)
def _hash_get_int64(hkeys, hvals, cap, key):
    if cap == 0:
        return -1
    idx = int((key * 11400714819323198485) & (cap - 1))
    k = hkeys[idx]
    if k == -1:
        return -1
    if k == key:
        return hvals[idx]
    # Probe
    start = idx
    while True:
        idx = (idx + 1) & (cap - 1)
        k = hkeys[idx]
        if k == -1:
            return -1
        if k == key:
            return hvals[idx]
        if idx == start:
            return -1

@njit(cache=True, nogil=True)
def ccl_occupied_voxels_3d(occupied_lin, counts_pts, dims, boundary_conditions, connectivity):
    """
    Connected-component labeling over occupied voxels represented by linear indices.

    Args:
        occupied_lin: (M,) int64 unique sorted/unsorted voxel linear indices where occupancy >= threshold
        counts_pts: (M,) int64 number of points per occupied voxel
        dims: (3,) int64 array of grid dimensions
        boundary_conditions: (3,) int32 array (PERIODIC/NEUMANN/DIRICHLET)
        connectivity: 6, 18, or 26

    Returns:
        voxel_labels: (M,) int64 component id per occupied voxel (0..K-1)
        comp_sizes_pts: (K_max,) int64 total points per component (first n_comp entries valid)
        n_comp: int64 number of components
    """
    M = occupied_lin.size
    voxel_labels = -np.ones(M, dtype=np.int64)
    comp_sizes_pts = np.zeros(max(1, M), dtype=np.int64)
    n_comp = 0
    if M == 0:
        return voxel_labels, comp_sizes_pts, 0

    # Build hash for membership queries
    hkeys, hvals, cap = _build_hash_int64(occupied_lin)

    dims0 = int(dims[0]); dims1 = int(dims[1]); dims2 = int(dims[2])
    is_per_x = boundary_conditions[0] == PERIODIC
    is_per_y = boundary_conditions[1] == PERIODIC
    is_per_z = boundary_conditions[2] == PERIODIC

    # BFS queue
    q = np.empty(M, dtype=np.int64)

    for start in range(M):
        if voxel_labels[start] != -1:
            continue
        # Start new component
        cid = n_comp
        n_comp += 1
        head = 0
        tail = 0
        voxel_labels[start] = cid
        q[tail] = start
        tail += 1
        total_pts = 0

        while head < tail:
            vi = q[head]
            head += 1
            total_pts += counts_pts[vi]

            vlin = occupied_lin[vi]
            x = int(vlin // (dims1 * dims2))
            rem = int(vlin % (dims1 * dims2))
            y = int(rem // dims2)
            z = int(rem % dims2)

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        if connectivity == 6:
                            if abs(dx) + abs(dy) + abs(dz) != 1:
                                continue
                        elif connectivity == 18:
                            if abs(dx) + abs(dy) + abs(dz) > 2:
                                continue
                        # Neighbor coordinates with BCs
                        nx = x + dx
                        ny = y + dy
                        nz = z + dz
                        if is_per_x:
                            if nx < 0:
                                nx += dims0
                            elif nx >= dims0:
                                nx -= dims0
                        else:
                            if nx < 0 or nx >= dims0:
                                continue
                        if is_per_y:
                            if ny < 0:
                                ny += dims1
                            elif ny >= dims1:
                                ny -= dims1
                        else:
                            if ny < 0 or ny >= dims1:
                                continue
                        if is_per_z:
                            if nz < 0:
                                nz += dims2
                            elif nz >= dims2:
                                nz -= dims2
                        else:
                            if nz < 0 or nz >= dims2:
                                continue
                        nlin = (nx * dims1 + ny) * dims2 + nz
                        ni = _hash_get_int64(hkeys, hvals, cap, nlin)
                        if ni != -1 and voxel_labels[ni] == -1:
                            voxel_labels[ni] = cid
                            q[tail] = ni
                            tail += 1

        comp_sizes_pts[cid] = total_pts

    return voxel_labels, comp_sizes_pts, n_comp
