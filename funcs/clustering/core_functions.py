"""
Core Numba JIT compiled functions for friends-of-friends clustering.
These are the performance-critical numerical kernels optimized for speed.

Author: James R. Beattie
"""
import numpy as np
from numba import njit, prange, types
from numba.typed import List
import numba
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
# Optimized neighbor finding functions (Phase 5 - Memory Optimized)
##########################################################################################

@njit([types.Tuple([types.int64[:], types.int64[:]])(
    types.float32[:, :], types.float32, types.float32[:], types.int32[:], types.int64),
    types.Tuple([types.int64[:], types.int64[:]])(
    types.float64[:, :], types.float64, types.float64[:], types.int32[:], types.int64)],
      parallel=True, fastmath=True, cache=True, nogil=True)
def find_neighbors_2d_optimized(positions, linking_length, box_size, boundary_conditions, max_edges):
    """
    Phase 5: Memory-optimized neighbor finding with inline distance calculation.
    
    Combines single-pass algorithm + inline distance + memory optimizations for best performance.
    Expected ~5x speedup over sequential through multiple optimizations.
    """
    n_particles = positions.shape[0]
    num_threads = numba.config.NUMBA_NUM_THREADS
    
    # Memory optimization: Larger buffer size for cache alignment
    max_possible_edges = min(max_edges, (n_particles * (n_particles - 1)) // 2)
    edges_per_thread = max_possible_edges // num_threads + 2000  # Optimal buffer size
    
    # Thread-local pre-allocated arrays
    thread_edges_i = np.empty((num_threads, edges_per_thread), dtype=np.int64)
    thread_edges_j = np.empty((num_threads, edges_per_thread), dtype=np.int64) 
    thread_counts = np.zeros(num_threads, dtype=np.int64)
    
    # Pre-compute boundary condition checks and optimized constants
    is_periodic_x = boundary_conditions[0] == PERIODIC
    is_periodic_y = boundary_conditions[1] == PERIODIC
    box_x = box_size[0]
    box_y = box_size[1]
    half_box_x = box_x * 0.5
    half_box_y = box_y * 0.5
    linking_length_sq = linking_length * linking_length
    
    # Single pass with inline distance calculation
    for i in prange(n_particles):
        thread_id = numba.get_thread_id()
        local_count = 0
        
        for j in range(i + 1, n_particles):
            # Calculate coordinate differences
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            
            # Inline distance calculation with boundary conditions
            if is_periodic_x:
                if dx > half_box_x:
                    dx -= box_x
                elif dx < -half_box_x:
                    dx += box_x
            
            if is_periodic_y:
                if dy > half_box_y:
                    dy -= box_y
                elif dy < -half_box_y:
                    dy += box_y
            
            # Calculate squared distance (avoid sqrt when possible)
            dist_sq = dx * dx + dy * dy
            
            if dist_sq <= linking_length_sq:
                # Store in thread-local arrays to avoid race conditions
                current_idx = thread_counts[thread_id] + local_count
                if current_idx < edges_per_thread:
                    thread_edges_i[thread_id, current_idx] = i
                    thread_edges_j[thread_id, current_idx] = j
                    local_count += 1
        
        # Update thread count atomically after processing particle i
        thread_counts[thread_id] += local_count
    
    # Merge thread results sequentially (no race conditions)
    total_edges = np.sum(thread_counts)
    total_edges = min(total_edges, max_possible_edges)
    
    edges_i = np.empty(total_edges, dtype=np.int64)
    edges_j = np.empty(total_edges, dtype=np.int64)
    
    # Memory optimization: Block copy for better cache performance
    edge_idx = 0
    for t in range(num_threads):
        count = min(thread_counts[t], edges_per_thread)
        count = min(count, total_edges - edge_idx)  # Don't exceed total
        
        # Block copy optimization 
        if count > 0:
            end_idx = edge_idx + count
            edges_i[edge_idx:end_idx] = thread_edges_i[t, :count]
            edges_j[edge_idx:end_idx] = thread_edges_j[t, :count]
            edge_idx = end_idx
    
    return edges_i[:edge_idx], edges_j[:edge_idx]


@njit([types.Tuple([types.int64[:], types.int64[:]])(
    types.float32[:, :], types.float32, types.float32[:], types.int32[:], types.int64),
    types.Tuple([types.int64[:], types.int64[:]])(
    types.float64[:, :], types.float64, types.float64[:], types.int32[:], types.int64)],
      parallel=True, fastmath=True, cache=True, nogil=True)
def find_neighbors_3d_optimized(positions, linking_length, box_size, boundary_conditions, max_edges):
    """
    Phase 5: Memory-optimized 3D neighbor finding with inline distance calculation.
    """
    n_particles = positions.shape[0]
    num_threads = numba.config.NUMBA_NUM_THREADS
    
    # Memory optimization: Larger buffer size for cache alignment
    max_possible_edges = min(max_edges, (n_particles * (n_particles - 1)) // 2)
    edges_per_thread = max_possible_edges // num_threads + 2000  # Optimal buffer size
    
    # Thread-local pre-allocated arrays
    thread_edges_i = np.empty((num_threads, edges_per_thread), dtype=np.int64)
    thread_edges_j = np.empty((num_threads, edges_per_thread), dtype=np.int64) 
    thread_counts = np.zeros(num_threads, dtype=np.int64)
    
    # Pre-compute boundary condition checks and optimized constants
    is_periodic_x = boundary_conditions[0] == PERIODIC
    is_periodic_y = boundary_conditions[1] == PERIODIC
    is_periodic_z = boundary_conditions[2] == PERIODIC
    box_x = box_size[0]
    box_y = box_size[1]
    box_z = box_size[2]
    half_box_x = box_x * 0.5
    half_box_y = box_y * 0.5
    half_box_z = box_z * 0.5
    linking_length_sq = linking_length * linking_length
    
    # Single pass with inline distance calculation
    for i in prange(n_particles):
        thread_id = numba.get_thread_id()
        local_count = 0
        
        for j in range(i + 1, n_particles):
            # Calculate coordinate differences
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            
            # Inline distance calculation with boundary conditions
            if is_periodic_x:
                if dx > half_box_x:
                    dx -= box_x
                elif dx < -half_box_x:
                    dx += box_x
            
            if is_periodic_y:
                if dy > half_box_y:
                    dy -= box_y
                elif dy < -half_box_y:
                    dy += box_y
            
            if is_periodic_z:
                if dz > half_box_z:
                    dz -= box_z
                elif dz < -half_box_z:
                    dz += box_z
            
            # Calculate squared distance (avoid sqrt when possible)
            dist_sq = dx * dx + dy * dy + dz * dz
            
            if dist_sq <= linking_length_sq:
                # Store in thread-local arrays to avoid race conditions
                current_idx = thread_counts[thread_id] + local_count
                if current_idx < edges_per_thread:
                    thread_edges_i[thread_id, current_idx] = i
                    thread_edges_j[thread_id, current_idx] = j
                    local_count += 1
        
        # Update thread count atomically after processing particle i
        thread_counts[thread_id] += local_count
    
    # Merge thread results sequentially (no race conditions)
    total_edges = np.sum(thread_counts)
    total_edges = min(total_edges, max_possible_edges)
    
    edges_i = np.empty(total_edges, dtype=np.int64)
    edges_j = np.empty(total_edges, dtype=np.int64)
    
    # Memory optimization: Block copy for better cache performance
    edge_idx = 0
    for t in range(num_threads):
        count = min(thread_counts[t], edges_per_thread)
        count = min(count, total_edges - edge_idx)  # Don't exceed total
        
        # Block copy optimization 
        if count > 0:
            end_idx = edge_idx + count
            edges_i[edge_idx:end_idx] = thread_edges_i[t, :count]
            edges_j[edge_idx:end_idx] = thread_edges_j[t, :count]
            edge_idx = end_idx
    
    return edges_i[:edge_idx], edges_j[:edge_idx]


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
    
    # Estimate maximum possible edges
    max_edges = (n_particles * (n_particles - 1)) // 2
    
    # For large datasets, limit to reasonable maximum
    if max_edges > 10000000:  # 10M edges max
        max_edges = 10000000
    
    # Use optimized neighbor finding (Phase 5)
    edges_i, edges_j = find_neighbors_2d_optimized(positions, linking_length, box_size, boundary_conditions, max_edges)
    
    # Build clusters using Union-Find
    parent = np.arange(n_particles, dtype=type(min_cluster_size))
    
    # Process all edges
    for k in range(len(edges_i)):
        i = edges_i[k]
        j = edges_j[k]
        union_nodes(parent, i, j)
    
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
    
    # Estimate maximum possible edges
    max_edges = (n_particles * (n_particles - 1)) // 2
    
    # For large datasets, limit to reasonable maximum
    if max_edges > 10000000:  # 10M edges max
        max_edges = 10000000
    
    # Use optimized neighbor finding (Phase 5)
    edges_i, edges_j = find_neighbors_3d_optimized(positions, linking_length, box_size, boundary_conditions, max_edges)
    
    # Build clusters using Union-Find
    parent = np.arange(n_particles, dtype=type(min_cluster_size))
    
    # Process all edges
    for k in range(len(edges_i)):
        i = edges_i[k]
        j = edges_j[k]
        union_nodes(parent, i, j)
    
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

