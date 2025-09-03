"""
PLASMAtools Clustering Operations Module

This module provides friends-of-friends clustering operations using Numba
for performance optimization. It supports both 2D and 3D clustering with
various boundary conditions and precision options.

Author: James R. Beattie
"""

import numpy as np
from .constants import *
from .core_functions import *

PARALLEL_AVAILABLE = True

class ClusteringOperations():
    """
    Friends-of-Friends clustering using Numba kernels.
    
    This class provides high-performance clustering algorithms optimized for
    plasma simulation data. It supports 2D and 3D spatial clustering with
    periodic, Neumann, and Dirichlet boundary conditions.
    """
    
    def __init__(
        self,
        num_of_dims: int = 3,
        use_numba: bool = True,
        precision: str = 'float32',
        use_parallel: bool = False) -> None:
        """
        Initialize clustering operations.
        
        Args:
            num_of_dims: Number of spatial dimensions (2 or 3)
            use_numba: Whether to use Numba JIT compilation (recommended)
            precision: Numerical precision ('float32' or 'float64')
            use_parallel: Whether to use parallel implementation (requires numba.config.NUMBA_NUM_THREADS > 1)
        """
        
        if num_of_dims not in [2, 3]:
            raise ValueError("num_of_dims must be 2 or 3")
        if precision not in ['float32', 'float64']:
            raise ValueError("precision must be 'float32' or 'float64'")
            
        self.num_of_dims = num_of_dims
        self.use_numba = use_numba
        self.precision = precision
        self.use_parallel = use_parallel and PARALLEL_AVAILABLE
        
        if use_parallel and not PARALLEL_AVAILABLE:
            print("Warning: Parallel implementation not available, falling back to sequential")
        
        # Set data types based on precision
        if precision == 'float32':
            self.float_dtype = np.float32
            self.int_dtype = np.int32
        else:
            self.float_dtype = np.float64
            self.int_dtype = np.int64
    
    def friends_of_friends(
        self,
        positions: np.ndarray,
        linking_length: float = DEFAULT_LINKING_LENGTH,
        box_size: np.ndarray = None,
        boundary_conditions: np.ndarray = None,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE) -> np.ndarray:
        """
        Perform friends-of-friends clustering on particle positions.
        
        Args:
            positions: (N, D) array of particle positions where D is 2 or 3
            linking_length: Maximum distance for particles to be considered friends
            box_size: (D,) array of domain sizes for each dimension. If None, 
                     inferred from data range
            boundary_conditions: (D,) array of boundary condition types per dimension.
                               If None, defaults to periodic for all dimensions
            min_cluster_size: Minimum number of particles required to form a cluster
        
        Returns:
            cluster_labels: (N,) array of cluster IDs. -1 indicates noise/isolated particles
        """
        
        # Validate input dimensions
        if positions.ndim != 2:
            raise ValueError("positions must be a 2D array (N, D)")
        
        n_particles, spatial_dims = positions.shape
        if spatial_dims != self.num_of_dims:
            raise ValueError(f"positions must have {self.num_of_dims} spatial dimensions")
        
        # Handle empty input
        if n_particles == 0:
            return np.array([], dtype=self.int_dtype)
        
        # Convert to appropriate precision
        positions = positions.astype(self.float_dtype)
        linking_length = self.float_dtype(linking_length)
        min_cluster_size = self.int_dtype(min_cluster_size)
        
        # Set default box size if not provided and shift positions into [0, box)
        if box_size is None:
            pos_min = np.min(positions, axis=0)
            pos_max = np.max(positions, axis=0)
            data_range = pos_max - pos_min
            
            # Add padding to ensure proper boundary condition handling
            # Use at least 2x the linking length as padding, or 10% of data range
            padding = np.maximum(2 * linking_length, 0.1 * data_range)
            box_size = (data_range + 2 * padding).astype(self.float_dtype)
            
            # Ensure minimum box size is at least 4x linking length in each dimension
            min_box_size = 4 * linking_length
            box_size = np.maximum(box_size, min_box_size).astype(self.float_dtype)

            # Shift positions so they are within [0, box) to enable efficient cell-list hashing
            origin = (pos_min - padding).astype(self.float_dtype)
            positions = (positions - origin).astype(self.float_dtype)
        else:
            box_size = np.array(box_size, dtype=self.float_dtype)
        
        # Set default boundary conditions if not provided (all Neumann for general use)
        if boundary_conditions is None:
            boundary_conditions = np.full(self.num_of_dims, NEUMANN, dtype=np.int32)
        else:
            boundary_conditions = np.array(boundary_conditions, dtype=np.int32)
        
        # Validate array sizes
        if box_size.shape[0] != self.num_of_dims:
            raise ValueError(f"box_size must have {self.num_of_dims} elements")
        if boundary_conditions.shape[0] != self.num_of_dims:
            raise ValueError(f"boundary_conditions must have {self.num_of_dims} elements")
        
        # Call appropriate core function based on dimensions and precision
        if not self.use_numba:
            return self._fof_numpy(positions, linking_length, box_size, 
                                 boundary_conditions, min_cluster_size)
        
        # Ensure arrays are contiguous to avoid numba compilation issues
        positions = np.ascontiguousarray(positions)
        box_size = np.ascontiguousarray(box_size)
        boundary_conditions = np.ascontiguousarray(boundary_conditions)
        
        try:
            # Use the optimized implementations from core_functions.py
            if self.num_of_dims == 2:
                return fof_2d(positions, linking_length, box_size, 
                             boundary_conditions, min_cluster_size)
            else:  # 3D
                return fof_3d(positions, linking_length, box_size, 
                             boundary_conditions, min_cluster_size)
        except Exception as e:
            # If numba compilation fails, fall back to numpy implementation
            print(f"Warning: Numba FOF failed ({e}), falling back to numpy implementation")
            return self._fof_numpy(positions, linking_length, box_size, 
                                 boundary_conditions, min_cluster_size)

    def friends_of_friends_grid(
        self,
        positions: np.ndarray,
        linking_length: float = DEFAULT_LINKING_LENGTH,
        box_size: np.ndarray = None,
        boundary_conditions: np.ndarray = None,
        voxel_size: float = None,
        connectivity: int = None,
        min_points_per_voxel: int = 1,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    ) -> np.ndarray:
        """
        Discretized FOF via voxelization + connected components on a grid.

        This approximates FOF by binning points into voxels of size `voxel_size`
        (default ~ linking_length/2), labeling connected voxel components using
        26-connectivity (3D) or 8-connectivity (2D), and mapping voxel labels
        back to points. Components below `min_cluster_size` (measured in points)
        are marked as noise (-1).

        This method is memory efficient for sparse occupancy because it uses a
        sparse BFS over occupied voxels (no full dense 3D arrays required).
        """
        if positions.ndim != 2:
            raise ValueError("positions must be a 2D array (N, D)")
        n_particles, spatial_dims = positions.shape
        if spatial_dims != self.num_of_dims:
            raise ValueError(f"positions must have {self.num_of_dims} spatial dimensions")
        if n_particles == 0:
            return np.array([], dtype=self.int_dtype)

        # Defaults
        if voxel_size is None:
            # Half the linking length gives conservative connectivity
            voxel_size = float(linking_length) * 0.5
            if voxel_size <= 0:
                voxel_size = float(linking_length)
        if connectivity is None:
            connectivity = 8 if self.num_of_dims == 2 else 26

        # Precision and copies
        positions = positions.astype(self.float_dtype)
        linking_length = self.float_dtype(linking_length)
        min_cluster_size = int(min_cluster_size)
        min_points_per_voxel = int(min_points_per_voxel)

        # Box and BCs
        if box_size is None:
            pos_min = np.min(positions, axis=0)
            pos_max = np.max(positions, axis=0)
            data_range = pos_max - pos_min
            padding = np.maximum(2 * linking_length, 0.1 * data_range)
            box_size = (data_range + 2 * padding).astype(self.float_dtype)
            # Ensure minimum box size is at least 4x linking length
            min_box_size = 4 * linking_length
            box_size = np.maximum(box_size, min_box_size).astype(self.float_dtype)
            origin = (pos_min - padding).astype(self.float_dtype)
            pos = (positions - origin).astype(self.float_dtype)
        else:
            box_size = np.array(box_size, dtype=self.float_dtype)
            pos = positions

        if boundary_conditions is None:
            boundary_conditions = np.full(self.num_of_dims, NEUMANN, dtype=np.int32)
        else:
            boundary_conditions = np.array(boundary_conditions, dtype=np.int32)

        # Grid shape
        dims = np.maximum(1, np.ceil(box_size / self.float_dtype(voxel_size)).astype(np.int64))
        # Avoid pathological memory via dense volume by operating sparsely

        # Compute voxel indices for each point
        inv_vs = self.float_dtype(1.0) / self.float_dtype(voxel_size)
        idx = np.floor(pos * inv_vs).astype(np.int64)

        # Apply BC to indices: periodic wraps, neumann clamp
        for d in range(self.num_of_dims):
            if boundary_conditions[d] == PERIODIC:
                idx[:, d] %= dims[d]
            else:
                idx[:, d] = np.minimum(np.maximum(idx[:, d], 0), dims[d] - 1)

        # Map voxel (linear index) -> count of points using vectorized unique
        if self.num_of_dims == 2:
            lin = idx[:, 0] * dims[1] + idx[:, 1]
        else:
            lin = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]

        uniq, inv, cnt = np.unique(lin, return_inverse=True, return_counts=True)
        occ_mask = cnt >= min_points_per_voxel
        if not np.any(occ_mask):
            return np.full(n_particles, -1, dtype=self.int_dtype)

        if self.num_of_dims == 2:
            # Retain original Python path for 2D for now
            occupied = set(uniq[occ_mask].tolist())
            # Build neighbor offsets as before
            offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
            if connectivity == 4:
                offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            component_id = {}
            comp_sizes_pts = []
            current_label = 0

            def neighbors_from_lin_2d(vlin2):
                x = vlin2 // dims[1]
                y = vlin2 % dims[1]
                for dx, dy in offsets:
                    nx = x + dx
                    ny = y + dy
                    if boundary_conditions[0] == PERIODIC:
                        nx %= dims[0]
                    elif nx < 0 or nx >= dims[0]:
                        continue
                    if boundary_conditions[1] == PERIODIC:
                        ny %= dims[1]
                    elif ny < 0 or ny >= dims[1]:
                        continue
                    yield nx * dims[1] + ny

            counts_map = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}
            for v in uniq[occ_mask]:
                v = int(v)
                if v in component_id:
                    continue
                q = [v]
                component_id[v] = current_label
                total_pts = counts_map.get(v, 0)
                qi = 0
                while qi < len(q):
                    cur = q[qi]
                    qi += 1
                    for nb in neighbors_from_lin_2d(cur):
                        if nb in occupied and nb not in component_id:
                            component_id[nb] = current_label
                            q.append(nb)
                            total_pts += counts_map.get(nb, 0)
                comp_sizes_pts.append(total_pts)
                current_label += 1

            comp_keep = {cid for cid, sz in enumerate(comp_sizes_pts) if sz >= min_cluster_size}
            out = np.full(n_particles, -1, dtype=self.int_dtype)
            # Vectorized back-mapping using inv -> uniq index
            # Fallback simple loop for 2D mapping
            for i in range(n_particles):
                vlin_i = int(lin[i])
                cid = component_id.get(vlin_i, -1)
                if cid in comp_keep:
                    out[i] = cid
            return out

        # 3D path with Numba-accelerated sparse CCL
        occ_lin = uniq[occ_mask].astype(np.int64, copy=False)
        occ_counts = cnt[occ_mask].astype(np.int64, copy=False)
        dims64 = dims.astype(np.int64, copy=False)
        # Compute components on occupied voxels
        voxel_labels, comp_sizes_pts, n_comp = ccl_occupied_voxels_3d(
            occ_lin.astype(np.int64), occ_counts.astype(np.int64), dims64, boundary_conditions.astype(np.int32), int(connectivity)
        )

        # Keep components by total points
        if n_comp == 0:
            return np.full(n_particles, -1, dtype=self.int_dtype)
        keep = np.zeros(n_comp, dtype=np.bool_)
        for cid in range(n_comp):
            if comp_sizes_pts[cid] >= int(min_cluster_size):
                keep[cid] = True

        # Remap kept component ids to 0..K-1
        comp_map = -np.ones(n_comp, dtype=np.int64)
        next_id = 0
        for cid in range(n_comp):
            if keep[cid]:
                comp_map[cid] = next_id
                next_id += 1

        # Build uniq -> component id map (-1 for non-occupied or dropped)
        uniq_to_comp = -np.ones(uniq.shape[0], dtype=np.int64)
        occ_idx = np.where(occ_mask)[0]
        for j in range(occ_idx.size):
            cid = voxel_labels[j]
            if cid >= 0 and keep[cid]:
                uniq_to_comp[occ_idx[j]] = comp_map[cid]

        # Map each point using inverse indices
        comp_for_point = uniq_to_comp[inv]
        out = np.full(n_particles, -1, dtype=self.int_dtype)
        # Assign only for kept components
        sel = comp_for_point >= 0
        out[sel] = comp_for_point[sel].astype(self.int_dtype, copy=False)
        return out
    
    def _fof_numpy(
        self,
        positions: np.ndarray,
        linking_length: float,
        box_size: np.ndarray,
        boundary_conditions: np.ndarray,
        min_cluster_size: int) -> np.ndarray:
        """
        Fallback NumPy implementation (slower but more compatible).
        """
        n_particles = positions.shape[0]
        
        # Initialize Union-Find structure
        parent = np.arange(n_particles, dtype=self.int_dtype)
        
        def find_root(node):
            if parent[node] != node:
                parent[node] = find_root(parent[node])
            return parent[node]
        
        def union_nodes(node1, node2):
            root1 = find_root(node1)
            root2 = find_root(node2)
            if root1 != root2:
                parent[root2] = root1
        
        # Find all pairs within linking length
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                # Calculate distance with boundary conditions
                dx = positions[i] - positions[j]
                
                # Apply periodic boundary conditions
                for dim in range(self.num_of_dims):
                    if boundary_conditions[dim] == PERIODIC:
                        if dx[dim] > 0.5 * box_size[dim]:
                            dx[dim] -= box_size[dim]
                        elif dx[dim] < -0.5 * box_size[dim]:
                            dx[dim] += box_size[dim]
                
                dist = np.linalg.norm(dx)
                
                if dist <= linking_length:
                    union_nodes(i, j)
        
        # Assign final cluster labels
        cluster_labels = np.full(n_particles, -1, dtype=self.int_dtype)
        cluster_sizes = np.zeros(n_particles, dtype=self.int_dtype)
        
        # Count cluster sizes
        for i in range(n_particles):
            root = find_root(i)
            cluster_sizes[root] += 1
        
        # Assign labels only to clusters meeting minimum size
        cluster_id = 0
        cluster_map = np.full(n_particles, -1, dtype=self.int_dtype)
        
        for i in range(n_particles):
            root = find_root(i)
            if cluster_sizes[root] >= min_cluster_size:
                if cluster_map[root] == -1:
                    cluster_map[root] = cluster_id
                    cluster_id += 1
                cluster_labels[i] = cluster_map[root]
        
        return cluster_labels
    
    def get_cluster_properties(
        self,
        positions: np.ndarray,
        cluster_labels: np.ndarray,
        radius_method: str = 'effective90',
        box_size: np.ndarray | None = None,
        boundary_conditions: np.ndarray | None = None) -> dict:
        """
        Calculate properties of identified clusters.
        
        Args:
            positions: (N, D) array of particle positions
            cluster_labels: (N,) array of cluster IDs from friends_of_friends()
            radius_method: Method for calculating cluster radius:
                         - 'effective90': radius containing 90% of points (best for SN extent)
                         - 'effective50': radius containing 50% of points (half-mass radius)
                         - 'maximum': maximum distance from center (shock front)
                         - 'rms': root-mean-square radius (original method)
            box_size: Optional domain lengths per axis. If provided with
                      periodic boundary conditions, centers are computed with
                      circular means and distances use minimal-image deltas.
            boundary_conditions: Optional BCs per axis; used only if box_size
                                  is provided to handle periodicity.
        
        Returns:
            Dictionary with cluster properties:
            - n_clusters: number of clusters found
            - cluster_sizes: array of cluster sizes
            - cluster_centers: array of cluster center positions
            - cluster_radii: array of cluster radii using specified method
        """
        
        # Get unique cluster IDs (excluding noise label -1)
        unique_labels = np.unique(cluster_labels)
        cluster_ids = unique_labels[unique_labels >= 0]
        n_clusters = len(cluster_ids)
        
        if n_clusters == 0:
            return {
                'n_clusters': 0,
                'cluster_sizes': np.array([]),
                'cluster_centers': np.array([]).reshape(0, self.num_of_dims),
                'cluster_radii': np.array([])
            }
        
        # Calculate properties for each cluster
        cluster_sizes = np.zeros(n_clusters, dtype=self.int_dtype)
        cluster_centers = np.zeros((n_clusters, self.num_of_dims), dtype=self.float_dtype)
        cluster_radii = np.zeros(n_clusters, dtype=self.float_dtype)

        # Optional periodic handling
        periodic_axes = None
        L = None
        if box_size is not None and boundary_conditions is not None:
            L = np.array(box_size, dtype=self.float_dtype)
            periodic_axes = np.array([bc == PERIODIC for bc in np.array(boundary_conditions, dtype=np.int32)])
        
        for i, cluster_id in enumerate(cluster_ids):
            mask = cluster_labels == cluster_id
            cluster_positions = positions[mask]
            
            cluster_sizes[i] = np.sum(mask)
            # Center: use circular mean for periodic axes when available
            if periodic_axes is not None:
                for d in range(self.num_of_dims):
                    if periodic_axes[d]:
                        ang = (2.0 * np.pi * cluster_positions[:, d]) / L[d]
                        s = np.sin(ang).sum(dtype=self.float_dtype)
                        c = np.cos(ang).sum(dtype=self.float_dtype)
                        theta = np.arctan2(s, c)
                        if theta < 0:
                            theta += 2.0 * np.pi
                        cluster_centers[i, d] = (theta / (2.0 * np.pi)) * L[d]
                    else:
                        cluster_centers[i, d] = np.mean(cluster_positions[:, d], dtype=self.float_dtype)
            else:
                cluster_centers[i] = np.mean(cluster_positions, axis=0)
            
            # Calculate radius using specified method
            if cluster_sizes[i] > 1:
                deltas = cluster_positions - cluster_centers[i]
                if periodic_axes is not None:
                    for d in range(self.num_of_dims):
                        if periodic_axes[d]:
                            deltas[:, d] = ((deltas[:, d] + 0.5 * L[d]) % L[d]) - 0.5 * L[d]
                distances = np.linalg.norm(deltas, axis=1)
                
                if radius_method == 'effective90':
                    # Effective radius containing 90% of points (best for supernova extent)
                    distances_sorted = np.sort(distances)
                    idx_90 = int(0.9 * len(distances_sorted))
                    cluster_radii[i] = distances_sorted[idx_90]
                elif radius_method == 'effective50':
                    # Half-mass radius containing 50% of points  
                    distances_sorted = np.sort(distances)
                    idx_50 = int(0.5 * len(distances_sorted))
                    cluster_radii[i] = distances_sorted[idx_50]
                elif radius_method == 'maximum':
                    # Maximum radius (shock front extent)
                    cluster_radii[i] = np.max(distances)
                elif radius_method == 'rms':
                    # Root-mean-square radius (original method)
                    cluster_radii[i] = np.sqrt(np.mean(distances**2))
                else:
                    raise ValueError(f"Unknown radius_method: {radius_method}")
            else:
                cluster_radii[i] = 0.0
        
        return {
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_sizes,
            'cluster_centers': cluster_centers,
            'cluster_radii': cluster_radii
        }
    
    def filter_clusters_by_size(
        self,
        cluster_labels: np.ndarray,
        min_size: int,
        max_size: int = None) -> np.ndarray:
        """
        Filter clusters by size, marking others as noise.
        
        Args:
            cluster_labels: (N,) array of cluster IDs
            min_size: minimum cluster size to keep
            max_size: maximum cluster size to keep (None for no upper limit)
        
        Returns:
            filtered_labels: (N,) array with filtered cluster IDs
        """
        
        filtered_labels = cluster_labels.copy()
        unique_labels = np.unique(cluster_labels)
        cluster_ids = unique_labels[unique_labels >= 0]
        
        for cluster_id in cluster_ids:
            mask = cluster_labels == cluster_id
            cluster_size = np.sum(mask)
            
            should_remove = cluster_size < min_size
            if max_size is not None:
                should_remove = should_remove or cluster_size > max_size
            
            if should_remove:
                filtered_labels[mask] = -1
        
        return filtered_labels
    
    def binary_mask_to_positions(self, binary_mask, grid_spacing=1.0):
        """
        Convert a binary mask (2D or 3D) to particle positions for clustering.
        
        Args:
            binary_mask: 2D or 3D boolean or binary array
            grid_spacing: Physical spacing between grid points (scalar or array)
        
        Returns:
            positions: (N, D) array of positions where mask is True
        """
        # Get indices where mask is True
        indices = np.where(binary_mask)
        
        # Convert indices to positions
        positions = np.column_stack(indices).astype(self.float_dtype)
        
        # Scale by grid spacing if provided
        if np.isscalar(grid_spacing):
            positions *= grid_spacing
        else:
            positions *= np.array(grid_spacing, dtype=self.float_dtype)
        
        return positions
    
    def apply_morphological_filter(
        self,
        mask: np.ndarray,
        min_thickness: int = 2,
        remove_small_objects_size: int = None,
        operation: str = 'open',
        periodic_axes: tuple = None) -> np.ndarray:
        """
        Apply morphological filtering to remove thin shell-like features.
        
        This is particularly useful for removing shock fronts and other thin
        structures that shouldn't be identified as separate clusters in 
        supernova analysis.
        
        Args:
            mask: Binary mask (2D or 3D)
            min_thickness: Number of iterations for the morphology operation.
            remove_small_objects_size: Minimum size for connected components.
                                      If None, uses min_thickness^ndim * 8
            operation: 'open' to remove thin links; 'close' to bridge small gaps
            periodic_axes: tuple of booleans per axis; when True, wrap-pad that axis
                           so morphology respects periodic boundaries
        
        Returns:
            Filtered binary mask with thin structures removed
        """
        try:
            from scipy import ndimage
        except ImportError:
            print("Warning: scipy not available, skipping morphological filtering")
            return mask
        
        ndim = mask.ndim
        if ndim not in [2, 3]:
            raise ValueError(f"Mask must be 2D or 3D, got {ndim}D")
        
        original_count = np.sum(mask)
        if original_count == 0:
            return mask
        
        # Generate structuring element for morphological operations
        # Use connectivity=1 (4-conn in 2D, 6-conn in 3D) for more aggressive filtering
        structure = ndimage.generate_binary_structure(ndim, connectivity=1)
        
        # Optional periodic padding so morphology respects wrap-around connectivity
        if periodic_axes is None:
            periodic_axes = tuple([False] * ndim)
        pad_n = max(1, int(min_thickness)) if min_thickness > 0 else 1
        pad_spec = []
        for ax in range(ndim):
            if periodic_axes[ax]:
                pad_spec.append((pad_n, pad_n))
            else:
                pad_spec.append((0, 0))
        if any(pw[0] > 0 or pw[1] > 0 for pw in pad_spec):
            mask_work = np.pad(mask, pad_spec, mode='wrap')
        else:
            mask_work = mask.copy()

        # Step 1: Morphological operation
        # 'open' removes thin connections; 'close' bridges small gaps
        if operation == 'close':
            if min_thickness > 0:
                mask_morph = ndimage.binary_closing(
                    mask_work,
                    structure=structure,
                    iterations=min_thickness
                )
            else:
                mask_morph = mask_work.copy()
        else:  # 'open' (default)
            if min_thickness > 0:
                mask_morph = ndimage.binary_opening(
                    mask_work,
                    structure=structure,
                    iterations=min_thickness
                )
            else:
                mask_morph = mask_work.copy()

        # Step 2: Fill small holes to maintain bulk structure
        mask_filled = ndimage.binary_fill_holes(mask_morph)
        
        # Step 3: Remove small disconnected components
        if remove_small_objects_size is None:
            # Default minimum size based on minimum thickness
            remove_small_objects_size = (min_thickness ** ndim) * 8
        
        if remove_small_objects_size > 0:
            # Label connected components
            labeled, num_features = ndimage.label(mask_filled, structure=structure)
            
            if num_features > 0:
                # Count size of each component
                component_sizes = ndimage.sum(mask_filled, labeled, range(1, num_features + 1))
                
                # Remove small components
                small_components = np.where(component_sizes < remove_small_objects_size)[0] + 1
                for comp_id in small_components:
                    mask_filled[labeled == comp_id] = False
        
        final_count = np.sum(mask_filled)
        removed_pixels = original_count - final_count
        
        if removed_pixels > 0:
            print(f"  Morphological filter removed {removed_pixels:,} pixels "
                  f"({100*removed_pixels/original_count:.1f}% of original)")
        
        # Crop back if padded
        if any(pw[0] > 0 or pw[1] > 0 for pw in pad_spec):
            slices = []
            for ax in range(ndim):
                if periodic_axes[ax]:
                    slices.append(slice(pad_spec[ax][0], mask_filled.shape[ax] - pad_spec[ax][1]))
                else:
                    slices.append(slice(0, mask_filled.shape[ax]))
            mask_final = mask_filled[tuple(slices)]
        else:
            mask_final = mask_filled
        return mask_final
    
    def separate_overlapping_clusters(
        self,
        positions: np.ndarray,
        labels: np.ndarray,
        field_values: np.ndarray = None,
        min_separation: float = 5.0,
        size_threshold: int = None,
        max_peaks: int = 3,
        max_clusters_to_process: int = 5) -> np.ndarray:
        """
        Separate overlapping clusters using optimized peak detection.
        
        This is particularly useful for separating nearby supernova explosions
        that get merged into single clusters. Uses spatial hashing for O(N log N) performance.
        
        Args:
            positions: (N, D) array of particle positions
            labels: (N,) array of cluster labels from friends_of_friends
            field_values: (N,) array of field values at each position (e.g., temperature).
                         If None, uses spatial density for peak detection.
            min_separation: Minimum distance between peaks to consider them separate
            size_threshold: Only apply separation to clusters larger than this.
                          If None, applies to clusters > 10x min_cluster_size
            max_peaks: Maximum number of peaks to find per cluster (prevents over-segmentation)
            max_clusters_to_process: Only process the N largest clusters (for performance)
        
        Returns:
            New cluster labels with overlapping clusters separated
        """
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            print("Warning: scipy not available, skipping cluster separation")
            return labels
        
        if size_threshold is None:
            size_threshold = 10 * DEFAULT_MIN_CLUSTER_SIZE
        
        new_labels = labels.copy()
        unique_labels = np.unique(labels[labels >= 0])
        next_label = np.max(labels) + 1 if len(labels) > 0 else 0
        
        # Calculate cluster sizes and get the largest ones
        cluster_sizes = []
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            cluster_sizes.append((cluster_id, cluster_size))
        
        # Sort by size (largest first) and take only the top N
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        clusters_to_process = cluster_sizes[:max_clusters_to_process]
        
        print(f"Processing only the {len(clusters_to_process)} largest clusters for separation:")
        for i, (cluster_id, size) in enumerate(clusters_to_process):
            print(f"  Rank {i+1}: Cluster {cluster_id} with {size} points")
        
        separated_clusters = 0
        
        for cluster_id, cluster_size in clusters_to_process:
            cluster_mask = labels == cluster_id
            cluster_positions = positions[cluster_mask]
            cluster_size = len(cluster_positions)
            
            # Only process large clusters
            if cluster_size < size_threshold:
                continue
            
            print(f"  Processing cluster {cluster_id} ({cluster_size} points)...")
            
            # Get field values for this cluster
            if field_values is not None:
                cluster_values = field_values[cluster_mask]
            else:
                # Fast density estimation using KD-tree
                tree = cKDTree(cluster_positions)
                cluster_values = np.array([len(tree.query_ball_point(pos, min_separation)) 
                                         for pos in cluster_positions])
            
            # Find local maxima efficiently using spatial sorting
            # Sort by field value (descending)
            value_order = np.argsort(cluster_values)[::-1]
            
            # Find peaks by checking if each point is maximum in its neighborhood
            peaks = []
            peak_values = []
            tree = cKDTree(cluster_positions)
            
            for idx in value_order:
                pos = cluster_positions[idx]
                value = cluster_values[idx]
                
                # Find neighbors within min_separation
                neighbor_indices = tree.query_ball_point(pos, min_separation)
                neighbor_values = cluster_values[neighbor_indices]
                
                # This is a peak if it has the maximum value in its neighborhood
                if value >= np.max(neighbor_values):
                    peaks.append(idx)
                    peak_values.append(value)
                    
                    # Early termination if we have enough peaks
                    if len(peaks) >= max_peaks:  # Limit to avoid over-segmentation
                        break
            
            # Remove peaks that are too close to each other
            if len(peaks) > 1:
                final_peaks = [peaks[0]]  # Always keep the strongest
                
                for peak_idx in peaks[1:]:
                    peak_pos = cluster_positions[peak_idx]
                    
                    # Check distance to all previously accepted peaks
                    too_close = False
                    for accepted_peak_idx in final_peaks:
                        accepted_pos = cluster_positions[accepted_peak_idx]
                        if np.linalg.norm(peak_pos - accepted_pos) < min_separation:
                            too_close = True
                            break
                    
                    if not too_close:
                        final_peaks.append(peak_idx)
                
                # Only separate if we found multiple well-separated peaks
                if len(final_peaks) > 1:
                    print(f"    Found {len(final_peaks)} well-separated peaks, separating...")
                    
                    # Fast nearest neighbor assignment using vectorized operations
                    peak_positions = cluster_positions[final_peaks]
                    
                    # Vectorized distance calculation
                    distances = np.linalg.norm(
                        cluster_positions[:, np.newaxis, :] - peak_positions[np.newaxis, :, :],
                        axis=2
                    )
                    nearest_peaks = np.argmin(distances, axis=1)
                    
                    # Assign new labels
                    original_indices = np.where(cluster_mask)[0]
                    for i, nearest_peak in enumerate(nearest_peaks):
                        if nearest_peak == 0:
                            new_labels[original_indices[i]] = cluster_id  # Keep original
                        else:
                            new_labels[original_indices[i]] = next_label + nearest_peak - 1
                    
                    next_label += len(final_peaks) - 1
                    separated_clusters += 1
                else:
                    print(f"    No well-separated peaks found, keeping as single cluster")
            else:
                print(f"    Single peak found, keeping as single cluster")
        
        if separated_clusters > 0:
            print(f"Successfully separated {separated_clusters} overlapping cluster pairs")
        else:
            print("No overlapping clusters found that could be separated")
        
        return new_labels
    
    def cluster_3d_field(
        self,
        field: np.ndarray,
        threshold: float = None,
        threshold_type: str = 'greater',
        grid_spacing: float = 1.0,
        linking_length: float = DEFAULT_LINKING_LENGTH,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
        morphological_filter: bool = False,
        min_thickness: int = 2,
        morph_operation: str = 'open',
        separate_overlapping: bool = False,
        min_separation: float = 5.0,
        max_peaks: int = 3,
        max_clusters_to_process: int = 5,
        return_field: bool = True,
        method: str = 'grid',
        connectivity: int = 26,
        boundary_conditions: np.ndarray = None,
        split_large_clusters: bool = False,
        split_size_factor: float = 10.0,
        split_min_separation_vox: int = 5,
        radius_method: str = 'effective90') -> dict:
        """
        Cluster a 3D field directly via grid connected components or FOF.
        
        This is a convenience method that combines thresholding, optional morphological
        filtering, clustering (grid-native or FOF), and optional mapping back to the grid.
        
        Args:
            field: 3D array to cluster
            threshold: Threshold value for creating binary mask. If None, uses median
            threshold_type: 'greater' or 'less' - how to apply threshold
            grid_spacing: Physical spacing between grid points
            linking_length: Maximum distance for particles to be friends (FOF method)
            min_cluster_size: Minimum number of particles for valid cluster
            morphological_filter: If True, apply morphological filtering
            min_thickness: Iterations for morphology
            morph_operation: 'open' to remove thin links; 'close' to bridge small gaps
            separate_overlapping: If True, attempt to separate overlapping clusters
                                using temperature peak detection (useful for nearby SNe)
            min_separation: Minimum distance between peaks for cluster separation
            max_peaks: Maximum number of peaks per cluster (prevents over-segmentation)
            max_clusters_to_process: Only process the N largest clusters (for performance)
            return_field: If True, return cluster labels mapped to 3D grid
            method: 'grid' (default) for grid-native 3D connected-components; 'fof' for particle FOF
            connectivity: Neighborhood connectivity for grid method (3D: 6/18/26; 2D: 4/8)
            boundary_conditions: (D,) array of BCs for grid method; default [PERIODIC, PERIODIC, NEUMANN] in 3D
            radius_method: 'effective90' (default), 'effective50', 'maximum', or 'rms' for cluster radius (grid mode)
        
        Returns:
            Dictionary containing:
            - 'labels': cluster labels for active positions
            - 'mask': binary mask used for clustering (after optional filtering)
            - 'mask_original': original binary mask before filtering (if filtering applied)
            - 'positions': particle positions
            - 'properties': cluster properties (sizes, centers, radii)
            - 'cluster_field': (optional) cluster labels mapped to 3D grid
        """
        # Validate input
        if field.ndim != self.num_of_dims:
            raise ValueError(f"Field must be {self.num_of_dims}D, got {field.ndim}D")
        
        # Apply threshold to create binary mask
        if threshold is None:
            threshold = np.median(field)
            print(f"Using median threshold: {threshold:.4f}")
        
        if threshold_type == 'greater':
            mask = field > threshold
        elif threshold_type == 'less':
            mask = field < threshold
        else:
            raise ValueError("threshold_type must be 'greater' or 'less'")
        
        n_active = np.sum(mask)
        print(f"Binary mask: {n_active:,} active points ({100*n_active/mask.size:.1f}%)")
        
        # Store original mask for output
        mask_original = mask.copy() if morphological_filter else None
        
        # Apply morphological filtering if requested
        if morphological_filter:
            print("Applying morphological filtering to remove thin shells...")
            mask = self.apply_morphological_filter(
                mask, 
                min_thickness=min_thickness,
                remove_small_objects_size=None,  # Use default sizing
                operation=morph_operation,
                periodic_axes=tuple([bc == PERIODIC for bc in boundary_conditions])
            )
            n_active_filtered = np.sum(mask)
            if n_active_filtered != n_active:
                print(f"Filtered mask: {n_active_filtered:,} active points "
                      f"({100*n_active_filtered/mask.size:.1f}% of grid)")
        
        if np.sum(mask) == 0:
            print("Warning: No points remain after filtering")
            result = {
                'labels': np.array([], dtype=self.int_dtype),
                'mask': mask,
                'positions': np.array([]).reshape(0, self.num_of_dims),
                'properties': {
                    'n_clusters': 0,
                    'cluster_sizes': np.array([]),
                    'cluster_centers': np.array([]).reshape(0, self.num_of_dims),
                    'cluster_radii': np.array([])
                },
                'cluster_field': np.full(mask.shape, -1, dtype=self.int_dtype) if return_field else None
            }
            if mask_original is not None:
                result['mask_original'] = mask_original
            return result
        
        # Default BCs for grid method: periodic XY, Neumann Z
        if boundary_conditions is None:
            if self.num_of_dims == 3:
                boundary_conditions = np.array([PERIODIC, PERIODIC, NEUMANN], dtype=np.int32)
            elif self.num_of_dims == 2:
                boundary_conditions = np.array([PERIODIC, PERIODIC], dtype=np.int32)
            else:
                boundary_conditions = np.full(self.num_of_dims, NEUMANN, dtype=np.int32)

        if method == 'grid':
            grid_labels = self._cluster_3d_mask_sparse(mask, connectivity=connectivity,
                                                       boundary_conditions=boundary_conditions,
                                                       min_cluster_size=min_cluster_size)
            # Build positions for compatibility
            positions = self.binary_mask_to_positions(mask, grid_spacing)
            # Properties from grid
            properties = self._properties_from_grid_labels(grid_labels, grid_spacing, radius_method=radius_method, boundary_conditions=boundary_conditions)
            labels = properties.pop('_point_labels', None)
            # Optionally split very large clusters using peak-based seeded growth
            if split_large_clusters:
                try:
                    grid_labels = self._split_large_clusters_grid(
                        field, grid_labels,
                        size_threshold=max(int(split_size_factor * min_cluster_size), min_cluster_size*2),
                        min_separation_vox=int(split_min_separation_vox),
                        max_peaks=max_peaks,
                        max_clusters_to_process=max_clusters_to_process,
                    )
                    properties = self._properties_from_grid_labels(grid_labels, grid_spacing, radius_method=radius_method, boundary_conditions=boundary_conditions)
                except Exception as e:
                    print(f"Warning: grid cluster splitting failed: {e}")
        elif method == 'fof':
            # Convert mask to positions
            positions = self.binary_mask_to_positions(mask, grid_spacing)
            labels = self.friends_of_friends(
                positions=positions,
                linking_length=linking_length,
                min_cluster_size=min_cluster_size
            )
            properties = self.get_cluster_properties(positions, labels)
        else:
            raise ValueError("method must be 'grid' or 'fof'")
        
        # Apply cluster separation if requested
        if separate_overlapping and method == 'fof':
            print("Applying cluster separation for overlapping explosions...")
            
            # Extract field values at cluster positions for peak detection
            indices = np.where(mask)
            if len(indices[0]) == len(positions):
                # Get field values at cluster positions for better separation
                field_values_at_positions = field[indices]
            else:
                field_values_at_positions = None
            
            labels = self.separate_overlapping_clusters(
                positions=positions,
                labels=labels,
                field_values=field_values_at_positions,
                min_separation=min_separation,
                size_threshold=min_cluster_size * 10,  # Only separate large clusters
                max_peaks=max_peaks,
                max_clusters_to_process=max_clusters_to_process
            )
        
        result = {
            'labels': labels if labels is not None else np.array([], dtype=self.int_dtype),
            'mask': mask,
            'positions': positions,
            'properties': properties
        }
        
        # Include original mask if morphological filtering was applied
        if mask_original is not None:
            result['mask_original'] = mask_original
        
        # Optionally map back to 3D grid
        if return_field:
            if method == 'grid':
                result['cluster_field'] = grid_labels
            else:
                cluster_field = np.full(mask.shape, -1, dtype=self.int_dtype)
                indices = np.where(mask)
                cluster_field[indices] = labels
                result['cluster_field'] = cluster_field

        return result

    def _split_large_clusters_grid(self, field: np.ndarray, labels_grid: np.ndarray,
                                   size_threshold: int,
                                   min_separation_vox: int = 5,
                                   max_peaks: int = 3,
                                   max_clusters_to_process: int = 5,
                                   split_passes: int = 2,
                                   split_bridge_thickness: int = 1,
                                   split_connectivity: int = 6,
                                   min_cluster_size: int = None) -> np.ndarray:
        """
        Split overly large grid-connected clusters by detecting multiple peaks
        within each large component and assigning voxels via multi-source BFS.

        Args:
            field: 3D scalar field used to find peaks (e.g., temperature).
            labels_grid: current grid labels (-1 for background).
            size_threshold: process clusters with voxel count >= this.
            min_separation_vox: minimum peak separation in voxels.
            max_peaks: max peaks per cluster to seed (prevents over-segmentation).
            max_clusters_to_process: process only the largest N clusters.
        """
        try:
            from scipy import ndimage
        except Exception:
            print("Warning: scipy not available; skipping grid splitting")
            return labels_grid

        current = labels_grid.copy()
        uniq = np.unique(current)
        uniq = uniq[uniq >= 0]
        if uniq.size == 0:
            return current

        if min_cluster_size is None:
            # Keep any positive value; final filtering happens in properties step
            min_cluster_size = 1

        next_label = int(np.max(uniq)) + 1
        structure = ndimage.generate_binary_structure(3, 1)
        cc_structure = ndimage.generate_binary_structure(3, 1 if split_connectivity == 6 else 2)

        for pass_idx in range(max(1, int(split_passes))):
            # Recompute sizes each pass
            uniq = np.unique(current)
            uniq = uniq[uniq >= 0]
            sizes = np.zeros(int(np.max(uniq)) + 1, dtype=np.int64)
            for lab in uniq:
                sizes[lab] = np.sum(current == lab)
            large = [(int(lab), int(sizes[lab])) for lab in uniq if sizes[lab] >= int(size_threshold)]
            if not large:
                break
            large.sort(key=lambda x: x[1], reverse=True)
            large = large[:max_clusters_to_process]
            print(f"Split pass {pass_idx+1}: processing {len(large)} large clusters (>= {size_threshold} vox)")

            for lab, sz in large:
                mask = current == lab
                if not np.any(mask):
                    continue
                coords = np.array(np.where(mask))
                mins = coords.min(axis=1)
                maxs = coords.max(axis=1) + 1
                sx, sy, sz_ = [slice(int(mins[d]), int(maxs[d])) for d in range(3)]
                submask = mask[sx, sy, sz_]
                subfield = field[sx, sy, sz_]

                # Smooth and find peaks (candidate seeds)
                subfield_s = ndimage.gaussian_filter(subfield.astype(np.float32), sigma=1.0)
                foot = np.ones((max(1, int(min_separation_vox)),) * 3, dtype=bool)
                local_max = (subfield_s == ndimage.maximum_filter(subfield_s, footprint=foot)) & submask
                peak_indices = np.array(np.where(local_max)).T

                # Morphological opening to break thin bridges and get seed regions
                opened = ndimage.binary_opening(submask, structure=structure, iterations=max(1, int(split_bridge_thickness)))
                labeled_opened, n_open = ndimage.label(opened, structure=cc_structure)

                seeds = []
                if n_open > 1:
                    # Use top voxel in each opened component as a seed
                    for cid in range(1, n_open + 1):
                        comp_mask = labeled_opened == cid
                        if not np.any(comp_mask):
                            continue
                        idxs = np.array(np.where(comp_mask)).T
                        vals = subfield_s[comp_mask]
                        if idxs.shape[0] == 0:
                            continue
                        best = idxs[np.argmax(vals)]
                        seeds.append(tuple(best.tolist()))
                elif peak_indices.shape[0] >= 2:
                    # Fallback to multiple peaks as seeds
                    peak_vals = subfield_s[local_max]
                    order = np.argsort(peak_vals)[::-1]
                    peak_indices = peak_indices[order][:max_peaks]
                    for p in peak_indices:
                        seeds.append((int(p[0]), int(p[1]), int(p[2])))
                else:
                    continue  # Nothing to split

                # Multi-source BFS on the original submask
                from collections import deque
                sublabels = -np.ones(submask.shape, dtype=np.int32)
                q = deque()
                for k, (px, py, pz) in enumerate(seeds):
                    sublabels[px, py, pz] = k
                    q.append((px, py, pz))
                neighbors = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
                while q:
                    x, y, z = q.popleft()
                    lab_k = sublabels[x, y, z]
                    for dx, dy, dz in neighbors:
                        nx = x + dx; ny = y + dy; nz = z + dz
                        if nx < 0 or ny < 0 or nz < 0 or nx >= submask.shape[0] or ny >= submask.shape[1] or nz >= submask.shape[2]:
                            continue
                        if not submask[nx, ny, nz]:
                            continue
                        if sublabels[nx, ny, nz] == -1:
                            sublabels[nx, ny, nz] = lab_k
                            q.append((nx, ny, nz))

                n_new = int(np.max(sublabels) + 1)
                if n_new <= 1:
                    continue
                # Apply split to global image
                for k in range(n_new):
                    sel = (sublabels == k)
                    if not np.any(sel):
                        continue
                    if k == 0:
                        current[sx, sy, sz_][sel] = lab
                    else:
                        current[sx, sy, sz_][sel] = next_label
                        next_label += 1

        # Final small-component filtering and label compaction
        uniq = np.unique(current); uniq = uniq[uniq >= 0]
        if uniq.size == 0:
            return current
        counts = np.zeros(int(np.max(uniq)) + 1, dtype=np.int64)
        for lab in uniq:
            counts[lab] = np.sum(current == lab)
        for lab in uniq:
            if counts[lab] < int(min_cluster_size):
                current[current == lab] = -1
        # Compact labels to 0..K-1
        uniq = np.unique(current); uniq = uniq[uniq >= 0]
        remap = {lab: i for i, lab in enumerate(uniq)}
        it = np.nditer(current, flags=['multi_index', 'refs_ok'], op_flags=['readwrite'])
        while not it.finished:
            v = int(it[0])
            if v >= 0:
                it[0][...] = remap[v]
            it.iternext()

        return current

    def _cluster_3d_mask_sparse(self, mask: np.ndarray, connectivity: int = 26,
                                boundary_conditions: np.ndarray = None,
                                min_cluster_size: int = 1) -> np.ndarray:
        """
        Sparse connected-component labeling on a boolean mask with boundary conditions.
        Returns label grid (same shape) with -1 for background.
        """
        if mask.ndim != self.num_of_dims:
            raise ValueError(f"Mask must be {self.num_of_dims}D")
        dims = np.array(mask.shape, dtype=np.int64)
        if boundary_conditions is None:
            boundary_conditions = np.full(self.num_of_dims, NEUMANN, dtype=np.int32)
        else:
            boundary_conditions = np.array(boundary_conditions, dtype=np.int32)

        # Indices of active voxels
        idx = np.where(mask)
        if len(idx[0]) == 0:
            return np.full(mask.shape, -1, dtype=self.int_dtype)

        labels_grid = np.full(mask.shape, -1, dtype=self.int_dtype)

        # Fast path: 3D Numba-accelerated sparse CCL over occupied voxels
        if self.num_of_dims == 3:
            x, y, z = idx[0].astype(np.int64), idx[1].astype(np.int64), idx[2].astype(np.int64)
            lin = (x * dims[1] + y) * dims[2] + z
            # Count per occupied voxel is 1 (grid method counts voxels)
            ones = np.ones(lin.shape[0], dtype=np.int64)
            from .core_functions import ccl_occupied_voxels_3d
            voxel_labels, comp_sizes_pts, n_comp = ccl_occupied_voxels_3d(
                lin.astype(np.int64, copy=False),
                ones,
                dims.astype(np.int64, copy=False),
                boundary_conditions.astype(np.int32, copy=False),
                int(connectivity)
            )
            # Keep components with size >= min_cluster_size
            keep = np.zeros(max(n_comp, 1), dtype=np.bool_)
            for cid in range(n_comp):
                if comp_sizes_pts[cid] >= int(min_cluster_size):
                    keep[cid] = True
            # Remap kept components to 0..K-1
            comp_map = -np.ones(max(n_comp, 1), dtype=np.int64)
            next_id = 0
            for cid in range(n_comp):
                if keep[cid]:
                    comp_map[cid] = next_id
                    next_id += 1
            # Fill grid
            for i in range(lin.shape[0]):
                cid = voxel_labels[i]
                if cid >= 0:
                    rid = comp_map[cid]
                    if rid >= 0:
                        labels_grid[x[i], y[i], z[i]] = self.int_dtype(rid)
            return labels_grid

        # 2D fallback: keep existing Python implementation for now
        # Compute linear indices and adjacency as in original, but lighter
        lin = idx[0] * dims[1] + idx[1]
        voxels = set(lin.tolist())
        if connectivity == 4:
            offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        else:
            offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]

        def neighbors_from_lin_2d(vlin):
            x0 = vlin // dims[1]
            y0 = vlin % dims[1]
            for dx, dy in offsets:
                nx = x0 + dx
                ny = y0 + dy
                if boundary_conditions[0] == PERIODIC:
                    nx %= dims[0]
                elif nx < 0 or nx >= dims[0]:
                    continue
                if boundary_conditions[1] == PERIODIC:
                    ny %= dims[1]
                elif ny < 0 or ny >= dims[1]:
                    continue
                yield nx * dims[1] + ny

        comp_id = {}
        cur_label = 0
        for v in voxels:
            if v in comp_id:
                continue
            q = [v]
            comp_id[v] = cur_label
            qi = 0
            while qi < len(q):
                cur = q[qi]
                qi += 1
                for nb in neighbors_from_lin_2d(cur):
                    if (nb in voxels) and (nb not in comp_id):
                        comp_id[nb] = cur_label
                        q.append(nb)
            cur_label += 1

        counts = np.zeros(cur_label, dtype=np.int64)
        for v in voxels:
            counts[comp_id[v]] += 1
        keep = counts >= int(min_cluster_size)

        for x0, y0 in zip(idx[0], idx[1]):
            vlin = x0 * dims[1] + y0
            cid = comp_id.get(int(vlin), -1)
            labels_grid[x0, y0] = cid if (cid != -1 and keep[cid]) else -1

        # Compact labels to 0..K-1
        uniq = np.unique(labels_grid)
        uniq = uniq[uniq >= 0]
        remap = {lab: i for i, lab in enumerate(uniq)}
        if uniq.size > 0:
            it = np.nditer(labels_grid, flags=['multi_index', 'refs_ok'], op_flags=['readwrite'])
            while not it.finished:
                v = int(it[0])
                if v >= 0:
                    it[0][...] = remap[v]
                it.iternext()

        return labels_grid

    def _properties_from_grid_labels(self, labels_grid: np.ndarray, grid_spacing: float = 1.0, radius_method: str = 'effective90', boundary_conditions: np.ndarray = None) -> dict:
        """
        Compute cluster properties from grid labels and return optional point labels.
        """
        props = {
            'n_clusters': 0,
            'cluster_sizes': np.array([]),
            'cluster_centers': np.array([]).reshape(0, self.num_of_dims),
            'cluster_radii': np.array([]),
            '_point_labels': None,
            'cluster_centers_vox': np.array([]).reshape(0, self.num_of_dims),
        }
        if labels_grid is None:
            return props
        uniq = np.unique(labels_grid)
        uniq = uniq[uniq >= 0]
        k = int(uniq.size)
        if k == 0:
            return props

        sizes = np.zeros(k, dtype=self.int_dtype)
        centers_vox = np.zeros((k, self.num_of_dims), dtype=self.float_dtype)
        radii = np.zeros(k, dtype=self.float_dtype)
        dims = np.array(labels_grid.shape, dtype=self.float_dtype)
        # Handle periodic axes for distances
        if boundary_conditions is None:
            periodic_axes = np.zeros(self.num_of_dims, dtype=np.bool_)
        else:
            periodic_axes = np.array([bc == PERIODIC for bc in boundary_conditions])

        coords = np.array(np.nonzero(labels_grid >= 0)).T.astype(self.float_dtype)
        point_labels = labels_grid[labels_grid >= 0].astype(self.int_dtype)

        # First pass: sizes per component
        for lab in point_labels:
            sizes[lab] += 1

        # Compute cluster centers. For periodic axes, use circular mean so
        # clusters that wrap around a boundary get a correct center.
        for i in range(k):
            pts = coords[point_labels == i]
            if pts.size == 0:
                continue
            for d in range(self.num_of_dims):
                if periodic_axes[d]:
                    L = dims[d]
                    # Convert voxel coordinates to angles on unit circle
                    ang = (2.0 * np.pi * pts[:, d]) / L
                    s = np.sin(ang).sum(dtype=self.float_dtype)
                    c = np.cos(ang).sum(dtype=self.float_dtype)
                    theta = np.arctan2(s, c)
                    if theta < 0:
                        theta += 2.0 * np.pi
                    centers_vox[i, d] = (theta / (2.0 * np.pi)) * L
                else:
                    centers_vox[i, d] = np.mean(pts[:, d], dtype=self.float_dtype)

        # Radius per cluster
        for i in range(k):
            pts = coords[point_labels == i]
            if pts.size == 0:
                continue
            deltas = pts - centers_vox[i]
            # Apply minimal-image convention for periodic axes
            for d in range(self.num_of_dims):
                if periodic_axes[d]:
                    L = dims[d]
                    deltas[:, d] = ((deltas[:, d] + 0.5 * L) % L) - 0.5 * L
            dist2 = np.sum(deltas * deltas, axis=1)
            if radius_method == 'rms':
                radii[i] = np.sqrt(np.mean(dist2))
            elif radius_method == 'maximum':
                radii[i] = np.sqrt(np.max(dist2))
            elif radius_method == 'effective50':
                d = np.sqrt(dist2)
                if d.size > 0:
                    radii[i] = float(np.percentile(d, 50.0))
                else:
                    radii[i] = 0.0
            else:  # 'effective90' default
                d = np.sqrt(dist2)
                if d.size > 0:
                    radii[i] = float(np.percentile(d, 90.0))
                else:
                    radii[i] = 0.0

        # Scale by spacing
        if np.isscalar(grid_spacing):
            centers = centers_vox * self.float_dtype(grid_spacing)
            radii *= self.float_dtype(grid_spacing)
        else:
            spacing = np.array(grid_spacing, dtype=self.float_dtype)
            centers = centers_vox * spacing
            radii *= self.float_dtype(np.mean(spacing))

        props['n_clusters'] = k
        props['cluster_sizes'] = sizes
        props['cluster_centers'] = centers
        props['cluster_radii'] = radii
        props['cluster_centers_vox'] = centers_vox
        props['_point_labels'] = point_labels
        return props
    
    def save_clustering_results(self, filename, results, compression='gzip', compression_opts=4):
        """
        Save clustering results to HDF5 file.
        
        Args:
            filename: Output filename (with or without .h5 extension)
            results: Dictionary from cluster_3d_field or manual clustering
            compression: HDF5 compression type ('gzip', 'lzf', None)
            compression_opts: Compression level (1-9 for gzip, ignored for lzf)
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for saving. Install with: pip install h5py")
        
        if not filename.endswith('.h5'):
            filename = filename + '.h5'
        
        with h5py.File(filename, 'w') as f:
            # Save main data with compression
            if 'mask' in results:
                f.create_dataset('mask', data=results['mask'], 
                               compression=compression, compression_opts=compression_opts)
            if 'positions' in results:
                f.create_dataset('positions', data=results['positions'],
                               compression=compression, compression_opts=compression_opts)
            if 'labels' in results:
                f.create_dataset('labels', data=results['labels'],
                               compression=compression, compression_opts=compression_opts)
            if 'cluster_field' in results and results['cluster_field'] is not None:
                f.create_dataset('cluster_field', data=results['cluster_field'],
                               compression=compression, compression_opts=compression_opts)
            
            # Save cluster properties
            if 'properties' in results:
                props = results['properties']
                props_group = f.create_group('properties')
                props_group.create_dataset('cluster_sizes', data=props['cluster_sizes'])
                props_group.create_dataset('cluster_centers', data=props['cluster_centers'])
                props_group.create_dataset('cluster_radii', data=props['cluster_radii'])
                props_group.attrs['n_clusters'] = props['n_clusters']
            
            # Save any additional metadata
            meta_group = f.create_group('metadata')
            meta_group.attrs['num_of_dims'] = self.num_of_dims
            meta_group.attrs['precision'] = self.precision
            
        print(f"Clustering results saved to {filename}")
    
    def load_clustering_results(self, filename):
        """
        Load clustering results from HDF5 file.
        
        Args:
            filename: Input filename (with or without .h5 extension)
        
        Returns:
            Dictionary containing saved clustering data
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for loading. Install with: pip install h5py")
        
        if not filename.endswith('.h5'):
            filename = filename + '.h5'
        
        with h5py.File(filename, 'r') as f:
            results = {}
            
            # Load main data
            if 'mask' in f:
                results['mask'] = f['mask'][:]
            if 'positions' in f:
                results['positions'] = f['positions'][:]
            if 'labels' in f:
                results['labels'] = f['labels'][:]
            if 'cluster_field' in f:
                results['cluster_field'] = f['cluster_field'][:]
            
            # Load cluster properties
            if 'properties' in f:
                results['properties'] = {
                    'n_clusters': f['properties'].attrs['n_clusters'],
                    'cluster_sizes': f['properties/cluster_sizes'][:],
                    'cluster_centers': f['properties/cluster_centers'][:],
                    'cluster_radii': f['properties/cluster_radii'][:]
                }
            
        return results
    
    def compute_radial_profile(
        self,
        field: np.ndarray,
        center: np.ndarray,
        n_bins: int = 50,
        r_max: float = None,
        r_min: float = 0.0,
        grid_spacing: float = 1.0,
        weights: np.ndarray = None,
        method: str = 'mean',
        return_std: bool = True) -> dict:
        """
        Compute radial profile of a field around a given center point.
        
        Perfect for analyzing supernova explosion profiles in temperature,
        density, pressure, velocity, etc.
        
        Args:
            field: 2D or 3D field to compute profile from
            center: Center point coordinates (in grid units)
            n_bins: Number of radial bins
            r_max: Maximum radius (in physical units). If None, uses half the minimum box dimension
            r_min: Minimum radius (in physical units), default 0
            grid_spacing: Physical spacing between grid points (scalar or array)
            weights: Optional weight field for weighted averages
            method: 'mean', 'median', 'sum', or 'max' for bin statistics
            return_std: If True, also return standard deviation in each bin
        
        Returns:
            Dictionary containing:
            - 'radius': Bin center radii
            - 'profile': Radial profile values
            - 'std': Standard deviation in each bin (if return_std=True)
            - 'counts': Number of points in each bin
            - 'bin_edges': Edges of radial bins
        """
        # Validate input dimensions
        if field.ndim not in [2, 3]:
            raise ValueError(f"Field must be 2D or 3D, got {field.ndim}D")
        
        ndim = field.ndim
        center = np.asarray(center, dtype=self.float_dtype)
        
        if len(center) != ndim:
            raise ValueError(f"Center must have {ndim} coordinates")
        
        # Create coordinate grids
        if ndim == 2:
            ny, nx = field.shape
            y, x = np.ogrid[0:ny, 0:nx]
            if np.isscalar(grid_spacing):
                dx = dy = grid_spacing
            else:
                dx, dy = grid_spacing
            distances = np.sqrt(((x - center[0]) * dx)**2 + 
                              ((y - center[1]) * dy)**2)
        else:  # 3D
            nz, ny, nx = field.shape
            z, y, x = np.ogrid[0:nz, 0:ny, 0:nx]
            if np.isscalar(grid_spacing):
                dx = dy = dz = grid_spacing
            else:
                dx, dy, dz = grid_spacing
            distances = np.sqrt(((x - center[0]) * dx)**2 + 
                              ((y - center[1]) * dy)**2 + 
                              ((z - center[2]) * dz)**2)
        
        # Determine maximum radius if not specified
        if r_max is None:
            if ndim == 2:
                r_max = 0.5 * min(nx * dx, ny * dy)
            else:
                r_max = 0.5 * min(nx * dx, ny * dy, nz * dz)
        
        # Create radial bins
        bin_edges = np.linspace(r_min, r_max, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Flatten arrays for easier binning
        distances_flat = distances.flatten()
        field_flat = field.flatten()
        
        # Initialize output arrays
        profile = np.zeros(n_bins, dtype=self.float_dtype)
        counts = np.zeros(n_bins, dtype=self.int_dtype)
        
        if return_std:
            std_profile = np.zeros(n_bins, dtype=self.float_dtype)
        
        # Compute profile for each bin
        for i in range(n_bins):
            mask = (distances_flat >= bin_edges[i]) & (distances_flat < bin_edges[i+1])
            
            if np.any(mask):
                bin_values = field_flat[mask]
                counts[i] = len(bin_values)
                
                if weights is not None:
                    weights_flat = weights.flatten()
                    bin_weights = weights_flat[mask]
                else:
                    bin_weights = None
                
                # Compute statistics based on method
                if method == 'mean':
                    if bin_weights is not None:
                        profile[i] = np.average(bin_values, weights=bin_weights)
                    else:
                        profile[i] = np.mean(bin_values)
                elif method == 'median':
                    profile[i] = np.median(bin_values)
                elif method == 'sum':
                    profile[i] = np.sum(bin_values)
                elif method == 'max':
                    profile[i] = np.max(bin_values)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                if return_std:
                    std_profile[i] = np.std(bin_values)
        
        result = {
            'radius': bin_centers,
            'profile': profile,
            'counts': counts,
            'bin_edges': bin_edges
        }
        
        if return_std:
            result['std'] = std_profile
        
        return result
    
    def compute_cluster_radial_profiles(
        self,
        cluster_id: int,
        cluster_results: dict,
        field_dict: dict,
        n_bins: int = 50,
        r_max: float = None,
        grid_spacing: float = 1.0,
        method: str = 'mean') -> dict:
        """
        Compute radial profiles for multiple fields around a specific cluster.
        
        This is a convenience method for analyzing supernova properties,
        computing profiles for temperature, density, pressure, velocity, etc.
        all at once around a detected cluster center.
        
        Args:
            cluster_id: ID of the cluster to analyze
            cluster_results: Results dictionary from cluster_3d_field()
            field_dict: Dictionary of fields to compute profiles for
                       e.g., {'temperature': T, 'density': rho, 'pressure': P}
            n_bins: Number of radial bins
            r_max: Maximum radius. If None, uses 3x cluster radius
            grid_spacing: Physical spacing between grid points
            method: Statistical method for binning
        
        Returns:
            Dictionary with profile for each field, plus cluster info
        """
        # Validate cluster exists
        if cluster_id >= cluster_results['properties']['n_clusters']:
            raise ValueError(f"Cluster {cluster_id} does not exist")
        
        # Get cluster center
        cluster_center = cluster_results['properties']['cluster_centers'][cluster_id]
        cluster_radius = cluster_results['properties']['cluster_radii'][cluster_id]
        
        # Set default r_max based on cluster size
        if r_max is None:
            r_max = max(3 * cluster_radius, 10.0) * grid_spacing
        
        # Compute profiles for each field
        profiles = {
            'cluster_id': cluster_id,
            'cluster_center': cluster_center,
            'cluster_radius': cluster_radius,
            'cluster_size': cluster_results['properties']['cluster_sizes'][cluster_id]
        }
        
        for field_name, field_data in field_dict.items():
            print(f"Computing radial profile for {field_name}...")
            profile = self.compute_radial_profile(
                field=field_data,
                center=cluster_center,
                n_bins=n_bins,
                r_max=r_max,
                grid_spacing=grid_spacing,
                method=method,
                return_std=True
            )
            profiles[field_name] = profile
        
        # Use the radius from the first field (they're all the same)
        first_field = list(field_dict.keys())[0]
        profiles['radius'] = profiles[first_field]['radius']
        
        return profiles
    
    def extract_spherical_region(
        self,
        field: np.ndarray,
        center: np.ndarray,
        radius: float,
        grid_spacing: float = 1.0,
        fill_value: float = np.nan) -> dict:
        """
        Extract a spherical region around a point from a field.
        
        Useful for isolating individual supernova regions for detailed analysis.
        
        Args:
            field: 2D or 3D field to extract from
            center: Center point coordinates (in grid units)
            radius: Radius of sphere to extract (in physical units)
            grid_spacing: Physical spacing between grid points
            fill_value: Value to use outside the sphere
        
        Returns:
            Dictionary containing:
            - 'data': Extracted region (same shape as input)
            - 'mask': Boolean mask of points inside sphere
            - 'bbox': Bounding box of the sphere (for cropping)
        """
        ndim = field.ndim
        center = np.asarray(center, dtype=self.float_dtype)
        
        # Create distance field
        if ndim == 2:
            ny, nx = field.shape
            y, x = np.ogrid[0:ny, 0:nx]
            if np.isscalar(grid_spacing):
                dx = dy = grid_spacing
            else:
                dx, dy = grid_spacing
            distances = np.sqrt(((x - center[0]) * dx)**2 + 
                              ((y - center[1]) * dy)**2)
        else:  # 3D
            nz, ny, nx = field.shape
            z, y, x = np.ogrid[0:nz, 0:ny, 0:nx]
            if np.isscalar(grid_spacing):
                dx = dy = dz = grid_spacing
            else:
                dx, dy, dz = grid_spacing
            distances = np.sqrt(((x - center[0]) * dx)**2 + 
                              ((y - center[1]) * dy)**2 + 
                              ((z - center[2]) * dz)**2)
        
        # Create mask for sphere
        mask = distances <= radius
        
        # Extract region
        extracted = np.full_like(field, fill_value)
        extracted[mask] = field[mask]
        
        # Calculate bounding box for efficient cropping
        indices = np.where(mask)
        if len(indices[0]) > 0:
            if ndim == 2:
                bbox = (
                    (indices[0].min(), indices[0].max() + 1),
                    (indices[1].min(), indices[1].max() + 1)
                )
            else:
                bbox = (
                    (indices[0].min(), indices[0].max() + 1),
                    (indices[1].min(), indices[1].max() + 1),
                    (indices[2].min(), indices[2].max() + 1)
                )
        else:
            bbox = None
        
        return {
            'data': extracted,
            'mask': mask,
            'bbox': bbox,
            'center': center,
            'radius': radius
        }

    def save_cluster_cubes(self, filename_base: str, results: dict, fields: dict,
                           cube_size: int = 64,
                           periodic_axes: tuple = None,
                           include_derived: bool = True,
                           compression: str = 'gzip', compression_opts: int = 4) -> str:
        """
        Save per-cluster local cubes around cluster centers to an HDF5 file.

        Args:
            filename_base: Base path (without .h5) used for output file name.
            results: dict from cluster_3d_field (must include 'cluster_field' and 'properties').
            fields: mapping of name -> 3D ndarray (e.g., temperature, density, pressure, vx, vy, vz).
            cube_size: side length of cube (voxels, even enforced).
            periodic_axes: tuple of booleans length D for periodicity (default: (True, True, False) in 3D).
            include_derived: if True, compute and store u_c/u_s modes and vorticity/baroclinic cubes.
            compression: HDF5 compression algorithm.
            compression_opts: compression level for gzip.

        Returns:
            Output HDF5 filepath.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required. Install with: pip install h5py")

        if 'cluster_field' not in results or results['cluster_field'] is None:
            raise ValueError("results must include 'cluster_field' from grid clustering")
        props = results.get('properties', {})
        if 'cluster_centers_vox' not in props:
            # Recompute from labels if needed
            props = self._properties_from_grid_labels(results['cluster_field'], grid_spacing=1.0)

        grid = results['cluster_field']
        dims = grid.shape
        if periodic_axes is None:
            periodic_axes = (True, True, False) if self.num_of_dims == 3 else tuple([True]*self.num_of_dims)

        # Prepare derived variables if requested
        velocity = None
        if include_derived:
            try:
                vx = fields.get('vx'); vy = fields.get('vy'); vz = fields.get('vz')
                if vx is not None and vy is not None and vz is not None:
                    velocity = np.array([vx, vy, vz])
                    from PLASMAtools.funcs.derived_vars import DerivedVars as DV
                    dvf = DV()
                    density = fields.get('density')
                    pressure = fields.get('pressure')
                    if density is not None and pressure is not None:
                        omega, _c, _s, baro, _, _ = dvf.vorticity_decomp(
                            velocity, density_scalar_field=np.array([density]), pressure_scalar_field=np.array([pressure])
                        )
                    else:
                        omega = None; baro = None
                    u_c, u_s = dvf.helmholtz_decomposition(velocity)
                else:
                    include_derived = False
            except Exception as e:
                print(f"Warning: failed to compute derived fields: {e}")
                include_derived = False

        def extract_cube(field, cx, cy, cz, half):
            cube = np.zeros((2*half, 2*half, 2*half), dtype=field.dtype)
            nx, ny, nz = dims
            # X
            x_idx = np.arange(cx - half, cx + half)
            x_wrapped = (x_idx % nx) if periodic_axes[0] else np.clip(x_idx, 0, nx-1)
            # Y
            y_idx = np.arange(cy - half, cy + half)
            y_wrapped = (y_idx % ny) if periodic_axes[1] else np.clip(y_idx, 0, ny-1)
            # Z
            z_idx = np.arange(cz - half, cz + half)
            if self.num_of_dims == 3 and periodic_axes[2]:
                z_wrapped = z_idx % nz
                for ix, xi in enumerate(x_wrapped):
                    for iy, yi in enumerate(y_wrapped):
                        cube[ix, iy, :] = field[xi, yi, z_wrapped]
            else:
                z_start = max(0, cz - half)
                z_end = min(nz, cz + half)
                z_cube_start = max(0, half - cz)
                z_cube_end = 2*half - max(0, (cz + half) - nz)
                for ix, xi in enumerate(x_wrapped):
                    for iy, yi in enumerate(y_wrapped):
                        cube[ix, iy, z_cube_start:z_cube_end] = field[xi, yi, z_start:z_end]
            return cube

        n_clusters = int(props.get('n_clusters', 0))
        centers_vox = props.get('cluster_centers_vox', np.array([]).reshape(0, self.num_of_dims))
        out_path = filename_base if filename_base.endswith('.h5') else filename_base + '_cubes.h5'
        # Enforce even size
        cube_size = int(cube_size)
        if cube_size % 2 != 0:
            cube_size += 1
        half = cube_size // 2

        with h5py.File(out_path, 'w') as hf:
            hf.attrs['n_clusters'] = n_clusters
            hf.attrs['cube_size'] = cube_size
            hf.attrs['fields'] = ','.join(fields.keys())

            for i in range(n_clusters):
                grp = hf.create_group(f'cluster_{i:02d}')
                cx, cy, cz = centers_vox[i]
                cx = int(round(float(cx))); cy = int(round(float(cy))); cz = int(round(float(cz)))

                # Base fields
                for name, arr in fields.items():
                    if arr is None:
                        continue
                    cube = extract_cube(arr, cx, cy, cz, half)
                    grp.create_dataset(name, data=cube, compression=compression, compression_opts=compression_opts)

                # Cluster mask
                mask_cube = extract_cube((grid == i).astype(np.uint8), cx, cy, cz, half) > 0
                grp.create_dataset('cluster_mask', data=mask_cube, compression=compression, compression_opts=compression_opts)

                # Derived
                if include_derived and velocity is not None:
                    try:
                        ucx = extract_cube(u_c[0], cx, cy, cz, half); grp.create_dataset('u_cx', data=ucx, compression=compression, compression_opts=compression_opts)
                        ucy = extract_cube(u_c[1], cx, cy, cz, half); grp.create_dataset('u_cy', data=ucy, compression=compression, compression_opts=compression_opts)
                        ucz = extract_cube(u_c[2], cx, cy, cz, half); grp.create_dataset('u_cz', data=ucz, compression=compression, compression_opts=compression_opts)
                        usx = extract_cube(u_s[0], cx, cy, cz, half); grp.create_dataset('u_sx', data=usx, compression=compression, compression_opts=compression_opts)
                        usy = extract_cube(u_s[1], cx, cy, cz, half); grp.create_dataset('u_sy', data=usy, compression=compression, compression_opts=compression_opts)
                        usz = extract_cube(u_s[2], cx, cy, cz, half); grp.create_dataset('u_sz', data=usz, compression=compression, compression_opts=compression_opts)
                        if 'density' in fields and 'pressure' in fields:
                            from PLASMAtools.funcs.derived_vars import DerivedVars as DV
                            dvf = DV()
                            try:
                                _ = omega  # try reuse if defined
                            except NameError:
                                omega, _c, _s, baro, _, _ = dvf.vorticity_decomp(
                                    velocity, density_scalar_field=np.array([fields['density']]), pressure_scalar_field=np.array([fields['pressure']])
                                )
                            ox = extract_cube(omega[0], cx, cy, cz, half); grp.create_dataset('omega_x', data=ox, compression=compression, compression_opts=compression_opts)
                            oy = extract_cube(omega[1], cx, cy, cz, half); grp.create_dataset('omega_y', data=oy, compression=compression, compression_opts=compression_opts)
                            oz = extract_cube(omega[2], cx, cy, cz, half); grp.create_dataset('omega_z', data=oz, compression=compression, compression_opts=compression_opts)
                            om = np.sqrt(ox**2 + oy**2 + oz**2); grp.create_dataset('omega_mag', data=om, compression=compression, compression_opts=compression_opts)
                            bx = extract_cube(baro[0], cx, cy, cz, half); grp.create_dataset('baro_x', data=bx, compression=compression, compression_opts=compression_opts)
                            by = extract_cube(baro[1], cx, cy, cz, half); grp.create_dataset('baro_y', data=by, compression=compression, compression_opts=compression_opts)
                            bz = extract_cube(baro[2], cx, cy, cz, half); grp.create_dataset('baro_z', data=bz, compression=compression, compression_opts=compression_opts)
                            bm = np.sqrt(bx**2 + by**2 + bz**2); grp.create_dataset('baro_mag', data=bm, compression=compression, compression_opts=compression_opts)
                    except Exception as e:
                        print(f"Warning: failed to save derived cubes for cluster {i}: {e}")

        return out_path
