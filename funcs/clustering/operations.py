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

# All optimized functions are now in core_functions.py
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
        
        # Set default box size if not provided
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
        radius_method: str = 'effective90') -> dict:
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
        
        for i, cluster_id in enumerate(cluster_ids):
            mask = cluster_labels == cluster_id
            cluster_positions = positions[mask]
            
            cluster_sizes[i] = np.sum(mask)
            cluster_centers[i] = np.mean(cluster_positions, axis=0)
            
            # Calculate radius using specified method
            if cluster_sizes[i] > 1:
                distances = np.linalg.norm(cluster_positions - cluster_centers[i], axis=1)
                
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
        remove_small_objects_size: int = None) -> np.ndarray:
        """
        Apply morphological filtering to remove thin shell-like features.
        
        This is particularly useful for removing shock fronts and other thin
        structures that shouldn't be identified as separate clusters in 
        supernova analysis.
        
        Args:
            mask: Binary mask (2D or 3D)
            min_thickness: Minimum thickness in pixels for valid regions.
                          Structures thinner than this will be removed.
            remove_small_objects_size: Minimum size for connected components.
                                      If None, uses min_thickness^ndim * 8
        
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
        
        # Step 1: Opening operation (erosion followed by dilation)
        # This removes thin protrusions and breaks thin connections
        if min_thickness > 0:
            mask_opened = ndimage.binary_opening(
                mask, 
                structure=structure, 
                iterations=min_thickness
            )
        else:
            mask_opened = mask.copy()
        
        # Step 2: Fill small holes that might have been created
        # This helps maintain the bulk structure of legitimate clusters
        mask_filled = ndimage.binary_fill_holes(mask_opened)
        
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
        
        return mask_filled
    
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
        separate_overlapping: bool = False,
        min_separation: float = 5.0,
        max_peaks: int = 3,
        max_clusters_to_process: int = 5,
        return_field: bool = True) -> dict:
        """
        Cluster a 3D field directly by thresholding and friends-of-friends.
        
        This is a convenience method that combines thresholding, optional morphological
        filtering, position extraction, clustering, and optional mapping back to the grid.
        
        Args:
            field: 3D array to cluster
            threshold: Threshold value for creating binary mask. If None, uses median
            threshold_type: 'greater' or 'less' - how to apply threshold
            grid_spacing: Physical spacing between grid points
            linking_length: Maximum distance for particles to be friends
            min_cluster_size: Minimum number of particles for valid cluster
            morphological_filter: If True, apply morphological filtering to remove
                                thin shell-like artifacts (useful for supernova analysis)
            min_thickness: Minimum thickness in pixels for morphological filtering
            separate_overlapping: If True, attempt to separate overlapping clusters
                                using temperature peak detection (useful for nearby SNe)
            min_separation: Minimum distance between peaks for cluster separation
            max_peaks: Maximum number of peaks per cluster (prevents over-segmentation)
            max_clusters_to_process: Only process the N largest clusters (for performance)
            return_field: If True, return cluster labels mapped to 3D grid
        
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
                remove_small_objects_size=None  # Use default sizing
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
        
        # Convert mask to positions
        positions = self.binary_mask_to_positions(mask, grid_spacing)
        
        # Perform clustering
        labels = self.friends_of_friends(
            positions=positions,
            linking_length=linking_length,
            min_cluster_size=min_cluster_size
        )
        
        # Apply cluster separation if requested
        if separate_overlapping:
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
        
        # Get cluster properties
        properties = self.get_cluster_properties(positions, labels)
        
        result = {
            'labels': labels,
            'mask': mask,
            'positions': positions,
            'properties': properties
        }
        
        # Include original mask if morphological filtering was applied
        if mask_original is not None:
            result['mask_original'] = mask_original
        
        # Optionally map back to 3D grid
        if return_field:
            cluster_field = np.full(mask.shape, -1, dtype=self.int_dtype)
            indices = np.where(mask)
            cluster_field[indices] = labels
            result['cluster_field'] = cluster_field
        
        return result
    
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