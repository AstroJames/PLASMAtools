"""
Cluster Morphology Analysis Module

This module provides tools for analyzing the shape, structure, and morphology
of clusters identified by the friends-of-friends algorithm. Works with both
particle data and gridded cell data.

Author: James R. Beattie
"""

import numpy as np
from numba import njit, prange
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata, RegularGridInterpolator
from typing import Dict, Tuple, Optional, Union
import warnings


class ClusterMorphology:
    """
    Analyze morphological properties of labeled clusters.
    
    This class provides comprehensive shape and structure analysis for both:
    - Particle data: positions are arbitrary coordinates
    - Cell data: positions are cell centers from a regular/AMR grid
    
    Features include:
    - Covariance matrix and principal axes
    - Effective dimensions (length, widths)
    - Shape parameters (elongation, triaxiality, sphericity)
    - Profile extraction along principal axes
    - Convexity and compactness measures
    - Density analysis
    """
    
    def __init__(self, precision: str = 'float64', data_type: str = 'auto'):
        """
        Initialize morphology analyzer.
        
        Args:
            precision: Numerical precision ('float32' or 'float64')
            data_type: 'particles', 'cells', or 'auto' (auto-detect)
        """
        if precision not in ['float32', 'float64']:
            raise ValueError("precision must be 'float32' or 'float64'")
        if data_type not in ['particles', 'cells', 'auto']:
            raise ValueError("data_type must be 'particles', 'cells', or 'auto'")
            
        self.precision = precision
        self.float_dtype = np.float32 if precision == 'float32' else np.float64
        self.int_dtype = np.int32 if precision == 'float32' else np.int64
        self.data_type = data_type
    
    def analyze_clusters(
        self, 
        positions: np.ndarray, 
        labels: np.ndarray,
        weights: Optional[np.ndarray] = None,
        min_elements: int = 10,
        grid_spacing: Optional[np.ndarray] = None
    ) -> Dict[int, Dict]:
        """
        Analyze morphology of all clusters.
        
        Args:
            positions: (N, D) array of particle positions or cell centers
            labels: (N,) array of cluster labels from FOF
            weights: (N,) array of weights (e.g., mass for particles, volume for cells)
            min_elements: Minimum cluster size to analyze
            grid_spacing: (D,) array of grid spacing for cell data
            
        Returns:
            Dictionary mapping cluster_id to morphology properties
        """
        positions = positions.astype(self.float_dtype)
        unique_labels = np.unique(labels)
        cluster_ids = unique_labels[unique_labels >= 0]
        
        # Auto-detect data type if needed
        if self.data_type == 'auto':
            self._detect_data_type(positions, grid_spacing)
        
        # Default weights
        if weights is None:
            if self.data_type == 'cells' and grid_spacing is not None:
                # For cells, weight by cell volume
                weights = np.prod(grid_spacing).astype(self.float_dtype)
            else:
                # Equal weights for particles
                weights = np.ones(len(positions), dtype=self.float_dtype)
        else:
            weights = weights.astype(self.float_dtype)
        
        results = {}
        
        for cluster_id in cluster_ids:
            mask = labels == cluster_id
            n_elements = np.sum(mask)
            
            if n_elements < min_elements:
                continue
                
            cluster_positions = positions[mask]
            cluster_weights = weights[mask] if weights.ndim > 0 else weights
            
            # Analyze this cluster
            morphology = self.analyze_single_cluster(
                cluster_positions, 
                cluster_weights,
                grid_spacing
            )
            morphology['n_elements'] = n_elements
            morphology['cluster_id'] = cluster_id
            morphology['total_weight'] = np.sum(cluster_weights)
            
            results[int(cluster_id)] = morphology
        
        return results
    
    def _detect_data_type(self, positions: np.ndarray, grid_spacing: Optional[np.ndarray]):
        """Auto-detect whether data is particles or cells."""
        if grid_spacing is not None:
            self.data_type = 'cells'
        else:
            # Check if positions lie on a regular grid
            n_dims = positions.shape[1]
            is_regular = True
            
            for dim in range(n_dims):
                unique_coords = np.unique(positions[:, dim])
                if len(unique_coords) > 1:
                    diffs = np.diff(unique_coords)
                    if not np.allclose(diffs, diffs[0], rtol=1e-5):
                        is_regular = False
                        break
            
            self.data_type = 'cells' if is_regular else 'particles'
    
    def analyze_single_cluster(
        self, 
        positions: np.ndarray,
        weights: Union[np.ndarray, float],
        grid_spacing: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Analyze morphology of a single cluster.
        
        Args:
            positions: (N, D) array of positions for one cluster
            weights: (N,) array of weights or scalar weight
            grid_spacing: (D,) array of grid spacing for cell data
            
        Returns:
            Dictionary of morphological properties
        """
        n_elements, n_dims = positions.shape
        
        # Ensure weights is array
        if np.isscalar(weights):
            weights = np.full(n_elements, weights, dtype=self.float_dtype)
        
        # Weighted center of mass
        total_weight = np.sum(weights)
        if total_weight > 0:
            center_of_mass = np.sum(positions * weights[:, np.newaxis], axis=0) / total_weight
        else:
            center_of_mass = np.mean(positions, axis=0)
        
        # Covariance matrix analysis
        cov_props = self._compute_covariance_properties(
            positions, center_of_mass, weights
        )
        
        # Shape metrics
        shape_metrics = self._compute_shape_metrics(
            positions, 
            weights,
            cov_props['eigenvalues'],
            cov_props['eigenvectors'],
            grid_spacing
        )
        
        # Combine results
        results = {
            'center_of_mass': center_of_mass,
            'n_dims': n_dims,
            'data_type': self.data_type,
            **cov_props,
            **shape_metrics
        }
        
        return results
    
    def _compute_covariance_properties(
        self, 
        positions: np.ndarray, 
        center: np.ndarray,
        weights: np.ndarray
    ) -> Dict:
        """
        Compute weighted covariance matrix and derived properties.
        
        Args:
            positions: (N, D) array of positions
            center: (D,) center of mass
            weights: (N,) array of weights
            
        Returns:
            Dictionary with covariance properties
        """
        # Center positions
        centered = positions - center
        
        # Weighted covariance matrix
        total_weight = np.sum(weights)
        if total_weight > 0:
            # Weighted covariance: C_ij = sum(w * (x_i - mu_i)(x_j - mu_j)) / sum(w)
            weighted_centered = centered * np.sqrt(weights[:, np.newaxis])
            cov_matrix = (weighted_centered.T @ weighted_centered) / total_weight
        else:
            cov_matrix = np.cov(centered.T, bias=True)
        
        # Eigenanalysis
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by decreasing eigenvalue
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Effective dimensions (2-sigma extent)
        std_devs = np.sqrt(np.maximum(eigenvalues, 0))  # Avoid negative due to numerical errors
        length = 2.0 * std_devs[0]
        
        if len(eigenvalues) >= 2:
            width1 = 2.0 * std_devs[1]
        else:
            width1 = 0.0
            
        if len(eigenvalues) >= 3:
            width2 = 2.0 * std_devs[2]
        else:
            width2 = 0.0
        
        return {
            'covariance_matrix': cov_matrix,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'std_devs': std_devs,
            'length': length,
            'width1': width1,
            'width2': width2
        }
    
    def _compute_shape_metrics(
        self, 
        positions: np.ndarray,
        weights: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        grid_spacing: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute various shape metrics.
        
        Args:
            positions: (N, D) array of positions
            weights: (N,) array of weights
            eigenvalues: Sorted eigenvalues of covariance matrix
            eigenvectors: Corresponding eigenvectors
            grid_spacing: Grid spacing for cell data
            
        Returns:
            Dictionary of shape metrics
        """
        n_dims = positions.shape[1]
        metrics = {}
        
        # Elongation
        if len(eigenvalues) >= 2 and eigenvalues[-1] > 0:
            metrics['elongation'] = np.sqrt(eigenvalues[0] / eigenvalues[-1])
        else:
            metrics['elongation'] = 1.0
        
        # Triaxiality (only for 3D)
        if n_dims == 3 and len(eigenvalues) == 3:
            denominator = eigenvalues[0] - eigenvalues[2]
            if denominator > 0:
                metrics['triaxiality'] = (eigenvalues[0] - eigenvalues[1]) / denominator
            else:
                metrics['triaxiality'] = 0.0
        
        # Planarity (for 3D)
        if n_dims == 3 and len(eigenvalues) == 3:
            denominator = eigenvalues[0] + eigenvalues[1]
            if denominator > 0:
                metrics['planarity'] = (eigenvalues[1] - eigenvalues[2]) / denominator
            else:
                metrics['planarity'] = 0.0
        
        # Anisotropy
        if len(eigenvalues) >= 2:
            total_variance = np.sum(eigenvalues)
            if total_variance > 0:
                metrics['anisotropy'] = 1.0 - (eigenvalues[-1] / eigenvalues[0])
            else:
                metrics['anisotropy'] = 0.0
        
        # Volume estimation
        if self.data_type == 'cells' and grid_spacing is not None:
            # For cells, volume is sum of cell volumes
            cell_volume = np.prod(grid_spacing)
            metrics['volume'] = np.sum(weights) * cell_volume / np.sum(weights > 0)
        else:
            # For particles, use convex hull
            if positions.shape[0] >= n_dims + 1:
                try:
                    hull = ConvexHull(positions)
                    metrics['convex_hull_volume'] = hull.volume
                    
                    # Effective density
                    if hull.volume > 0:
                        metrics['effective_density'] = np.sum(weights) / hull.volume
                    else:
                        metrics['effective_density'] = np.inf
                        
                except Exception:
                    # ConvexHull can fail for degenerate cases
                    metrics['convex_hull_volume'] = 0.0
                    metrics['effective_density'] = 0.0
        
        # Sphericity (if we have volume estimate)
        if 'convex_hull_volume' in metrics or 'volume' in metrics:
            volume = metrics.get('convex_hull_volume', metrics.get('volume', 0))
            if volume > 0 and n_dims == 3:
                # For sphere: V = 4/3 * pi * r^3, so r = (3V/4pi)^(1/3)
                # Surface area of sphere: A = 4 * pi * r^2
                r_sphere = (3 * volume / (4 * np.pi)) ** (1/3)
                a_sphere = 4 * np.pi * r_sphere**2
                
                # Approximate surface area from eigenvalues (ellipsoid approximation)
                # This is approximate but works for rough estimates
                if eigenvalues[0] > 0 and eigenvalues[1] > 0 and eigenvalues[2] > 0:
                    a = np.sqrt(eigenvalues[0])
                    b = np.sqrt(eigenvalues[1])
                    c = np.sqrt(eigenvalues[2])
                    # Approximate ellipsoid surface area (Knud Thomsen formula)
                    p = 1.6075
                    a_ellipsoid = 4 * np.pi * ((a**p * b**p + a**p * c**p + b**p * c**p) / 3)**(1/p)
                    
                    metrics['sphericity'] = a_sphere / a_ellipsoid
                else:
                    metrics['sphericity'] = 0.0
        
        return metrics
    
    def extract_profile(
        self,
        field_data: Union[np.ndarray, Dict],
        positions: np.ndarray,
        labels: np.ndarray,
        cluster_id: int,
        direction: str = 'major',
        n_samples: int = 100,
        extent_factor: float = 3.0,
        weights: Optional[np.ndarray] = None,
        grid_info: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract interpolated profile along principal axis.
        
        Args:
            field_data: For particles: (N,) or (N, D) array of field values
                       For cells: dict with 'data' array and optionally 'grid' info
            positions: (N, D) array of positions
            labels: (N,) array of cluster labels
            cluster_id: ID of cluster to analyze
            direction: 'major', 'intermediate', or 'minor' axis
            n_samples: Number of points to sample along profile
            extent_factor: How many standard deviations to extend profile
            weights: Optional weights for averaging
            grid_info: Optional dict with grid information for cell data
            
        Returns:
            (distances, profile) arrays
        """
        # Get cluster positions
        mask = labels == cluster_id
        cluster_positions = positions[mask]
        
        # Handle field data
        if isinstance(field_data, dict):
            # Cell data with grid info
            cluster_field = field_data['data'][mask]
            if 'grid' in field_data:
                grid_info = field_data['grid']
        else:
            # Simple array
            cluster_field = field_data[mask] if field_data.ndim == 1 else field_data[mask]
        
        # Get morphology
        morphology = self.analyze_single_cluster(
            cluster_positions, 
            weights[mask] if weights is not None else np.ones(len(cluster_positions))
        )
        
        # Select direction
        direction_map = {
            'major': 0,
            'intermediate': 1,
            'minor': 2
        }
        axis_idx = direction_map.get(direction, 0)
        
        if axis_idx >= len(morphology['eigenvalues']):
            raise ValueError(f"Direction '{direction}' not available for {morphology['n_dims']}D data")
        
        # Get trajectory parameters
        center = morphology['center_of_mass']
        direction_vector = morphology['eigenvectors'][:, axis_idx]
        extent = extent_factor * morphology['std_devs'][axis_idx]
        
        # Sample points along trajectory
        s = np.linspace(-extent, extent, n_samples)
        sample_points = center + s[:, np.newaxis] * direction_vector
        
        # Interpolation method depends on data type
        if self.data_type == 'cells' and grid_info is not None:
            # Use regular grid interpolation for cell data
            profile = self._interpolate_regular_grid(
                cluster_field, cluster_positions, sample_points, grid_info
            )
        else:
            # Use scattered interpolation for particles or unstructured cells
            profile = self._interpolate_scattered(
                cluster_field, cluster_positions, sample_points
            )
        
        return s, profile
    
    def _interpolate_scattered(
        self,
        field_values: np.ndarray,
        positions: np.ndarray,
        sample_points: np.ndarray
    ) -> np.ndarray:
        """Interpolate using scattered data interpolation."""
        if field_values.ndim == 1:
            # Scalar field
            profile = griddata(
                positions, 
                field_values, 
                sample_points,
                method='linear',
                fill_value=np.nan
            )
        else:
            # Vector field - interpolate each component
            profile = np.zeros((len(sample_points), field_values.shape[1]))
            for i in range(field_values.shape[1]):
                profile[:, i] = griddata(
                    positions,
                    field_values[:, i],
                    sample_points,
                    method='linear',
                    fill_value=np.nan
                )
        return profile
    
    def _interpolate_regular_grid(
        self,
        field_values: np.ndarray,
        positions: np.ndarray,
        sample_points: np.ndarray,
        grid_info: Dict
    ) -> np.ndarray:
        """Interpolate using regular grid interpolation (more efficient for cell data)."""
        # This is a placeholder - would need actual grid structure
        # For now, fall back to scattered interpolation
        return self._interpolate_scattered(field_values, positions, sample_points)
    
    def compute_density_profile(
        self,
        positions: np.ndarray,
        labels: np.ndarray,
        cluster_id: int,
        weights: Optional[np.ndarray] = None,
        n_bins: int = 50,
        density_type: str = 'radial'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute density profile of cluster.
        
        Args:
            positions: (N, D) array of positions
            labels: (N,) array of cluster labels
            cluster_id: ID of cluster to analyze
            weights: Optional weights (mass/volume)
            n_bins: Number of radial bins
            density_type: 'radial' or 'ellipsoidal'
            
        Returns:
            (distances, density) arrays
        """
        # Get cluster data
        mask = labels == cluster_id
        cluster_positions = positions[mask]
        cluster_weights = weights[mask] if weights is not None else np.ones(np.sum(mask))
        
        # Get morphology
        morphology = self.analyze_single_cluster(cluster_positions, cluster_weights)
        center = morphology['center_of_mass']
        
        if density_type == 'radial':
            # Simple radial distance
            distances = np.linalg.norm(cluster_positions - center, axis=1)
        else:
            # Ellipsoidal distance using covariance matrix
            centered = cluster_positions - center
            cov_inv = np.linalg.inv(morphology['covariance_matrix'])
            distances = np.sqrt(np.sum(centered @ cov_inv * centered, axis=1))
        
        # Compute weighted density in bins
        hist, bin_edges = np.histogram(distances, bins=n_bins, weights=cluster_weights)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        # Volume of shells
        if positions.shape[1] == 2:
            # 2D: area of annulus
            shell_volumes = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
        else:
            # 3D: volume of spherical shell
            shell_volumes = (4/3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
        
        # Density = total weight / volume
        density = hist / shell_volumes
        
        return bin_centers, density