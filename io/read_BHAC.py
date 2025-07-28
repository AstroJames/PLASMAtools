import numpy as np
import struct
import xml.etree.ElementTree as ET
import time
import base64
from typing import Union, List, Tuple, Dict
from pykdtree.kdtree import KDTree
from numba import njit, prange, types

DEFAULT_KD_TREE_LEAFSIZE = 16  # leaf size for KD trees
DEFAULT_KD_NEIGHBOURS = 4  # number of neighbours for interpolation

# Define signatures for better performance
sig_linear_interp_multi = types.float32[:,:](
    types.int32[:,:],      # neighbor_indices
    types.float32[:,:],    # neighbor_distances  
    types.float32[:,:],    # all_values
    types.int32,           # k
    types.int32            # n_vars
)

sig_linear_interp_single = types.float32[:](
    types.int32[:,:],      # neighbor_indices
    types.float32[:,:],    # neighbor_distances
    types.float32[:],      # values
    types.int32            # k
)


@njit(sig_linear_interp_multi, parallel=True, cache=True, fastmath=True)
def _linear_interpolate_multiple_vars(neighbor_indices, neighbor_distances, all_values, k, n_vars):
    """
    Interpolate multiple variables at once to maximize cache reuse.
    This is the key optimization - process all variables in a single pass.
    """
    n_points = neighbor_indices.shape[0]
    results = np.empty((n_vars, n_points), dtype=np.float32)
    
    for i in prange(n_points):
        # Calculate weights once for all variables
        if neighbor_distances[i, 0] < 1e-10:
            # Exact match
            idx = neighbor_indices[i, 0]
            for v in range(n_vars):
                results[v, i] = all_values[v, idx]
        else:
            # Calculate inverse distance weights
            inv_distances = 1.0 / (neighbor_distances[i, :k] + 1e-10)
            weight_sum = np.sum(inv_distances)
            weights = inv_distances / weight_sum
            # Apply weights to all variables
            for v in range(n_vars):
                weighted_sum = 0.0
                for j in range(k):
                    idx = neighbor_indices[i, j]
                    weighted_sum += all_values[v, idx] * weights[j]
                results[v, i] = weighted_sum
    return results


@njit(sig_linear_interp_single, parallel=True, cache=True, fastmath=True)
def _linear_interpolate_with_neighbors(neighbor_indices, neighbor_distances, values, k):
    """
    Perform linear interpolation using pre-computed nearest neighbors.
    Uses inverse distance weighting.
    """
    
    n_points = neighbor_indices.shape[0]
    result = np.empty(n_points, dtype=np.float32)
    for i in prange(n_points):
        if neighbor_distances[i, 0] < 1e-10:
            # Exact match with a source point
            result[i] = values[neighbor_indices[i, 0]]
        else:
            # Inverse distance weighting
            weights = 1.0 / (neighbor_distances[i, :k] + 1e-10)
            weight_sum = np.sum(weights)
            weighted_values = 0.0
            for j in range(k):
                weighted_values += values[neighbor_indices[i, j]] * weights[j]
            result[i] = weighted_values / weight_sum
    return result


class reformat_BHAC_field():
    """
    Performance-optimized class to reformat the BHAC field.
    Includes both scipy.interpolate.griddata and Numba-based fast interpolation.
    """
    
    def __init__(
        self,
        file_name: str,
        verbose: bool = False) -> None:
        """
        Initialize the class.
        
        Args:
            file_name: Path to the VTU file
            verbose: Whether to print progress information
        """
        self.file_name = file_name
        self.cells_per_piece = 0
        self.total_cells = 0
        self.verbose = verbose
        self.num_pieces = 0
        self.data = {}
        self.data_names = []
        
        # Cache for parsed content to avoid re-reading
        self._content = None
        self._xml_root = None
        self._data_start = None
        
        
    def __enter__(
        self) -> 'reformat_BHAC_field':
        """Context manager entry."""
        
        return self
        
        
    def __exit__(
        self, 
        exc_type : Union[type, None],
        exc_val : Union[BaseException, None],
        exc_tb) -> None:
        """Context manager exit - clean up memory."""
        
        self.cleanup()
        
        
    def cleanup(
        self) -> None:
        """Free up memory by clearing data structures."""
        
        self.data.clear()
        self._content = None
        self._xml_root = None


    def _parse_file_once(
        self) -> None:
        """Parse the file structure once and cache results."""
        
        if self._content is not None:
            return
            
        with open(self.file_name, 'rb') as f:
            self._content = f.read()

        appended_data_start = self._content.find(b'<AppendedData encoding="raw">')
        if appended_data_start == -1:
            raise ValueError("AppendedData section not found")

        self._data_start = self._content.find(b'_', appended_data_start) + 1
        xml_content = self._content[:appended_data_start].decode('utf-8', errors='ignore')
        self._xml_root = ET.fromstring(xml_content + '</VTKFile>')


    def fast_vtu_reader(
        self, 
        attr: Union[str, List, set] = 'all') -> Dict[str, np.ndarray]:
        """Optimized VTU reader based on M. Grehan's implementation."""
        
        self._parse_file_once()
        pieces = self._xml_root.findall('.//Piece')
        self.num_pieces = len(pieces)
        self.cells_per_piece = int(pieces[0].get('NumberOfCells'))
        self.total_cells = self.cells_per_piece * self.num_pieces
        
        if self.verbose:
            print(f"Number of Pieces: {self.num_pieces}")
            print(f"Cells per piece: {self.cells_per_piece}")
            print(f"Total number of cells: {self.total_cells}")

        data = {}
        # Determine which attributes to read
        if attr == 'all':
            data_array_names = set()
            for piece in pieces:
                for data_array in piece.findall('.//DataArray'):
                    name = data_array.get('Name')
                    if name and name != 'types':
                        data_array_names.add(name)
        else:
            if isinstance(attr, (list, set)):
                data_array_names = set(attr)
            else:
                data_array_names = {attr}
            # Always need connectivity and offsets for cell center calculation
            data_array_names.update(
                ['connectivity', 'offsets'])
        # Read Points first
        self._read_points_optimized(
            pieces, 
            data)
        # Read other data arrays
        for name in data_array_names:
            if name not in ['xpoint', 'ypoint', 'zpoint']:
                self._read_data_array_optimized(
                    name, 
                    pieces, 
                    data)

        self.data = data
        self.data_names = list(data.keys())
        return data


    def _read_points_optimized(
        self, 
        pieces: List[ET.Element], 
        data: Dict[str, np.ndarray]) -> None:
        """Optimized points reading with memory pre-allocation."""
        
        # Pre-calculate total size
        total_points = sum(int(piece.get('NumberOfPoints')) for piece in pieces)
        # Pre-allocate arrays
        points = np.empty(
            (total_points, 3), 
            dtype=np.float32)
        offset = 0
        for piece in pieces:
            points_data_array = piece.find('.//Points/DataArray')
            if points_data_array is None:
                raise ValueError("Points data not found")
            num_points = int(piece.get('NumberOfPoints'))
            dtype_str = points_data_array.get('type')
            # Get raw data
            raw_data = self._read_raw_data_optimized(
                points_data_array)
            # Parse data with correct dtype
            if dtype_str == 'Float32':
                parsed_data = np.frombuffer(
                    raw_data, 
                    dtype=np.float32)
            elif dtype_str == 'Float64':
                parsed_data = np.frombuffer(
                    raw_data,
                    dtype=np.float64).astype(np.float32)
            else:
                raise ValueError(f"Unsupported data type for Points: {dtype_str}")
            # Reshape and store
            points[offset:offset + num_points] = parsed_data.reshape(-1, 3)
            offset += num_points
        # Split coordinates
        data['xpoint'] = points[:, 0]
        data['ypoint'] = points[:, 1]
        data['zpoint'] = points[:, 2]
        if self.verbose:
            print(f"Extracted {total_points} points")


    def _read_data_array_optimized(
        self, 
        name: str, 
        pieces: List[ET.Element], 
        data: Dict[str, np.ndarray]) -> None:
        """Optimized data array reading with minimal allocations."""
        
        arrays_to_concat = []
        dtype = None
        for piece in pieces:
            piece_data_array = piece.find(f".//DataArray[@Name='{name}']")
            if piece_data_array is None:
                continue
            # Determine dtype once
            if dtype is None:
                dtype_str = piece_data_array.get('type')
                dtype_map = {
                    'Float32': np.float32, 
                    'Float64': np.float64,
                    'Int32': np.int32, 
                    'Int64': np.int64}
                dtype = dtype_map.get(dtype_str)
                if dtype is None:
                    raise ValueError(f"Unsupported data type: {dtype_str}")
            # Get and parse raw data
            raw_data = self._read_raw_data_optimized(
                piece_data_array)
            parsed_data = np.frombuffer(
                raw_data, 
                dtype=dtype)
            arrays_to_concat.append(
                parsed_data)
        if arrays_to_concat:
            # Efficient concatenation
            if len(arrays_to_concat) == 1:
                data[name] = arrays_to_concat[0]
            else:
                data[name] = np.concatenate(arrays_to_concat)


    def _read_raw_data_optimized(
        self, 
        data_array: ET.Element) -> bytes:
        """Optimized raw data reading."""
        
        format_type = data_array.get('format')
        if format_type == 'appended':
            offset = int(data_array.get('offset', '0'))
            size_and_data_start = self._data_start + offset
            size = struct.unpack('<I', self._content[size_and_data_start:size_and_data_start + 4])[0]
            return self._content[size_and_data_start + 4:size_and_data_start + 4 + size]
        elif format_type == 'ascii':
            text_data = data_array.text.strip().split()
            dtype_map = {
                'Float32': np.float32, 
                'Float64': np.float64,
                'Int32': np.int32, 
                'Int64': np.int64}
            dtype = dtype_map[data_array.get('type')]
            return np.array(
                text_data, 
                dtype=dtype).tobytes()
        else:  # base64
            return base64.b64decode(
                data_array.text.strip())

    def calculate_cell_centers(
        self) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized cell center calculation based on your working implementation."""
        
        if 'connectivity' not in self.data or 'offsets' not in self.data:
            raise ValueError("Missing connectivity or offsets data")
        base_conn = self.data['connectivity'][:np.max(self.data['offsets'])]
        num_iterations = int(4 * self.total_cells / np.max(self.data['offsets']))
        # Use broadcasting to create mod_conn without a loop
        offsets_array = np.arange(num_iterations, dtype=np.int32) * (np.max(base_conn) + 1)
        mod_conn = base_conn[np.newaxis, :] + offsets_array[:, np.newaxis]
        # Reshape efficiently
        cell_vertices = mod_conn.ravel()[:self.total_cells * 4].reshape(self.total_cells, 4)
        # centers
        cell_centers_x = np.mean(self.data['xpoint'][cell_vertices], axis=1)
        cell_centers_y = np.mean(self.data['ypoint'][cell_vertices], axis=1)
        return cell_centers_x, cell_centers_y

    def optimize_memory_layout(
        self) -> None:
        """Optimize data layout for better cache performance."""
        
        # Ensure all arrays are C-contiguous and properly aligned
        for key in self.data:
            if isinstance(self.data[key], np.ndarray):
                if not self.data[key].flags['C_CONTIGUOUS']:
                    self.data[key] = np.ascontiguousarray(self.data[key])
                # Convert to float32 where appropriate to reduce memory bandwidth
                if key in ['xpoint', 'ypoint', 'zpoint'] and self.data[key].dtype == np.float64:
                    self.data[key] = self.data[key].astype(np.float32, copy=False)


    def interpolate_uniform_grid(
        self,
        var_name: str,
        n_grid_x : int,
        n_grid_y : int,
        k_neighbors: int = DEFAULT_KD_NEIGHBOURS) -> np.ndarray:
        """Optimized interpolation using KD-tree and fast Numba functions."""
        
        print('===============================')
        print(f"Starting to read file: {self.file_name}")
        start_time = time.time()
        self.fast_vtu_reader({var_name})
        self.optimize_memory_layout()  # ensure memory is optimized
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished reading file: {self.file_name}")
        print(f"Time taken to read: {elapsed_time:.4f} seconds")
        print('===============================')

        print('===============================')
        print(f"Started finding cell centers")
        start_time = time.time()
        center_x, center_y = self.calculate_cell_centers()     
        # Ensure centers are float32 and contiguous (memory optimization)
        center_x = np.ascontiguousarray(
            center_x, 
            dtype=np.float32)
        center_y = np.ascontiguousarray(
            center_y, 
            dtype=np.float32)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished finding cell centers")
        print(f"Time taken to get centers: {elapsed_time:.4f} seconds")
        print('===============================')

        # Create uniform grid
        grid_x = np.linspace(
            center_x.min(),
            center_x.max(),
            n_grid_x, 
            dtype=np.float32)
        grid_y = np.linspace(
            center_y.min(), 
            center_y.max(), 
            n_grid_y, 
            dtype=np.float32)
        grid_x, grid_y = np.meshgrid(
            grid_x, 
            grid_y, 
            indexing='ij')

        print('===============================')
        print(f"Started KD-tree interpolating")
        start_time = time.time()
        # Build KD-tree
        source_points = np.column_stack(
            (center_x, center_y))
        tree = KDTree(
            source_points,
            leafsize=DEFAULT_KD_TREE_LEAFSIZE)
        # Query neighbors
        grid_shape = grid_x.shape
        grid_points = np.column_stack(
            (grid_x.ravel(), grid_y.ravel()))
        distances, indices = tree.query(
            grid_points, 
            k=k_neighbors)
        # Cast to single precision types
        values = np.ascontiguousarray(
            self.data[var_name], 
            dtype=np.float32)
        # Interpolate using Numba function
        grid = _linear_interpolate_with_neighbors(
            indices.astype(np.int32),
            distances.astype(np.float32),
            values,
            k_neighbors
        ).reshape(grid_shape)
        # Check for NaN values
        nan_count = np.sum(np.isnan(grid))
        if nan_count > 0:
            print(f"{nan_count} NaN values...")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished KD-tree interpolating")
        print(f"Time taken to interpolate: {elapsed_time:.4f} seconds")
        print('===============================')
        return grid

    def interpolate_multiple_vars(
        self,
        var_names: List[str],
        n_grid_x: int,
        n_grid_y: int,
        k_neighbors: int = DEFAULT_KD_NEIGHBOURS) -> Dict[str, np.ndarray]:
        """
        Highly optimized version that interpolates all variables in one pass.
        """
        print('===============================')
        print(f"Starting to read file for {len(var_names)} variables: {self.file_name}")
        start_time = time.time()
        self.fast_vtu_reader(var_names)
        self.optimize_memory_layout()  # ensure memory is optimized
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished reading file: {self.file_name}")
        print(f"Time taken to read: {elapsed_time:.4f} seconds")
        print('===============================')
        
        print('===============================')
        print(f"Started finding cell centers")
        start_time = time.time()
        center_x, center_y = self.calculate_cell_centers()
        # Ensure centers are float32 and contiguous
        center_x = np.ascontiguousarray(
            center_x, 
            dtype=np.float32)
        center_y = np.ascontiguousarray(
            center_y, 
            dtype=np.float32)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished finding cell centers")
        print(f"Time taken to get centers: {elapsed_time:.4f} seconds")
        print('===============================')
        
        # Create uniform grid
        grid_x = np.linspace(
            center_x.min(),
            center_x.max(),
            n_grid_x,
            dtype=np.float32)
        grid_y = np.linspace(
            center_y.min(),
            center_y.max(),
            n_grid_y,
            dtype=np.float32)
        grid_x, grid_y = np.meshgrid(
            grid_x, 
            grid_y, 
            indexing='ij')
        
        print('===============================')
        print(f"Building KD-tree for {len(var_names)} variables")
        tree_start_time = time.time()
        # Build KD-tree
        source_points = np.column_stack(
            (center_x, center_y))
        tree = KDTree(
            source_points,
            leafsize=DEFAULT_KD_TREE_LEAFSIZE)
        # Query neighbors
        grid_shape = grid_x.shape
        grid_points = np.column_stack(
            (grid_x.ravel(), grid_y.ravel()))
        distances, indices = tree.query(
            grid_points, 
            k=k_neighbors)
        # Cast to single precision types
        distances = distances.astype(np.float32)
        indices = indices.astype(np.int32)
        tree_elapsed = time.time() - tree_start_time
        print(f"KD-tree built and queried in {tree_elapsed:.4f} seconds")
        # Stack all variable values for efficient processing
        print(f"Preparing data for interpolation...")
        prep_start = time.time()
        # Create a single array with all variable values (memory optimization)
        all_values = np.empty(
            (len(var_names), len(center_x)), 
            dtype=np.float32)
        for i, var_name in enumerate(var_names):
            all_values[i] = np.ascontiguousarray(
                self.data[var_name],
                dtype=np.float32)
        prep_elapsed = time.time() - prep_start
        print(f"Data preparation completed in {prep_elapsed:.4f} seconds")
        print(f"Started interpolating {len(var_names)} variables")
        interp_start_time = time.time()
        # Interpolate all variables at once using Numba function
        interpolated_values = _linear_interpolate_multiple_vars(
            indices, 
            distances, 
            all_values, 
            k_neighbors, 
            len(var_names))
        # Reshape results
        results = {}
        for i, var_name in enumerate(var_names):
            results[var_name] = interpolated_values[i].reshape(grid_shape)
        end_time = time.time()
        total_interp_time = end_time - interp_start_time
        total_time = end_time - tree_start_time
        print(f"Finished interpolating {len(var_names)} variables")
        print(f"Time taken to interpolate: {total_interp_time:.4f} seconds")
        print(f"Total time including KD-tree: {total_time:.4f} seconds")
        print('===============================')
        return results

    def diagnose_data_structure(self):
        """Print diagnostic information about the loaded data structure."""
        print("\n=== BHAC Data Structure Diagnostic ===")
        print(f"File: {self.file_name}")
        print(f"Number of pieces: {self.num_pieces}")
        print(f"Cells per piece: {self.cells_per_piece}")
        print(f"Total cells: {self.total_cells}")
        print("\nLoaded data arrays:")
        for name, array in self.data.items():
            if isinstance(array, np.ndarray):
                print(f"  {name}: shape={array.shape}, dtype={array.dtype}")
                if len(array) > 0:
                    print(f"    min={np.min(array):.6f}, max={np.max(array):.6f}")
        print("=== End Diagnostic ===\n")