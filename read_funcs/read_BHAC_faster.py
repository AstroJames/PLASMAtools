import numpy as np
import struct
import xml.etree.ElementTree as ET
import time
import base64
from scipy.interpolate import griddata
from typing import Optional, Union, List, Tuple, Dict
import warnings

# Import numba for fast interpolation
try:
    import numba as nb
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    print("Warning: Numba not found.")
    HAS_NUMBA = False

# Numba-based fast interpolation functions
@njit(nopython=True, parallel=True, cache=True)
def _linear_interpolate_2d(points_x, points_y, values, grid_x, grid_y):
    """Fast 2D linear interpolation using inverse distance weighting."""
    ny, nx = grid_x.shape
    result = np.full((ny, nx), np.nan, dtype=np.float32)
    
    for i in prange(ny):
        for j in prange(nx):
            target_x = grid_x[i, j]
            target_y = grid_y[i, j]
            
            # Find closest points and do inverse distance weighting
            total_weight = 0.0
            weighted_sum = 0.0
            min_dist_sq = 1e10
            closest_value = 0.0
            
            # Search all points - could be optimized with spatial indexing for very large datasets
            for k in range(len(points_x)):
                dx = points_x[k] - target_x
                dy = points_y[k] - target_y
                dist_sq = dx*dx + dy*dy
                
                # Track closest point for fallback
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_value = values[k]
                
                # Only use nearby points for interpolation
                if dist_sq < min_dist_sq * 16:  # Within 4x closest distance
                    if dist_sq > 1e-12:  # Avoid division by zero
                        weight = 1.0 / (dist_sq + 1e-8)
                        weighted_sum += values[k] * weight
                        total_weight += weight
                    else:
                        # Point is essentially at grid location
                        result[i, j] = values[k]
                        total_weight = -1
                        break
            
            if total_weight > 0:
                result[i, j] = weighted_sum / total_weight
            elif total_weight == 0:
                result[i, j] = closest_value
            # If total_weight < 0, we already set the result
    
    return result

@njit(nopython=True, parallel=True, cache=True)
def _nearest_interpolate_2d(points_x, points_y, values, grid_x, grid_y):
    """Ultra-fast nearest neighbor interpolation."""
    ny, nx = grid_x.shape
    result = np.empty((ny, nx), dtype=np.float32)
    
    for i in prange(ny):
        for j in prange(nx):
            target_x = grid_x[i, j]
            target_y = grid_y[i, j]
            
            min_dist_sq = 1e10
            closest_value = 0.0
            
            for k in range(len(points_x)):
                dx = points_x[k] - target_x
                dy = points_y[k] - target_y
                dist_sq = dx*dx + dy*dy
                
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_value = values[k]
            
            result[i, j] = closest_value
    
    return result

class reformat_BHAC_field():
    """
    Performance-optimized class to reformat the BHAC field.
    Includes both scipy.interpolate.griddata and Numba-based fast interpolation.
    """
    
    def __init__(self,
                 file_name: str,
                 verbose: bool = False,
                 chunk_size: int = 1024 * 1024 * 50) -> None:
        """
        Initialize the class.
        
        Args:
            file_name: Path to the VTU file
            verbose: Whether to print progress information
            chunk_size: Size of chunks for reading large files (default: 50MB)
        """
        self.file_name = file_name
        self.cells_per_piece = 0
        self.total_cells = 0
        self.verbose = verbose
        self.num_pieces = 0
        self.data = {}
        self.data_names = []
        self.chunk_size = chunk_size
        
        # Cache for parsed content to avoid re-reading
        self._content = None
        self._xml_root = None
        self._data_start = None
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up memory."""
        self.cleanup()
        
    def cleanup(self):
        """Free up memory by clearing data structures."""
        self.data.clear()
        self._content = None
        self._xml_root = None

    def _parse_file_once(self):
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

    def fast_vtu_reader(self, attr: Union[str, List, set] = 'all') -> Dict[str, np.ndarray]:
        """Optimized VTU reader based on your working implementation."""
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
            data_array_names.update(['connectivity', 'offsets'])

        # Read Points first (optimized)
        self._read_points_optimized(pieces, data)

        # Read other data arrays (optimized)
        for name in data_array_names:
            if name not in ['xpoint', 'ypoint', 'zpoint']:
                self._read_data_array_optimized(name, pieces, data)

        self.data = data
        self.data_names = list(data.keys())
        return data

    def _read_points_optimized(self, pieces: List[ET.Element], data: Dict[str, np.ndarray]):
        """Optimized points reading with memory pre-allocation."""
        # Pre-calculate total size
        total_points = sum(int(piece.get('NumberOfPoints')) for piece in pieces)
        
        # Pre-allocate arrays
        points = np.empty((total_points, 3), dtype=np.float32)
        
        offset = 0
        for piece in pieces:
            points_data_array = piece.find('.//Points/DataArray')
            if points_data_array is None:
                raise ValueError("Points data not found")

            num_points = int(piece.get('NumberOfPoints'))
            dtype_str = points_data_array.get('type')
            
            # Get raw data
            raw_data = self._read_raw_data_optimized(points_data_array)
            
            # Parse data with correct dtype
            if dtype_str == 'Float32':
                parsed_data = np.frombuffer(raw_data, dtype=np.float32)
            elif dtype_str == 'Float64':
                parsed_data = np.frombuffer(raw_data, dtype=np.float64).astype(np.float32)
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

    def _read_data_array_optimized(self, name: str, pieces: List[ET.Element], data: Dict[str, np.ndarray]):
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
                    'Int64': np.int64
                }
                dtype = dtype_map.get(dtype_str)
                if dtype is None:
                    raise ValueError(f"Unsupported data type: {dtype_str}")

            # Get and parse raw data
            raw_data = self._read_raw_data_optimized(piece_data_array)
            parsed_data = np.frombuffer(raw_data, dtype=dtype)
            arrays_to_concat.append(parsed_data)

        if arrays_to_concat:
            # Efficient concatenation
            if len(arrays_to_concat) == 1:
                data[name] = arrays_to_concat[0]
            else:
                data[name] = np.concatenate(arrays_to_concat)

    def _read_raw_data_optimized(self, data_array: ET.Element) -> bytes:
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
                'Int64': np.int64
            }
            dtype = dtype_map[data_array.get('type')]
            return np.array(text_data, dtype=dtype).tobytes()
        else:  # base64
            return base64.b64decode(data_array.text.strip())

    def calculate_cell_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized cell center calculation based on your working implementation."""
        if 'connectivity' not in self.data or 'offsets' not in self.data:
            raise ValueError("Missing connectivity or offsets data")

        # Your original approach - much simpler and faster!
        base_conn = self.data['connectivity'][:np.max(self.data['offsets'])]
        num_iterations = int(4 * self.total_cells / np.max(self.data['offsets']))

        # Optimized broadcasting approach
        offsets_array = np.arange(num_iterations, dtype=np.int32) * (np.max(base_conn) + 1)
        mod_conn = base_conn[np.newaxis, :] + offsets_array[:, np.newaxis]
        
        # Reshape efficiently
        cell_vertices = mod_conn.ravel()[:self.total_cells * 4].reshape(self.total_cells, 4)

        # Vectorized calculation
        cell_centers_x = np.mean(self.data['xpoint'][cell_vertices], axis=1)
        cell_centers_y = np.mean(self.data['ypoint'][cell_vertices], axis=1)
        
        return cell_centers_x, cell_centers_y

    def fill_nan_optimized(self, grid: np.ndarray) -> np.ndarray:
        """Optimized NaN filling using scipy.interpolate.griddata."""
        nan_mask = np.isnan(grid)
        
        # Early return if no NaNs
        if not np.any(nan_mask):
            return grid
            
        not_nan_mask = ~nan_mask
        
        # More efficient coordinate generation
        nan_coords = np.column_stack(np.where(nan_mask))
        valid_coords = np.column_stack(np.where(not_nan_mask))
        
        # Use griddata for NaN filling
        grid[nan_mask] = griddata(valid_coords,
                                  grid[not_nan_mask],
                                  nan_coords,
                                  method='nearest')
        return grid

    # ORIGINAL METHODS (scipy-based)
    def interpolate_uniform_grid(self,
                                 var_name: str = "b1",
                                 n_grid_x: int = 2048,
                                 n_grid_y: int = 2048,
                                 method: str = 'linear') -> np.ndarray:
        """Original interpolation method using scipy.interpolate.griddata."""
        print('===============================')
        print(f"Starting to read file: {self.file_name}")
        start_time = time.time()
        
        self.fast_vtu_reader({var_name})

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished reading file: {self.file_name}")
        print(f"Time taken to read: {elapsed_time:.4f} seconds")
        print('===============================')

        print('===============================')
        print(f"Started finding cell centers")
        start_time = time.time()

        center_x, center_y = self.calculate_cell_centers()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished finding cell centers")
        print(f"Time taken to get centers: {elapsed_time:.4f} seconds")
        print('===============================')

        # Create uniform grid
        grid_x = np.linspace(center_x.min(), center_x.max(), n_grid_x)
        grid_y = np.linspace(center_y.min(), center_y.max(), n_grid_y)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y, indexing='xy')

        print('===============================')
        print(f"Started interpolating (scipy griddata)")
        start_time = time.time()

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Qhull')
            grid = griddata((center_x, center_y), 
                           self.data[var_name], 
                           (grid_x, grid_y), 
                           method=method,
                           rescale=True)

        # Fill NaNs if needed
        if np.any(np.isnan(grid)):
            grid = self.fill_nan_optimized(grid)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished interpolating")
        print(f"Time taken to interpolate: {elapsed_time:.4f} seconds")
        print('===============================')

        return grid

    def interpolate_multiple_vars(self,
                                  var_names: List[str],
                                  n_grid_x: int = 2048,
                                  n_grid_y: int = 2048,
                                  method: str = 'linear') -> Dict[str, np.ndarray]:
        """Multiple variable interpolation using scipy.interpolate.griddata."""
        print('===============================')
        print(f"Starting to read file for {len(var_names)} variables: {self.file_name}")
        start_time = time.time()
        
        self.fast_vtu_reader(var_names)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished reading file: {self.file_name}")
        print(f"Time taken to read: {elapsed_time:.4f} seconds")
        print('===============================')

        print('===============================')
        print(f"Started finding cell centers")
        start_time = time.time()

        center_x, center_y = self.calculate_cell_centers()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished finding cell centers")
        print(f"Time taken to get centers: {elapsed_time:.4f} seconds")
        print('===============================')

        # Create uniform grid once
        grid_x = np.linspace(center_x.min(), center_x.max(), n_grid_x)
        grid_y = np.linspace(center_y.min(), center_y.max(), n_grid_y)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y, indexing='xy')

        print('===============================')
        print(f"Started interpolating {len(var_names)} variables (scipy griddata)")
        start_time = time.time()

        results = {}
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Qhull')
            
            for var_name in var_names:
                if var_name in self.data:
                    grid = griddata((center_x, center_y), 
                                   self.data[var_name], 
                                   (grid_x, grid_y), 
                                   method=method,
                                   rescale=True)
                    
                    if np.any(np.isnan(grid)):
                        grid = self.fill_nan_optimized(grid)
                        
                    results[var_name] = grid

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished interpolating {len(var_names)} variables")
        print(f"Time taken to interpolate: {elapsed_time:.4f} seconds")
        print('===============================')

        return results

    # NUMBA-BASED FAST METHODS
    def interpolate_uniform_grid_numba(self,
                                       var_name: str = "b1",
                                       n_grid_x: int = 2048,
                                       n_grid_y: int = 2048,
                                       method: str = 'linear') -> np.ndarray:
        """Ultra-fast interpolation using Numba-compiled functions."""
        if not HAS_NUMBA:
            print("Warning: Numba not available, falling back to scipy method")
            return self.interpolate_uniform_grid(var_name, n_grid_x, n_grid_y, method)
        
        print('===============================')
        print(f"Starting to read file: {self.file_name}")
        start_time = time.time()
        
        self.fast_vtu_reader({var_name})

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished reading file: {self.file_name}")
        print(f"Time taken to read: {elapsed_time:.4f} seconds")
        print('===============================')

        print('===============================')
        print(f"Started finding cell centers")
        start_time = time.time()

        center_x, center_y = self.calculate_cell_centers()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished finding cell centers")
        print(f"Time taken to get centers: {elapsed_time:.4f} seconds")
        print('===============================')

        # Create uniform grid
        grid_x = np.linspace(center_x.min(), center_x.max(), n_grid_x)
        grid_y = np.linspace(center_y.min(), center_y.max(), n_grid_y)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y, indexing='xy')

        print('===============================')
        print(f"Started Numba interpolating (method: {method})")
        start_time = time.time()

        # Convert to float32 for speed
        center_x = center_x.astype(np.float32)
        center_y = center_y.astype(np.float32)
        values = self.data[var_name].astype(np.float32)
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)

        # Use appropriate Numba function
        if method == 'nearest':
            grid = _fast_nearest_interpolate_2d(center_x, center_y, values, grid_x, grid_y)
        else:  # linear or other
            grid = fast_linear_interpolate_optimized(center_x, center_y, values, grid_x, grid_y)

        # Handle any remaining NaNs
        nan_count = np.sum(np.isnan(grid))
        if nan_count > 0:
            print(f"Filling {nan_count} NaN values with nearest neighbor...")
            if method != 'nearest':
                grid_nearest = _fast_nearest_interpolate_2d(center_x, center_y, values, grid_x, grid_y)
                nan_mask = np.isnan(grid)
                grid[nan_mask] = grid_nearest[nan_mask]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished Numba interpolating")
        print(f"Time taken to interpolate: {elapsed_time:.4f} seconds")
        print('===============================')

        return grid

    def interpolate_multiple_vars_numba(self,
                                        var_names: List[str],
                                        n_grid_x: int = 2048,
                                        n_grid_y: int = 2048,
                                        method: str = 'linear') -> Dict[str, np.ndarray]:
        """Ultra-fast multiple variable interpolation using Numba."""
        if not HAS_NUMBA:
            print("Warning: Numba not available, falling back to scipy method")
            return self.interpolate_multiple_vars(var_names, n_grid_x, n_grid_y, method)
        
        print('===============================')
        print(f"Starting to read file for {len(var_names)} variables: {self.file_name}")
        start_time = time.time()
        
        self.fast_vtu_reader(var_names)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished reading file: {self.file_name}")
        print(f"Time taken to read: {elapsed_time:.4f} seconds")
        print('===============================')

        print('===============================')
        print(f"Started finding cell centers")
        start_time = time.time()

        center_x, center_y = self.calculate_cell_centers()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished finding cell centers")
        print(f"Time taken to get centers: {elapsed_time:.4f} seconds")
        print('===============================')

        # Create uniform grid once
        grid_x = np.linspace(center_x.min(), center_x.max(), n_grid_x)
        grid_y = np.linspace(center_y.min(), center_y.max(), n_grid_y)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y, indexing='xy')

        # Convert to float32 for speed
        center_x = center_x.astype(np.float32)
        center_y = center_y.astype(np.float32)
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)

        print('===============================')
        print(f"Started Numba interpolating {len(var_names)} variables")
        start_time = time.time()

        results = {}
        for i, var_name in enumerate(var_names):
            if var_name in self.data:
                print(f"  Interpolating {var_name} ({i+1}/{len(var_names)})...")
                var_start_time = time.time()
                
                values = self.data[var_name].astype(np.float32)
                
                if method == 'nearest':
                    grid = _nearest_interpolate_2d(center_x, center_y, values, grid_x, grid_y)
                else:
                    grid = _linear_interpolate_2d(center_x, center_y, values, grid_x, grid_y)
                
                # Handle NaNs
                nan_count = np.sum(np.isnan(grid))
                if nan_count > 0 and method != 'nearest':
                    grid_nearest = _nearest_interpolate_2d(center_x, center_y, values, grid_x, grid_y)
                    nan_mask = np.isnan(grid)
                    grid[nan_mask] = grid_nearest[nan_mask]
                
                results[var_name] = grid
                
                var_elapsed = time.time() - var_start_time
                print(f"    {var_name} completed in {var_elapsed:.4f} seconds")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished Numba interpolating {len(var_names)} variables")
        print(f"Time taken to interpolate: {elapsed_time:.4f} seconds")
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