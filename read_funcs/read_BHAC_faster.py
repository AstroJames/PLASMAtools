import numpy as np
import struct
import xml.etree.ElementTree as ET
import time
import base64
from scipy.interpolate import griddata
from typing import Optional, Union, List, Tuple, Dict
import warnings

class reformat_BHAC_field():
    """
    Optimized class to reformat the BHAC field with improved performance and memory efficiency.
    
    Class functions based on Michael Grehan's implementation of a fast BHAC reader.
    Thanks Michael! 
    """
    
    def __init__(self,
                 file_name: str,
                 verbose: bool = False,
                 chunk_size: int = 1024 * 1024 * 10) -> None:
        """
        Initialize the class.
        
        Args:
            file_name: Path to the VTU file
            verbose: Whether to print progress information
            chunk_size: Size of chunks for reading large files (default: 10MB)
        """
        self.file_name = file_name
        self.cells_per_piece = 0
        self.total_cells = 0
        self.verbose = verbose
        self.num_pieces = 0
        self.data = {}
        self.data_names = []
        self.chunk_size = chunk_size
        self._xml_root = None
        self._appended_data_start = None
        self._data_start = None
        self._content = None
        
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
        
    def _parse_header(self):
        """Parse the XML header only once and cache results."""
        if self._xml_root is not None:
            return
            
        with open(self.file_name, 'rb') as f:
            # Read file in chunks to find AppendedData section
            chunk = f.read(self.chunk_size)
            appended_pos = chunk.find(b'<AppendedData encoding="raw">')
            
            if appended_pos == -1:
                # Need to read more
                content = chunk
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        raise ValueError("AppendedData section not found")
                    appended_pos = content.find(b'<AppendedData encoding="raw">')
                    if appended_pos != -1:
                        break
                    content += chunk
            else:
                content = chunk
                
            self._appended_data_start = appended_pos
            self._data_start = content.find(b'_', appended_pos) + 1
            
            # Parse XML
            xml_content = content[:appended_pos].decode('utf-8', errors='ignore')
            self._xml_root = ET.fromstring(xml_content + '</VTKFile>')
            
            # Store full content for later use (can be optimized further if needed)
            f.seek(0)
            self._content = f.read()
            
    def _get_data_info(self) -> Dict[str, Dict]:
        """Extract data array information from XML without reading the data."""
        self._parse_header()
        
        pieces = self._xml_root.findall('.//Piece')
        self.num_pieces = len(pieces)
        self.cells_per_piece = int(pieces[0].get('NumberOfCells'))
        self.total_cells = self.cells_per_piece * self.num_pieces
        
        if self.verbose:
            print(f"Number of Pieces: {self.num_pieces}")
            print(f"Cells per piece: {self.cells_per_piece}")
            print(f"Total number of cells: {self.total_cells}")
            
        # Collect metadata about data arrays
        data_info = {}
        for piece in pieces:
            for data_array in piece.findall('.//DataArray'):
                name = data_array.get('Name')
                if name and name not in data_info:
                    data_info[name] = {
                        'dtype': data_array.get('type'),
                        'format': data_array.get('format'),
                        'offsets': []
                    }
                    
        return data_info
        
    def fast_vtu_reader(self, 
                        attr: Union[str, List[str]] = 'all') -> Dict[str, np.ndarray]:
        """
        Optimized VTU reader with selective data loading.
        
        Args:
            attr: 'all' to read all attributes, or list of specific attribute names
            
        Returns:
            Dictionary of data arrays
        """
        self._parse_header()
        
        pieces = self._xml_root.findall('.//Piece')
        
        # Determine which attributes to read
        if attr == 'all':
            data_array_names = set()
            for piece in pieces:
                for data_array in piece.findall('.//DataArray'):
                    name = data_array.get('Name')
                    if name and name != 'types':
                        data_array_names.add(name)
        else:
            data_array_names = set(attr) if isinstance(attr, list) else {attr}
            # Always need connectivity and offsets for cell center calculation
            data_array_names.update(['connectivity', 'offsets'])
            
        # Always read points for cell center calculation
        self._read_points_optimized(pieces)
            
        # Read other data arrays
        for name in data_array_names:
            if name not in ['xpoint', 'ypoint', 'zpoint']:
                try:
                    self._read_data_array_optimized(name, pieces)
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Failed to read data array '{name}': {e}")
                    # Continue with other arrays
                    continue
                
        self.data_names = list(self.data.keys())
        
        # Validate that we have minimum required data
        required = ['xpoint', 'ypoint', 'connectivity', 'offsets']
        missing = [r for r in required if r not in self.data]
        if missing:
            raise ValueError(f"Missing required data arrays: {missing}")
            
        return self.data
        
    def _read_points_optimized(self, pieces: List[ET.Element]):
        """Optimized points reading with pre-allocation."""
        # Calculate total points needed
        total_points = 0
        points_info = []
        
        for piece in pieces:
            points_data_array = piece.find('.//Points/DataArray')
            if points_data_array is None:
                raise ValueError("Points data not found")
                
            num_points = int(piece.get('NumberOfPoints'))
            total_points += num_points
            
            points_info.append({
                'data_array': points_data_array,
                'num_points': num_points
            })
            
        # Pre-allocate arrays
        dtype_map = {'Float32': np.float32, 'Float64': np.float64}
        dtype = dtype_map[points_info[0]['data_array'].get('type')]
        points = np.empty((total_points, 3), dtype=dtype)
        
        # Fill arrays
        offset = 0
        for info in points_info:
            data_array = info['data_array']
            num_points = info['num_points']
            
            raw_data = self._read_raw_data(data_array)
            parsed_data = np.frombuffer(raw_data, dtype=dtype).reshape(-1, 3)
            
            points[offset:offset + num_points] = parsed_data
            offset += num_points
            
        self.data['xpoint'] = points[:, 0]
        self.data['ypoint'] = points[:, 1]
        self.data['zpoint'] = points[:, 2]
        
        if self.verbose:
            print(f"Extracted {total_points} points")
            
    def _read_data_array_optimized(self, name: str, pieces: List[ET.Element]):
        """Optimized data array reading with dynamic allocation."""
        # First pass: collect all data arrays and determine dtype
        data_arrays = []
        dtype = None
        
        for piece in pieces:
            data_array = piece.find(f".//DataArray[@Name='{name}']")
            if data_array is None:
                continue
                
            data_arrays.append(data_array)
            
            if dtype is None:
                dtype_str = data_array.get('type')
                dtype_map = {
                    'Float32': np.float32, 
                    'Float64': np.float64,
                    'Int32': np.int32, 
                    'Int64': np.int64
                }
                dtype = dtype_map.get(dtype_str)
                if dtype is None:
                    raise ValueError(f"Unsupported data type: {dtype_str}")
        
        if not data_arrays:
            return
            
        # Read all data first to determine actual sizes
        parsed_arrays = []
        total_size = 0
        
        for data_array in data_arrays:
            raw_data = self._read_raw_data(data_array)
            parsed_data = np.frombuffer(raw_data, dtype=dtype)
            parsed_arrays.append(parsed_data)
            total_size += len(parsed_data)
            
        # Now concatenate efficiently
        if len(parsed_arrays) == 1:
            self.data[name] = parsed_arrays[0]
        else:
            # Pre-allocate final array
            result = np.empty(total_size, dtype=dtype)
            offset = 0
            
            for parsed_data in parsed_arrays:
                size = len(parsed_data)
                result[offset:offset + size] = parsed_data
                offset += size
                
            self.data[name] = result
        
    def _read_raw_data(self, data_array: ET.Element) -> bytes:
        """Read raw data from a DataArray element."""
        format = data_array.get('format')
        
        if format == 'appended':
            offset = int(data_array.get('offset', '0'))
            size = struct.unpack('<I', 
                               self._content[self._data_start + offset:self._data_start + offset + 4])[0]
            return self._content[self._data_start + offset + 4:self._data_start + offset + 4 + size]
        elif format == 'ascii':
            # ASCII format needs different handling
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
        """
        Optimized calculation of cell centers using vectorized operations.
        Handles various cell types and offset formats.
        """
        if 'connectivity' not in self.data or 'offsets' not in self.data:
            raise ValueError("Missing connectivity or offsets data")
            
        connectivity = self.data['connectivity']
        offsets = self.data['offsets']
        xpoints = self.data['xpoint']
        ypoints = self.data['ypoint']
        
        if self.verbose:
            print(f"Debug: Connectivity shape: {connectivity.shape}, Offsets shape: {offsets.shape}")
            print(f"Debug: Points shape: {xpoints.shape}, Total cells: {self.total_cells}")
            
        # Check if we have the expected number of cells
        if len(offsets) != self.total_cells:
            print(f"Warning: Expected {self.total_cells} cells but found {len(offsets)} offsets")
            self.total_cells = len(offsets)
            
        # Method 1: Try using offsets directly (most general approach)
        try:
            cell_centers_x = np.zeros(self.total_cells)
            cell_centers_y = np.zeros(self.total_cells)
            
            for i in range(self.total_cells):
                if i == 0:
                    start = 0
                    end = offsets[0]
                else:
                    start = offsets[i-1]
                    end = offsets[i]
                    
                cell_verts = connectivity[start:end]
                
                # Ensure indices are within bounds
                if np.any(cell_verts >= len(xpoints)):
                    print(f"Warning: Cell {i} has out-of-bounds vertex indices")
                    continue
                    
                cell_centers_x[i] = np.mean(xpoints[cell_verts])
                cell_centers_y[i] = np.mean(ypoints[cell_verts])
                
            return cell_centers_x, cell_centers_y
            
        except Exception as e:
            if self.verbose:
                print(f"Method 1 failed: {e}")
                
        # Method 2: Try assuming regular grid structure
        try:
            # Determine vertices per cell from first offset difference
            if len(offsets) > 1:
                vertices_per_cell = offsets[0]  # First offset gives vertices in first cell
            else:
                vertices_per_cell = 4  # Default to quad
                
            if self.verbose:
                print(f"Debug: Vertices per cell: {vertices_per_cell}")
                
            # Reshape connectivity for vectorized operations
            num_cells = len(connectivity) // vertices_per_cell
            if num_cells > self.total_cells:
                num_cells = self.total_cells
                
            cell_vertices = connectivity[:num_cells * vertices_per_cell].reshape(num_cells, vertices_per_cell)
            
            # Check bounds
            max_vertex_idx = np.max(cell_vertices)
            if max_vertex_idx >= len(xpoints):
                raise ValueError(f"Vertex index {max_vertex_idx} exceeds points array size {len(xpoints)}")
                
            # Vectorized calculation
            cell_centers_x = np.mean(xpoints[cell_vertices], axis=1)
            cell_centers_y = np.mean(ypoints[cell_vertices], axis=1)
            
            return cell_centers_x[:self.total_cells], cell_centers_y[:self.total_cells]
            
        except Exception as e:
            if self.verbose:
                print(f"Method 2 failed: {e}")
            raise ValueError(f"Failed to calculate cell centers: {e}")
        
    def fill_nan_optimized(self, grid: np.ndarray) -> np.ndarray:
        """Optimized NaN filling using vectorized operations."""
        nan_mask = np.isnan(grid)
        
        # Early return if no NaNs
        if not np.any(nan_mask):
            return grid
            
        not_nan_mask = ~nan_mask
        
        # Use more efficient coordinate generation
        nan_coords = np.column_stack(np.where(nan_mask))
        valid_coords = np.column_stack(np.where(not_nan_mask))
        
        # Fill NaNs
        grid[nan_mask] = griddata(valid_coords,
                                  grid[not_nan_mask],
                                  nan_coords,
                                  method='nearest')
        return grid
        
    def interpolate_uniform_grid(self,
                                 var_name: str = "b1",
                                 n_grid_x: int = 2048,
                                 n_grid_y: int = 2048,
                                 method: str = 'linear',
                                 bounds: Optional[Tuple[float, float, float, float]] = None,
                                 debug: bool = False) -> np.ndarray:
        """
        Interpolates the cell center data onto a uniform 2D grid with optimizations.
        
        Args:
            var_name: Variable name to interpolate
            n_grid_x: Number of grid points in x direction
            n_grid_y: Number of grid points in y direction
            method: Interpolation method ('linear', 'nearest', 'cubic')
            bounds: Optional (xmin, xmax, ymin, ymax) to limit interpolation region
            debug: Enable debug output
            
        Returns:
            Interpolated grid
        """
        if debug:
            print(f"Debug: Requested variable: {var_name}")
            
        print('===============================')
        print(f"Starting to read file: {self.file_name}")
        start_time = time.time()
        
        # Read only required data
        self.fast_vtu_reader([var_name])
        
        if debug:
            print(f"Debug: Loaded data arrays: {list(self.data.keys())}")
            print(f"Debug: Variable '{var_name}' shape: {self.data.get(var_name, 'NOT FOUND')}")
            if var_name in self.data:
                print(f"Debug: Variable '{var_name}' dtype: {self.data[var_name].dtype}")
                print(f"Debug: Variable '{var_name}' min/max: {self.data[var_name].min():.6f}/{self.data[var_name].max():.6f}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished reading file: {self.file_name}")
        print(f"Time taken to read: {elapsed_time:.4f} seconds")
        print('===============================')
        
        # Check if variable exists
        if var_name not in self.data:
            available = [k for k in self.data.keys() if k not in ['xpoint', 'ypoint', 'zpoint', 'connectivity', 'offsets']]
            raise ValueError(f"Variable '{var_name}' not found in data. Available variables: {available}")
        
        print('===============================')
        print(f"Started finding cell centers")
        start_time = time.time()
        
        try:
            center_x, center_y = self.calculate_cell_centers()
        except Exception as e:
            print(f"Error calculating cell centers: {e}")
            print(f"Available data keys: {list(self.data.keys())}")
            if 'connectivity' in self.data:
                print(f"Connectivity shape: {self.data['connectivity'].shape}")
            if 'offsets' in self.data:
                print(f"Offsets shape: {self.data['offsets'].shape}")
                print(f"First few offsets: {self.data['offsets'][:10] if len(self.data['offsets']) > 0 else 'empty'}")
            raise
        
        if debug:
            print(f"Debug: Cell centers shape: {center_x.shape}")
            print(f"Debug: Cell centers x range: [{center_x.min():.6f}, {center_x.max():.6f}]")
            print(f"Debug: Cell centers y range: [{center_y.min():.6f}, {center_y.max():.6f}]")
        
        # Check if we got valid cell centers
        if len(center_x) == 0 or len(center_y) == 0:
            raise ValueError("No cell centers calculated. Check data integrity.")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished finding cell centers")
        print(f"Time taken to get centers: {elapsed_time:.4f} seconds")
        print('===============================')
        
        # Apply bounds if specified
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            mask = (center_x >= xmin) & (center_x <= xmax) & \
                   (center_y >= ymin) & (center_y <= ymax)
            center_x = center_x[mask]
            center_y = center_y[mask]
            var_data = self.data[var_name][mask]
        else:
            var_data = self.data[var_name]
            # Check for empty arrays before calling min/max
            if len(center_x) == 0 or len(center_y) == 0:
                raise ValueError("Cell centers are empty. Cannot proceed with interpolation.")
            xmin, xmax = center_x.min(), center_x.max()
            ymin, ymax = center_y.min(), center_y.max()
            
        # Check data consistency
        if len(var_data) != len(center_x):
            raise ValueError(f"Data size mismatch: variable has {len(var_data)} values but there are {len(center_x)} cell centers")
            
        # Create uniform grid
        grid_x = np.linspace(xmin, xmax, n_grid_x)
        grid_y = np.linspace(ymin, ymax, n_grid_y)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y, indexing='xy')
        
        print('===============================')
        print(f"Started interpolating")
        start_time = time.time()
        
        # Use rescale option for better performance
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Qhull')
            grid = griddata((center_x, center_y), 
                           var_data, 
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
        
    def diagnose_data_structure(self):
        """Print diagnostic information about the loaded data structure."""
        print("\n=== BHAC Data Structure Diagnostic ===")
        print(f"File: {self.file_name}")
        print(f"Number of pieces: {self.num_pieces}")
        print(f"Cells per piece: {self.cells_per_piece}")
        print(f"Total cells: {self.total_cells}")
        
        print("\nLoaded data arrays:")
        for name, array in self.data.items():
            print(f"  {name}: shape={array.shape}, dtype={array.dtype}")
            if len(array) > 0:
                print(f"    min={np.min(array):.6f}, max={np.max(array):.6f}")
                
        if 'connectivity' in self.data and 'offsets' in self.data:
            print("\nConnectivity analysis:")
            conn = self.data['connectivity']
            offs = self.data['offsets']
            print(f"  Total connectivity entries: {len(conn)}")
            print(f"  Total offset entries: {len(offs)}")
            if len(offs) > 0:
                print(f"  First offset: {offs[0]}")
                if len(offs) > 1:
                    print(f"  Second offset: {offs[1]}")
                    print(f"  Implied vertices per cell: {offs[0]}")
                print(f"  Max connectivity index: {np.max(conn)}")
                
        if 'xpoint' in self.data:
            print(f"\nTotal points: {len(self.data['xpoint'])}")
            
        print("=== End Diagnostic ===\n")
        
    def interpolate_multiple_vars(self,
                                  var_names: List[str],
                                  n_grid_x: int = 2048,
                                  n_grid_y: int = 2048,
                                  method: str = 'linear') -> Dict[str, np.ndarray]:
        """
        Efficiently interpolate multiple variables by reusing cell centers.
        
        Args:
            var_names: List of variable names to interpolate
            n_grid_x: Number of grid points in x direction
            n_grid_y: Number of grid points in y direction
            method: Interpolation method
            
        Returns:
            Dictionary of interpolated grids
        """
        # Read all required data at once
        self.fast_vtu_reader(var_names)
        
        # Calculate cell centers once
        center_x, center_y = self.calculate_cell_centers()
        
        # Create uniform grid once
        grid_x = np.linspace(center_x.min(), center_x.max(), n_grid_x)
        grid_y = np.linspace(center_y.min(), center_y.max(), n_grid_y)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y, indexing='xy')
        
        # Interpolate each variable
        results = {}
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
                
        return results