import numpy as np
import struct
import xml.etree.ElementTree as ET
import time
import base64
from scipy.interpolate import griddata

class reformat_BHAC_field():
    """
    Class to reformat the BHAC field to the correct format for the BHAC class.
    """
    
    def __init__(self,
                 file_name  : str,
                 verbose    : bool = False) -> None:
        """
        Initialize the class.
        """
        
        self.file_name = file_name
        self.cells_per_piece = 0
        self.total_cells = 0
        self.verbose = verbose
        self.num_pieces = 0
        self.data = []
        self.data_names = []
        
        
    def fast_vtu_reader(self, 
                        attr : str ='all') -> np.ndarray:
        
        with open(self.file_name, 'rb') as f:
            content = f.read()

        appended_data_start = content.find(b'<AppendedData encoding="raw">')
        if appended_data_start == -1:
            raise ValueError("AppendedData section not found")

        data_start = content.find(b'_', appended_data_start) + 1
        xml_content = content[:appended_data_start].decode('utf-8', errors='ignore')
        root = ET.fromstring(xml_content + '</VTKFile>')

        pieces = root.findall('.//Piece')
        
        self.num_pieces = len(pieces)
        self.cells_per_piece = int(pieces[0].get('NumberOfCells'))
        self.total_cells = self.cells_per_piece * self.num_pieces
        
        if self.verbose:
            print(f"Number of Pieces: {self.num_pieces}")
            print(f"Cells per piece: {self.cells_per_piece}")
            print(f"Total number of cells: {self.total_cells}")

        data = {}
        # Get all unique DataArray names
        data_array_names = set()
        for piece in pieces:
            for data_array in piece.findall('.//DataArray'):
                data_array_names.add(data_array.get('Name'))

        # Read Points (x, y, z coordinates)
        points_data = []
        for piece in pieces:
            points_data_array = piece.find('.//Points/DataArray')
            if points_data_array is None:
                raise ValueError("Points data not found")

            dtype = points_data_array.get('type')
            format = points_data_array.get('format')

            if format == 'appended':
                offset = int(points_data_array.get('offset', '0'))
                size = struct.unpack('<I', content[data_start+offset:data_start+offset+4])[0]
                raw_data = content[data_start+offset+4:data_start+offset+4+size]
            elif format == 'ascii':
                raw_data = points_data_array.text.strip().split()
            else:  # Assume inline base64
                raw_data = base64.b64decode(points_data_array.text.strip())
            if dtype == 'Float32':
                parsed_data = np.frombuffer(raw_data, dtype=np.float32) if format != 'ascii' else np.array(raw_data, dtype=np.float32)
            elif dtype == 'Float64':
                parsed_data = np.frombuffer(raw_data, dtype=np.float64) if format != 'ascii' else np.array(raw_data, dtype=np.float64)
            else:
                raise ValueError(f"Unsupported data type for Points: {dtype}")
            points_data.append(parsed_data)

        if points_data:
            points = np.concatenate(points_data).reshape(-1, 3)  # Assuming 3D points (x, y, z)
            data['xpoint'], data['ypoint'], data['zpoint'] = points[:, 0], points[:, 1], points[:, 2]
            if self.verbose:
                print(f"Extracted {len(data['xpoint'])} points")

        # Handle attributes
        if attr == 'all':
            data_array_names.discard(None)
            data_array_names.discard('types')
        else:
            data_array_names = attr
            data_array_names.add('connectivity')
            data_array_names.add('offsets')

        # Read data arrays
        for name in data_array_names:
            combined_data = []
            # Read data array for each piece
            for piece in pieces:
                piece_data_array = piece.find(f".//DataArray[@Name='{name}']")
                if piece_data_array is None:
                    continue
                dtype = piece_data_array.get('type')
                format = piece_data_array.get('format')
                if format == 'appended':
                    offset = int(piece_data_array.get('offset', '0'))
                    size = struct.unpack('<I', content[data_start+offset:data_start+offset+4])[0]
                    raw_data = content[data_start+offset+4:data_start+offset+4+size]
                elif format == 'ascii':
                    raw_data = piece_data_array.text.strip().split()
                else:
                    raw_data = base64.b64decode(piece_data_array.text.strip())
                if dtype == 'Float32':
                    parsed_data = np.frombuffer(raw_data, dtype=np.float32) if format != 'ascii' else np.array(raw_data, dtype=np.float32)
                elif dtype == 'Float64':
                    parsed_data = np.frombuffer(raw_data, dtype=np.float64) if format != 'ascii' else np.array(raw_data, dtype=np.float64)
                elif dtype == 'Int32':
                    parsed_data = np.frombuffer(raw_data, dtype=np.int32) if format != 'ascii' else np.array(raw_data, dtype=np.int32)
                elif dtype == 'Int64':
                    parsed_data = np.frombuffer(raw_data, dtype=np.int64) if format != 'ascii' else np.array(raw_data, dtype=np.int64)
                else:
                    raise ValueError(f"Unsupported data type: {dtype}")
                combined_data.append(parsed_data)
            if combined_data:
                data[name] = np.concatenate(combined_data)



        self.data_names = list(data.keys())
        self.data = data
    
    def calculate_cell_centers(self):
        """
        Interpolates the corner verticies to cell centers.
        """

        # Create mod_conn array using broadcasting instead of a for loop
        base_conn       = self.data['connectivity'][:np.max(self.data['offsets'])]  # Base mod_conn for the first set
        num_iterations  = int(4 * self.total_cells / np.max(self.data['offsets']))  # Number of iterations

        # Use broadcasting to create mod_conn without a loop
        offsets_array   = np.arange(num_iterations) * (np.max(base_conn) + 1)  # Calculate all offsets at once
        mod_conn        = base_conn + offsets_array[:, None]  # Broadcast and add offsets
        
        # Flatten mod_conn to a 1D array
        # Reshape mod_conn to group cell vertices (ncells x 4)
        cell_vertices = mod_conn.ravel()[:self.total_cells * 4].reshape(self.total_cells, 4)  # Only take enough entries for ncells

        # Vectorized calculation of cell centers
        cell_centers_x = np.mean(self.data['xpoint'][cell_vertices], axis=1)
        cell_centers_y = np.mean(self.data['ypoint'][cell_vertices], axis=1)
        
        return cell_centers_x, cell_centers_y
    
    def fill_nan(self, grid):
        nan_mask = np.isnan(grid)
        not_nan_mask = ~nan_mask
        grid[nan_mask] = griddata((np.where(not_nan_mask)[0], 
                                   np.where(not_nan_mask)[1]),
                                  grid[not_nan_mask],
                                  (np.where(nan_mask)[0], np.where(nan_mask)[1]),
                                  method='nearest')
        return grid
    
    def interpolate_uniform_grid(self,
                                 var_name  : str = "b1",
                                 n_grid_x  : int = 2048,
                                 n_grid_y  : int = 2048,
                                 method    : str = 'linear'):
        """
        Interpolates the cell center data onto a uniform 2D grid.
        """
        
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

        # Create a uniform grid based on the range of x and y
        grid_x = np.linspace(center_x.min(), center_x.max(), n_grid_x)
        grid_y = np.linspace(center_y.min(), center_y.max(), n_grid_y)
        
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)

        print('===============================')
        print(f"Started interpolating")
        start_time = time.time()

        # Interpolate point data (point_rho) onto the uniform grid
        grid = self.fill_nan(griddata((center_x, center_y), self.data[var_name], (grid_x, grid_y), method=method))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished interpolating")
        print(f"Time taken to interpolate: {elapsed_time:.4f} seconds")
        print('===============================')

        return grid
        
        