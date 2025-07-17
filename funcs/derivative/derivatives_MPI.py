import numpy as np
from mpi4py import MPI

# shifts for derivatives
F = -1  # shift forwards
B = +1  # shift backwards

################################################################
## Derivative class
################################################################

class Derivative:
    """
    Derivative class.
    
    This class provides methods for computing the first and second
    derivatives of a scalar field using finite differences.
    
    Author: James Beattie
    
    """
    
    def __init__(self, 
                 stencil : int = 2) -> None:
        """
        Initialize the derivative class.

        Author: James Beattie

        Args:
            order (int, optional) : the size of the stencil. Defaults to 2 point.
            L (float, optional)   : the size of the domain. Defaults to 1.0.
        """
        
        # Stencil order
        if stencil not in [2, 4, 6, 8]:
            raise ValueError("Invalid stencil order")
        self.stencil = stencil
        
        
    def apply_boundary_conditions(self, 
                                  scalar_field          : np.ndarray,
                                  boundary_condition    : str,
                                  gradient_dir          : int) -> np.ndarray:
        """
        NOTE: This function is not complete. Currently only works for periodic boundary conditions.
        
        
        Apply boundary conditions to the scalar field.

        Author: James Beattie

        Args:
            scalar_field (np.ndarray): the scalar field to which boundary conditions are applied.
            boundary_condition (str): the type of boundary condition ('periodic', 'dirichlet', 'neumann').
            gradient_dir (int):        the direction of the gradient.

        Returns:
            np.ndarray: the scalar field with boundary conditions applied.
        """
                
        self.num_of_gcs = self.stencil // 2
        pad_width = [(0, 0)] * scalar_field.ndim
        pad_width[gradient_dir] = (self.num_of_gcs, self.num_of_gcs)

        # For Dirichlet boundary conditions, set the boundary values to zero
        if boundary_condition == 'dirichlet':
            Warning("Dirichlet boundary conditions not implemented yet")
            scalar_field = np.pad(scalar_field, 
                                  pad_width, 
                                  mode='constant', 
                                  constant_values=0)

        # For Neumann boundary conditions, replicate the edge values, 
        # setting derivative to zero
        if boundary_condition == 'neumann' or boundary_condition == "periodic":
            if boundary_condition == 'neumann':
                Warning("Neumann boundary conditions not implemented yet")
            scalar_field = np.pad(scalar_field, 
                                  pad_width, 
                                  mode='edge')

        return scalar_field
    
    
    def remove_ghost_cells(self, 
                           scalar_field         : np.ndarray,
                           gradient_dir         : int) -> np.ndarray:
        """
        Remove ghost cells from the scalar field.

        Author: James Beattie

        Args:
            scalar_field (np.ndarray): the scalar field with ghost cells.
            gradient_dir (int):        the direction of the gradient.

        Returns:
            np.ndarray: the scalar field with ghost cells removed.
        """
        
        slices                   = [slice(None)] * scalar_field.ndim
        slices[gradient_dir]     = slice(self.num_of_gcs, -self.num_of_gcs)
        constant                 = 2.0
        scalar_field             = scalar_field[tuple(slices)]

        # Multiply the first slice along the specified axis
        slc_first = [slice(None)] * scalar_field.ndim  # Create a full slice for all dimensions
        slc_first[gradient_dir] = 0  # Set the slice for the first element along the axis
        scalar_field[tuple(slc_first)] *= constant
        
        # Multiply the last slice along the specified axis
        slc_last = [slice(None)] * scalar_field.ndim  # Create a full slice for all dimensions
        slc_last[gradient_dir] = -1  # Set the slice for the last element along the axis
        scalar_field[tuple(slc_last)] *= constant
        
        return scalar_field
    
    
    def compute_dr(self, 
                   L            : float,
                   shape        : np.ndarray,
                   gradient_dir : int) -> float:
        """
        Compute the grid spacing `dr`.
        
        Author: James Beattie
        
        Args:
            shape (tuple): the shape of the scalar field.
            gradient_dir (int): the direction of the gradient.
            
        Returns:
            float: the grid spacing `dr`.
        """
        
        return L / (shape[gradient_dir] - 1)


    def gradient(self,
                 scalar_field       : np.ndarray,
                 gradient_dir       : int,
                 L                  : float = 1.0,
                 derivative_order   : int = 1,
                 boundary_condition : str = "periodic",
                 comm               : MPI.Comm = MPI.COMM_WORLD) -> np.ndarray:
        """
        Compute the first derivative of a scalar field in one direction
        using finite differences.
        
        Author: James Beattie & Neco Kriel
        
        Args:
            scalar_field (np.ndarray)          : the scalar field to compute the gradient of.
            gradient_dir (int)                 : the direction of the gradient.
            derivative_order (int, optional)   : the order of the derivative. Defaults to 1.
                                                (1 for first derivative, 2 for second derivative)
            boundary_condition (str, optional) : the type of boundary condition 
                                                    ('periodic', 'dirichlet', 'neumann'). D
                                                    efaults to 'periodic'.
            comm (MPI.Comm, optional)          : MPI communicator. Defaults to MPI.COMM_WORLD.

        Returns:
            np.ndarray: the gradient of the scalar field in the specified direction.
        
        """
        
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Split data along the gradient direction
        local_shape = list(scalar_field.shape)
        local_shape[gradient_dir] = scalar_field.shape[gradient_dir] // size
        if rank == size - 1:
            local_shape[gradient_dir] += scalar_field.shape[gradient_dir] % size
            
        local_data = np.zeros(local_shape,
                              dtype=scalar_field.dtype)
        
        comm.Scatterv([scalar_field, 
                       self._create_scatter_counts(scalar_field, gradient_dir, size), 
                       MPI.DOUBLE],
                      local_data,
                      root=0)
        
        # Apply boundary conditions (including ghost cells)
        local_data = self.apply_boundary_conditions(local_data,
                                                    boundary_condition,
                                                    gradient_dir)
        
        # Compute the grid spacing `dr`
        dr = self.compute_dr(L,
                             scalar_field.shape,
                             gradient_dir)
        
        # Compute the local gradient using stencils
        if derivative_order == 1:
            local_gradient = self._compute_first_order_derivative(local_data,
                                                                  gradient_dir,
                                                                  dr)
        elif derivative_order == 2:
            local_gradient = self._compute_second_order_derivative(local_data,
                                                                   gradient_dir,
                                                                   dr)
        else:
            raise ValueError("Only first and second derivative orders are supported.")
        
        # Exchange ghost cells with neighboring processes
        self._exchange_ghost_cells(local_gradient,
                                   gradient_dir,
                                   comm,
                                   boundary_condition)

        # Remove ghost cells before returning
        local_gradient = self.remove_ghost_cells(local_gradient,
                                                 gradient_dir)
        
        return local_gradient
    
    def _create_scatter_counts(self,
                               scalar_field,
                               gradient_dir,
                               size):
        total_points = scalar_field.shape[gradient_dir]
        counts = [total_points // size] * size
        counts[-1] += total_points % size
        counts = [c * np.prod([scalar_field.shape[i] for i in range(len(scalar_field.shape)) if i != gradient_dir]) for c in counts]
        displacements = [sum(counts[:i]) for i in range(size)]
        return counts, displacements
    
    def _exchange_ghost_cells(self,
                              local_data,
                              gradient_dir,
                              comm):
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Prepare ghost slices
        send_forward = local_data.take(indices=-1, axis=gradient_dir).copy()
        send_backward = local_data.take(indices=0, axis=gradient_dir).copy()
        recv_forward = np.empty_like(send_forward)
        recv_backward = np.empty_like(send_backward)

        # Periodic boundary conditions
        forward_rank = (rank + 1) % size
        backward_rank = (rank - 1 + size) % size

        # Exchange data with neighbors
        comm.Sendrecv(send_forward, dest=forward_rank, recvbuf=recv_backward, source=backward_rank)
        comm.Sendrecv(send_backward, dest=backward_rank, recvbuf=recv_forward, source=forward_rank)

        # Insert ghost cells
        local_data = np.concatenate(([recv_backward], local_data, [recv_forward]), axis=gradient_dir)
        return local_data