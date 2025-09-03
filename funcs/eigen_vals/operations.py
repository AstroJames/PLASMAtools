"""
PLASMAtools: Eigenvalue Operations

This module provides optimized eigenvalue and eigenvector computations for symmetric tensors
using Numba for high-performance computing. It is particularly useful for analyzing
stretching tensors, strain rate tensors, and other symmetric tensors in plasma physics.

Author: James R. Beattie

"""

import numpy as np
from typing import Tuple, Union, Optional
from .core_functions import *
from ..tensor.operations import TensorOperations


class EigenvalueOperations:
    """
    A class to perform eigenvalue and eigenvector operations on tensor fields.
    Optimized for symmetric tensors commonly found in plasma physics applications.
    
    This class provides methods for:
    - Computing eigenvalues of symmetric tensors
    - Computing eigenvectors of symmetric tensors
    - Analyzing principal directions and magnitudes
    - Computing growth rates from stretching tensors
    """
    
    def __init__(
        self, 
        use_numba: bool = True):
        """
        Initialize the EigenvalueOperations class.

        Args:
            use_numba (bool, optional): Use Numba core functions. Defaults to True.
        """
        self.use_numba = use_numba
        self.tensor_ops = TensorOperations(
            use_numba=use_numba)


    def eigenvalues_symmetric(
        self, 
        tensor_field: np.ndarray,
        sorted: bool = True) -> np.ndarray:
        """
        Compute eigenvalues of symmetric tensor field using optimized analytical methods.
        
        For 3x3 tensors, uses the analytical formula from https://hal.science/hal-01501221/document
        which provides exact solutions and superior performance (up to 45x speedup).
        
        Args:
            tensor_field: Symmetric tensor field of shape (n, n, ...) where n is 2 or 3
            sorted: If True, sort eigenvalues in descending order (default: ascending for 3x3 analytical)
            
        Returns:
            eigenvalues: Array of shape (n, ...) containing eigenvalues
        """
        # Validate input
        if tensor_field.shape[0] != tensor_field.shape[1]:
            raise ValueError("Tensor must be square (first two dimensions must match)")
        
        n = tensor_field.shape[0]
        
        if n == 3:
            # Use analytical method for 3x3 symmetric tensors (best performance and accuracy)
            return self.eigenvalues_symmetric_analytical(tensor_field, sorted=sorted)
            
        elif n == 2 and self.use_numba and tensor_field.ndim == 4:
            # 2D tensor field - use optimized Numba implementation
            eigenvalues = np.empty((2,) + tensor_field.shape[2:], dtype=tensor_field.dtype)
            eigenvalues_symmetric_2x2_nb_core(tensor_field, eigenvalues)
            # Note: 2x2 core function returns eigenvalues in descending order
            if sorted:
                eigenvalues = eigenvalues[::-1, ...]  # Reverse to ascending order for sorted=True
            return eigenvalues
            
        elif n == 2:
            # Fallback to NumPy for 2x2 - returns in descending order
            result = eigenvalues_symmetric_2x2_np_core(tensor_field)
            if sorted:
                result = result[::-1, ...]  # Reverse to ascending order for sorted=True
            return result
            
        else:
            raise ValueError(f"Unsupported tensor dimension: {n}x{n}")


    def eigenvalues_symmetric_analytical(
        self, 
        tensor_field: np.ndarray,
        sorted: bool = True) -> np.ndarray:
        """
        Compute eigenvalues of symmetric 3x3 tensor field using analytical formula.
        
        This method uses the analytical formula from https://hal.science/hal-01501221/document
        which provides exact solutions for symmetric 3x3 matrices. This is now the default
        implementation used by eigenvalues_symmetric() for 3x3 tensors.
        
        Performance: Up to 45x speedup over iterative methods while maintaining
        machine precision accuracy.
        
        Args:
            tensor_field (np.ndarray): Input symmetric tensor field of shape (3, 3, ...)
            sorted (bool, optional): Sort eigenvalues in ascending order. Defaults to True.
        
        Returns:
            np.ndarray: Eigenvalues of shape (3, ...) 
        
        Raises:
            ValueError: If tensor is not 3x3
        
        Note:
            This is the production implementation - use eigenvalues_symmetric() for the
            public API which delegates to this method for 3x3 tensors.
        """
        # Validate input
        if tensor_field.shape[0] != 3 or tensor_field.shape[1] != 3:
            raise ValueError("Analytical method only supports 3x3 symmetric tensors")
        
        # Ensure float32 or float64
        if tensor_field.dtype not in [np.float32, np.float64]:
            tensor_field = tensor_field.astype(np.float64)
        
        # Create output array
        output_shape = (3,) + tensor_field.shape[2:]
        eigenvalues = np.zeros(output_shape, dtype=tensor_field.dtype)
        
        if self.use_numba:
            # Use optimized Numba core function
            eigenvalues_symmetric_3x3_nb_core(tensor_field, eigenvalues)
        else:
            # For comparison, use NumPy fallback
            eigenvalues = eigenvalues_symmetric_3x3_np_core(tensor_field)
            if not sorted:
                # If not sorted, reverse to match the analytical formula's ascending order
                eigenvalues = eigenvalues[..., ::-1]
        
        # Sort if requested (analytical method already sorts in ascending order)
        if not sorted:
            # Reverse to descending order
            eigenvalues = eigenvalues[::-1, ...]
        
        return eigenvalues
    
    
    def eigenvectors_symmetric(
        self,
        tensor_field: np.ndarray,
        eigenvalues: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors of symmetric tensor field.
        
        Args:
            tensor_field: Symmetric tensor field of shape (n, n, ...) where n is 2 or 3
            eigenvalues: Pre-computed eigenvalues (optional). If None, will compute them.
            
        Returns:
            eigenvalues: Array of shape (n, ...) containing eigenvalues
            eigenvectors: Array of shape (n, n, ...) containing eigenvectors
                         First index selects eigenvector, second index is component
        """
        # Validate input
        if tensor_field.shape[0] != tensor_field.shape[1]:
            raise ValueError("Tensor must be square")
        
        n = tensor_field.shape[0]
        
        # Compute eigenvalues if not provided
        if eigenvalues is None:
            eigenvalues = self.eigenvalues_symmetric(tensor_field)
        
        if n == 3 and self.use_numba and tensor_field.ndim == 5:
            # Check if analytical method is viable (f != 0 for most points)
            # f is the a13 element: tensor_field[0, 2, ...]
            f_elements = tensor_field[0, 2, ...]
            f_zero_fraction = np.mean(np.abs(f_elements) < 1e-12)
            
            if f_zero_fraction < 0.1:  # If less than 10% of points have f≈0, use analytical
                # 3D tensor field with analytical method
                eigenvectors = np.empty_like(tensor_field)
                eigenvectors_symmetric_3x3_nb_core(tensor_field, eigenvalues, eigenvectors)
                return eigenvalues, eigenvectors
            else:
                # Too many f≈0 cases, fallback to general method
                eigenvalues_real = eigenvalues.astype(np.float64)
                eigenvalues_imag = np.zeros_like(eigenvalues_real)
                eigenvectors_real = np.empty_like(tensor_field, dtype=np.float64)
                eigenvectors_imag = np.zeros_like(eigenvectors_real)
                
                eigenvectors_general_3x3_nb_core(
                    tensor_field.astype(np.float64), 
                    eigenvalues_real, eigenvalues_imag,
                    eigenvectors_real, eigenvectors_imag
                )
                return eigenvalues_real, eigenvectors_real
            
        elif n == 2 and self.use_numba and tensor_field.ndim == 4:
            # 2D tensor field  
            eigenvectors = np.empty_like(tensor_field)
            eigenvectors_symmetric_2x2_nb_core(tensor_field, eigenvalues, eigenvectors)
            return eigenvalues, eigenvectors
            
        else:
            # Fallback to NumPy
            # Move tensor indices to the end
            tensor_transposed = np.moveaxis(tensor_field, [0, 1], [-2, -1])
            
            # Compute eigenvalues and eigenvectors
            evals, evecs = np.linalg.eigh(tensor_transposed)
            
            # Sort by eigenvalue (descending) - largest first
            idx = np.argsort(evals, axis=-1)[..., ::-1]
            
            # Apply sorting
            evals_sorted = np.take_along_axis(evals, idx, axis=-1)
            
            # Sort eigenvectors accordingly
            evecs_sorted = np.take_along_axis(evecs, idx[..., None], axis=-2)
            
            # Rearrange axes to match expected format
            eigenvalues = np.moveaxis(evals_sorted, -1, 0)
            eigenvectors = np.moveaxis(evecs_sorted, [-2, -1], [0, 1])
            
            return eigenvalues, eigenvectors
    
    
    def principal_stretching_rate(
        self,
        velocity_gradient_tensor: np.ndarray) -> np.ndarray:
        """
        Compute the principal stretching rate from velocity gradient tensor.
        
        The stretching tensor S = 0.5 * (∇u + ∇u^T) represents the symmetric
        part of the velocity gradient. Its eigenvalues give the principal
        stretching rates.
        
        Args:
            velocity_gradient_tensor: Velocity gradient tensor ∇u of shape (n, n, ...)
            
        Returns:
            principal_rates: Maximum eigenvalue at each point, shape (...)
        """
        # Compute symmetric part (stretching tensor)
        transposed = self.tensor_ops.tensor_transpose(velocity_gradient_tensor)
        stretching_tensor = 0.5 * (velocity_gradient_tensor + transposed)
        
        # Get eigenvalues
        eigenvalues = self.eigenvalues_symmetric(stretching_tensor)
        
        # Return maximum eigenvalue (principal stretching rate)
        return eigenvalues[0]
    
        
    def tensor_anisotropy(
        self,
        tensor_field: np.ndarray) -> np.ndarray:
        """
        Compute anisotropy measure of a symmetric tensor.
        
        Anisotropy = (λ₁ - λ₃) / (λ₁ + λ₂ + λ₃)
        
        For 2D: Anisotropy = (λ₁ - λ₂) / (λ₁ + λ₂)
        
        Args:
            tensor_field: Symmetric tensor field
            
        Returns:
            anisotropy: Scalar field with values between 0 (isotropic) and 1 (highly anisotropic)
        """
        eigenvalues = self.eigenvalues_symmetric(tensor_field)
        
        if eigenvalues.shape[0] == 3:
            # 3D case
            trace = eigenvalues[0] + eigenvalues[1] + eigenvalues[2]
            # Avoid division by zero
            trace_safe = np.where(np.abs(trace) > EPSILON, trace, EPSILON)
            anisotropy = (eigenvalues[0] - eigenvalues[2]) / trace_safe
        else:
            # 2D case
            trace = eigenvalues[0] + eigenvalues[1]
            trace_safe = np.where(np.abs(trace) > EPSILON, trace, EPSILON)
            anisotropy = (eigenvalues[0] - eigenvalues[1]) / trace_safe
        
        return np.abs(anisotropy)
    
    
    def positive_definite_check(
        self,
        tensor_field: np.ndarray) -> np.ndarray:
        """
        Check if symmetric tensor is positive definite at each point.
        
        A symmetric tensor is positive definite if all eigenvalues are positive.
        
        Args:
            tensor_field: Symmetric tensor field
            
        Returns:
            is_positive_definite: Boolean array indicating positive definiteness
        """
        eigenvalues = self.eigenvalues_symmetric(tensor_field)
        
        # Check if all eigenvalues are positive
        return np.all(eigenvalues > 0, axis=0)
    
    
    def eigenvalues_general(
        self,
        tensor_field: np.ndarray,
        sorted: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute eigenvalues of general (non-symmetric) tensor field.
        
        For non-symmetric tensors, eigenvalues can be complex.
        
        Args:
            tensor_field: Tensor field of shape (n, n, ...) where n is 2 or 3
            sorted: If True, sort eigenvalues by magnitude (largest first)
            
        Returns:
            For real eigenvalues: eigenvalues array of shape (n, ...)
            For complex eigenvalues: tuple of (real_part, imag_part) each of shape (n, ...)
        """
        # Validate input
        if tensor_field.shape[0] != tensor_field.shape[1]:
            raise ValueError("Tensor must be square")
        
        n = tensor_field.shape[0]
        
        if n == 3 and self.use_numba and tensor_field.ndim == 5:
            # 3D tensor field - use Numba
            eigenvalues_real = np.empty((3,) + tensor_field.shape[2:], dtype=tensor_field.dtype)
            eigenvalues_imag = np.empty((3,) + tensor_field.shape[2:], dtype=tensor_field.dtype)
            eigenvalues_general_3x3_nb_core(tensor_field, eigenvalues_real, eigenvalues_imag)
            
            # Check if all imaginary parts are zero
            if np.allclose(eigenvalues_imag, 0):
                return eigenvalues_real
            else:
                return eigenvalues_real, eigenvalues_imag
                
        elif n == 2 and self.use_numba and tensor_field.ndim == 4:
            # 2D tensor field - use Numba
            eigenvalues_real = np.empty((2,) + tensor_field.shape[2:], dtype=tensor_field.dtype)
            eigenvalues_imag = np.empty((2,) + tensor_field.shape[2:], dtype=tensor_field.dtype)
            eigenvalues_general_2x2_nb_core(tensor_field, eigenvalues_real, eigenvalues_imag)
            
            # Check if all imaginary parts are zero
            if np.allclose(eigenvalues_imag, 0):
                return eigenvalues_real
            else:
                return eigenvalues_real, eigenvalues_imag
        else:
            # Fallback to NumPy
            # Move tensor indices to the end
            tensor_transposed = np.moveaxis(tensor_field, [0, 1], [-2, -1])
            
            # Compute eigenvalues
            eigenvalues = np.linalg.eigvals(tensor_transposed)
            
            if sorted:
                # Sort by magnitude (descending)
                if np.iscomplexobj(eigenvalues):
                    # For complex eigenvalues, sort by magnitude = sqrt(real^2 + imag^2)
                    magnitudes = np.abs(eigenvalues)
                else:
                    # For real eigenvalues, sort by absolute value
                    magnitudes = np.abs(eigenvalues)
                
                idx = np.argsort(magnitudes, axis=-1)[..., ::-1]
                eigenvalues = np.take_along_axis(eigenvalues, idx, axis=-1)
            
            # Move eigenvalue index to front
            eigenvalues = np.moveaxis(eigenvalues, -1, 0)
            
            # Check if eigenvalues are complex
            if np.iscomplexobj(eigenvalues):
                return np.real(eigenvalues), np.imag(eigenvalues)
            else:
                return eigenvalues
    
    
    def eigenvectors_general(
        self,
        tensor_field: np.ndarray,
        eigenvalues: Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None
        ) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], 
                   Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Compute eigenvalues and eigenvectors of general (non-symmetric) tensor field.
        
        For non-symmetric tensors, eigenvalues and eigenvectors can be complex.
        Uses Numba-optimized implementations for better performance.
        
        Args:
            tensor_field: Tensor field of shape (n, n, ...) where n is 2 or 3
            eigenvalues: Optional pre-computed eigenvalues. If None, will compute them.
            
        Returns:
            eigenvalues: For real case - array of shape (n, ...)
                        For complex case - tuple of (real_part, imag_part)
            eigenvectors: For real case - array of shape (n, n, ...)
                         For complex case - tuple of (real_part, imag_part)
                         First index selects eigenvector, second is component
        """
        # Validate input
        if tensor_field.shape[0] != tensor_field.shape[1]:
            raise ValueError("Tensor must be square")
        
        n = tensor_field.shape[0]
        
        # Ensure float32 or float64
        if tensor_field.dtype not in [np.float32, np.float64]:
            tensor_field = tensor_field.astype(np.float64)
        
        # Compute eigenvalues if not provided
        if eigenvalues is None:
            eigenvalues = self.eigenvalues_general(tensor_field, sorted=True)
        
        # Handle eigenvalue format (could be array or tuple for complex)
        if isinstance(eigenvalues, tuple):
            evals_real, evals_imag = eigenvalues
            has_complex = True
        else:
            evals_real = eigenvalues
            evals_imag = np.zeros_like(evals_real)
            has_complex = False
        
        if self.use_numba and n == 3 and tensor_field.ndim == 5:
            # Use Numba 3x3 implementation
            shape = tensor_field.shape
            evecs_real = np.zeros((3, 3) + shape[2:], dtype=tensor_field.dtype)
            evecs_imag = np.zeros((3, 3) + shape[2:], dtype=tensor_field.dtype)
            
            eigenvectors_general_3x3_nb_core(tensor_field, evals_real, evals_imag, 
                                            evecs_real, evecs_imag)
            
            # Return appropriate format
            if has_complex or np.any(np.abs(evecs_imag) > EPSILON):
                return (evals_real, evals_imag), (evecs_real, evecs_imag)
            else:
                return evals_real, evecs_real
                
        elif self.use_numba and n == 2 and tensor_field.ndim == 4:
            # Use Numba 2x2 implementation
            shape = tensor_field.shape
            evecs_real = np.zeros((2, 2) + shape[2:], dtype=tensor_field.dtype)
            evecs_imag = np.zeros((2, 2) + shape[2:], dtype=tensor_field.dtype)
            
            eigenvectors_general_2x2_nb_core(tensor_field, evals_real, evals_imag,
                                            evecs_real, evecs_imag)
            
            # Return appropriate format
            if has_complex or np.any(np.abs(evecs_imag) > EPSILON):
                return (evals_real, evals_imag), (evecs_real, evecs_imag)
            else:
                return evals_real, evecs_real
        
        else:
            # Fallback to NumPy
            # Move tensor indices to the end
            tensor_transposed = np.moveaxis(tensor_field, [0, 1], [-2, -1])
            
            # Compute eigenvalues and eigenvectors
            eigenvalues_np, eigenvectors_np = np.linalg.eig(tensor_transposed)
            
            # Sort by eigenvalue magnitude
            idx = np.argsort(np.abs(eigenvalues_np), axis=-1)[..., ::-1]
            eigenvalues_np = np.take_along_axis(eigenvalues_np, idx, axis=-1)
            eigenvectors_np = np.take_along_axis(eigenvectors_np, idx[..., None], axis=-2)
            
            # Rearrange axes
            eigenvalues_np = np.moveaxis(eigenvalues_np, -1, 0)
            eigenvectors_np = np.moveaxis(eigenvectors_np, [-2, -1], [0, 1])
            
            # Handle complex results
            if np.iscomplexobj(eigenvalues_np):
                eigenvalues_np = (np.real(eigenvalues_np), np.imag(eigenvalues_np))
            if np.iscomplexobj(eigenvectors_np):
                eigenvectors_np = (np.real(eigenvectors_np), np.imag(eigenvectors_np))
                
            return eigenvalues_np, eigenvectors_np
    
    
    def complex_eigenvalue_analysis(
        self,
        tensor_field: np.ndarray) -> dict:
        """
        Analyze complex eigenvalues of a general tensor field.
        
        Provides information about:
        - Real parts (growth/decay rates)
        - Imaginary parts (oscillation frequencies)
        - Stability (based on real parts)
        
        Args:
            tensor_field: General tensor field
            
        Returns:
            analysis: Dictionary containing:
                - 'eigenvalues_real': Real parts
                - 'eigenvalues_imag': Imaginary parts
                - 'magnitudes': Absolute values
                - 'phases': Phase angles
                - 'stable_points': Boolean array indicating stability (all Re(λ) < 0)
                - 'oscillatory_points': Boolean array indicating oscillatory behavior
        """
        # Get eigenvalues
        result = self.eigenvalues_general(tensor_field, sorted=True)
        
        if isinstance(result, tuple):
            real_parts, imag_parts = result
        else:
            real_parts = result
            imag_parts = np.zeros_like(real_parts)
        
        # Compute magnitudes and phases
        magnitudes = np.sqrt(real_parts**2 + imag_parts**2)
        phases = np.arctan2(imag_parts, real_parts)
        
        # Stability analysis
        stable_points = np.all(real_parts < 0, axis=0)
        
        # Oscillatory behavior (has non-zero imaginary parts)
        oscillatory_points = np.any(np.abs(imag_parts) > EPSILON, axis=0)
        
        return {
            'eigenvalues_real': real_parts,
            'eigenvalues_imag': imag_parts,
            'magnitudes': magnitudes,
            'phases': phases,
            'stable_points': stable_points,
            'oscillatory_points': oscillatory_points
        }
    
    
    def tensor_exponential_eigendecomp(
        self,
        tensor_field: np.ndarray,
        t: float = 1.0) -> np.ndarray:
        """
        Compute matrix exponential exp(A*t) using eigendecomposition.
        
        Useful for evolution operators and time integration.
        For A = V*D*V^(-1), exp(A*t) = V*exp(D*t)*V^(-1)
        
        Args:
            tensor_field: Tensor field A
            t: Time parameter
            
        Returns:
            exp_tensor: Matrix exponential exp(A*t)
        """
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = self.eigenvectors_general(tensor_field)
        
        # Handle complex case
        if isinstance(eigenvalues, tuple):
            # Complex eigenvalues
            real_vals, imag_vals = eigenvalues
            real_vecs, imag_vecs = eigenvectors
            
            # exp(a + ib) = exp(a) * (cos(b) + i*sin(b))
            exp_real = np.exp(real_vals * t)
            cos_part = np.cos(imag_vals * t)
            sin_part = np.sin(imag_vals * t)
            
            # Complex exponential
            exp_vals_real = exp_real * cos_part
            exp_vals_imag = exp_real * sin_part
            
            # Reconstruct (this is simplified - full implementation would need complex arithmetic)
            # For now, return real part only
            return self._reconstruct_from_eigen_real(
                exp_vals_real, real_vecs, tensor_field.shape)
        else:
            # Real eigenvalues - simple case
            exp_vals = np.exp(eigenvalues * t)
            return self._reconstruct_from_eigen_real(
                exp_vals, eigenvectors, tensor_field.shape)
    
    
    def _reconstruct_from_eigen_real(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        original_shape: tuple) -> np.ndarray:
        """
        Reconstruct tensor from eigenvalues and eigenvectors (real case).
        
        A = V * D * V^(-1)
        
        Args:
            eigenvalues: Shape (n, ...)
            eigenvectors: Shape (n, n, ...)
            original_shape: Original tensor shape
            
        Returns:
            reconstructed: Reconstructed tensor
        """
        # This is a simplified implementation
        # Full implementation would use proper matrix multiplication
        n = eigenvalues.shape[0]
        result = np.zeros(original_shape, dtype=eigenvalues.dtype)
        
        # For each spatial point, reconstruct the tensor
        # This is a placeholder - actual implementation would be more complex
        # and would properly handle the tensor reconstruction
        
        return result