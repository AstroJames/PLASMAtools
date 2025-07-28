from numba import njit, prange
import numpy as np
from .constants import *
from typing import Tuple
from ..tensor.operations import TensorOperations

##########################################################################################
# Core numba JIT functions for eigenvalue operations
##########################################################################################


@njit([eigenvalues_symmetric_2x2_2d_sig_32, eigenvalues_symmetric_2x2_2d_sig_64], 
      parallel=True, fastmath=True, cache=True)
def eigenvalues_symmetric_2x2_nb_core(tensor_field, eigenvalues):
    """
    Compute eigenvalues of symmetric 2x2 tensors using analytical formula.
    
    For 2x2 symmetric matrices:
    λ = (trace ± √(trace² - 4*det)) / 2
    
    Args:
        tensor_field: Input symmetric tensor (2, 2, Nx, Ny)
        eigenvalues: Output eigenvalues (2, Nx, Ny), sorted in descending order
    """
    Nx, Ny = tensor_field.shape[2], tensor_field.shape[3]
    
    for i in prange(Nx):
        for j in range(Ny):
            # Extract matrix elements
            a11 = tensor_field[0, 0, i, j]
            a22 = tensor_field[1, 1, i, j]
            a12 = tensor_field[0, 1, i, j]
            
            # Compute trace and determinant
            trace = a11 + a22
            det = a11 * a22 - a12 * a12
            
            # Compute discriminant
            discriminant = trace * trace - 4.0 * det
            
            if discriminant < 0:
                discriminant = 0  # Should not happen for symmetric matrices
            
            sqrt_disc = np.sqrt(discriminant)
            
            # Compute eigenvalues (already sorted)
            eigenvalues[0, i, j] = 0.5 * (trace + sqrt_disc)
            eigenvalues[1, i, j] = 0.5 * (trace - sqrt_disc)


@njit(parallel=True, fastmath=True, cache=True)
def eigenvectors_symmetric_3x3_nb_core(tensor_field, eigenvalues, eigenvectors):
    """
    Compute eigenvectors for symmetric 3x3 tensors using analytical method.
    
    Based on the analytical formulation from https://hal.science/hal-01501221/document
    This is much more efficient than solving (A - λI)v = 0 numerically.
    
    Args:
        tensor_field: Input symmetric tensor (3, 3, Nx, Ny, Nz)
        eigenvalues: Input eigenvalues (3, Nx, Ny, Nz) - must be sorted
        eigenvectors: Output eigenvectors (3, 3, Nx, Ny, Nz)
                     First index is eigenvector number, second is component
    """
    Nx, Ny, Nz = tensor_field.shape[2], tensor_field.shape[3], tensor_field.shape[4]
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Extract matrix elements using same notation as derived_vars
                a = tensor_field[0, 0, i, j, k]  # a11
                b = tensor_field[1, 1, i, j, k]  # a22  
                c = tensor_field[2, 2, i, j, k]  # a33
                d = tensor_field[0, 1, i, j, k]  # a12
                e = tensor_field[1, 2, i, j, k]  # a23
                f = tensor_field[0, 2, i, j, k]  # a13
                
                # Extract sorted eigenvalues
                lambda1 = eigenvalues[0, i, j, k]  # smallest
                lambda2 = eigenvalues[1, i, j, k]  # middle
                lambda3 = eigenvalues[2, i, j, k]  # largest
                
                # Compute eigenvectors using analytical formulation
                # Check for degenerate cases to avoid division by zero
                
                # Eigenvector 1
                denom1 = f * (b - lambda1) - d * e
                if abs(denom1) > EPSILON:
                    m1 = (d * (c - lambda1) - e * f) / denom1
                    vec1_x = (lambda1 - c - e * m1) / f if abs(f) > EPSILON else 1.0
                    vec1_y = m1
                    vec1_z = 1.0
                else:
                    # Handle degenerate case - use standard approach
                    if abs(d) > EPSILON:
                        vec1_x = d
                        vec1_y = lambda1 - a
                        vec1_z = 0.0
                    else:
                        vec1_x = 1.0
                        vec1_y = 0.0
                        vec1_z = 0.0
                
                # Normalize eigenvector 1
                norm1 = np.sqrt(vec1_x*vec1_x + vec1_y*vec1_y + vec1_z*vec1_z)
                if norm1 > EPSILON:
                    eigenvectors[0, 0, i, j, k] = vec1_x / norm1
                    eigenvectors[0, 1, i, j, k] = vec1_y / norm1
                    eigenvectors[0, 2, i, j, k] = vec1_z / norm1
                else:
                    eigenvectors[0, 0, i, j, k] = 1.0
                    eigenvectors[0, 1, i, j, k] = 0.0
                    eigenvectors[0, 2, i, j, k] = 0.0
                
                # Eigenvector 2
                denom2 = f * (b - lambda2) - d * e
                if abs(denom2) > EPSILON:
                    m2 = (d * (c - lambda2) - e * f) / denom2
                    vec2_x = (lambda2 - c - e * m2) / f if abs(f) > EPSILON else 1.0
                    vec2_y = m2
                    vec2_z = 1.0
                else:
                    # Handle degenerate case
                    if abs(d) > EPSILON:
                        vec2_x = d
                        vec2_y = lambda2 - a
                        vec2_z = 0.0
                    else:
                        vec2_x = 0.0
                        vec2_y = 1.0
                        vec2_z = 0.0
                
                # Normalize eigenvector 2
                norm2 = np.sqrt(vec2_x*vec2_x + vec2_y*vec2_y + vec2_z*vec2_z)
                if norm2 > EPSILON:
                    eigenvectors[1, 0, i, j, k] = vec2_x / norm2
                    eigenvectors[1, 1, i, j, k] = vec2_y / norm2
                    eigenvectors[1, 2, i, j, k] = vec2_z / norm2
                else:
                    eigenvectors[1, 0, i, j, k] = 0.0
                    eigenvectors[1, 1, i, j, k] = 1.0
                    eigenvectors[1, 2, i, j, k] = 0.0
                
                # Eigenvector 3
                denom3 = f * (b - lambda3) - d * e
                if abs(denom3) > EPSILON:
                    m3 = (d * (c - lambda3) - e * f) / denom3
                    vec3_x = (lambda3 - c - e * m3) / f if abs(f) > EPSILON else 1.0
                    vec3_y = m3
                    vec3_z = 1.0
                else:
                    # Handle degenerate case
                    if abs(d) > EPSILON:
                        vec3_x = d
                        vec3_y = lambda3 - a
                        vec3_z = 0.0
                    else:
                        vec3_x = 0.0
                        vec3_y = 0.0
                        vec3_z = 1.0
                
                # Normalize eigenvector 3
                norm3 = np.sqrt(vec3_x*vec3_x + vec3_y*vec3_y + vec3_z*vec3_z)
                if norm3 > EPSILON:
                    eigenvectors[2, 0, i, j, k] = vec3_x / norm3
                    eigenvectors[2, 1, i, j, k] = vec3_y / norm3
                    eigenvectors[2, 2, i, j, k] = vec3_z / norm3
                else:
                    eigenvectors[2, 0, i, j, k] = 0.0
                    eigenvectors[2, 1, i, j, k] = 0.0
                    eigenvectors[2, 2, i, j, k] = 1.0


@njit(parallel=True, fastmath=True, cache=True)
def eigenvectors_symmetric_2x2_nb_core(tensor_field, eigenvalues, eigenvectors):
    """
    Compute eigenvectors for symmetric 2x2 tensors given eigenvalues.
    
    Args:
        tensor_field: Input symmetric tensor (2, 2, Nx, Ny)
        eigenvalues: Input eigenvalues (2, Nx, Ny)
        eigenvectors: Output eigenvectors (2, 2, Nx, Ny)
    """
    Nx, Ny = tensor_field.shape[2], tensor_field.shape[3]
    
    for i in prange(Nx):
        for j in range(Ny):
            a11 = tensor_field[0, 0, i, j]
            a22 = tensor_field[1, 1, i, j]
            a12 = tensor_field[0, 1, i, j]
            
            for n in range(2):
                lam = eigenvalues[n, i, j]
                
                # For 2x2 case, if a12 ≠ 0, eigenvector is [a12, λ - a11]
                if abs(a12) > EPSILON:
                    v1 = a12
                    v2 = lam - a11
                else:
                    # Matrix is diagonal
                    if n == 0:
                        v1 = 1.0
                        v2 = 0.0
                    else:
                        v1 = 0.0
                        v2 = 1.0
                
                # Normalize
                norm = np.sqrt(v1*v1 + v2*v2)
                if norm > EPSILON:
                    eigenvectors[n, 0, i, j] = v1 / norm
                    eigenvectors[n, 1, i, j] = v2 / norm
                else:
                    eigenvectors[n, 0, i, j] = 1.0
                    eigenvectors[n, 1, i, j] = 0.0


# NumPy fallback functions for non-uniform or special cases
def eigenvalues_symmetric_3x3_np_core(tensor_field):
    """
    NumPy fallback for eigenvalue computation of symmetric 3x3 tensors.
    
    Args:
        tensor_field: Input tensor field of shape (3, 3, ...)
        
    Returns:
        eigenvalues: Array of shape (3, ...) with eigenvalues sorted in descending order
    """
    # Initialize tensor operations for consistent tensor manipulation
    tensor_ops = TensorOperations(use_numba=False)
    
    # Move tensor indices to the end for np.linalg.eigh
    tensor_transposed = np.moveaxis(tensor_field, [0, 1], [-2, -1])
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(tensor_transposed)
    
    # Sort in descending order and move axis back
    eigenvalues = np.sort(eigenvalues, axis=-1)[..., ::-1]  # Reverse only the eigenvalue axis
    eigenvalues = np.moveaxis(eigenvalues, -1, 0)
    
    return eigenvalues


@njit([eigenvalues_general_3x3_3d_sig_32, eigenvalues_general_3x3_3d_sig_64], 
      parallel=True, fastmath=True, cache=True)
def eigenvalues_general_3x3_nb_core(tensor_field, eigenvalues_real, eigenvalues_imag):
    """
    Compute eigenvalues of general (non-symmetric) 3x3 tensors.
    
    Uses the characteristic polynomial and cubic formula.
    For general matrices, eigenvalues can be complex.
    
    Args:
        tensor_field: Input tensor (3, 3, Nx, Ny, Nz)
        eigenvalues_real: Output real parts (3, Nx, Ny, Nz)
        eigenvalues_imag: Output imaginary parts (3, Nx, Ny, Nz)
    """
    Nx, Ny, Nz = tensor_field.shape[2], tensor_field.shape[3], tensor_field.shape[4]
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Extract matrix elements
                a11 = tensor_field[0, 0, i, j, k]
                a12 = tensor_field[0, 1, i, j, k]
                a13 = tensor_field[0, 2, i, j, k]
                a21 = tensor_field[1, 0, i, j, k]
                a22 = tensor_field[1, 1, i, j, k]
                a23 = tensor_field[1, 2, i, j, k]
                a31 = tensor_field[2, 0, i, j, k]
                a32 = tensor_field[2, 1, i, j, k]
                a33 = tensor_field[2, 2, i, j, k]
                
                # Compute coefficients of characteristic polynomial
                # Use monic form: λ³ + aλ² + bλ + c = 0 (following NumPy convention)
                trace = a11 + a22 + a33
                
                # Sum of principal 2x2 minors
                minor_sum = (a11*a22 - a12*a21) + (a11*a33 - a13*a31) + (a22*a33 - a23*a32)
                
                # Determinant
                det = (a11*(a22*a33 - a23*a32) - 
                      a12*(a21*a33 - a23*a31) + 
                      a13*(a21*a32 - a22*a31))
                
                # Monic form coefficients: λ³ + a*λ² + b*λ + c = 0
                a_coeff = -trace      # coefficient of λ²
                b_coeff = minor_sum   # coefficient of λ
                c_coeff = -det        # constant term
                
                # Convert to depressed cubic: t³ + pt + q = 0
                # Substitution: λ = t - a_coeff/3 = t + trace/3
                p = b_coeff - a_coeff*a_coeff/3.0
                q = 2.0*a_coeff*a_coeff*a_coeff/27.0 - a_coeff*b_coeff/3.0 + c_coeff
                
                # Use standard Cardano discriminant: Δ = q²/4 + p³/27
                discriminant_standard = q*q/4.0 + p*p*p/27.0
                
                if discriminant_standard <= 0:
                    # Three real roots - use trigonometric method
                    if abs(p) < EPSILON:
                        # Special case: p ≈ 0
                        lambda1 = lambda2 = lambda3 = -a_coeff/3.0
                    else:
                        # Ensure -p/3.0 is positive for sqrt
                        if p >= 0:
                            # If p >= 0, we have issues - fallback to equal roots
                            lambda1 = lambda2 = lambda3 = -a_coeff/3.0
                        else:
                            m = 2.0 * np.sqrt(-p/3.0)
                            # Clamp argument to arccos to valid range [-1, 1]
                            arg = 3.0*q/(p*m)
                            arg = max(-1.0, min(1.0, arg))
                            theta = np.arccos(arg) / 3.0
                            
                            lambda1 = m * np.cos(theta) - a_coeff/3.0
                            lambda2 = m * np.cos(theta - 2.0*np.pi/3.0) - a_coeff/3.0
                            lambda3 = m * np.cos(theta + 2.0*np.pi/3.0) - a_coeff/3.0
                    
                    # Sort by actual value (descending) - largest first
                    if lambda1 < lambda2:
                        lambda1, lambda2 = lambda2, lambda1
                    if lambda1 < lambda3:
                        lambda1, lambda3 = lambda3, lambda1
                    if lambda2 < lambda3:
                        lambda2, lambda3 = lambda3, lambda2
                    
                    eigenvalues_real[0, i, j, k] = lambda1
                    eigenvalues_real[1, i, j, k] = lambda2
                    eigenvalues_real[2, i, j, k] = lambda3
                    eigenvalues_imag[0, i, j, k] = 0.0
                    eigenvalues_imag[1, i, j, k] = 0.0
                    eigenvalues_imag[2, i, j, k] = 0.0
                else:
                    # One real root and two complex conjugate roots
                    # Use standard Cardano formula with cube roots
                    sqrt_discriminant = np.sqrt(discriminant_standard)
                    u3 = -q/2.0 + sqrt_discriminant
                    v3 = -q/2.0 - sqrt_discriminant
                    
                    # Take cube roots carefully
                    u = u3**(1.0/3.0) if u3 >= 0 else -((-u3)**(1.0/3.0))
                    v = v3**(1.0/3.0) if v3 >= 0 else -((-v3)**(1.0/3.0))
                    
                    # The three roots in depressed form are:
                    # t1 = u + v (real)
                    # t2 = -0.5*(u + v) + 0.5*sqrt(3)*(u - v)*i (complex)
                    # t3 = -0.5*(u + v) - 0.5*sqrt(3)*(u - v)*i (complex conjugate)
                    
                    t_real = u + v
                    t_complex_real = -0.5 * (u + v)
                    t_complex_imag = 0.5 * np.sqrt(3.0) * (u - v)
                    
                    # Back-transform to original eigenvalues: λ = t - a_coeff/3
                    real_root = t_real - a_coeff/3.0
                    complex_real = t_complex_real - a_coeff/3.0
                    complex_imag = abs(t_complex_imag)  # Take absolute value for magnitude
                    
                    # Sort by real part (descending), then by imaginary part 
                    # For consistent ordering: larger real part first
                    if real_root >= complex_real:
                        # Real eigenvalue is largest
                        eigenvalues_real[0, i, j, k] = real_root
                        eigenvalues_real[1, i, j, k] = complex_real
                        eigenvalues_real[2, i, j, k] = complex_real
                        eigenvalues_imag[0, i, j, k] = 0.0
                        # Match NumPy convention: negative imaginary part first
                        eigenvalues_imag[1, i, j, k] = -complex_imag
                        eigenvalues_imag[2, i, j, k] = complex_imag
                    else:
                        # Complex eigenvalues have larger real part
                        eigenvalues_real[0, i, j, k] = complex_real
                        eigenvalues_real[1, i, j, k] = complex_real
                        eigenvalues_real[2, i, j, k] = real_root
                        eigenvalues_imag[0, i, j, k] = -complex_imag  # negative first
                        eigenvalues_imag[1, i, j, k] = complex_imag   # positive second
                        eigenvalues_imag[2, i, j, k] = 0.0


@njit([eigenvalues_general_2x2_2d_sig_32, eigenvalues_general_2x2_2d_sig_64], 
      parallel=True, fastmath=True, cache=True)
def eigenvalues_general_2x2_nb_core(tensor_field, eigenvalues_real, eigenvalues_imag):
    """
    Compute eigenvalues of general (non-symmetric) 2x2 tensors.
    
    For 2x2 matrices: λ = (trace ± √(trace² - 4*det)) / 2
    
    Args:
        tensor_field: Input tensor (2, 2, Nx, Ny)
        eigenvalues_real: Output real parts (2, Nx, Ny)
        eigenvalues_imag: Output imaginary parts (2, Nx, Ny)
    """
    Nx, Ny = tensor_field.shape[2], tensor_field.shape[3]
    
    for i in prange(Nx):
        for j in range(Ny):
            # Extract matrix elements
            a11 = tensor_field[0, 0, i, j]
            a12 = tensor_field[0, 1, i, j]
            a21 = tensor_field[1, 0, i, j]
            a22 = tensor_field[1, 1, i, j]
            
            # Compute trace and determinant
            trace = a11 + a22
            det = a11 * a22 - a12 * a21
            
            # Compute discriminant
            discriminant = trace * trace - 4.0 * det
            
            if discriminant >= 0:
                # Real eigenvalues
                sqrt_disc = np.sqrt(discriminant)
                lambda1 = 0.5 * (trace + sqrt_disc)
                lambda2 = 0.5 * (trace - sqrt_disc)
                
                # Sort by actual value (descending)
                if lambda1 >= lambda2:
                    eigenvalues_real[0, i, j] = lambda1
                    eigenvalues_real[1, i, j] = lambda2
                else:
                    eigenvalues_real[0, i, j] = lambda2
                    eigenvalues_real[1, i, j] = lambda1
                
                eigenvalues_imag[0, i, j] = 0.0
                eigenvalues_imag[1, i, j] = 0.0
            else:
                # Complex eigenvalues
                real_part = 0.5 * trace
                imag_part = 0.5 * np.sqrt(-discriminant)
                
                # Match NumPy convention: when magnitudes are equal, negative imaginary first
                eigenvalues_real[0, i, j] = real_part
                eigenvalues_real[1, i, j] = real_part
                eigenvalues_imag[0, i, j] = -imag_part  # negative first
                eigenvalues_imag[1, i, j] = imag_part   # positive second


def eigenvalues_symmetric_2x2_np_core(tensor_field):
    """
    NumPy fallback for eigenvalue computation of symmetric 2x2 tensors.
    
    Args:
        tensor_field: Input tensor field of shape (2, 2, ...)
        
    Returns:
        eigenvalues: Array of shape (2, ...) with eigenvalues sorted in descending order
    """
    # Initialize tensor operations for consistent tensor manipulation
    tensor_ops = TensorOperations(use_numba=False)
    
    # Move tensor indices to the end
    tensor_transposed = np.moveaxis(tensor_field, [0, 1], [-2, -1])
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(tensor_transposed)
    
    # Sort in descending order and move axis back
    eigenvalues = np.sort(eigenvalues, axis=-1)[..., ::-1]  # Reverse only the eigenvalue axis
    eigenvalues = np.moveaxis(eigenvalues, -1, 0)
    
    return eigenvalues


@njit([eigenvalues_symmetric_analytical_3x3_3d_sig_32, eigenvalues_symmetric_analytical_3x3_3d_sig_64], 
      parallel=True, fastmath=True, cache=True)
def eigenvalues_symmetric_3x3_nb_core(tensor_field, eigenvalues):
    """
    Compute eigenvalues of symmetric 3x3 tensors using analytical formula from:
    https://hal.science/hal-01501221/document
    
    This is a Numba-optimized port of the symmetric_eigvals function from derived_vars.
    The analytical method provides exact solutions for symmetric 3x3 matrices.
    
    Args:
        tensor_field: Input symmetric tensor (3, 3, Nx, Ny, Nz)
        eigenvalues: Output eigenvalues (3, Nx, Ny, Nz), sorted in ascending order
    """
    Nx, Ny, Nz = tensor_field.shape[2], tensor_field.shape[3], tensor_field.shape[4]
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Extract matrix elements
                a = tensor_field[0, 0, i, j, k]  # a11
                b = tensor_field[1, 1, i, j, k]  # a22
                c = tensor_field[2, 2, i, j, k]  # a33
                d = tensor_field[0, 1, i, j, k]  # a12
                e = tensor_field[1, 2, i, j, k]  # a23
                f = tensor_field[0, 2, i, j, k]  # a13
                
                # Compute the analytical formulation from hal-01501221
                # First compute x1 and x2
                x1 = a*a + b*b + c*c - a*b - a*c - b*c + 3.0*(d*d + f*f + e*e)
                x2 = (-(2.0*a - b - c)*(2.0*b - a - c)*(2.0*c - a - b) + 
                      9.0*((2.0*c - a - b)*d*d + (2.0*b - a - c)*f*f + (2.0*a - b - c)*e*e) - 
                      54.0*d*e*f)
                
                # Compute phi based on x2 value
                phi = 0.0
                if x2 > 0.0:
                    # x2 > 0
                    sqrt_term = np.sqrt(4.0*x1*x1*x1 - x2*x2)
                    phi = np.arctan(sqrt_term / x2)
                elif x2 == 0.0:
                    # x2 = 0
                    phi = np.pi / 2.0
                else:
                    # x2 < 0
                    sqrt_term = np.sqrt(4.0*x1*x1*x1 - x2*x2)
                    phi = np.arctan(sqrt_term / x2) + np.pi
                
                # Compute the three eigenvalues
                sqrt_x1 = np.sqrt(x1)
                lambda1 = (a + b + c - 2.0*sqrt_x1*np.cos(phi/3.0)) / 3.0
                lambda2 = (a + b + c + 2.0*sqrt_x1*np.cos((phi - np.pi)/3.0)) / 3.0
                lambda3 = (a + b + c + 2.0*sqrt_x1*np.cos((phi + np.pi)/3.0)) / 3.0
                
                # Sort eigenvalues in descending order (largest first)
                if lambda1 < lambda2:
                    lambda1, lambda2 = lambda2, lambda1
                if lambda1 < lambda3:
                    lambda1, lambda3 = lambda3, lambda1
                if lambda2 < lambda3:
                    lambda2, lambda3 = lambda3, lambda2
                
                eigenvalues[0, i, j, k] = lambda1
                eigenvalues[1, i, j, k] = lambda2
                eigenvalues[2, i, j, k] = lambda3


@njit([eigenvectors_general_2x2_2d_sig_32, eigenvectors_general_2x2_2d_sig_64], 
      parallel=True, fastmath=True, cache=True)
def eigenvectors_general_2x2_nb_core(tensor_field, eigenvalues_real, eigenvalues_imag, 
                                    eigenvectors_real, eigenvectors_imag):
    """
    Compute eigenvectors of general (non-symmetric) 2x2 tensors given eigenvalues.
    
    For a 2x2 matrix A with eigenvalue λ, the eigenvector v satisfies:
    (A - λI)v = 0
    
    Args:
        tensor_field: Input tensor (2, 2, Nx, Ny)
        eigenvalues_real: Input eigenvalues real parts (2, Nx, Ny) 
        eigenvalues_imag: Input eigenvalues imaginary parts (2, Nx, Ny)
        eigenvectors_real: Output eigenvectors real parts (2, 2, Nx, Ny)
        eigenvectors_imag: Output eigenvectors imaginary parts (2, 2, Nx, Ny)
    """
    Nx, Ny = tensor_field.shape[2], tensor_field.shape[3]
    
    for i in prange(Nx):
        for j in range(Ny):
            # Extract matrix elements
            a11 = tensor_field[0, 0, i, j]
            a12 = tensor_field[0, 1, i, j]
            a21 = tensor_field[1, 0, i, j]
            a22 = tensor_field[1, 1, i, j]
            
            for n in range(2):
                lambda_real = eigenvalues_real[n, i, j]
                lambda_imag = eigenvalues_imag[n, i, j]
                
                # Solve (A - λI)v = 0
                # For 2x2: [a11-λ  a12] [v1] = [0]
                #          [a21  a22-λ] [v2]   [0]
                
                if abs(lambda_imag) < EPSILON:
                    # Real eigenvalue case
                    if abs(a12) > EPSILON:
                        # Use first row: (a11 - λ)v1 + a12*v2 = 0
                        # So v2 = -(a11 - λ)*v1 / a12
                        v1 = 1.0
                        v2 = -(a11 - lambda_real) / a12
                    elif abs(a21) > EPSILON:
                        # Use second row: a21*v1 + (a22 - λ)v2 = 0
                        # So v1 = -(a22 - λ)*v2 / a21
                        v1 = -(a22 - lambda_real) / a21
                        v2 = 1.0
                    else:
                        # Diagonal case or degenerate
                        if n == 0:
                            v1, v2 = 1.0, 0.0
                        else:
                            v1, v2 = 0.0, 1.0
                    
                    # Normalize
                    norm = np.sqrt(v1*v1 + v2*v2)
                    if norm > EPSILON:
                        eigenvectors_real[n, 0, i, j] = v1 / norm
                        eigenvectors_real[n, 1, i, j] = v2 / norm
                    else:
                        eigenvectors_real[n, 0, i, j] = 1.0
                        eigenvectors_real[n, 1, i, j] = 0.0
                    
                    eigenvectors_imag[n, 0, i, j] = 0.0
                    eigenvectors_imag[n, 1, i, j] = 0.0
                    
                else:
                    # Complex eigenvalue case
                    # For complex conjugate pairs, compute one and derive the other
                    if n == 0:  # First of conjugate pair
                        # Solve (A - λI)v = 0 where λ = λr + iλi
                        # (a11 - λr - iλi)v1 + a12*v2 = 0
                        
                        if abs(a12) > EPSILON:
                            # v2 = -(a11 - λr - iλi)*v1 / a12
                            # Let v1 = 1, then v2 = -(a11 - λr - iλi) / a12
                            
                            v1_real = 1.0
                            v1_imag = 0.0
                            
                            denom = a12
                            v2_real = -(a11 - lambda_real) / denom
                            v2_imag = lambda_imag / denom
                            
                        elif abs(a21) > EPSILON:
                            # Use second row: a21*v1 + (a22 - λr - iλi)*v2 = 0
                            # v1 = -(a22 - λr - iλi)*v2 / a21
                            
                            v2_real = 1.0
                            v2_imag = 0.0
                            
                            denom = a21
                            v1_real = -(a22 - lambda_real) / denom
                            v1_imag = lambda_imag / denom
                            
                        else:
                            # Degenerate case
                            v1_real, v1_imag = 1.0, 0.0
                            v2_real, v2_imag = 0.0, 0.0
                        
                        # Normalize complex vector
                        norm_sq = v1_real*v1_real + v1_imag*v1_imag + v2_real*v2_real + v2_imag*v2_imag
                        norm = np.sqrt(norm_sq)
                        
                        if norm > EPSILON:
                            eigenvectors_real[n, 0, i, j] = v1_real / norm
                            eigenvectors_imag[n, 0, i, j] = v1_imag / norm
                            eigenvectors_real[n, 1, i, j] = v2_real / norm
                            eigenvectors_imag[n, 1, i, j] = v2_imag / norm
                        else:
                            eigenvectors_real[n, 0, i, j] = 1.0
                            eigenvectors_imag[n, 0, i, j] = 0.0
                            eigenvectors_real[n, 1, i, j] = 0.0
                            eigenvectors_imag[n, 1, i, j] = 0.0
                            
                    else:  # n == 1, second of conjugate pair
                        # Complex conjugate of the first eigenvector
                        eigenvectors_real[n, 0, i, j] = eigenvectors_real[0, 0, i, j]
                        eigenvectors_imag[n, 0, i, j] = -eigenvectors_imag[0, 0, i, j]
                        eigenvectors_real[n, 1, i, j] = eigenvectors_real[0, 1, i, j]
                        eigenvectors_imag[n, 1, i, j] = -eigenvectors_imag[0, 1, i, j]


@njit([eigenvectors_general_3x3_3d_sig_32, eigenvectors_general_3x3_3d_sig_64], 
      parallel=True, fastmath=True, cache=True)
def eigenvectors_general_3x3_nb_core(tensor_field, eigenvalues_real, eigenvalues_imag,
                                    eigenvectors_real, eigenvectors_imag):
    """
    Compute eigenvectors of general (non-symmetric) 3x3 tensors given eigenvalues.
    
    For a 3x3 matrix A with eigenvalue λ, the eigenvector v satisfies:
    (A - λI)v = 0
    
    This is more complex than 2x2 as we need to solve a 3x3 homogeneous system.
    
    Args:
        tensor_field: Input tensor (3, 3, Nx, Ny, Nz)
        eigenvalues_real: Input eigenvalues real parts (3, Nx, Ny, Nz)
        eigenvalues_imag: Input eigenvalues imaginary parts (3, Nx, Ny, Nz)
        eigenvectors_real: Output eigenvectors real parts (3, 3, Nx, Ny, Nz)
        eigenvectors_imag: Output eigenvectors imaginary parts (3, 3, Nx, Ny, Nz)
    """
    Nx, Ny, Nz = tensor_field.shape[2], tensor_field.shape[3], tensor_field.shape[4]
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Extract matrix elements
                a11 = tensor_field[0, 0, i, j, k]
                a12 = tensor_field[0, 1, i, j, k]
                a13 = tensor_field[0, 2, i, j, k]
                a21 = tensor_field[1, 0, i, j, k]
                a22 = tensor_field[1, 1, i, j, k]
                a23 = tensor_field[1, 2, i, j, k]
                a31 = tensor_field[2, 0, i, j, k]
                a32 = tensor_field[2, 1, i, j, k]
                a33 = tensor_field[2, 2, i, j, k]
                
                for n in range(3):
                    lambda_real = eigenvalues_real[n, i, j, k]
                    lambda_imag = eigenvalues_imag[n, i, j, k]
                    
                    if abs(lambda_imag) < EPSILON:
                        # Real eigenvalue case
                        # Solve (A - λI)v = 0
                        # Use cross product method for robustness
                        
                        # Create matrix (A - λI)
                        m11 = a11 - lambda_real
                        m12 = a12
                        m13 = a13
                        m21 = a21
                        m22 = a22 - lambda_real
                        m23 = a23
                        m31 = a31
                        m32 = a32
                        m33 = a33 - lambda_real
                        
                        # Use robust null space computation
                        # Strategy: For a singular matrix, find the best 2x2 submatrix and solve
                        
                        # Compute all 2x2 determinants (minors)
                        det_12_12 = m11*m22 - m12*m21  # rows 1,2, cols 1,2
                        det_12_13 = m11*m23 - m13*m21  # rows 1,2, cols 1,3
                        det_12_23 = m12*m23 - m13*m22  # rows 1,2, cols 2,3
                        
                        det_13_12 = m11*m32 - m12*m31  # rows 1,3, cols 1,2
                        det_13_13 = m11*m33 - m13*m31  # rows 1,3, cols 1,3
                        det_13_23 = m12*m33 - m13*m32  # rows 1,3, cols 2,3
                        
                        det_23_12 = m21*m32 - m22*m31  # rows 2,3, cols 1,2
                        det_23_13 = m21*m33 - m23*m31  # rows 2,3, cols 1,3
                        det_23_23 = m22*m33 - m23*m32  # rows 2,3, cols 2,3
                        
                        # Find the 2x2 submatrix with largest determinant
                        max_det = 0.0
                        max_idx = -1
                        
                        if abs(det_12_12) > max_det:
                            max_det = abs(det_12_12)
                            max_idx = 0
                        if abs(det_12_13) > max_det:
                            max_det = abs(det_12_13)
                            max_idx = 1
                        if abs(det_12_23) > max_det:
                            max_det = abs(det_12_23)
                            max_idx = 2
                        if abs(det_13_12) > max_det:
                            max_det = abs(det_13_12)
                            max_idx = 3
                        if abs(det_13_13) > max_det:
                            max_det = abs(det_13_13)
                            max_idx = 4
                        if abs(det_13_23) > max_det:
                            max_det = abs(det_13_23)
                            max_idx = 5
                        if abs(det_23_12) > max_det:
                            max_det = abs(det_23_12)
                            max_idx = 6
                        if abs(det_23_13) > max_det:
                            max_det = abs(det_23_13)
                            max_idx = 7
                        if abs(det_23_23) > max_det:
                            max_det = abs(det_23_23)
                            max_idx = 8
                        
                        # Based on which 2x2 submatrix is best conditioned, compute null vector
                        if max_det < EPSILON:
                            # All 2x2 submatrices are singular - special handling
                            # Check if matrix is nearly diagonal
                            is_diagonal = (abs(m12) < EPSILON and abs(m13) < EPSILON and 
                                         abs(m21) < EPSILON and abs(m23) < EPSILON and 
                                         abs(m31) < EPSILON and abs(m32) < EPSILON)
                            
                            if is_diagonal:
                                # For diagonal (A - λI), eigenvector has 1 in position of smallest diagonal
                                if abs(m11) <= abs(m22) and abs(m11) <= abs(m33):
                                    v1, v2, v3 = 1.0, 0.0, 0.0
                                elif abs(m22) <= abs(m33):
                                    v1, v2, v3 = 0.0, 1.0, 0.0
                                else:
                                    v1, v2, v3 = 0.0, 0.0, 1.0
                            else:
                                # Use row with smallest norm
                                norm1 = m11*m11 + m12*m12 + m13*m13
                                norm2 = m21*m21 + m22*m22 + m23*m23
                                norm3 = m31*m31 + m32*m32 + m33*m33
                                
                                if norm1 <= norm2 and norm1 <= norm3:
                                    if abs(m12) > EPSILON:
                                        v1, v2, v3 = m12, -m11, 0.0
                                    elif abs(m13) > EPSILON:
                                        v1, v2, v3 = m13, 0.0, -m11
                                    else:
                                        v1, v2, v3 = 0.0, 1.0, 0.0
                                elif norm2 <= norm3:
                                    if abs(m22) > EPSILON:
                                        v1, v2, v3 = m22, -m21, 0.0
                                    elif abs(m23) > EPSILON:
                                        v1, v2, v3 = 0.0, m23, -m22
                                    else:
                                        v1, v2, v3 = 1.0, 0.0, 0.0
                                else:
                                    if abs(m32) > EPSILON:
                                        v1, v2, v3 = m32, -m31, 0.0
                                    elif abs(m33) > EPSILON:
                                        v1, v2, v3 = 0.0, m33, -m32
                                    else:
                                        v1, v2, v3 = 1.0, 0.0, 0.0
                        else:
                            # Use the best 2x2 submatrix to find null vector
                            if max_idx == 0:
                                # rows 1,2, cols 1,2 - set v3 free
                                v3 = 1.0
                                rhs1 = -m13 * v3
                                rhs2 = -m23 * v3
                                v1 = (rhs1*m22 - rhs2*m12) / det_12_12
                                v2 = (rhs2*m11 - rhs1*m21) / det_12_12
                                
                            elif max_idx == 1:
                                # rows 1,2, cols 1,3 - set v2 free
                                v2 = 1.0
                                rhs1 = -m12 * v2
                                rhs2 = -m22 * v2
                                v1 = (rhs1*m23 - rhs2*m13) / det_12_13
                                v3 = (rhs2*m11 - rhs1*m21) / det_12_13
                                
                            elif max_idx == 2:
                                # rows 1,2, cols 2,3 - set v1 free
                                v1 = 1.0
                                rhs1 = -m11 * v1
                                rhs2 = -m21 * v1
                                v2 = (rhs1*m23 - rhs2*m13) / det_12_23
                                v3 = (rhs2*m12 - rhs1*m22) / det_12_23
                                
                            elif max_idx == 3:
                                # rows 1,3, cols 1,2 - set v3 free
                                v3 = 1.0
                                rhs1 = -m13 * v3
                                rhs3 = -m33 * v3
                                v1 = (rhs1*m32 - rhs3*m12) / det_13_12
                                v2 = (rhs3*m11 - rhs1*m31) / det_13_12
                                
                            elif max_idx == 4:
                                # rows 1,3, cols 1,3 - set v2 free
                                v2 = 1.0
                                rhs1 = -m12 * v2
                                rhs3 = -m32 * v2
                                v1 = (rhs1*m33 - rhs3*m13) / det_13_13
                                v3 = (rhs3*m11 - rhs1*m31) / det_13_13
                                
                            elif max_idx == 5:
                                # rows 1,3, cols 2,3 - set v1 free
                                v1 = 1.0
                                rhs1 = -m11 * v1
                                rhs3 = -m31 * v1
                                v2 = (rhs1*m33 - rhs3*m13) / det_13_23
                                v3 = (rhs3*m12 - rhs1*m32) / det_13_23
                                
                            elif max_idx == 6:
                                # rows 2,3, cols 1,2 - set v3 free
                                v3 = 1.0
                                rhs2 = -m23 * v3
                                rhs3 = -m33 * v3
                                v1 = (rhs2*m32 - rhs3*m22) / det_23_12
                                v2 = (rhs3*m21 - rhs2*m31) / det_23_12
                                
                            elif max_idx == 7:
                                # rows 2,3, cols 1,3 - set v2 free
                                v2 = 1.0
                                rhs2 = -m22 * v2
                                rhs3 = -m32 * v2
                                v1 = (rhs2*m33 - rhs3*m23) / det_23_13
                                v3 = (rhs3*m21 - rhs2*m31) / det_23_13
                                
                            else:  # max_idx == 8
                                # rows 2,3, cols 2,3 - set v1 free
                                v1 = 1.0
                                rhs2 = -m21 * v1
                                rhs3 = -m31 * v1
                                v2 = (rhs2*m33 - rhs3*m23) / det_23_23
                                v3 = (rhs3*m22 - rhs2*m32) / det_23_23
                        
                        # Normalize
                        norm = np.sqrt(v1*v1 + v2*v2 + v3*v3)
                        if norm > EPSILON:
                            eigenvectors_real[n, 0, i, j, k] = v1 / norm
                            eigenvectors_real[n, 1, i, j, k] = v2 / norm
                            eigenvectors_real[n, 2, i, j, k] = v3 / norm
                        else:
                            # Degenerate case - use standard basis vector
                            if n == 0:
                                eigenvectors_real[n, 0, i, j, k] = 1.0
                                eigenvectors_real[n, 1, i, j, k] = 0.0
                                eigenvectors_real[n, 2, i, j, k] = 0.0
                            elif n == 1:
                                eigenvectors_real[n, 0, i, j, k] = 0.0
                                eigenvectors_real[n, 1, i, j, k] = 1.0
                                eigenvectors_real[n, 2, i, j, k] = 0.0
                            else:
                                eigenvectors_real[n, 0, i, j, k] = 0.0
                                eigenvectors_real[n, 1, i, j, k] = 0.0
                                eigenvectors_real[n, 2, i, j, k] = 1.0
                        
                        eigenvectors_imag[n, 0, i, j, k] = 0.0
                        eigenvectors_imag[n, 1, i, j, k] = 0.0
                        eigenvectors_imag[n, 2, i, j, k] = 0.0
                        
                    else:
                        # Complex eigenvalue case
                        # For complex conjugate pairs, we compute one and derive the other
                        
                        # Check if this is part of a conjugate pair
                        # For a typical 3x3 matrix with complex eigenvalues, we have:
                        # λ0 = real eigenvalue
                        # λ1 = a + bi (complex)  
                        # λ2 = a - bi (conjugate)
                        
                        # We need to identify which eigenvalue we're processing
                        is_conjugate_pair = False
                        conjugate_index = -1
                        
                        # Check if there's a conjugate eigenvalue
                        for nn in range(3):
                            if nn != n:
                                other_real = eigenvalues_real[nn, i, j, k]
                                other_imag = eigenvalues_imag[nn, i, j, k]
                                
                                # Check if it's a conjugate
                                real_diff = abs(lambda_real - other_real)
                                imag_sum = abs(lambda_imag + other_imag)
                                
                                if real_diff < EPSILON and imag_sum < EPSILON and abs(lambda_imag) > EPSILON:
                                    is_conjugate_pair = True
                                    conjugate_index = nn
                                    break
                        
                        if is_conjugate_pair and n > conjugate_index:
                            # This is the second eigenvalue of a conjugate pair
                            # Use the conjugate of the first eigenvector
                            eigenvectors_real[n, 0, i, j, k] = eigenvectors_real[conjugate_index, 0, i, j, k]
                            eigenvectors_imag[n, 0, i, j, k] = -eigenvectors_imag[conjugate_index, 0, i, j, k]
                            eigenvectors_real[n, 1, i, j, k] = eigenvectors_real[conjugate_index, 1, i, j, k]
                            eigenvectors_imag[n, 1, i, j, k] = -eigenvectors_imag[conjugate_index, 1, i, j, k]
                            eigenvectors_real[n, 2, i, j, k] = eigenvectors_real[conjugate_index, 2, i, j, k]
                            eigenvectors_imag[n, 2, i, j, k] = -eigenvectors_imag[conjugate_index, 2, i, j, k]
                        else:
                            # Compute the eigenvector using complex null space method
                            # Real parts of (A - λ_r*I)
                            mr11 = a11 - lambda_real
                            mr12 = a12
                            mr13 = a13
                            mr21 = a21
                            mr22 = a22 - lambda_real
                            mr23 = a23
                            mr31 = a31
                            mr32 = a32
                            mr33 = a33 - lambda_real
                            
                            # Imaginary parts: only diagonal is non-zero (-λ_i)
                            mi11, mi12, mi13 = -lambda_imag, 0.0, 0.0
                            mi21, mi22, mi23 = 0.0, -lambda_imag, 0.0
                            mi31, mi32, mi33 = 0.0, 0.0, -lambda_imag
                            
                            # Compute all 9 complex 2x2 determinants and find the best one
                            determinants = []
                            
                            # rows 1,2, cols 1,2
                            det_r = mr11*mr22 - mr12*mr21 - mi11*mi22 + mi12*mi21
                            det_i = mr11*mi22 + mi11*mr22 - mr12*mi21 - mi12*mr21
                            det_mag = det_r*det_r + det_i*det_i
                            determinants.append((0, det_r, det_i, det_mag))
                            
                            # rows 1,2, cols 1,3
                            det_r = mr11*mr23 - mr13*mr21 - mi11*mi23 + mi13*mi21
                            det_i = mr11*mi23 + mi11*mr23 - mr13*mi21 - mi13*mr21
                            det_mag = det_r*det_r + det_i*det_i
                            determinants.append((1, det_r, det_i, det_mag))
                            
                            # rows 1,2, cols 2,3
                            det_r = mr12*mr23 - mr13*mr22 - mi12*mi23 + mi13*mi22
                            det_i = mr12*mi23 + mi12*mr23 - mr13*mi22 - mi13*mr22
                            det_mag = det_r*det_r + det_i*det_i
                            determinants.append((2, det_r, det_i, det_mag))
                            
                            # rows 1,3, cols 1,2
                            det_r = mr11*mr32 - mr12*mr31 - mi11*mi32 + mi12*mi31
                            det_i = mr11*mi32 + mi11*mr32 - mr12*mi31 - mi12*mr31
                            det_mag = det_r*det_r + det_i*det_i
                            determinants.append((3, det_r, det_i, det_mag))
                            
                            # rows 1,3, cols 1,3
                            det_r = mr11*mr33 - mr13*mr31 - mi11*mi33 + mi13*mi31
                            det_i = mr11*mi33 + mi11*mr33 - mr13*mi31 - mi13*mr31
                            det_mag = det_r*det_r + det_i*det_i
                            determinants.append((4, det_r, det_i, det_mag))
                            
                            # rows 1,3, cols 2,3
                            det_r = mr12*mr33 - mr13*mr32 - mi12*mi33 + mi13*mi32
                            det_i = mr12*mi33 + mi12*mr33 - mr13*mi32 - mi13*mr32
                            det_mag = det_r*det_r + det_i*det_i
                            determinants.append((5, det_r, det_i, det_mag))
                            
                            # rows 2,3, cols 1,2
                            det_r = mr21*mr32 - mr22*mr31 - mi21*mi32 + mi22*mi31
                            det_i = mr21*mi32 + mi21*mr32 - mr22*mi31 - mi22*mr31
                            det_mag = det_r*det_r + det_i*det_i
                            determinants.append((6, det_r, det_i, det_mag))
                            
                            # rows 2,3, cols 1,3
                            det_r = mr21*mr33 - mr23*mr31 - mi21*mi33 + mi23*mi31
                            det_i = mr21*mi33 + mi21*mr33 - mr23*mi31 - mi23*mr31
                            det_mag = det_r*det_r + det_i*det_i
                            determinants.append((7, det_r, det_i, det_mag))
                            
                            # rows 2,3, cols 2,3
                            det_r = mr22*mr33 - mr23*mr32 - mi22*mi33 + mi23*mi32
                            det_i = mr22*mi33 + mi22*mr33 - mr23*mi32 - mi23*mr32
                            det_mag = det_r*det_r + det_i*det_i
                            determinants.append((8, det_r, det_i, det_mag))
                            
                            # Find the best determinant
                            best_idx = 0
                            best_mag = determinants[0][3]
                            for idx in range(1, 9):
                                if determinants[idx][3] > best_mag:
                                    best_mag = determinants[idx][3]
                                    best_idx = idx
                            
                            best_det_idx, best_det_r, best_det_i, best_det_mag = determinants[best_idx]
                            
                            if best_det_mag > EPSILON*EPSILON:
                                # Use the best 2x2 submatrix
                                det_inv_r = best_det_r / best_det_mag
                                det_inv_i = -best_det_i / best_det_mag
                                
                                if best_det_idx == 0:
                                    # rows 1,2, cols 1,2 - set v3 free
                                    v3r, v3i = 1.0, 0.0
                                    rhs1r = -mr13 * v3r + mi13 * v3i
                                    rhs1i = -mr13 * v3i - mi13 * v3r
                                    rhs2r = -mr23 * v3r + mi23 * v3i
                                    rhs2i = -mr23 * v3i - mi23 * v3r
                                    
                                    temp1r = rhs1r*mr22 - rhs1i*mi22 - rhs2r*mr12 + rhs2i*mi12
                                    temp1i = rhs1r*mi22 + rhs1i*mr22 - rhs2r*mi12 - rhs2i*mr12
                                    v1r = temp1r*det_inv_r - temp1i*det_inv_i
                                    v1i = temp1r*det_inv_i + temp1i*det_inv_r
                                    
                                    temp2r = rhs2r*mr11 - rhs2i*mi11 - rhs1r*mr21 + rhs1i*mi21
                                    temp2i = rhs2r*mi11 + rhs2i*mr11 - rhs1r*mi21 - rhs1i*mr21
                                    v2r = temp2r*det_inv_r - temp2i*det_inv_i
                                    v2i = temp2r*det_inv_i + temp2i*det_inv_r
                                    
                                elif best_det_idx == 1:
                                    # rows 1,2, cols 1,3 - set v2 free
                                    v2r, v2i = 1.0, 0.0
                                    rhs1r = -mr12 * v2r + mi12 * v2i
                                    rhs1i = -mr12 * v2i - mi12 * v2r
                                    rhs2r = -mr22 * v2r + mi22 * v2i
                                    rhs2i = -mr22 * v2i - mi22 * v2r
                                    
                                    temp1r = rhs1r*mr23 - rhs1i*mi23 - rhs2r*mr13 + rhs2i*mi13
                                    temp1i = rhs1r*mi23 + rhs1i*mr23 - rhs2r*mi13 - rhs2i*mr13
                                    v1r = temp1r*det_inv_r - temp1i*det_inv_i
                                    v1i = temp1r*det_inv_i + temp1i*det_inv_r
                                    
                                    temp3r = rhs2r*mr11 - rhs2i*mi11 - rhs1r*mr21 + rhs1i*mi21
                                    temp3i = rhs2r*mi11 + rhs2i*mr11 - rhs1r*mi21 - rhs1i*mr21
                                    v3r = temp3r*det_inv_r - temp3i*det_inv_i
                                    v3i = temp3r*det_inv_i + temp3i*det_inv_r
                                    
                                elif best_det_idx == 2:
                                    # rows 1,2, cols 2,3 - set v1 free
                                    v1r, v1i = 1.0, 0.0
                                    rhs1r = -mr11 * v1r + mi11 * v1i
                                    rhs1i = -mr11 * v1i - mi11 * v1r
                                    rhs2r = -mr21 * v1r + mi21 * v1i
                                    rhs2i = -mr21 * v1i - mi21 * v1r
                                    
                                    temp2r = rhs1r*mr23 - rhs1i*mi23 - rhs2r*mr13 + rhs2i*mi13
                                    temp2i = rhs1r*mi23 + rhs1i*mr23 - rhs2r*mi13 - rhs2i*mr13
                                    v2r = temp2r*det_inv_r - temp2i*det_inv_i
                                    v2i = temp2r*det_inv_i + temp2i*det_inv_r
                                    
                                    temp3r = rhs2r*mr12 - rhs2i*mi12 - rhs1r*mr22 + rhs1i*mi22
                                    temp3i = rhs2r*mi12 + rhs2i*mr12 - rhs1r*mi22 - rhs1i*mr22
                                    v3r = temp3r*det_inv_r - temp3i*det_inv_i
                                    v3i = temp3r*det_inv_i + temp3i*det_inv_r
                                    
                                elif best_det_idx == 3:
                                    # rows 1,3, cols 1,2 - set v3 free
                                    v3r, v3i = 1.0, 0.0
                                    rhs1r = -mr13 * v3r + mi13 * v3i
                                    rhs1i = -mr13 * v3i - mi13 * v3r
                                    rhs3r = -mr33 * v3r + mi33 * v3i
                                    rhs3i = -mr33 * v3i - mi33 * v3r
                                    
                                    temp1r = rhs1r*mr32 - rhs1i*mi32 - rhs3r*mr12 + rhs3i*mi12
                                    temp1i = rhs1r*mi32 + rhs1i*mr32 - rhs3r*mi12 - rhs3i*mr12
                                    v1r = temp1r*det_inv_r - temp1i*det_inv_i
                                    v1i = temp1r*det_inv_i + temp1i*det_inv_r
                                    
                                    temp2r = rhs3r*mr11 - rhs3i*mi11 - rhs1r*mr31 + rhs1i*mi31
                                    temp2i = rhs3r*mi11 + rhs3i*mr11 - rhs1r*mi31 - rhs1i*mr31
                                    v2r = temp2r*det_inv_r - temp2i*det_inv_i
                                    v2i = temp2r*det_inv_i + temp2i*det_inv_r
                                    
                                elif best_det_idx == 4:
                                    # rows 1,3, cols 1,3 - set v2 free
                                    v2r, v2i = 1.0, 0.0
                                    rhs1r = -mr12 * v2r + mi12 * v2i
                                    rhs1i = -mr12 * v2i - mi12 * v2r
                                    rhs3r = -mr32 * v2r + mi32 * v2i
                                    rhs3i = -mr32 * v2i - mi32 * v2r
                                    
                                    temp1r = rhs1r*mr33 - rhs1i*mi33 - rhs3r*mr13 + rhs3i*mi13
                                    temp1i = rhs1r*mi33 + rhs1i*mr33 - rhs3r*mi13 - rhs3i*mr13
                                    v1r = temp1r*det_inv_r - temp1i*det_inv_i
                                    v1i = temp1r*det_inv_i + temp1i*det_inv_r
                                    
                                    temp3r = rhs3r*mr11 - rhs3i*mi11 - rhs1r*mr31 + rhs1i*mi31
                                    temp3i = rhs3r*mi11 + rhs3i*mr11 - rhs1r*mi31 - rhs1i*mr31
                                    v3r = temp3r*det_inv_r - temp3i*det_inv_i
                                    v3i = temp3r*det_inv_i + temp3i*det_inv_r
                                    
                                elif best_det_idx == 5:
                                    # rows 1,3, cols 2,3 - set v1 free
                                    v1r, v1i = 1.0, 0.0
                                    rhs1r = -mr11 * v1r + mi11 * v1i
                                    rhs1i = -mr11 * v1i - mi11 * v1r
                                    rhs3r = -mr31 * v1r + mi31 * v1i
                                    rhs3i = -mr31 * v1i - mi31 * v1r
                                    
                                    temp2r = rhs1r*mr33 - rhs1i*mi33 - rhs3r*mr13 + rhs3i*mi13
                                    temp2i = rhs1r*mi33 + rhs1i*mr33 - rhs3r*mi13 - rhs3i*mr13
                                    v2r = temp2r*det_inv_r - temp2i*det_inv_i
                                    v2i = temp2r*det_inv_i + temp2i*det_inv_r
                                    
                                    temp3r = rhs3r*mr12 - rhs3i*mi12 - rhs1r*mr32 + rhs1i*mi32
                                    temp3i = rhs3r*mi12 + rhs3i*mr12 - rhs1r*mi32 - rhs1i*mr32
                                    v3r = temp3r*det_inv_r - temp3i*det_inv_i
                                    v3i = temp3r*det_inv_i + temp3i*det_inv_r
                                    
                                elif best_det_idx == 6:
                                    # rows 2,3, cols 1,2 - set v3 free
                                    v3r, v3i = 1.0, 0.0
                                    rhs2r = -mr23 * v3r + mi23 * v3i
                                    rhs2i = -mr23 * v3i - mi23 * v3r
                                    rhs3r = -mr33 * v3r + mi33 * v3i
                                    rhs3i = -mr33 * v3i - mi33 * v3r
                                    
                                    temp1r = rhs2r*mr32 - rhs2i*mi32 - rhs3r*mr22 + rhs3i*mi22
                                    temp1i = rhs2r*mi32 + rhs2i*mr32 - rhs3r*mi22 - rhs3i*mr22
                                    v1r = temp1r*det_inv_r - temp1i*det_inv_i
                                    v1i = temp1r*det_inv_i + temp1i*det_inv_r
                                    
                                    temp2r = rhs3r*mr21 - rhs3i*mi21 - rhs2r*mr31 + rhs2i*mi31
                                    temp2i = rhs3r*mi21 + rhs3i*mr21 - rhs2r*mi31 - rhs2i*mr31
                                    v2r = temp2r*det_inv_r - temp2i*det_inv_i
                                    v2i = temp2r*det_inv_i + temp2i*det_inv_r
                                    
                                elif best_det_idx == 7:
                                    # rows 2,3, cols 1,3 - set v2 free
                                    v2r, v2i = 1.0, 0.0
                                    rhs2r = -mr22 * v2r + mi22 * v2i
                                    rhs2i = -mr22 * v2i - mi22 * v2r
                                    rhs3r = -mr32 * v2r + mi32 * v2i
                                    rhs3i = -mr32 * v2i - mi32 * v2r
                                    
                                    temp1r = rhs2r*mr33 - rhs2i*mi33 - rhs3r*mr23 + rhs3i*mi23
                                    temp1i = rhs2r*mi33 + rhs2i*mr33 - rhs3r*mi23 - rhs3i*mr23
                                    v1r = temp1r*det_inv_r - temp1i*det_inv_i
                                    v1i = temp1r*det_inv_i + temp1i*det_inv_r
                                    
                                    temp3r = rhs3r*mr21 - rhs3i*mi21 - rhs2r*mr31 + rhs2i*mi31
                                    temp3i = rhs3r*mi21 + rhs3i*mr21 - rhs2r*mi31 - rhs2i*mr31
                                    v3r = temp3r*det_inv_r - temp3i*det_inv_i
                                    v3i = temp3r*det_inv_i + temp3i*det_inv_r
                                    
                                else:  # best_det_idx == 8
                                    # rows 2,3, cols 2,3 - set v1 free
                                    v1r, v1i = 1.0, 0.0
                                    rhs2r = -mr21 * v1r + mi21 * v1i
                                    rhs2i = -mr21 * v1i - mi21 * v1r
                                    rhs3r = -mr31 * v1r + mi31 * v1i
                                    rhs3i = -mr31 * v1i - mi31 * v1r
                                    
                                    temp2r = rhs2r*mr33 - rhs2i*mi33 - rhs3r*mr23 + rhs3i*mi23
                                    temp2i = rhs2r*mi33 + rhs2i*mr33 - rhs3r*mi23 - rhs3i*mr23
                                    v2r = temp2r*det_inv_r - temp2i*det_inv_i
                                    v2i = temp2r*det_inv_i + temp2i*det_inv_r
                                    
                                    temp3r = rhs3r*mr22 - rhs3i*mi22 - rhs2r*mr32 + rhs2i*mi32
                                    temp3i = rhs3r*mi22 + rhs3i*mr22 - rhs2r*mi32 - rhs2i*mr32
                                    v3r = temp3r*det_inv_r - temp3i*det_inv_i
                                    v3i = temp3r*det_inv_i + temp3i*det_inv_r
                            else:
                                # All determinants are too small - use fallback
                                v1r, v1i = 1.0, 0.0
                                v2r, v2i = 0.0, 0.5 * lambda_imag / abs(lambda_imag) if abs(lambda_imag) > EPSILON else 0.0
                                v3r, v3i = 0.0, -0.5 * lambda_imag / abs(lambda_imag) if abs(lambda_imag) > EPSILON else 0.0
                            
                            # Normalize the complex eigenvector
                            norm_sq = v1r*v1r + v1i*v1i + v2r*v2r + v2i*v2i + v3r*v3r + v3i*v3i
                            if norm_sq > EPSILON:
                                norm = np.sqrt(norm_sq)
                                eigenvectors_real[n, 0, i, j, k] = v1r / norm
                                eigenvectors_imag[n, 0, i, j, k] = v1i / norm
                                eigenvectors_real[n, 1, i, j, k] = v2r / norm
                                eigenvectors_imag[n, 1, i, j, k] = v2i / norm
                                eigenvectors_real[n, 2, i, j, k] = v3r / norm
                                eigenvectors_imag[n, 2, i, j, k] = v3i / norm
                            else:
                                # Fallback
                                eigenvectors_real[n, 0, i, j, k] = 1.0
                                eigenvectors_real[n, 1, i, j, k] = 0.0
                                eigenvectors_real[n, 2, i, j, k] = 0.0
                                eigenvectors_imag[n, 0, i, j, k] = 0.0
                                eigenvectors_imag[n, 1, i, j, k] = 0.0
                                eigenvectors_imag[n, 2, i, j, k] = 0.0

