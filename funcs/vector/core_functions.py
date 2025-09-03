from numba import njit, prange
import numpy as np
from .constants import *

##########################################################################################
# Core numba JIT functions for vector operations
##########################################################################################

@njit([sig_dot_3d_f32, sig_dot_3d_f64], parallel=True, fastmath=True, cache=True)
def vector_dot_product_3D_nb_core(
    vec1, 
    vec2):
    """
    Compute dot product of two 3D vector fields
    vec1, vec2: shape (3, Nx, Ny, Nz)
    returns: shape (Nx, Ny, Nz)
    """
    Nx, Ny, Nz = vec1.shape[1], vec1.shape[2], vec1.shape[3]
    out = np.zeros((Nx, Ny, Nz), dtype=vec1.dtype)
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                out[i, j, k] = (vec1[0, i, j, k] * vec2[0, i, j, k] +
                               vec1[1, i, j, k] * vec2[1, i, j, k] +
                               vec1[2, i, j, k] * vec2[2, i, j, k])
    
    return out


@njit([sig_dot_2d_f32, sig_dot_2d_f64], parallel=True, fastmath=True, cache=True)
def vector_dot_product_2D_nb_core(
    vec1,
    vec2):
    """
    Compute dot product of two 2D vector fields
    vec1, vec2: shape (2, Nx, Ny)
    returns: shape (Nx, Ny)
    """
    Nx, Ny = vec1.shape[1], vec1.shape[2]
    out = np.zeros((Nx, Ny), dtype=vec1.dtype)
    
    for i in prange(Nx):
        for j in range(Ny):
            out[i, j] = (vec1[0, i, j] * vec2[0, i, j] +
                        vec1[1, i, j] * vec2[1, i, j])
    
    return out


@njit([sig_mag_3d_f32, sig_mag_3d_f64], parallel=True, fastmath=True, cache=True)
def vector_magnitude_3D_nb_core(
    vec):
    """
    Compute magnitude of 3D vector field
    vec: shape (3, Nx, Ny, Nz)
    returns: shape (Nx, Ny, Nz)
    """
    Nx, Ny, Nz = vec.shape[1], vec.shape[2], vec.shape[3]
    out = np.zeros((Nx, Ny, Nz), dtype=vec.dtype)
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                out[i, j, k] = np.sqrt(vec[0, i, j, k]**2 + 
                                      vec[1, i, j, k]**2 + 
                                      vec[2, i, j, k]**2)
    
    return out


@njit([sig_mag_2d_f32, sig_mag_2d_f64], parallel=True, fastmath=True, cache=True)
def vector_magnitude_2D_nb_core(
    vec):
    """
    Compute magnitude of 2D vector field
    vec: shape (2, Nx, Ny)
    returns: shape (Nx, Ny)
    """
    Nx, Ny = vec.shape[1], vec.shape[2]
    out = np.zeros((Nx, Ny), dtype=vec.dtype)
    
    for i in prange(Nx):
        for j in range(Ny):
            out[i, j] = np.sqrt(vec[0, i, j]**2 + vec[1, i, j]**2)
    
    return out


@njit([sig_cross_3d_f32, sig_cross_3d_f64], parallel=True, fastmath=True, cache=True)
def vector_cross_product_3D_nb_core(
    vec1, 
    vec2):
    """
    Compute cross product of two 3D vector fields
    vec1, vec2: shape (3, Nx, Ny, Nz)
    returns: shape (3, Nx, Ny, Nz)
    """
    Nx, Ny, Nz = vec1.shape[1], vec1.shape[2], vec1.shape[3]
    out = np.zeros_like(vec1)
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Component 0: v1_y * v2_z - v1_z * v2_y
                out[0, i, j, k] = (vec1[1, i, j, k] * vec2[2, i, j, k] - 
                                  vec1[2, i, j, k] * vec2[1, i, j, k])
                
                # Component 1: v1_z * v2_x - v1_x * v2_z
                out[1, i, j, k] = (vec1[2, i, j, k] * vec2[0, i, j, k] - 
                                  vec1[0, i, j, k] * vec2[2, i, j, k])
                
                # Component 2: v1_x * v2_y - v1_y * v2_x
                out[2, i, j, k] = (vec1[0, i, j, k] * vec2[1, i, j, k] - 
                                  vec1[1, i, j, k] * vec2[0, i, j, k])
    
    return out


@njit([sig_cross_2d_f32, sig_cross_2d_f64], parallel=True, fastmath=True, cache=True)
def vector_cross_product_2D_nb_core(
    vec1, 
    vec2):
    """
    Compute cross product of two 2D vector fields (returns scalar)
    vec1, vec2: shape (2, Nx, Ny)
    returns: shape (Nx, Ny)
    """
    Nx, Ny = vec1.shape[1], vec1.shape[2]
    out = np.zeros((Nx, Ny), dtype=vec1.dtype)
    
    for i in prange(Nx):
        for j in range(Ny):
            out[i, j] = vec1[0, i, j] * vec2[1, i, j] - vec1[1, i, j] * vec2[0, i, j]
    
    return out


@njit([sig_norm_3d_f32, sig_norm_3d_f64], parallel=True, fastmath=True, cache=True)
def vector_normalize_3D_nb_core(
    vec, 
    epsilon):
    """
    Normalize a 3D vector field
    vec: shape (3, Nx, Ny, Nz)
    epsilon: small value to avoid division by zero
    returns: shape (3, Nx, Ny, Nz)
    """
    Nx, Ny, Nz = vec.shape[1], vec.shape[2], vec.shape[3]
    out = np.zeros_like(vec)
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                mag = np.sqrt(vec[0, i, j, k]**2 + 
                             vec[1, i, j, k]**2 + 
                             vec[2, i, j, k]**2)
                
                if mag > epsilon:
                    inv_mag = 1.0 / mag
                    out[0, i, j, k] = vec[0, i, j, k] * inv_mag
                    out[1, i, j, k] = vec[1, i, j, k] * inv_mag
                    out[2, i, j, k] = vec[2, i, j, k] * inv_mag
                else:
                    out[0, i, j, k] = 0.0
                    out[1, i, j, k] = 0.0
                    out[2, i, j, k] = 0.0
    
    return out


@njit([sig_norm_2d_f32, sig_norm_2d_f64], parallel=True, fastmath=True, cache=True)
def vector_normalize_2D_nb_core(
    vec, 
    epsilon):
    """
    Normalize a 2D vector field
    vec: shape (2, Nx, Ny)
    epsilon: small value to avoid division by zero
    returns: shape (2, Nx, Ny)
    """
    Nx, Ny = vec.shape[1], vec.shape[2]
    out = np.zeros_like(vec)
    
    for i in prange(Nx):
        for j in range(Ny):
            mag = np.sqrt(vec[0, i, j]**2 + vec[1, i, j]**2)
            
            if mag > epsilon:
                inv_mag = 1.0 / mag
                out[0, i, j] = vec[0, i, j] * inv_mag
                out[1, i, j] = vec[1, i, j] * inv_mag
            else:
                out[0, i, j] = 0.0
                out[1, i, j] = 0.0
    
    return out


@njit([sig_triple_3d_f32, sig_triple_3d_f64], parallel=True, fastmath=True, cache=True)
def vector_triple_product_3D_nb_core(
    vec1, 
    vec2, 
    vec3):
    """
    Compute scalar triple product: vec1 · (vec2 × vec3)
    vec1, vec2, vec3: shape (3, Nx, Ny, Nz)
    returns: shape (Nx, Ny, Nz)
    """
    Nx, Ny, Nz = vec1.shape[1], vec1.shape[2], vec1.shape[3]
    out = np.zeros((Nx, Ny, Nz), dtype=vec1.dtype)
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Compute vec2 × vec3
                cross_x = vec2[1, i, j, k] * vec3[2, i, j, k] - vec2[2, i, j, k] * vec3[1, i, j, k]
                cross_y = vec2[2, i, j, k] * vec3[0, i, j, k] - vec2[0, i, j, k] * vec3[2, i, j, k]
                cross_z = vec2[0, i, j, k] * vec3[1, i, j, k] - vec2[1, i, j, k] * vec3[0, i, j, k]
                
                # Dot with vec1
                out[i, j, k] = (vec1[0, i, j, k] * cross_x + 
                               vec1[1, i, j, k] * cross_y + 
                               vec1[2, i, j, k] * cross_z)
    
    return out


@njit([sig_angle_3d_f32, sig_angle_3d_f64], parallel=True, fastmath=True, cache=True)
def vector_angle_3D_nb_core(
    vec1, 
    vec2, 
    epsilon):
    """
    Compute angle between two 3D vector fields (in radians)
    vec1, vec2: shape (3, Nx, Ny, Nz)
    epsilon: small value to avoid division by zero
    returns: shape (Nx, Ny, Nz)
    """
    Nx, Ny, Nz = vec1.shape[1], vec1.shape[2], vec1.shape[3]
    out = np.zeros((Nx, Ny, Nz), dtype=vec1.dtype)
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Compute magnitudes
                mag1 = np.sqrt(vec1[0, i, j, k]**2 + vec1[1, i, j, k]**2 + vec1[2, i, j, k]**2)
                mag2 = np.sqrt(vec2[0, i, j, k]**2 + vec2[1, i, j, k]**2 + vec2[2, i, j, k]**2)
                
                if mag1 > epsilon and mag2 > epsilon:
                    # Compute dot product
                    dot = (vec1[0, i, j, k] * vec2[0, i, j, k] +
                          vec1[1, i, j, k] * vec2[1, i, j, k] +
                          vec1[2, i, j, k] * vec2[2, i, j, k])
                    
                    # Compute angle (clamp to avoid numerical issues)
                    cos_angle = dot / (mag1 * mag2)
                    # Clamp between -1 and 1
                    if cos_angle > 1.0:
                        cos_angle = 1.0
                    elif cos_angle < -1.0:
                        cos_angle = -1.0
                    
                    out[i, j, k] = np.arccos(cos_angle)
                else:
                    out[i, j, k] = 0.0
    
    return out


@njit([sig_angle_2d_f32, sig_angle_2d_f64], parallel=True, fastmath=True, cache=True)
def vector_angle_2D_nb_core(
    vec1, 
    vec2, 
    epsilon):
    """
    Compute angle between two 2D vector fields (in radians)
    vec1, vec2: shape (2, Nx, Ny)
    epsilon: small value to avoid division by zero
    returns: shape (Nx, Ny)
    """
    Nx, Ny = vec1.shape[1], vec1.shape[2]
    out = np.zeros((Nx, Ny), dtype=vec1.dtype)
    
    for i in prange(Nx):
        for j in range(Ny):
            # Compute magnitudes
            mag1 = np.sqrt(vec1[0, i, j]**2 + vec1[1, i, j]**2)
            mag2 = np.sqrt(vec2[0, i, j]**2 + vec2[1, i, j]**2)
            
            if mag1 > epsilon and mag2 > epsilon:
                # Compute dot product
                dot = vec1[0, i, j] * vec2[0, i, j] + vec1[1, i, j] * vec2[1, i, j]
                
                # Compute angle (clamp to avoid numerical issues)
                cos_angle = dot / (mag1 * mag2)
                if cos_angle > 1.0:
                    cos_angle = 1.0
                elif cos_angle < -1.0:
                    cos_angle = -1.0
                
                out[i, j] = np.arccos(cos_angle)
            else:
                out[i, j] = 0.0
    
    return out


@njit([sig_extract_component_3d_f32, sig_extract_component_3d_f64], parallel=True, fastmath=True, cache=True)
def extract_vector_component_3D_nb_core(
    vec, 
    component):
    """
    Extract a single component from a vector field
    vec: shape (3, Nx, Ny, Nz)
    component: 0, 1, or 2
    returns: shape (Nx, Ny, Nz)
    """
    Nx, Ny, Nz = vec.shape[1], vec.shape[2], vec.shape[3]
    out = np.zeros((Nx, Ny, Nz), dtype=vec.dtype)
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                out[i, j, k] = vec[component, i, j, k]
    
    return out


@njit([sig_project_3d_f32, sig_project_3d_f64], parallel=True, fastmath=True, cache=True)
def vector_projection_3D_nb_core(
    vec_a, 
    vec_b):
    """
    Project vector A onto vector B: proj_B(A) = (A·B/|B|²) * B
    vec_a, vec_b: shape (3, Nx, Ny, Nz)
    returns: shape (3, Nx, Ny, Nz)
    """
    Nx, Ny, Nz = vec_a.shape[1], vec_a.shape[2], vec_a.shape[3]
    out = np.zeros_like(vec_a)
    epsilon = 1e-10
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Compute B·B
                b_dot_b = (vec_b[0, i, j, k]**2 + 
                          vec_b[1, i, j, k]**2 + 
                          vec_b[2, i, j, k]**2)
                
                if b_dot_b > epsilon:
                    # Compute A·B
                    a_dot_b = (vec_a[0, i, j, k] * vec_b[0, i, j, k] +
                              vec_a[1, i, j, k] * vec_b[1, i, j, k] +
                              vec_a[2, i, j, k] * vec_b[2, i, j, k])
                    
                    # Compute projection
                    factor = a_dot_b / b_dot_b
                    out[0, i, j, k] = factor * vec_b[0, i, j, k]
                    out[1, i, j, k] = factor * vec_b[1, i, j, k]
                    out[2, i, j, k] = factor * vec_b[2, i, j, k]
                else:
                    out[0, i, j, k] = 0.0
                    out[1, i, j, k] = 0.0
                    out[2, i, j, k] = 0.0
    
    return out


##########################################################################################
# Core numpy functions for vector operations
##########################################################################################


def vector_magnitude_np_core(
    vector_field : np.ndarray) -> np.ndarray:
    """
    Compute the vector magnitude of a vector.        
    """
    
    out = np.zeros_like(vector_field[0,...])
    out = np.sqrt(vector_dot_product_np_core(vector_field,
                                             vector_field))
    
    return out


def vector_dot_product_np_core(
    vector_field_1 : np.ndarray,
    vector_field_2 : np.ndarray) -> np.ndarray:
    """
    Compute the vector dot product of two vectors.
    """
    
    out = np.zeros_like(vector_field_1[0,...])
    out = np.einsum("i...,i...->...",
                    vector_field_1,
                    vector_field_2)
    
    return out


def vector_cross_product_np_core(
    vector_field_1 : np.ndarray,
    vector_field_2 : np.ndarray) -> np.ndarray:
    """
    Compute the vector cross product of two vectors.
    """
    
    out = np.zeros_like(vector_field_1)
    if vector_field_1.shape[0] == 1:
        ValueError("Vector cross product is not defined for 1D.")
    elif vector_field_1.shape[0] == 2:
        out = vector_field_1[X] * vector_field_2[Y] - vector_field_1[Y] * vector_field_2[X]
    elif vector_field_1.shape[0] == 3:
        out = np.array([
                    vector_field_1[Y] * vector_field_2[Z] - vector_field_1[Z] * vector_field_2[Y],
                    vector_field_1[Z] * vector_field_2[X] - vector_field_1[X] * vector_field_2[Z],
                    vector_field_1[X] * vector_field_2[Y] - vector_field_1[Y] * vector_field_2[X]]
                        )
        
    return out


def vector_projection_np_core(
    vector_field_1 : np.ndarray,
    vector_field_2 : np.ndarray) -> np.ndarray:
    """
    Compute the projection of a vector onto another vector.
    """
    
    out = np.zeros_like(vector_field_1)
    b_dot_b = vector_dot_product_np_core(vector_field_2,
                                        vector_field_2)
    a_dot_b = vector_dot_product_np_core(vector_field_1,
                                        vector_field_2)
    factor = np.zeros_like(b_dot_b)
    mask = b_dot_b > 1e-10
    factor[mask] = a_dot_b[mask] / b_dot_b[mask]
    out = vector_field_2 * factor[np.newaxis, ...] 
    
    return out


def vector_normalize_np_core(
    vector_field : np.ndarray,
    epsilon: float = 1e-10) -> np.ndarray:
    """
    Normalize a vector field.
    """
    
    out = np.zeros_like(vector_field)
    mag = vector_magnitude_np_core(vector_field)
    mask = mag > epsilon
    out[mask] = vector_field[mask] / mag[mask][..., np.newaxis]
    
    return out


def vector_angle_np_core(
    vector_field_1: np.ndarray,
    vector_field_2: np.ndarray,
    epsilon: float = DEFAULT_EPS) -> np.ndarray:
    """
    Compute the angle between two vector fields.
    """
    
    out = np.zeros_like(vector_field_1[0,...])
    mag1 = vector_magnitude_np_core(vector_field_1)
    mag2 = vector_magnitude_np_core(vector_field_2)
    mask = (mag1 > epsilon) & (mag2 > epsilon)
    dot_product = vector_dot_product_np_core(vector_field_1, vector_field_2)
    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    out[mask] = np.arccos(cos_angle[mask])
    
    return out


def vector_triple_product_np_core(
    vector_field_1: np.ndarray,
    vector_field_2: np.ndarray,
    vector_field_3: np.ndarray) -> np.ndarray:
    """
    Compute the scalar triple product: vec1 · (vec2 × vec3).
    """
    
    out = np.zeros_like(vector_field_1[0,...])
    cross_product = vector_cross_product_np_core(vector_field_2, vector_field_3)
    out = vector_dot_product_np_core(vector_field_1, cross_product)
    
    return out