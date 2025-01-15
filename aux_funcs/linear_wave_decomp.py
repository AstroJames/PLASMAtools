import numpy as np
from numpy.fft import fftn, ifftn

# Define constants and parameters
B0 = np.array([1.0, 0.0, 0.0])  # Mean magnetic field vector
B0_unit = B0 / np.linalg.norm(B0)  # Unit vector of mean magnetic field

# Plasma parameters
a_squared = 0.5  # Sound speed squared (adjust based on plasma properties)
V_A_squared = 1.0  # Alfvén speed squared (adjust based on plasma properties)
beta = 2 * a_squared / V_A_squared  # Plasma beta
gamma = 1.0  # Polytropic index for isothermal case

# Generate wavevector grid with consistent shapes
def generate_k_grid(N, L):
    """
    Generates wavevector grid in Fourier space with consistent shape.
    """
    k = np.fft.fftfreq(N, d=L/N)
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k_grid = np.stack([kx, ky, kz], axis=0)  # Shape: (3, N, N, N)
    return k_grid

# Compute the displacement vectors for slow, fast, and Alfvén modes
def compute_displacement_vectors(k_grid):
    """
    Computes displacement vectors for slow, fast, and Alfvén modes.

    Parameters:
    k_grid: ndarray
        Wavevector grid with shape (3, N, N, N).

    Returns:
    xi_s, xi_f, xi_A: ndarray
        Displacement vectors for slow, fast, and Alfvén modes, each with shape (3, N, N, N).
    """
    k_mag = np.linalg.norm(k_grid, axis=0)  # Magnitude of wavevector
    k_unit = k_grid / np.where(k_mag != 0, k_mag, 1)  # Unit vector of wavevector

    k_parallel = np.tensordot(B0_unit, k_unit, axes=1)[None, ...] * B0_unit[:, None, None, None]
    k_perp = k_unit - k_parallel  # Perpendicular component of k

    # Normalize k_perp to avoid division by zero
    k_perp_mag = np.linalg.norm(k_perp, axis=0)
    k_perp = k_perp / np.where(k_perp_mag != 0, k_perp_mag, 1)

    # Angle theta between k and B0
    cos_theta = np.tensordot(B0_unit, k_unit, axes=1)
    alpha = a_squared / V_A_squared
    D = (1 + alpha)**2 - 4 * alpha * cos_theta**2

    # Compute slow and fast mode displacement vectors
    xi_s = (-1 + alpha - np.sqrt(D))[None, ...] * k_parallel + (1 + alpha - np.sqrt(D))[None, ...] * k_perp
    xi_f = (-1 + alpha + np.sqrt(D))[None, ...] * k_parallel + (1 + alpha + np.sqrt(D))[None, ...] * k_perp

    # Normalize displacement vectors
    xi_s /= np.linalg.norm(xi_s, axis=0)
    xi_f /= np.linalg.norm(xi_f, axis=0)

    # Compute Alfvén mode displacement vector
    xi_A = np.cross(k_unit, B0_unit[:, None, None, None], axis=0)
    xi_A /= np.linalg.norm(xi_A, axis=0)

    return xi_s, xi_f, xi_A

# Project velocity field onto displacement vectors
def project_onto_modes(u_k, k_grid):
    """
    Projects the velocity field onto the displacement vectors for slow, fast, and Alfvén modes.

    Parameters:
    u_k: ndarray
        Fourier-transformed velocity field with shape (3, N, N, N).
    k_grid: ndarray
        Wavevector grid with shape (3, N, N, N).

    Returns:
    u_s, u_f, u_A: ndarray
        Projections of the velocity field onto the slow, fast, and Alfvén modes, each with shape (N, N, N).
    """
    xi_s, xi_f, xi_A = compute_displacement_vectors(k_grid)

    # Perform dot products for projection
    u_s = np.sum(u_k * xi_s, axis=0)  # Projection onto slow mode
    u_f = np.sum(u_k * xi_f, axis=0)  # Projection onto fast mode
    u_A = np.sum(u_k * xi_A, axis=0)  # Projection onto Alfvén mode

    return u_s, u_f, u_A

# Reconstruct real-space fields
def reconstruct_real_space(u_s, u_f, u_A):
    """
    Reconstructs real-space fields via inverse Fourier transform.

    Parameters:
    u_s, u_f, u_A: ndarray
        Fourier-transformed velocity components for slow, fast, and Alfvén modes.

    Returns:
    u_s_real, u_f_real, u_A_real: ndarray
        Real-space velocity fields for slow, fast, and Alfvén modes.
    """
    u_s_real = np.real(ifftn(u_s))
    u_f_real = np.real(ifftn(u_f))
    u_A_real = np.real(ifftn(u_A))

    return u_s_real, u_f_real, u_A_real

# Main function
def main():
    # Example grid parameters
    N = 64  # Grid size
    L = 1.0  # Physical domain size

    # Generate wavevector grid
    k_grid = generate_k_grid(N, L)

    # Example Fourier-transformed velocity field (random data for demonstration)
    u_k = np.random.randn(3, N, N, N)  # Replace with your actual data

    # Project onto MHD modes
    u_s, u_f, u_A = project_onto_modes(u_k, k_grid)

    # Reconstruct real-space fields
    u_s_real, u_f_real, u_A_real = reconstruct_real_space(u_s, u_f, u_A)

    print("Decomposition and reconstruction completed.")
    # Further analysis or visualization can be added here

if __name__ == "__main__":
    main()