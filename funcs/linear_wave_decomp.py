import numpy as np

def compute_displacement_vectors(k_grid : np.ndarray,
                                 alpha  : float
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes displacement vectors for slow, fast, and Alfvén modes.

    Parameters:
    k_grid: ndarray
        Wavevector grid with shape (3, N, N, N).

    Returns:
    xi_s, xi_f, xi_A: ndarray
        Displacement vectors for slow, fast, and Alfvén modes, each with shape (3, N, N, N).
    """
    
    # External magnetic field unit vector is always in the z-direction
    B0_unit = np.array([0.0, 0.0, 1.0])
    
    # compute magnitude of k and mask out the zero mode
    k_mag = np.linalg.norm(k_grid, axis=0)                     # shape (N,N,N)
    zero_mask = (k_mag == 0)                                   # mask for k=0
    # safe divisor for k
    k_mag_safe = k_mag.copy()
    k_mag_safe[zero_mask] = 1.0
    # unit vector
    k_unit = k_grid / k_mag_safe[None, ...]                    # shape (3,N,N,N)

    # projection of k onto B0 and perpendicular component
    cos_theta = np.tensordot(B0_unit, k_unit, axes=1)          # shape (N,N,N)
    k_parallel = cos_theta[None, ...] * B0_unit[:, None, None, None]
    k_perp = k_unit - k_parallel

    # normalize k_perp safely
    k_perp_mag = np.linalg.norm(k_perp, axis=0)
    k_perp_safe = k_perp_mag.copy()
    k_perp_safe[k_perp_mag == 0] = 1.0
    k_perp = k_perp / k_perp_safe[None, ...]

    # compute mode coefficients
    D = (1 + alpha)**2 - 4 * alpha * cos_theta**2
    sqrtD = np.sqrt(D)

    xi_s = (-1 + alpha - sqrtD)[None,...] * k_parallel \
           + (1 + alpha - sqrtD)[None,...] * k_perp
    xi_f = (-1 + alpha + sqrtD)[None,...] * k_parallel \
           + (1 + alpha + sqrtD)[None,...] * k_perp

    # normalize slow and fast safely
    norm_s = np.linalg.norm(xi_s, axis=0)
    norm_s_safe = norm_s.copy()
    norm_s_safe[norm_s == 0] = 1.0
    xi_s /= norm_s_safe[None,...]

    norm_f = np.linalg.norm(xi_f, axis=0)
    norm_f_safe = norm_f.copy()
    norm_f_safe[norm_f == 0] = 1.0
    xi_f /= norm_f_safe[None,...]

    # compute and normalize Alfvén mode
    xi_A = np.cross(k_unit, B0_unit[:, None, None, None], axis=0)
    norm_A = np.linalg.norm(xi_A, axis=0)
    norm_A_safe = norm_A.copy()
    norm_A_safe[norm_A == 0] = 1.0
    xi_A /= norm_A_safe[None,...]

    # explicitly zero out k=0 mode to avoid NaNs
    xi_s[:, zero_mask] = 0.0
    xi_f[:, zero_mask] = 0.0
    xi_A[:, zero_mask] = 0.0

    # fix degenerate propagation (parallel or perpendicular) so eigenvectors form a full basis
    tol = 1e-8
    # parallel: |cosθ|≈1, perpendicular: |cosθ|≈0
    deg_par = np.abs(cos_theta) > 1 - tol
    deg_perp = np.abs(cos_theta) < tol
    degenerate = (deg_par | deg_perp) & (~zero_mask)

    # complete to orthonormal basis at each degenerate k
    for idx in zip(*np.where(degenerate)):
        ii, jj, kk = idx
        # primary direction
        e1 = k_unit[:, ii, jj, kk]
        # choose a helper not parallel to e1
        helper = B0_unit.copy()
        if deg_par[ii, jj, kk]:
            helper = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(helper, e1)) > 1 - tol:
                helper = np.array([0.0, 1.0, 0.0])
        # build second and third orthonormal vectors
        e2 = np.cross(e1, helper)
        e2 /= np.linalg.norm(e2)
        e3 = np.cross(e2, e1)
        e3 /= np.linalg.norm(e3)
        # assign full basis
        xi_s[:, ii, jj, kk] = e1
        xi_A[:, ii, jj, kk] = e2
        xi_f[:, ii, jj, kk] = e3

    return xi_s, xi_f, xi_A

def generate_k_grid(N, L):
    """
    Generates a wavevector grid for an N^3 domain of length L.
    Returns an array of shape (3, N, N, N).
    """
    k_vals = np.fft.fftfreq(N, d=L/N)
    kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
    return np.stack([kx, ky, kz], axis=0)

def decompose_linear_modes(u        : np.ndarray,
                           k_grid   : np.ndarray,
                           alpha    : float = 0.0
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projects a real-space velocity field u(x) (shape (3, N, N, N))
    onto MHD modes in k-space, returning (u_s, u_f, u_A) in k-space.
    """
    # Fourier transform u to k-space
    u_k = np.fft.fftn(u, 
                      axes=(1,2,3),
                      norm="forward")

    # Compute displacement vectors in k-space
    xi_s, xi_f, xi_A = compute_displacement_vectors(k_grid,
                                                    alpha=alpha)

    # mask for k=0 modes
    zero_mask = (np.linalg.norm(k_grid, axis=0) == 0)

    # Build k-space projections by solving at each k: u_k = [xi_s,xi_f,xi_A] * coeffs
    N = k_grid.shape[1]
    # prepare empty k-space arrays
    u_s_k = np.zeros_like(u_k)
    u_f_k = np.zeros_like(u_k)
    u_A_k = np.zeros_like(u_k)
    # loop over all k
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if zero_mask[i, j, k]:
                    # retain mean in Alfvén projection
                    u_A_k[:, i, j, k] = u_k[:, i, j, k]
                    continue
                # form matrix of eigenvectors at this k
                M = np.stack([xi_s[:, i, j, k],
                              xi_f[:, i, j, k],
                              xi_A[:, i, j, k]], axis=1)  # shape (3,3)
                # Use least-squares solve to handle degenerate (singular) M
                coeff, *_ = np.linalg.lstsq(M, u_k[:, i, j, k], rcond=None)
                # assign each mode slice
                u_s_k[:, i, j, k] = coeff[0] * xi_s[:, i, j, k]
                u_f_k[:, i, j, k] = coeff[1] * xi_f[:, i, j, k]
                u_A_k[:, i, j, k] = coeff[2] * xi_A[:, i, j, k]

    # Inverse FFT back to real space (per component)
    u_s_r = np.real(np.fft.ifftn(u_s_k, axes=(1,2,3),norm="forward"))
    u_f_r = np.real(np.fft.ifftn(u_f_k, axes=(1,2,3),norm="forward"))
    u_A_r = np.real(np.fft.ifftn(u_A_k, axes=(1,2,3),norm="forward"))

    return u_s_r, u_f_r, u_A_r

if __name__ == "__main__":
    
    # Simple test/demo
    N = 64
    L = 1.0
    k_grid = generate_k_grid(N, L)
    # random velocity field for demonstration
    u = np.random.randn(3, N, N, N)
    u_s, u_f, u_A = decompose_linear_modes(u, k_grid)
    print("MHD decomposition demo complete. Shapes:", u_s.shape)