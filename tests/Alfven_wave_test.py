#!/usr/bin/env python3
import numpy as np
from PLASMAtools.aux_funcs.linear_wave_decomp import (
    generate_k_grid,
    compute_displacement_vectors,
    decompose_linear_modes
)

def test_pure_alfven_roundtrip():
    N, L = 64, 1.0
    alpha = 0.0  # cold plasma limit
    # Generate k-grid and compute eigenvectors
    k_grid = generate_k_grid(N, L)
    xi_s, xi_f, xi_A = compute_displacement_vectors(k_grid,
                                                    alpha)

    # Choose a single nonzero k-mode along y-direction (i=0,j=1,k=0)
    i, j, k = 0, 1, 0

    # Construct a real-space pure-Alfvén wave: u(x) = xi_A * cos(2π * y / L)
    u_real = np.zeros((3, N, N, N), dtype=float)
    y = np.arange(N)
    cos_y = np.cos(2 * np.pi * y / 2.0)
    # Broadcast the wave along x, z dimensions
    # xi_A[:,i,j,k] is the constant polarization direction
    pol = xi_A[:, i, j, k]  # shape (3,)
    # Fill u_real such that u_real[:, :, j, :] = pol[:, None] * cos_y[None, :]
    for comp in range(3):
        u_real[comp, :, :, :] = pol[comp] * cos_y[None, :, None]
    # Decompose back into modes
    u_s_r, u_f_r, u_A_r = decompose_linear_modes(u_real,
                                                 k_grid,
                                                 alpha)
    # Check that slow & fast modes vanish
    assert np.allclose(u_s_r, 0, atol=1e-12), "Slow mode leaked!"
    assert np.allclose(u_f_r, 0, atol=1e-12), "Fast mode leaked!"
    # Check that the Alfvén reconstruction matches the original
    assert np.allclose(u_A_r, u_real, atol=1e-12), "Alfvén reconstruction failed!"
    print("Pure Alfvén roundtrip test passed.")

def test_completeness():
    N, L = 64, 1.0
    alpha = 0.0  # cold plasma limit
    k_grid = generate_k_grid(N, L)
    u = np.random.randn(3, N, N, N)
    u_s, u_f, u_A = decompose_linear_modes(u,
                                           k_grid,
                                           alpha)
    u_recon = u_s + u_f + u_A
    assert np.allclose(u_recon, u, atol=1e-12), "Completeness failed!"
    print("Completeness passed.")

def test_pure_slow_roundtrip():
    N, L = 64, 1.0
    alpha = 0.0  # cold plasma limit
    k_grid = generate_k_grid(N, L)
    xi_s, xi_f, xi_A = compute_displacement_vectors(k_grid,alpha)
    i, j, k = 0, 1, 0
    u_k = np.zeros((3, N, N, N), dtype=complex)
    u_k[:, i, j, k] = xi_s[:, i, j, k]
    u_real = np.real(np.fft.ifftn(u_k, axes=(1,2,3), norm="forward"))
    u_s, u_f, u_A = decompose_linear_modes(u_real, k_grid)
    u_s_k = np.fft.fftn(u_s, axes=(1,2,3), norm="forward")
    u_f_k = np.fft.fftn(u_f, axes=(1,2,3), norm="forward")
    u_A_k = np.fft.fftn(u_A, axes=(1,2,3), norm="forward")
    assert np.allclose(u_f_k[:,i,j,k], 0, atol=1e-12), "Fast leaked!"
    assert np.allclose(u_A_k[:,i,j,k], 0, atol=1e-12), "Alfvén leaked!"
    assert np.allclose(u_s_k[:,i,j,k], xi_s[:,i,j,k], atol=1e-12), "Slow not recovered!"
    print("Pure slow roundtrip passed.")

def test_pure_fast_roundtrip():
    N, L = 64, 1.0
    alpha = 0.0  # cold plasma limit
    k_grid = generate_k_grid(N, L)
    xi_s, xi_f, xi_A = compute_displacement_vectors(k_grid,alpha)
    i, j, k = 0, 1, 0
    u_k = np.zeros((3, N, N, N), dtype=complex)
    u_k[:, i, j, k] = xi_f[:, i, j, k]
    u_real = np.real(np.fft.ifftn(u_k, axes=(1,2,3), norm="forward"))
    u_s, u_f, u_A = decompose_linear_modes(u_real, k_grid)
    u_s_k = np.fft.fftn(u_s, axes=(1,2,3), norm="forward")
    u_f_k = np.fft.fftn(u_f, axes=(1,2,3), norm="forward")
    u_A_k = np.fft.fftn(u_A, axes=(1,2,3), norm="forward")
    assert np.allclose(u_s_k[:,i,j,k], 0, atol=1e-12), "Slow leaked!"
    assert np.allclose(u_A_k[:,i,j,k], 0, atol=1e-12), "Alfvén leaked!"
    assert np.allclose(u_f_k[:,i,j,k], xi_f[:,i,j,k], atol=1e-12), "Fast not recovered!"
    print("Pure fast roundtrip passed.")

def test_energy_conservation():
    N, L = 64, 1.0
    alpha = 0.0  # cold plasma limit
    k_grid = generate_k_grid(N, L)
    u = np.random.randn(3, N, N, N)
    u_s, u_f, u_A = decompose_linear_modes(u, k_grid,alpha)
    E_tot = np.sum(u**2)
    E_modes = np.sum(u_s**2) + np.sum(u_f**2) + np.sum(u_A**2)
    assert np.allclose(E_tot, E_modes, atol=1e-12), "Energy not conserved!"
    print("Energy conservation passed.")

def test_alfven_divergence_free():
    N, L = 64, 1.0
    alpha = 0.0  # cold plasma limit
    dx = L/N
    k_grid = generate_k_grid(N, L)
    xi_s, xi_f, xi_A = compute_displacement_vectors(k_grid,alpha)
    i, j, k = 0, 1, 0
    u_k = np.zeros((3, N, N, N), dtype=complex)
    u_k[:, i, j, k] = xi_A[:, i, j, k]
    u_A = np.real(np.fft.ifftn(u_k, axes=(1,2,3), norm="forward"))
    div = (np.gradient(u_A[0], dx, axis=0) +
           np.gradient(u_A[1], dx, axis=1) +
           np.gradient(u_A[2], dx, axis=2))
    assert np.allclose(div, 0, atol=1e-12), "Alfvén not divergence free!"
    print("Alfvén divergence‐free passed.")

def test_A_orthogonality():
    N, L = 32, 1.0
    k_grid = generate_k_grid(N, L)
    xi_s, xi_f, xi_A = compute_displacement_vectors(k_grid)
    mag = np.linalg.norm(k_grid, axis=0)
    mask = mag > 0
    dot_sA = np.sum(xi_s * xi_A, axis=0)[mask]
    dot_fA = np.sum(xi_f * xi_A, axis=0)[mask]
    assert np.allclose(dot_sA, 0, atol=1e-12), "ξA·ξs nonzero!"
    assert np.allclose(dot_fA, 0, atol=1e-12), "ξA·ξf nonzero!"
    print("ξA orthogonal to ξs,f passed.")

def test_s_f_orthogonal_cold():
    N, L = 32, 1.0
    k_grid = generate_k_grid(N, L)
    xi_s, xi_f, xi_A = compute_displacement_vectors(k_grid, alpha=0.0)
    mag = np.linalg.norm(k_grid, axis=0)
    mask = mag > 0
    dot_sf = np.sum(xi_s * xi_f, axis=0)[mask]
    assert np.allclose(dot_sf, 0, atol=1e-12), "ξs·ξf not zero at α=0!"
    print("ξs ⟂ ξf in cold‐plasma passed.")

def test_random_statistics(num=10):
    N, L = 32, 1.0
    k_grid = generate_k_grid(N, L)
    sums = {"sf":0, "sA":0, "fA":0}
    for _ in range(num):
        u = np.random.randn(3, N, N, N)
        u_s, u_f, u_A = decompose_linear_modes(u, k_grid)
        sums["sf"] += np.sum(u_s * u_f)
        sums["sA"] += np.sum(u_s * u_A)
        sums["fA"] += np.sum(u_f * u_A)
    assert abs(sums["sf"]) < 1e-6, "sf correlation nonzero!"
    assert abs(sums["sA"]) < 1e-6, "sA correlation nonzero!"
    assert abs(sums["fA"]) < 1e-6, "fA correlation nonzero!"
    print("Random‐field statistics passed.")

if __name__ == "__main__":
    test_pure_alfven_roundtrip()
    test_completeness()
    test_pure_slow_roundtrip()
    test_pure_fast_roundtrip()
    test_energy_conservation()
    test_alfven_divergence_free()
    test_A_orthogonality()
    test_s_f_orthogonal_cold()
    test_random_statistics()