import numpy as np
import time
# adjust this import to your actual install path
from PLASMAtools.aux_funcs.derivatives_numba import Derivative, gradient_tensor_fused

# Analytical test functions
L = 2*np.pi
def sin1d(x):    return np.sin(x)
def dsin1d(x):   return np.cos(x)
def poly1d(x):   return x**10 + 3*x + 1
def dpoly1d(x):  return 10*x**9 + 3

# 1D grid
N1 = 200
x1 = np.linspace(0, L, N1, endpoint=False)

print("=== 1D first-derivative tests ===")
for stencil in (2,4,6,8):
    d = Derivative(stencil)
    f = sin1d(x1)[np.newaxis, :]
    df_num = d.gradient(f[0], gradient_dir=0, L=L, derivative_order=1, boundary_condition='periodic')
    df_exact = (2*np.pi/L)*dsin1d(2*np.pi*x1/L)
    err = np.max(np.abs(df_num - df_exact))
    print(f"stencil={stencil:2d}, periodic sin: max error = {err:.2e}")

print("\n=== 1D Dirichlet tests ===")
for stencil in (2,4,6,8):
    d = Derivative(stencil)
    f = poly1d(x1)[np.newaxis, :]
    df_num = d.gradient(f[0], gradient_dir=0, L=L, derivative_order=1, boundary_condition='dirichlet')
    df_exact = dpoly1d(x1)
    m = stencil//2
    err = np.max(np.abs(df_num[m:-m] - df_exact[m:-m]))
    print(f"stencil={stencil:2d}, dirichlet poly: interior max error = {err:.2e}")

# 3D fused vs separate
print("\n=== Fused 3D vs separate calls consistency ===")
N3 = 288
# correct grid spacing for periodic domain
dr = L / N3
x3 = np.linspace(0, L, N3, endpoint=False)
X3 = np.broadcast_to(x3, (N3,N3,N3))
f3 = np.tile(np.sin(X3)[None,:,:,:], (3,1,1,1))
d3 = Derivative(4)

# warm up compile
gradient_tensor_fused(f3, d3.offsets1, d3.coeffs1, dr, np.empty((3,3,N3,N3,N3)))

# fused timing
t0 = time.time()
G_fused = gradient_tensor_fused(f3, d3.offsets1, d3.coeffs1, dr, np.empty((3,3,N3,N3,N3)))
t_fused = time.time()-t0

# separate timing
out_sep = np.empty_like(G_fused)
t0 = time.time()
for i in range(3):
    for j in range(3):
        out_sep[i,j] = d3.gradient(f3[i], gradient_dir=j, L=L, derivative_order=1, boundary_condition='periodic')
t_sep = time.time()-t0

# --- Diagnostics start ---
# Compare grid spacing used
print("Diagnostic: dr used for fused:", dr)
for axis in range(3):
    dr_sep = d3.compute_dr(L, (N3, N3, N3), axis)
    print(f"Diagnostic: dr_sep axis {axis}:", dr_sep)

# errors
diff = np.max(np.abs(G_fused - out_sep))

print(f"Fused time:    {t_fused:.4f}s")
print(f"Separate time: {t_sep:.4f}s")
print(f"Max error fused vs sep: {diff:.2e}")