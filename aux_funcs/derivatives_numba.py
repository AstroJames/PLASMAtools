import numpy as np
from numba import njit, prange

# Numba-accelerated finite-difference along a given axis for 1D, 2D, 3D arrays
@njit(parallel=True, fastmath=True)
def _gradient_axis_numba_1d(f, dr, offsets, coeffs, bc_code, out):
    N0 = f.shape[0]
    # bc_code: 0=periodic, 1=dirichlet, 2=neumann
    for i in prange(N0):
        s = 0.0
        for t in range(offsets.shape[0]):
            idx = i + offsets[t]
            if idx < 0 or idx >= N0:
                if bc_code == 0:
                    ii = idx % N0
                    fval = f[ii]
                elif bc_code == 1:
                    fval = 0.0
                else:
                    # Neumann: clamp to edge
                    ii = 0 if idx < 0 else N0-1
                    fval = f[ii]
            else:
                fval = f[idx]
            s += coeffs[t] * fval
        out[i] = s / dr
    return out

@njit(parallel=True, fastmath=True)
def _gradient_axis_numba_2d(f, axis, dr, offsets, coeffs, bc_code, out):
    N0, N1 = f.shape
    # bc_code: 0=periodic, 1=dirichlet, 2=neumann
    if axis == 0:
        for i in prange(N0):
            for j in range(N1):
                s = 0.0
                for t in range(offsets.shape[0]):
                    idx = i + offsets[t]
                    if idx < 0 or idx >= N0:
                        if bc_code == 0:
                            ii = idx % N0
                            fval = f[ii, j]
                        elif bc_code == 1:
                            fval = 0.0
                        else:
                            ii = 0 if idx < 0 else N0-1
                            fval = f[ii, j]
                    else:
                        fval = f[idx, j]
                    s += coeffs[t] * fval
                out[i, j] = s / dr
    else:
        for i in prange(N0):
            for j in range(N1):
                s = 0.0
                for t in range(offsets.shape[0]):
                    idx = j + offsets[t]
                    if idx < 0 or idx >= N1:
                        if bc_code == 0:
                            jj = idx % N1
                            fval = f[i, jj]
                        elif bc_code == 1:
                            fval = 0.0
                        else:
                            jj = 0 if idx < 0 else N1-1
                            fval = f[i, jj]
                    else:
                        fval = f[i, idx]
                    s += coeffs[t] * fval
                out[i, j] = s / dr
    return out

@njit(parallel=True, fastmath=True)
def _gradient_axis_numba_3d(f, axis, dr, offsets, coeffs, bc_code, out):
    N0, N1, N2 = f.shape
    # bc_code: 0=periodic, 1=dirichlet, 2=neumann
    if axis == 0:
        for i in prange(N0):
            for j in range(N1):
                for k in range(N2):
                    s = 0.0
                    for t in range(offsets.shape[0]):
                        idx = i + offsets[t]
                        if idx < 0 or idx >= N0:
                            if bc_code == 0:
                                ii = idx % N0
                                fval = f[ii, j, k]
                            elif bc_code == 1:
                                fval = 0.0
                            else:
                                ii = 0 if idx < 0 else N0-1
                                fval = f[ii, j, k]
                        else:
                            fval = f[idx, j, k]
                        s += coeffs[t] * fval
                    out[i, j, k] = s / dr
    elif axis == 1:
        for i in prange(N0):
            for j in range(N1):
                for k in range(N2):
                    s = 0.0
                    for t in range(offsets.shape[0]):
                        idx = j + offsets[t]
                        if idx < 0 or idx >= N1:
                            if bc_code == 0:
                                jj = idx % N1
                                fval = f[i, jj, k]
                            elif bc_code == 1:
                                fval = 0.0
                            else:
                                jj = 0 if idx < 0 else N1-1
                                fval = f[i, jj, k]
                        else:
                            fval = f[i, idx, k]
                        s += coeffs[t] * fval
                    out[i, j, k] = s / dr
    else:
        for i in prange(N0):
            for j in range(N1):
                for k in range(N2):
                    s = 0.0
                    for t in range(offsets.shape[0]):
                        idx = k + offsets[t]
                        if idx < 0 or idx >= N2:
                            if bc_code == 0:
                                kk = idx % N2
                                fval = f[i, j, kk]
                            elif bc_code == 1:
                                fval = 0.0
                            else:
                                kk = 0 if idx < 0 else N2-1
                                fval = f[i, j, kk]
                        else:
                            fval = f[i, j, idx]
                        s += coeffs[t] * fval
                    out[i, j, k] = s / dr
    return out


@njit(parallel=True, fastmath=True)
def gradient_tensor_fused(f3, offsets, coeffs, dr, out):
    """
    Fused Numba kernel to compute the 3x3 gradient tensor of a 3D vector field.
    f3: shape (3, N, N, N)
    offsets, coeffs: 1st-derivative stencil
    dr: grid spacing
    out: preallocated array of shape (3,3,N,N,N)
    """
    M, N, P, Q = f3.shape
    for comp in range(M):
        for coord in range(M):  # derivative direction
            for i in prange(N):  # parallel over X-index for greater task granularity
                for j in range(P):
                    for k in range(Q):
                        s = 0.0
                        if coord == 0:
                            for t in range(offsets.shape[0]):
                                ii = (i + offsets[t]) % N
                                s += coeffs[t] * f3[comp, ii, j, k]
                        elif coord == 1:
                            for t in range(offsets.shape[0]):
                                jj = (j + offsets[t]) % P
                                s += coeffs[t] * f3[comp, i, jj, k]
                        else:
                            for t in range(offsets.shape[0]):
                                kk = (k + offsets[t]) % Q
                                s += coeffs[t] * f3[comp, i, j, kk]
                        out[comp, coord, i, j, k] = s / dr
    return out


class Derivative:
    """
    Derivative class using Numba for periodic finite-difference.

    Supports 1st and 2nd derivatives on 1D, 2D, or 3D arrays.
    """
    def __init__(self, stencil: int = 2) -> None:
        if stencil not in (2, 4, 6, 8):
            raise ValueError("Invalid stencil order: choose 2,4,6, or 8")
        self.stencil = stencil
        # first-derivative offsets & coeffs
        if stencil == 2:
            self.offsets1 = np.array([-1, +1], dtype=np.int64)
            self.coeffs1  = np.array([ -0.5, +0.5], dtype=np.float64)
            self.offsets2 = np.array([-1, 0, +1], dtype=np.int64)
            self.coeffs2  = np.array([1.0, -2.0, 1.0], dtype=np.float64)
        elif stencil == 4:
            self.offsets1 = np.array([-2, -1, +1, +2], dtype=np.int64)
            # 4th-order central difference: (f_{i-2} - 8f_{i-1} + 8f_{i+1} - f_{i+2})/(12*dx)
            self.coeffs1  = np.array([+1/12, -8/12, +8/12, -1/12], dtype=np.float64)
            self.offsets2 = np.array([-2, -1, 0, +1, +2], dtype=np.int64)
            self.coeffs2  = np.array([-1, 16, -30, 16, -1], dtype=np.float64)/12.0
        elif stencil == 6:
            self.offsets1 = np.array([-3,-2,-1,+1,+2,+3], dtype=np.int64)
            self.coeffs1  = np.array([-1/60, +9/60, -45/60, +45/60, -9/60, +1/60], dtype=np.float64)
            self.offsets2 = np.array([-3,-2,-1,0,+1,+2,+3], dtype=np.int64)
            self.coeffs2  = np.array([-2, +27, -270, +490, -270, +27, -2], dtype=np.float64)/180.0
        else:  # stencil==8
            self.offsets1 = np.array([-4,-3,-2,-1,+1,+2,+3,+4], dtype=np.int64)
            # 8th-order central difference: correct antisymmetric coefficients
            self.coeffs1 = np.array([+1/280,-4/105,+1/5,-4/5,+4/5,-1/5,+4/105,-1/280
            ], dtype=np.float64)
            self.offsets2 = np.array([-4,-3,-2,-1,0,+1,+2,+3,+4], dtype=np.int64)
            self.coeffs2  = np.array([-9, +128, -1008, +8064, -14350, +8064, -1008, +128, -9], dtype=np.float64)/5040.0

    def gradient(self,
                 scalar_field       : np.ndarray,
                 gradient_dir       : int,
                 L                  : float = 1.0,
                 derivative_order   : int = 1,
                 boundary_condition : str = "periodic") -> np.ndarray:
        """
        Compute the 1st or 2nd derivative along one axis using Numba.

        Only periodic BC is supported
        """
        # map BC string to code for Numba
        if boundary_condition == 'periodic':
            bc_code = 0
        elif boundary_condition == 'dirichlet':
            bc_code = 1
        else:
            bc_code = 2
        ndim = scalar_field.ndim
        if gradient_dir < 0 or gradient_dir >= ndim:
            raise ValueError("Invalid gradient_dir")
        scalar_field_shape = scalar_field.shape
        dr = self.compute_dr(L,
                             scalar_field_shape,
                             gradient_dir)
        # choose offsets/coeffs
        if derivative_order == 1:
            offsets, coeffs = self.offsets1, self.coeffs1
        else:
            offsets, coeffs = self.offsets2, self.coeffs2
        # allocate output
        out = np.empty_like(scalar_field)
        # dispatch by dimension
        if ndim == 1:
            return _gradient_axis_numba_1d(scalar_field, dr if derivative_order==1 else dr*dr,
                                           offsets, coeffs, bc_code, out)
        elif ndim == 2:
            return _gradient_axis_numba_2d(scalar_field, gradient_dir,
                                           dr if derivative_order==1 else dr*dr,
                                           offsets, coeffs, bc_code, out)
        elif ndim == 3:
            return _gradient_axis_numba_3d(scalar_field, gradient_dir,
                                           dr if derivative_order==1 else dr*dr,
                                           offsets, coeffs, bc_code, out)
        else:
            raise NotImplementedError("gradient only implemented for ndim<=3")

    def gradient_tensor_fast(self, 
                        f3: np.ndarray, 
                        L: float) -> np.ndarray:
        """
        Compute the full 3×3 gradient tensor in a single fused Numba call.
        f3: shape (3,N,N,N) vector field
        L: domain length (assumed cubic)
        Returns: out array shape (3,3,N,N,N)
        """
        # pick first-derivative offsets & coeffs
        offsets, coeffs = self.offsets1, self.coeffs1
        # compute uniform grid spacing via compute_dr (using spatial axis 1)
        dr = self.compute_dr(L, f3.shape, 1)
        N = f3.shape[1]
        # allocate output
        out = np.empty((f3.shape[0], f3.shape[0], N, N, N), dtype=f3.dtype)
        # call fused kernel
        return gradient_tensor_fused(f3, offsets, coeffs, dr, out)

    def compute_dr(self, 
                   L            : float,
                   shape        : tuple,
                   gradient_dir : int) -> float:
        # Uniform grid spacing for both periodic and non-periodic
        return L / shape[gradient_dir]