"""
    PLASMAtools: Spectral Functions

    Main functions for calculating power spectra, and performing spherical and cylindrical integration.
    
    This module provides JIT-compiled functions for efficient computation of power spectra
    and integration in spherical and cylindrical coordinates. It supports both 2D and 3D data
    and can handle both real and complex data types. The functions are optimized for performance
    using Numba and can utilize multi-threading for parallel execution.
    
    Author: James R. Beattie
    Collaborators: Anne Noer Kolborg
"""

## ###############################################################
## IMPORTS
## ###############################################################

import numpy as np
from .core_functions import *
from .utils import *

class SpectralOperations():
    
    
    def __init__(
        self, 
        L : float = [1.0, 1.0, 1.0],
        cache_plans : bool = False):
        """
        Initialize with optional FFTW plan caching.
        
        Args:
            cache_plans (bool): If True, use FFTW plan caching for faster repeated FFTs.
        Default is False, because caching is not always beneficial for single use FFTs. If
        you are running many FFTs, then caching is worth it, so set this to True.
        
        Author: James R. Beattie
        
        """
        self.fft_cache = FFTWPlanCache() if cache_plans else None
        self.L = L if isinstance(L, list) else [L, L, L]
        
    def _do_fft(
        self, 
        data, 
        axes, 
        forward=True, 
        real=False, 
        norm='forward') -> None:
        """
        Helper to use cached FFT plans if available.
        
        This ends up being a lot of overhead for small arrays, and if you're just
        running a single FFT, it is better to not cache the plans.
        If you are running many FFTs, then caching is worth it.
        
        """
        if self.fft_cache is not None:
            return self.fft_cache.execute_fft(
                data,
                axes,
                forward,
                real,
                norm)
        else:
            if real:
                if forward:
                    return rfftn(
                        data,
                        axes=axes,
                        norm=norm)
                else:
                    return irfftn(
                        data,
                        axes=axes,
                        norm=norm)
            else:
                if forward:
                    return fftn(
                        data,
                        axes=axes,
                        norm=norm)
                else:
                    return ifftn(
                        data,
                        axes=axes,
                        norm=norm)
    
    
    def compute_power_spectrum_2D(
        self, 
        field: np.ndarray,
        field_name : str = "field") -> np.ndarray:
        """
        Computes 2D power spectrum using cached FFT plans.
        """
        assert len(field.shape) == 3, "Field should be 2D"
        
        # Ensure data is float32 for memory efficiency
        field = ensure_float32(
            field, 
            field_name=field_name)

        field_fft = self._do_fft(
            field, 
            axes=(X_GRID_VEC, Y_GRID_VEC), 
            forward=True,
            real=np.isrealobj(field),
            norm='forward')
        
        out = np.sum(
            np.abs(field_fft)**2, 
            axis=0)
        
        # Handle real FFT
        if np.isrealobj(field) and field_fft.shape[-1] != field.shape[-1]:
            N = field.shape
            full_out = np.zeros((N[X_GRID_VEC],
                                 N[Y_GRID_VEC]), dtype=out.dtype)
            full_out[:, :out.shape[-1]] = out
            if N[Y_GRID_VEC] % 2 == 0:
                full_out[:, -N[Y_GRID_VEC]//2+1:] = out[:, 1:N[Y_GRID_VEC]//2][:, ::-1]
            else:
                full_out[:, -N[Y_GRID_VEC]//2:] = out[:, 1:N[Y_GRID_VEC]//2+1][:, ::-1]
            out = full_out
        
        return fftshift(
            out, 
            axes=(X, Y))


    def compute_power_spectrum_3D(
        self, 
        field: np.ndarray,
        field_name : str = "field",
        shear: bool = False,
        S: float | None = None,
        t: float | None = None,
        y_coords: np.ndarray | None = None) -> np.ndarray:
        """
        Computes the 3D power spectrum using cached FFT plans.

        Parameters
        ----------
        field : np.ndarray
            Array shaped (ncomp, Nx, Ny, Nz). Sum over components is performed internally.
        field_name : str
            Name for logging/error messages.
        shear : bool, optional
            If True, apply shearing-box phase correction (Route A) before completing the FFTs.
        S : float, optional
            Shear rate (S = q * Omega). Required if `shear` is True.
        t : float, optional
            Time at which the spectrum is computed. Required if `shear` is True.
        y_coords : np.ndarray, optional
            Physical y-coordinates of cell centers with shape (Ny,). If None and `shear` is True,
            a uniform grid in [0, Ly) is assumed based on self.L[Y].

        Returns
        -------
        np.ndarray
            Power spectrum cube with shape (Nx, Ny, Nz), fftshifted.

        Notes
        -----
        - For shearing-box analyses, this routine implements the phase-shift (de-shearing)
          method: FFT in x, multiply by exp[-i ky S t x], then FFT in y and z.

        
        """
        assert len(field.shape) == 4, "Field should be 3D"
        
        field = ensure_float32(field, field_name=field_name)
        is_real = np.isrealobj(field)

        if not shear:
            # -----------------------------
            # non-sheared power spectrum
            # -----------------------------
            field_fft = self._do_fft(
                field, 
                axes=(X_GRID_VEC,
                      Y_GRID_VEC,
                      Z_GRID_VEC), 
                forward=True, 
                real=is_real, 
                norm='forward')

            out = np.sum(np.abs(field_fft)**2, axis=0)

            # Handle real FFT output shape
            if is_real and field_fft.shape[-1] != field.shape[-1]:
                N = field.shape
                full_out = np.zeros((N[X_GRID_VEC], N[Y_GRID_VEC], N[Z_GRID_VEC]), dtype=out.dtype)
                full_out[:, :, :out.shape[-1]] = out
                if N[Z_GRID_VEC] % 2 == 0:
                    full_out[:, :, -N[Z_GRID_VEC]//2+1:] = out[:, :, 1:N[Z_GRID_VEC]//2][:, :, ::-1]
                else:
                    full_out[:, :, -N[Z_GRID_VEC]//2:] = out[:, :, 1:N[Z_GRID_VEC]//2+1][:, :, ::-1]
                out = full_out

            return fftshift(
                out, 
                axes=(X, Y, Z)
                )

        # --------------------------------------------------
        # Shear-corrected spectrum
        # --------------------------------------------------
        # Validate inputs
        if S is None or t is None:
            raise ValueError("When shear=True, both S (shear rate) and t (time) must be provided.")

        # Grid sizes and coordinates
        Nx = field.shape[X_GRID_VEC]
        Ny = field.shape[Y_GRID_VEC]
        Nz = field.shape[Z_GRID_VEC]

        if x_coords is None:
            # Build uniform physical grid in [0, Ly)
            Lx = self.L[X]
            dx = Lx / Nx
            x_coords = np.linspace(-Lx/2,
                                   Lx/2,
                                   Nx).astype(np.float32)
        else:
            if x_coords.shape[0] != Nx:
                raise ValueError("x_coords must have length Nx (the size of the first grid dimension).")

        # Physical wavenumbers consistent with numpy FFT convention (angular wavenumber)
        # kx = 2pi * m / Lx, where m are the FFT integer frequencies from fftfreq
        Ly = self.L[Y]
        ky_vals = TwoPi * fftfreq(Ny, d=Ly / Ny).astype(np.float32)  # shape: (Nx,)

        # 1) FFT in x only (keep y,z in real space for now)
        Fx = self._do_fft(
            field,
            axes=(X_GRID_VEC,),
            forward=True,
            real=is_real,
            norm='forward'
        )  # -> (ncomp, Nx, Ny, Nz)

        # 2) Apply de-shearing phase: exp[-i * kx * S * t * y]
        # Build broadcastable phase with shape (1, Nx, Ny, 1)
        phase_2d = np.exp(-1j * ky_vals[:, None] * (S * t) * x_coords[None, :])  # (Nx, Ny)
        phase = phase_2d[None, :, :, None]
        Fx = Fx * phase

        # 3) FFT in y and z
        Fxyz = self._do_fft(
            Fx,
            axes=(Y_GRID_VEC, Z_GRID_VEC),
            forward=True,
            real=False,   # complex due to phase
            norm='forward'
        )  # -> (ncomp, Nx, Ny, Nz_k)

        # 4) Power spectrum (sum over components)
        out = np.sum(np.abs(Fxyz)**2, axis=0)  # (Nx, Ny, Nz_k)

        # 5) If the input was real and we used an rFFT along z in _do_fft inside cached path,
        #    restore the Hermitian half to a full cube, mirroring exactly as in the original code.
        if is_real and Fxyz.shape[-1] != field.shape[-1]:
            full = np.zeros((Nx, Ny, Nz), dtype=out.dtype)
            full[:, :, :out.shape[-1]] = out
            if Nz % 2 == 0:
                full[:, :, -Nz//2+1:] = out[:, :, 1:Nz//2][:, :, ::-1]
            else:
                full[:, :, -Nz//2:] = out[:, :, 1:Nz//2+1][:, :, ::-1]
            out = full

        return fftshift(
            out,
            axes=(X, Y, Z)
            )
    

    def compute_tensor_power_spectrum(
        self, 
        field: np.ndarray,
        field_name : str = "field") -> np.ndarray:
        """
        Computes tensor power spectrum using cached FFT plans.
        """
        assert field.shape[:X_GRID_TENS] == (3, 3), "Field should be a 3D tensor field"
        
        # Ensure data is float32 for memory efficiency
        field = ensure_float32(
            field,
            field_name=field_name)
        
        field_fft = self._do_fft(
            field, 
            axes=(X_GRID_TENS, Y_GRID_TENS, Z_GRID_TENS), 
            forward=True,
            real=np.isrealobj(field),
            norm='forward')
        
        out = np.sum(
            np.abs(field_fft)**2, 
            axis=(N_COORDS_TENS, M_COORDS_TENS))
        
        # Handle real FFT
        if np.isrealobj(field) and field_fft.shape[-1] != field.shape[-1]:
            N = field.shape
            full_out = np.zeros((N[X_GRID_TENS],
                                 N[Y_GRID_TENS],
                                 N[Z_GRID_TENS]), dtype=out.dtype)
            full_out[:, :, :out.shape[-1]] = out
            if N[Z_GRID_TENS] % 2 == 0:
                full_out[:, :, -N[Z_GRID_TENS]//2+1:] = out[:, :, 1:N[Z_GRID_TENS]//2][:, :, ::-1]
            else:
                full_out[:, :, -N[Z_GRID_TENS]//2:] = out[:, :, 1:N[Z_GRID_TENS]//2+1][:, :, ::-1]
            out = full_out
        
        return fftshift(
            out, 
            axes=(X, Y, Z))


    def compute_mixed_spectrum_2D(
        self,
        field1: np.ndarray,
        field2: np.ndarray,
        field_name : str = "field") -> np.ndarray:
        """
        Compute 2D generic mixed variable power spectrum.
        
        Args:
            field1: First vector field (2, ny, nx)
            field2: Second vector field (2, ny, nx)
        
        Returns:
            Mixed power spectrum (ny, nx), fftshifted
        """
        assert len(field1.shape) == 3, "Field1 should be (2, ny, nx)"
        assert len(field2.shape) == 3, "Field2 should be (2, ny, nx)"
        assert field1.shape == field2.shape, "Fields must have same shape"
        
        # Ensure data is float32 for memory efficiency
        field1 = ensure_float32(
            field1,
            field_name=field_name)
        field2 = ensure_float32(
            field2,
            field_name=field_name)
        
        # Compute FFTs
        field1_fft = self._do_fft(
            field1, 
            axes=(X_GRID_VEC, Y_GRID_VEC), 
            forward=True,
            real=np.isrealobj(field1),
            norm='forward')
        field2_fft = self._do_fft(
            field2, 
            axes=(X_GRID_VEC, Y_GRID_VEC), 
            forward=True,
            real=np.isrealobj(field2),
            norm='forward')
        
        # Compute mixed spectrum
        mixed_spectrum = compute_mixed_spectrum_2D_core(
            field1_fft, 
            field2_fft)
        
        return fftshift(
            mixed_spectrum,
            axes=(X, Y))


    def compute_mixed_spectrum_3D(
        self,
        field1: np.ndarray, 
        field2: np.ndarray,
        field_name : str = "field") -> np.ndarray:
        """
        Compute generic mixed variable power spectrum: |field1(k) · field2*(k)|
        
        This is a general function for computing cross-spectra between any two
        vector fields.
        
        Args:
            field1: First vector field (3, nz, ny, nx)
            field2: Second vector field (3, nz, ny, nx)
        
        Returns:
            Mixed power spectrum (nz, ny, nx), fftshifted
        """
        assert len(field1.shape) == 4, "Field1 should be (3, nz, ny, nx)"
        assert len(field2.shape) == 4, "Field2 should be (3, nz, ny, nx)"
        assert field1.shape == field2.shape, "Fields must have same shape"
        
        # Ensure data is float32 for memory efficiency
        field1 = ensure_float32(
            field1,
            field_name=field_name)
        field2 = ensure_float32(
            field2,
            field_name=field_name)
        
        # Compute FFTs
        field1_fft = self._do_fft(
            field1, 
            axes=(X_GRID_VEC, Y_GRID_VEC, Z_GRID_VEC), 
            forward=True,
            real=np.isrealobj(field1),
            norm='forward')
        field2_fft = self._do_fft(
            field2, 
            axes=(X_GRID_VEC, Y_GRID_VEC, Z_GRID_VEC), 
            forward=True,
            real=np.isrealobj(field2), 
            norm='forward')
        
        # Compute mixed spectrum
        out = compute_mixed_spectrum_3D_core(
            field1_fft, 
            field2_fft)
        
        # Handle real FFT output shape
        if np.isrealobj(field1) and field1_fft.shape[-1] != field1.shape[-1]:
            # Restore full spectrum by mirroring
            N = field1.shape
            full_out = np.zeros((N[X_GRID_VEC],
                                 N[Y_GRID_VEC],
                                 N[Z_GRID_VEC]), dtype=out.dtype)
            full_out[:, :, :out.shape[-1]] = out
            # Mirror conjugate parts
            if N[Z_GRID_VEC] % 2 == 0:
                full_out[:, :, -N[Z_GRID_VEC]//2+1:] = out[:, :, 1:N[Z_GRID_VEC]//2][:, :, ::-1]
            else:
                full_out[:, :, -N[Z_GRID_VEC]//2:] = out[:, :, 1:N[Z_GRID_VEC]//2+1][:, :, ::-1]
            out = full_out
        
        return fftshift(
            out, 
            axes=(X, Y, Z))

    
    def spherical_integrate_2D(
        self, 
        field: np.ndarray, 
        bins: int = None,
        field_name : str = "field") -> tuple:
        """
        2D spherical integration using JIT-compiled functions.
        """
        N = field.shape[N_COORDS_VEC]
        if not bins:
            bins = N // DEFAULT_BINS_RATIO
            
        # Ensure data is float32 for memory efficiency
        field = ensure_float32(
            field, 
            field_name=field_name)
        
        # Use JIT function for distances
        r = compute_radial_distances_2D_core(
            field.shape)
        bin_edges = np.linspace(DEFAULT_BIN_MIN, bins, bins + 1)
        radial_sum = spherical_integrate_2D_core(
            field, 
            r, 
            bin_edges, 
            bins)    
        k_modes = np.ceil((bin_edges[:-1] + bin_edges[1:]) / 2)
    
        return k_modes, radial_sum


    def spherical_integrate_3D(
        self, 
        field: np.ndarray, 
        bins: int = None,
        field_name : str = "field",
        coords: str = "default",
        S: float | None = None,
        t: float | None = None) -> tuple:
        """
        3D spherical integration to produce an isotropic 1D spectrum.

        Parameters
        ----------
        field : np.ndarray
            Power spectrum cube P(kx, ky, kz) with shape (Nx, Ny, Nz), typically the output of
            `compute_power_spectrum_3D(...)` (already fftshifted in our pipeline).
        bins : int, optional
            Number of radial bins. Defaults to N//DEFAULT_BINS_RATIO.
        field_name : str
            Name for logging/error messages.
        coords : {"default", "shear", "physical"}
            - "default" (or "shear"): use the cube’s natural (kx0, ky, kz) indices (comoving shear coords).
            - "physical": use instantaneous physical wavevectors with kx_inst = kx0 + S * t * ky.
        S : float, optional
            Shear rate (S = q * Omega). Required if coords == "physical".
        t : float, optional
            Time at which to evaluate kx_inst. Required if coords == "physical".
        """
        N = field.shape[N_COORDS_VEC]
        if not bins:
            bins = N // DEFAULT_BINS_RATIO
            
        field = ensure_float32(field, field_name=field_name)
        
        # rest-frame box coordinates
        if coords in ("default", "shear"):
            r = compute_radial_distances_3D_core(field.shape)
            bin_edges = np.linspace(DEFAULT_BIN_MIN, bins, bins + 1, dtype=np.float32)
            radial_sum = spherical_integrate_3D_core(
                field, 
                r, 
                bin_edges, 
                bins
                )
            k_modes = np.ceil((bin_edges[:-1] + bin_edges[1:]) / 2)
            return k_modes, radial_sum
        
        # instantaneous physical coordinates (for shearing boxes)
        if coords == "physical":
            if S is None or t is None:
                raise ValueError("When coords='physical', both S (shear rate) and t (time) must be provided.")

            Nx, Ny, Nz = field.shape

            # 1D angular wavenumbers consistent with FFT conventions
            kx_1d = TwoPi * fftfreq(Nx, d=self.L[X]/Nx).astype(np.float32)
            ky_1d = TwoPi * fftfreq(Ny, d=self.L[Y]/Ny).astype(np.float32)
            kz_1d = TwoPi * fftfreq(Nz, d=self.L[Z]/Nz).astype(np.float32)

            # Conservative bin edges covering the shear mapping
            # kx_ext = float(np.max(np.abs(kx_1d)))
            # ky_ext = float(np.max(np.abs(ky_1d)))
            # kz_ext = float(np.max(np.abs(kz_1d)))
            # k_max = np.float32(np.sqrt((kx_ext + abs(S*t)*ky_ext)**2 + ky_ext**2 + kz_ext**2))
            bin_edges = np.linspace(DEFAULT_BIN_MIN, bins, bins + 1, dtype=np.float32)

            # Numba-accelerated shear-aware accumulation
            radial_sum = spherical_integrate_3D_shear_core(
                field.astype(np.float32),
                kx_1d, 
                ky_1d, 
                kz_1d,
                np.float32(S), 
                np.float32(t),
                bin_edges, 
                bins
            )
            k_modes = np.ceil((bin_edges[:-1] + bin_edges[1:]) / 2)
            return k_modes, radial_sum
        
        # Invalid option
        raise ValueError("coords must be one of: 'default', 'shear', or 'physical'.")




    def cylindrical_integrate(
        self, 
        field: np.ndarray, 
        bins_perp: int = 0,
        bins_para: int = 0) -> tuple:
        """
        Cylindrical integration using JIT-compiled functions.
        """
        N = field.shape[N_COORDS_VEC]
        if bins_perp == 0:
            bins_perp = N // DEFAULT_BINS_RATIO
        if bins_para == 0:
            bins_para = N // DEFAULT_BINS_RATIO
            
        # Ensure data is float32 for memory efficiency
        field = ensure_float32(field, field_name="field")
        
        # Use JIT function for distances
        k_perp, k_para = compute_cylindrical_distances_core(field.shape)
        
        bin_edges_perp = np.linspace(DEFAULT_BIN_MIN, bins_perp, bins_perp + 1)
        bin_edges_para = np.linspace(DEFAULT_BIN_MIN, bins_para, bins_para + 1)
        
        # Use JIT function for integration
        cylindrical_sum = cylindrical_integrate_core(
            field, k_perp, k_para,
            bin_edges_perp, bin_edges_para,
            bins_perp, bins_para
        )
        
        k_perp_modes = (bin_edges_perp[:-1] + bin_edges_perp[1:]) / 2
        k_para_modes = (bin_edges_para[:-1] + bin_edges_para[1:]) / 2
        
        return k_perp_modes, k_para_modes, cylindrical_sum


    def extract_isotropic_shell_X(
        self, 
        field: np.ndarray,
        k_minus_dk: float,
        k_plus_dk: float,
        filter: str = 'tophat',
        sigma: float = DEFAULT_SIGMA,
        field_name : str = "field") -> np.ndarray:
        """
        Extract shell using JIT-compiled filter application.
        """
        k_minus = TwoPi / self.L[X] * k_minus_dk
        k_plus = TwoPi / self.L[X] * k_plus_dk
        
        # Ensure data is float32 for memory efficiency
        field = ensure_float32(
            field, 
            field_name=field_name)
        
        # FFT
        field_fft = self._do_fft(
            field,
            axes=(X_GRID_VEC, Y_GRID_VEC, Z_GRID_VEC),
            forward=True, 
            real=False, 
            norm='forward')
        
        # Compute k magnitudes
        N = field.shape
        kx = TwoPi * fftfreq(N[X_GRID_VEC], d=self.L[X]/N[X_GRID_VEC])
        ky = TwoPi * fftfreq(N[Y_GRID_VEC], d=self.L[Y]/N[Y_GRID_VEC])
        kz = TwoPi * fftfreq(N[Z_GRID_VEC], d=self.L[Z]/N[Z_GRID_VEC])
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = np.sqrt(kx*kx + ky*ky + kz*kz).astype(np.float32)
        
        # Apply filter using JIT function
        filter_type = 0 if filter == 'tophat' else 1
        
        # Process each component
        for comp in range(field.shape[N_COORDS_VEC]):
            real_part = np.real(field_fft[comp]).astype(np.float32)
            imag_part = np.imag(field_fft[comp]).astype(np.float32)
            
            real_filtered, imag_filtered = compute_shell_filter_3D_core(
                real_part, imag_part, k_mag,
                k_minus, k_plus, filter_type, sigma
            )
            
            field_fft[comp] = real_filtered + 1j * imag_filtered
        
        # Inverse FFT
        result = self._do_fft(
            field_fft, 
            axes=(X_GRID_VEC, Y_GRID_VEC, Z_GRID_VEC),
            forward=False, 
            real=False, 
            norm='forward')
        
        return np.real(result)
    
    
    def extract_2D_isotropic_X(
        self,
        vector_field: np.ndarray,
        k_minus_dk: float,
        k_plus_dk: float) -> np.ndarray:
        """
        2D shell extraction (keeping original implementation for now).
        """
        N = vector_field.shape
        kx = TwoPi * fftfreq(N[X_GRID_VEC],d=self.L[X]/N[X_GRID_VEC])
        ky = TwoPi * fftfreq(N[Y_GRID_VEC], d=self.L[Y]/N[Y_GRID_VEC])
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        
        # Create filter
        k_mag = np.sqrt(kx*kx + ky*ky)
        mask = np.logical_and(k_mag >= k_minus_dk, k_mag <= k_plus_dk)
        mask = np.stack([mask, mask], axis=0)
        
        # Apply filter in Fourier space
        vector_fft = fftn(
            vector_field, 
            axes=(X_GRID_VEC, Y_GRID_VEC), 
            norm='forward')
        vector_fft *= mask
        
        result =np.real(
            ifftn(
                vector_fft, 
                axes=(X_GRID_VEC, Y_GRID_VEC), 
                norm='forward')
            )
        
        return result
    
    
    def helmholtz_decomposition(
        self,
        vector_field: np.ndarray,
        field_name: str = "field") -> tuple:
        """
        Computes the Helmholtz decomposition of a vector field
        using reduced spectral space (Nx, Ny, Nz//2+1), which saves memory
        and is more efficient for large fields.
        """
        
        # Validate input
        assert len(vector_field.shape) == 4, "Vector field should be (3, Nx, Ny, Nz)"
        assert vector_field.shape[0] == 3, "First dimension should be 3 (vector components)"
        
        vector_field = ensure_float32(
            vector_field, 
            field_name=field_name)
        
        # Real FFT to reduced spectral space
        vector_field_fft = self._do_fft(
            vector_field,
            axes=(1, 2, 3),
            forward=True,
            real=np.isrealobj(vector_field),
            norm='forward'
        )
        
        # Get reduced spectral shape
        spectral_shape = vector_field_fft.shape[1:]  # (Nx, Ny, Nz//2+1)
        
        # Compute wave numbers for reduced space
        kx, ky, kz, ksqr = compute_wave_numbers_reduced(
            spectral_shape, 
            self.L)
        
        # Perform decomposition in reduced space
        Fhat_irrot, Fhat_solen = helmholtz_decomposition_3D_nb_core(
            vector_field_fft, 
            kx, 
            ky, 
            kz, 
            ksqr
        )
        
        # Inverse real FFT directly from reduced space
        F_irrot = self._do_fft(
            Fhat_irrot,
            axes=(1, 2, 3),
            forward=False,
            real=True,
            norm='forward'
        ).astype(np.float32)
        
        F_solen = self._do_fft(
            Fhat_solen,
            axes=(1, 2, 3),
            forward=False,
            real=True,
            norm='forward'
        ).astype(np.float32)
        
        return F_irrot, F_solen
            
class GeneratedFields:
    """
    
    A class for generating various types of vector fields, including helical fields,
    isotropic power-law fields, and anisotropic power-law fields.
    
    """
    
    def __init__(
        self):
        """
        Initializes the GeneratedFields class.
        This class provides methods for generating vector fields with specific properties.
        
        TODO: need to fully implement the class and its methods.
        Currently, it contains a placeholder, with some basic methods, for the class.
        It is not fully functional yet.
        """
        pass
    
    
    @staticmethod
    def create_helical_field(
        N, 
        k_index,
        A_plus=1,
        A_minus=0):
        """
        Creates a vector field with known helical components.

        Parameters:
        N (int): Size of the grid in each dimension.
        k_index (tuple): The index of the wavevector to be used.

        Returns:
        np.ndarray: The generated vector field.
        """
        # Create an empty field in Fourier space
        field_fft = np.zeros(
            (N, N, N, 3), 
            dtype=complex)

        # Generate the wavevector
        L = 1
        kx, ky, kz = np.meshgrid(
            fftfreq(N, d=L/N), 
            fftfreq(N, d=L/N), 
            fftfreq(N, d=L/N), 
            indexing='ij')
        k = np.stack((kx, ky, kz), axis=-1)

        # Calculate h_plus and h_minus for the selected wavevector
        k_vector = k[k_index]
        k_norm = np.linalg.norm(k_vector,axis=-1,keepdims=True)
        k_hat = k_vector / k_norm

        z = np.array([0, 0, 1])
        e = np.cross(z, k_hat)
        e_norm = np.linalg.norm(e,axis=-1)
        e_hat = e / e_norm

        factor = 1/np.sqrt(2.0)
        e_cross_k = np.cross(e_hat,k_vector)
        e_cross_k_norm = np.linalg.norm(e_cross_k,axis=-1,keepdims=True)
        k_cross_e_cross_k = np.cross(k_vector, e_cross_k)
        k_cross_e_cross_k_norm = np.linalg.norm(k_cross_e_cross_k,axis=-1,keepdims=True)
        
        h_plus =  factor * e_cross_k / e_cross_k_norm + \
                    factor * 1j * k_cross_e_cross_k / k_cross_e_cross_k_norm
                    
        h_minus = factor * e_cross_k / e_cross_k_norm - \
                    factor * 1j * k_cross_e_cross_k / k_cross_e_cross_k_norm

        # Assign coefficients in Fourier space
        field_fft[k_index] = A_plus * h_plus + A_minus * h_minus

        # Perform inverse FFT to get the field in physical space
        field = ifftn(field_fft, axes=(0, 1, 2),norm="forward").real

        return field


    @staticmethod
    def generate_isotropic_powerlaw_field(
        size:  int,
        alpha: float = 5./3.) -> np.ndarray:
        """
        This computes a random field with a power-law power spectrum. The power spectrum
        is P(k) = k^-alpha. The field is generated in Fourier space, and then inverse
        transformed to real space.

        Author: James Beattie (2023)

        Args:
            size (int): the linear dimension of the 3D field
            alpha (float): the negative 1D power-law exponent used in Fourier space. Defaults to 5/3.
                            Note that I use the negative exponent, because the power spectrum is
                            proportional to k^-alpha, and note that I make the transformations between
                            3D Fourier transform exponent and 1D power spectrum exponent in the code.

        Returns:
            ifft field (np.ndarray): the inverse fft of the random field with a power-law power spectrum
        """
        # Create a grid of frequencies
        kx = np.fft.fftfreq(size)
        ky = np.fft.fftfreq(size)
        kz = np.fft.fftfreq(size)
        
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Calculate the magnitude of k for each frequency component
        k = np.sqrt(kx**2 + ky**2 + kz**2)
        k[0, 0, 0] = 1  # Avoid division by zero
        
        # Create a 3D grid of random complex numbers
        random_field = np.random.randn(size, size, size) + 1j * np.random.randn(size, size, size)
        
        # Adjust the amplitude of each frequency component to follow k^-5/3 (-11/3 in 3D)
        amplitude = np.where(k != 0, k**(-(alpha+2.0)/2.0), 0)
        adjusted_field = 10*random_field * amplitude

        return np.fft.ifftn(adjusted_field).real


    @staticmethod
    def generate_anisotropic_powerlaw_field(
        N:      int,
        alpha:  float = 5./3.,
        beta:   float = 5./3.,
        L:      float = 1.0) -> np.ndarray:
        """
        This computes a random field with a power-law power spectrum. The power spectrum
        is P(k) = k_perp^-alpha k_parallel^-beta. The field is generated in Fourier space, 
        and then inverse transformed to real space.
        
        Author: James Beattie

        Args:
            N (int): the linear dimension of the 3D field
            alpha (float): the negative 1D power-law exponent used in Fourier space for the 
                            perpendicular component. Defaults to 5/3.
                            Note that I make the transformations between 3D Fourier transform 
                            exponent and 1D power spectrum exponent in the code.
            beta (float): the negative 1D power-law exponent used in Fourier space for the 
                            parallel component. Defaults to 5/3.
            L (float): the physical size of the domain. Defaults to 1.0.

        Returns:
            ifft field (np.ndarray): the inverse fft of the random field with a power-law power spectrum
        """
        
        # Create a grid of frequencies
        kx = np.fft.fftfreq(N)
        ky = np.fft.fftfreq(N)
        kz = np.fft.rfftfreq(N)   
        
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Calculate the magnitude of k for each frequency component
        k_perp              = np.sqrt(kx**2 + ky**2)
        k_par               = np.abs(kz)
        k_perp[k_perp==0]   = np.inf   # Avoid division by zero
        k_par[k_par==0]     = np.inf   # Avoid division by zero
        
        
        # Adjust the amplitude of each frequency component to follow k^-5/3 (-11/3 in 3D)
        amplitude = k_perp**(-(alpha+1.0)/2.0)*k_par**(-beta/2.0)

        # Create a 3D grid of random complex numbers for phases
        rphase = np.exp(2j*np.pi*np.random.rand(*amplitude.shape))
        
        Fhalf = 10*rphase * amplitude
        Fhalf[0,0,0] = 0.0  # Set the zero mode to zero to avoid DC component
        
        # build full cube via Hermitian symmetry
        F = np.zeros((N, N, N), dtype=complex)
        F[:, :, :N//2+1] = Fhalf
        F[:, :, N//2+1:] = np.conj(Fhalf[:, :, 1:N//2][..., ::-1])

        return np.fft.ifftn(F,norm="forward").real
    
    
    @staticmethod
    def helical_decomposition(
        vector_field):
        """
        Performs a helical decomposition of a vector field.

        Parameters:
        velocity_field (array-like): The velocity field corresponding to each k, an array of shape (N, 3).

        Returns:
        u_plus (array): The component of the vector field in the direction of the right-handed helical component.
        u_minus (array): The component of the vector field in the direction of the left-handed helical component.
        
        TODO: this whole function needs to be updated to conform to 3,N,N,N vector fields instead of
            N,N,N,3 vector fields
        """
        # Convert inputs to numpy arrays
        vector_field = np.asarray(vector_field)
        
        # Take FFT of vector field
        vector_field_FFT = fftn(vector_field,
                                norm='forward',
                                axes=(0,1,2))
        N = vector_field.shape[0]  # Assuming a cubic domain
        L = 1  # The physical size of the domain
        kx = fftfreq(N, d=L/N)
        ky = fftfreq(N, d=L/N)
        kz = fftfreq(N, d=L/N)
        
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
        k = np.stack((kx, ky, kz), axis=-1)  # This will be of shape (N, N, N, 3)

        # Normalize k to get the unit wavevector
        k_norm = np.linalg.norm(k, axis=-1, keepdims=True)

        if np.any(np.isnan(k_norm)) or np.any(np.isinf(k_norm)):
            raise ValueError("NaN or Inf found in k_norm")    

        # Set the k_hat for zero wavevectors explicitly to zero (or some other appropriate value)
        k_hat = np.zeros_like(k)
        non_zero_indices = k_norm.squeeze() > 0  # Indices where the norm is non-zero
        k_hat[non_zero_indices] = k[non_zero_indices] / k_norm[non_zero_indices]
        
        if np.any(np.isnan(k_hat)) or np.any(np.isinf(k_hat)):
            raise ValueError("NaN or Inf found in k_hat")

        # Choose an arbitrary versor orthogonal to k
        z = np.array([0, 0, 1])
        e = np.cross(z, k_hat)
        e_norm = np.linalg.norm(e, axis=-1, keepdims=True)
        
        # Set the k_hat for zero wavevectors explicitly to zero (or some other appropriate value)
        e_hat = np.zeros_like(k)
        non_zero_indices = e_norm.squeeze() > 0  # Indices where the norm is non-zero
        e_hat[non_zero_indices] = e[non_zero_indices] / e_norm[non_zero_indices]

        # Ensure that e_hat is not a zero vector (which can happen if k is parallel to z)
        # In such a case, we can choose e_hat to be any vector orthogonal to k
        for i, e in enumerate(e_hat):
            if np.allclose(e, np.zeros_like(e)):
                # Choose a new e_hat that is not parallel to k
                if np.allclose(k_hat[i], np.array([1, 0, 0])) or np.allclose(k_hat[i], np.array([0, 0, 1])):
                    e_hat[i] = np.array([0, 1, 0])
                else:
                    e_hat[i] = np.array([1, 0, 0])

        # Calculate helical components    
        factor                 = 1/np.sqrt(2.0)
        e_cross_k              = np.cross(e_hat,k)
        e_cross_k_norm         = np.linalg.norm(e_cross_k,axis=-1,keepdims=True)
        k_cross_e_cross_k      = np.cross(k, e_cross_k)
        k_cross_e_cross_k_norm = np.linalg.norm(k_cross_e_cross_k,axis=-1,keepdims=True)
        
        h_plus =  factor * e_cross_k / e_cross_k_norm  + \
                    factor * 1j * k_cross_e_cross_k / k_cross_e_cross_k_norm
                    
        h_minus = factor * e_cross_k / e_cross_k_norm - \
                    factor * 1j * k_cross_e_cross_k / k_cross_e_cross_k_norm

        # test orthogonality 
        #print(np.abs(np.sum(h_plus * h_minus, axis=-1)))
        #print(np.sum(np.abs(h_minus * h_plus), axis=-1))

        # Project velocity field onto helical components
        u_plus = np.sum(vector_field_FFT * h_plus, axis=-1)
        u_minus = np.sum(vector_field_FFT * h_minus, axis=-1)

        # remove k = 0 mode   
        u_plus[np.isnan(u_plus)] = 0
        u_minus[np.isnan(u_minus)] = 0

        return u_plus, u_minus
    