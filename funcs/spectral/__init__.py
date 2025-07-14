"""
PLASMAtools Spectral Analysis Module

Provides optimized functions for computing a range of spectral operations,
including power spectra, spherical and cylindrical integration, and shell filtering.

This module is designed for high-performance spectral analysis in plasma physics and related fields.
"""

# Import main classes
from .operations import SpectralOperations
from .operations import GeneratedFields

# Import core functions for advanced users
from .core_functions import (
    compute_mixed_spectrum_3D_core,
    compute_mixed_spectrum_2D_core,
    spherical_integrate_3D_core,
    spherical_integrate_2D_core,
    cylindrical_integrate_core,
    compute_shell_filter_2D_core,
    compute_shell_filter_3D_core,
    compute_wave_numbers_reduced,
    helmholtz_decomposition_3D_nb_core,
)

# Version info
__version__ = "1.0.0"
__author__ = "James R. Beattie"

# Define public API
__all__ = [
    'SpectralOperations',
    # Core functions for advanced use
    'compute_mixed_spectrum_3D_core',
    'compute_mixed_spectrum_2D_core',
    'spherical_integrate_3D_core',
    'spherical_integrate_2D_core',
    'cylindrical_integrate_core',
    'compute_shell_filter_2D_core',
    'compute_shell_filter_3D_core',
    'compute_wave_numbers_reduced',
    'helmholtz_decomposition_3D_nb_core',
]