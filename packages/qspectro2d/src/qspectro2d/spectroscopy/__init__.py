"""
Spectroscopy package for qspectro2d.

This package provides computational tools for 1D and 2D spectroscopy simulations,
including pulse evolution calculations, polarization computations, and post-processing
routines for Fourier transforms and signal analysis.

Main components:
- e_field_1d: 1D field and polarization computation pipeline
- inhomogenity: Tools for handling inhomogeneous broadening
- post_processing: FFT and signal processing utilities
- simulation: High-level simulation runners and utilities
"""

# CORE CALCULATION FUNCTIONS
from .e_field_1d import (
    compute_evolution,
    parallel_compute_1d_e_comps,
)
from .polarization import complex_polarization

from .solver_check import check_the_solver

# INHOMOGENEOUS BROADENING
from .inhomogenity import (
    normalized_gauss,
    sample_from_gaussian,
)

# POST-PROCESSING FUNCTIONS
from .post_processing import (
    compute_spectra,
)

__all__ = [
    # Core calculations
    "complex_polarization",
    "compute_evolution",
    "check_the_solver",
    "parallel_compute_1d_e_comps",
    "phase_cycle_component",
    # Inhomogeneous broadening
    "normalized_gauss",
    "sample_from_gaussian",
    # Post-processing
    "compute_spectra",
]
