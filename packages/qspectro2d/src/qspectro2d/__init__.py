"""
QSpectro2D - Quantum 2D Electronic Spectroscopy Package

A comprehensive Python package for simulating 2D electronic spectroscopy
with Open Quantum Systems models. This package provides tools for:

- System parameter configuration and pulse sequence design
- OQS dynamics simulation with various bath models
- 1D and 2D spectroscopy calculations with inhomogeneous broadening (TODO not yet)
- Data visualization tools
- Configuration management and file I/O utilities

Main subpackages:
- config: Configuration settings and constants
- core: Fundamental simulation components (AtomicSystem, LaserPulseSequence, solvers, bath models)
- spectroscopy: 1D/2D spectroscopy calculations and post-processing
- utils: File I/O, units, and helper utilities
- visualization: Plotting and data visualization tools
"""

__version__ = "1.0"
__author__ = "Leopold"
__email__ = ""

# Silence a specific QuTiP FutureWarning about keyword-only args in brmesolve
import warnings as _warnings

_warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*c_ops, e_ops, args and options will be keyword only from qutip 5\.3.*",
    module=r"qutip\.solver\.brmesolve",
)


# EXPLICIT IMPORTS ONLY (no lazy imports)

# Core exports
from .core import (
    AtomicSystem,
    LaserPulse,
    LaserPulseSequence,
    e_pulses,
    pulse_envelopes,
    matrix_ODE_paper,
    power_spectrum_func_paper,
    power_spectrum_func_drude_lorentz,
    power_spectrum_func_ohmic,
)

from .utils.data_io import (
    save_simulation_data,
    load_simulation_data,
)
from .utils import generate_unique_plot_filename

# Spectroscopy exports (imported after data I/O to avoid partial init race)
from .spectroscopy import extend_time_domain_data, compute_spectra


# PUBLIC API - MOST COMMONLY USED
__all__ = [
    # Core classes - most important for users
    "AtomicSystem",
    "LaserPulse",
    "LaserPulseSequence",
    # Essential functions
    "e_pulses",
    "pulse_envelopes",
    "matrix_ODE_paper",
    # Bath functions
    "power_spectrum_func_paper",
    "power_spectrum_func_drude_lorentz",
    "power_spectrum_func_ohmic",
    # High-level simulation functions
    "complex_polarization",
    # Post-processing
    "extend_time_domain_data",
    "compute_spectra",
    # Data management
    "save_simulation_data",
    "load_simulation_data",
    # Plotting helpers
    "generate_unique_plot_filename",
]


# PACKAGE INFORMATION
def get_package_info():
    """
    Display package information and available modules.
    """
    info = f"""
QSpectro2D Package Information
=============================
Version: {__version__}
Author: {__author__}

Main subpackages:
- config: Configuration settings and constants
- core: Fundamental simulation components (AtomicSystem, LaserPulseSequence, solvers, bath models)
- spectroscopy: 1D/2D spectroscopy calculations and post-processing
- utils: File I/O, units, and helper utilities
- visualization: Plotting and data visualization tools

For detailed documentation, see individual module docstrings.
"""
    return info


def list_available_functions():
    """
    List all functions available in the main namespace.
    """
    return __all__
