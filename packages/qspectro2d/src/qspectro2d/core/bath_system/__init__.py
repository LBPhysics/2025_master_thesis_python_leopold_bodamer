"""
Bath functions module for qspectro2d package.

This module provides various models for bosonic baths commonly used in
quantum spectroscopy simulations. It includes spectral density functions
and power spectrum functions for different bath types:

- Drude-Lorentz bath: Classical model for phonon baths
- Ohmic bath: Simple ohmic dissipation model
- Paper bath: Specific model as defined in the research paper

Each bath type provides both spectral density and power spectrum functions
that are compatible with scalar and array inputs for efficient computation.
"""

# BATH FUNCTIONS

from .bath_fcts import (
    # Drude-Lorentz bath functions
    spectral_density_func_drude_lorentz,
    power_spectrum_func_drude_lorentz,
    # Ohmic bath functions
    spectral_density_func_ohmic,
    power_spectrum_func_ohmic,
    # Paper-specific bath functions
    spectral_density_func_paper,
    power_spectrum_func_paper,
    # extract info of the qutip bath
    extract_bath_parameters,
)


# PUBLIC API

__all__ = [
    # Drude-Lorentz bath
    "spectral_density_func_drude_lorentz",
    "power_spectrum_func_drude_lorentz",
    # Ohmic bath
    "spectral_density_func_ohmic",
    "power_spectrum_func_ohmic",
    # Paper bath
    "spectral_density_func_paper",
    "power_spectrum_func_paper",
    # extract bath parameters
    "extract_bath_parameters",
]


# VERSION INFO

__version__ = "1.0.0"
__author__ = "Leopold"
__email__ = ""
