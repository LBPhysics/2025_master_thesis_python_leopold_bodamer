"""
Core module for qspectro2d package.

This module provides the fundamental building blocks for 2D spectroscopy simulations:
- System parameters and configuration
- LaserPulse definitions and sequences
- LaserPulse field functions
- Solver functions for system dynamics
- RWA (Rotating Wave Approximation) utilities

The core module is designed to handle both single-atom and two-atom systems
with various bath models and pulse configurations.
"""

# SYSTEM PARAMETERS
from .atomic_system import AtomicSystem


# BATH SYSTEMS
from .bath_system import (
    spectral_density_func_drude_lorentz,
    spectral_density_func_ohmic,
    spectral_density_func_paper,
    power_spectrum_func_drude_lorentz,
    power_spectrum_func_ohmic,
    power_spectrum_func_paper,
)


# PULSE DEFINITIONS AND SEQUENCES
from .laser_system import (
    LaserPulse,
    LaserPulseSequence,
    # PULSE FIELD FUNCTIONS
    pulse_envelopes,
    e_pulses,
    epsilon_pulses,
)


# WHOLE MODULE CLASS AND specific Paper SOLVER FUNCTIONS
from .simulation import (
    SimulationModuleOQS,
    SimulationConfig,
    matrix_ODE_paper,
)


# PUBLIC API

__all__ = [
    # System configuration
    "AtomicSystem",
    # LaserPulse definitions
    "LaserPulse",
    "LaserPulseSequence",
    # Environment system
    "spectral_density_func_drude_lorentz",
    "spectral_density_func_ohmic",
    "spectral_density_func_paper",
    "power_spectrum_func_drude_lorentz",
    "power_spectrum_func_ohmic",
    "power_spectrum_func_paper",
    # Simulation module and configuration
    "SimulationModuleOQS",
    "SimulationConfig",
    # LaserPulse field functions
    "pulse_envelopes",
    "e_pulses",
    "epsilon_pulses",
    # Solver functions
    "matrix_ODE_paper",
]


# VERSION INFO

__version__ = "1.0.0"
__author__ = "Leopold Bodamer"
__email__ = ""
