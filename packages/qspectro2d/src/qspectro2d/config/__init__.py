"""Configuration package (simplified).

This package now exposes only:
  - Physics default validation helpers
  - Path utilities
  - Simple loader: `load_simulation` (returns SimulationModuleOQS)

The previous layered dataclass API (`models.py`, `loader.py`) was removed.
"""

from __future__ import annotations

from .default_simulation_params import validate_defaults, validate  # physics-level sanity
from ..utils.constants import HBAR, BOLTZMANN
from .create_sim_obj import (
    load_simulation,
    create_base_sim_oqs,
)

__all__ = [
    # constants
    "HBAR",
    "BOLTZMANN",
    # validation
    "validate_defaults",
    "validate",
    # loader
    "load_simulation",
    # Simulation utilities
    "create_base_sim_oqs",
]
