"""Configuration package (simplified).

This package now exposes only:
  - Physics default validation helpers
  - Path utilities
  - Simple loader: `load_simulation` (returns SimulationModuleOQS)

The previous layered dataclass API (`models.py`, `loader.py`) was removed.
"""

from __future__ import annotations

from .validate import validate_defaults, validate  # physics-level sanity
from ..utils.constants import HBAR, BOLTZMANN


def load_simulation(*args, **kwargs):
  """Lazy proxy to avoid import cycles with core modules."""
  from .create_sim_obj import load_simulation as _load_simulation

  return _load_simulation(*args, **kwargs)


def create_base_sim_oqs(*args, **kwargs):
  """Lazy proxy to avoid import cycles with core modules."""
  from .create_sim_obj import create_base_sim_oqs as _create_base_sim_oqs

  return _create_base_sim_oqs(*args, **kwargs)

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
