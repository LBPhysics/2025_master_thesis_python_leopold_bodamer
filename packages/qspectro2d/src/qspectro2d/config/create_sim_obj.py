"""Compatibility shim for legacy imports.

The canonical simulation builder API lives in config.factory.
"""

from __future__ import annotations

from .factory import (
    create_base_sim_oqs,
    load_simulation,
    load_simulation_atomic_system,
    load_simulation_bath,
    load_simulation_config,
    load_simulation_laser,
)

__all__ = [
    "create_base_sim_oqs",
    "load_simulation",
    "load_simulation_atomic_system",
    "load_simulation_bath",
    "load_simulation_config",
    "load_simulation_laser",
]
