"""
Atomic system module for qspectro2d package.

This module provides the AtomicSystem class for defining and managing
atomic system parameters including:
- N-atom systems (currently supports 1 and 2 atoms)
- Atomic frequencies and dipole moments
- System Hamiltonians and operators in canonical basis, eigenstates, and eigenvalues
- JSON serialization/deserialization

The AtomicSystem class handles both single atoms and coupled atom systems
with configurable coupling strengths and inhomogeneities.
"""

# Atomic-system models and helpers.

from .basis import pair_to_index
from .system import AtomicSystem

__all__ = ["AtomicSystem", "pair_to_index"]
