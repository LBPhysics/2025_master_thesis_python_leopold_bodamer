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

# ATOMIC SYSTEM CLASS

from .system_class import AtomicSystem


# PUBLIC API

__all__ = ["AtomicSystem"]


# VERSION INFO

__version__ = "1.0.0"
__author__ = "Leopold Bodamer"
__email__ = ""
