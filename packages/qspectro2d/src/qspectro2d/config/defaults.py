"""Central defaults for :mod:`qspectro2d.config`.

This module intentionally consolidates:
- "constants" (e.g. initial state)
- time/grid defaults
- supported option lists

Goal: fewer files, fewer cross-imports, less maintenance.
"""

from __future__ import annotations

# --- Generic defaults ---
INITIAL_STATE = "ground"

# --- Time and grid defaults ---
T_DET_MAX = 20.0  # Maximum detection time in fs
T_COH_MAX = T_DET_MAX  # Coherence time in fs
DT = 0.1  # Spacing between time grid points in fs
T_WAIT = 0.0  # Waiting time in fs

# --- Supported options ---
SUPPORTED_SOLVERS = ["lindblad", "redfield", "paper_eqs"]

SUPPORTED_BATHS = [
    "ohmic",
    "subohmic",
    "superohmic",
    "drudelorentz",
    "ohmic+lorentzian",
    "subohmic+lorentzian",
    "superohmic+lorentzian",
    "drudelorentz+lorentzian",
]

SUPPORTED_ENVELOPES = ["gaussian", "cos2"]

SUPPORTED_SIM_TYPES = ["0d", "1d", "2d"]

__all__ = [
    "INITIAL_STATE",
    "T_DET_MAX",
    "T_COH_MAX",
    "DT",
    "T_WAIT",
    "SUPPORTED_SOLVERS",
    "SUPPORTED_BATHS",
    "SUPPORTED_ENVELOPES",
    "SUPPORTED_SIM_TYPES",
]
