"""Simulation defaults for qspectro2d."""

from .defaults import SOLVER_OPTIONS

# === SIMULATION DEFAULTS ===
ODE_SOLVER = "redfield"
SIM_TYPE = "1d"

ALLOWED_SOLVER_OPTIONS = {
    "paper_eqs": [],
    "lindblad": [
        "atol",
        "rtol",
        "nsteps",
        "method",
        "max_step",
        "min_step",
    ],
    "redfield": [
        "atol",
        "rtol",
        "nsteps",
        "method",
        "max_step",
        "min_step",
        "sec_cutoff",
    ],
}

__all__ = [
    "ODE_SOLVER",
    "SIM_TYPE",
    "SOLVER_OPTIONS",
    "ALLOWED_SOLVER_OPTIONS",
]
