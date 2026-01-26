"""Simulation defaults for qspectro2d."""

# === SIMULATION DEFAULTS ===
ODE_SOLVER = "redfield"
SIM_TYPE = "1d"

SOLVER_OPTIONS = {
    "paper_eqs": {},
    "redfield": {
        "sec_cutoff": -1,
        "atol": 1e-4,
        "rtol": 1e-3,
        "nsteps": 200000,
        "method": "bdf",
    },
    "lindblad": {
        "atol": 1e-5,
        "rtol": 1e-3,
        "nsteps": 200000,
        "method": "bdf",
    },
}

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
