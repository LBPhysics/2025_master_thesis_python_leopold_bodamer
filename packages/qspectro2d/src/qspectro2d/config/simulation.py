"""Simulation defaults for qspectro2d."""

# === SIMULATION DEFAULTS ===
ODE_SOLVER = "redfield"
SIM_TYPE = "1d"

SOLVER_OPTIONS = {
    "heom": {
        "max_depth": 1,  # 1 for redfield limit, higher more accurate
        "atol": 1e-5,
        "rtol": 1e-3,
        "nsteps": 200000,
        "method": "bdf",
        # BATH Approximation options
        "approx_method": "prony",
        "Ni": 5,  # good approximation
        "Nr": 5,
        "combine": True,
        "separate": True,
        "n_t": 1000,
        "t_max": 500.0,
    },
    "redfield": {
        "sec_cutoff": -1,
        "atol": 1e-4,
        "rtol": 1e-3,
        "nsteps": 200000,
        "method": "bdf",
    },
    "montecarlo": {
        "ntraj": 2,
        "atol": 1e-5,
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
    "montecarlo": [
        "atol",
        "rtol",
        "nsteps",
        "method",
        "max_step",
        "min_step",
        "ntraj",
        "progress_bar",
    ],
    "heom": [
        "max_depth",
        "approx_method",
        "n_exp",
        "Ni",
        "Nr",
        "combine",
        "separate",
        "n_t",
        "t_max",
        "tag",
        "atol",
        "rtol",
        "nsteps",
        "method",
        "max_step",
        "min_step",
        "progress_bar",
    ],
    "paper_eqs": [],
}

__all__ = [
    "ODE_SOLVER",
    "SIM_TYPE",
    "SOLVER_OPTIONS",
    "ALLOWED_SOLVER_OPTIONS",
]
