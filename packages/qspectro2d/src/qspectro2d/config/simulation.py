"""
Simulation defaults for qspectro2d.
"""

# === SIMULATION DEFAULTS ===
ODE_SOLVER = "redfield"  # ODE solver to use
SIM_TYPE = "1d"
SOLVER_OPTIONS = {
    "heom": {
        "max_depth": 1, # 1 for redfield limit, higher more accurate
        "bath": {
            "approx_method": "prony",
            "Ni": 5, # good approximation
            "Nr": 5,
            "combine": True,
            "n_t": 1000,
        },
        "atol": 1e-5,
        "rtol": 1e-3,
        "nsteps": 200000,
        "method": "bdf",
    },
    "redfield": {
        "sec_cutoff": -1,
        "br_computation_method": "sparse",
        "atol": 1e-4,
        "rtol": 1e-3,
        "nsteps": 200000,
        "method": "bdf",
    },
    "montecarlo": {
        "ntraj": 64,
        "atol": 1e-5,
        "rtol": 1e-3,
        "nsteps": 200000,
        "method": "bdf",
    },
    "linblad": {
        "atol": 1e-5,
        "rtol": 1e-3,
        "nsteps": 200000,
        "method": "bdf",
    },
}
ALLOWED_SOLVER_OPTIONS = {
    "linblad": [
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
        "br_computation_method",
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
        "bath",
        "w_min",
        "w_max",
        "w_max_factor",
        "n_points",
        "n_exp",
        "atol",
        "rtol",
        "nsteps",
        "method",
        "max_step",
        "min_step",
    ],
    "paper_eqs": [],
}
