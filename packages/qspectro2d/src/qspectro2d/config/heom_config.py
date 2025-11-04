"""Default configuration values for the HEOM solver."""

DEFAULT_OPTIONS = {
    "max_depth": 1,
    "bath": {
        "approx_method": "prony",
        "Ni": 5,
        "Nr": 5,
        "combine": True,
        "n_t": 1000,
    },
    "atol": 1e-5,
    "rtol": 1e-3,
    "nsteps": 200000,
    "method": "bdf",
}

ALLOWED_OPTIONS = [
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
]

__all__ = ["DEFAULT_OPTIONS", "ALLOWED_OPTIONS"]
