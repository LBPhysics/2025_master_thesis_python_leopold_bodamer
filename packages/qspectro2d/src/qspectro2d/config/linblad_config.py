"""Default configuration values for the Lindblad solver."""

DEFAULT_OPTIONS = {
    "atol": 1e-5,
    "rtol": 1e-3,
    "nsteps": 200000,
    "method": "bdf",
}

ALLOWED_OPTIONS = [
    "atol",
    "rtol",
    "nsteps",
    "method",
    "max_step",
    "min_step",
]

__all__ = ["DEFAULT_OPTIONS", "ALLOWED_OPTIONS"]
