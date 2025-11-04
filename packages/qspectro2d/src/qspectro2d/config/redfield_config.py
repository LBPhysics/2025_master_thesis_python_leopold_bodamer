"""Default configuration values for the Bloch-Redfield solver."""

DEFAULT_OPTIONS = {
    "sec_cutoff": -1,
    "br_computation_method": "sparse",
    "atol": 1e-4,
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
    "sec_cutoff",
    "br_computation_method",
]

__all__ = ["DEFAULT_OPTIONS", "ALLOWED_OPTIONS"]
