"""
Supported options for qspectro2d.
"""

# supported solvers and bath models
SUPPORTED_SOLVERS = ["linblad", "redfield", "paper_eqs", "heom", "montecarlo"]

SUPPORTED_BATHS = ["ohmic", "drudelorentz"]

SUPPORTED_ENVELOPES = ["gaussian", "cos2"]

SUPPORTED_SIM_TYPES = ["0d", "1d", "2d"]
