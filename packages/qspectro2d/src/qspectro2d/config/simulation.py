"""Simulation defaults for qspectro2d."""

from .heom_config import DEFAULT_OPTIONS as HEOM_DEFAULTS, ALLOWED_OPTIONS as HEOM_ALLOWED
from .linblad_config import (
    DEFAULT_OPTIONS as LINBLAD_DEFAULTS,
    ALLOWED_OPTIONS as LINBLAD_ALLOWED,
)
from .montecarlo_config import (
    DEFAULT_OPTIONS as MONTECARLO_DEFAULTS,
    ALLOWED_OPTIONS as MONTECARLO_ALLOWED,
)
from .redfield_config import (
    DEFAULT_OPTIONS as REDFIELD_DEFAULTS,
    ALLOWED_OPTIONS as REDFIELD_ALLOWED,
)


# === SIMULATION DEFAULTS ===
ODE_SOLVER = "redfield"
SIM_TYPE = "1d"

SOLVER_OPTIONS = {
    "heom": HEOM_DEFAULTS,
    "redfield": REDFIELD_DEFAULTS,
    "montecarlo": MONTECARLO_DEFAULTS,
    "linblad": LINBLAD_DEFAULTS,
    "paper_eqs": {},
}

ALLOWED_SOLVER_OPTIONS = {
    "linblad": LINBLAD_ALLOWED,
    "redfield": REDFIELD_ALLOWED,
    "montecarlo": MONTECARLO_ALLOWED,
    "heom": HEOM_ALLOWED,
    "paper_eqs": [],
}

__all__ = [
    "ODE_SOLVER",
    "SIM_TYPE",
    "SOLVER_OPTIONS",
    "ALLOWED_SOLVER_OPTIONS",
]
