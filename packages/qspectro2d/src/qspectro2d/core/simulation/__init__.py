"""Simulation subpackage.

Modular simulation components. Legacy `simulation_class.py` has been
fully removed; import from here or concrete submodules directly.

Modules
-------
sim_config        : SimulationConfig dataclass & validation
builders          : Core helper functions (interaction Hamiltonians)
redfield          : Redfield tensor construction helpers
liouvillian_paper : Paper specific time–dependent Liouvillian builders
"""

from .paper_solver import matrix_ODE_paper
from .sim_config import SimulationConfig
from .simulation import SimulationModuleOQS

__all__ = [
    "SimulationConfig",
    "SimulationModuleOQS",
    "matrix_ODE_paper",
]
