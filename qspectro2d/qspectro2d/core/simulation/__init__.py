"""Simulation subpackage.

Modular simulation components. Legacy `simulation_class.py` has been
fully removed; import from here or concrete submodules directly.

Modules
-------
sim_config        : SimulationConfig dataclass & validation
builders          : Core helper functions (interaction Hamiltonians)
liouvillian_paper : Paper specific time–dependent Liouvillian builders
redfield          : Redfield tensor construction helpers
"""

from .sim_config import SimulationConfig
from .simulation_class import SimulationModuleOQS
from .liouvillian_paper import matrix_ODE_paper

__all__ = [
    "SimulationConfig",
    "SimulationModuleOQS",
    "matrix_ODE_paper",
]
