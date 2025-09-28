"""Simulation configuration data structures (renamed from `config.py`).

Primary immutable configuration object for simulations.
"""

from __future__ import annotations

from typing import List
from dataclasses import dataclass, asdict, field
import warnings


@dataclass
class SimulationConfig:
    """Primary configuration object for simulations.

    Focused immutable configuration object; no legacy compatibility paths.
    """

    ode_solver: str = "ME"
    # Solver and pulse/detection options
    solver_options: dict[str, float | int] = field(
        default_factory=lambda: {"nsteps": 200000, "atol": 1e-6, "rtol": 1e-4}
    )
    rwa_sl: bool = True

    dt: float = 0.1
    t_coh: float = 0.0
    t_wait: float = 0.0
    t_det_max: float = 100.0

    n_phases: int = 4
    n_inhomogen: int = 1

    # Inhomogeneous handling / bookkeeping
    inhom_enabled: bool = False  # True if current run loops over inhom configs
    inhom_averaged: bool = False  # True if data represent an average over inhom configs
    inhom_index: int = 0  # Current inhom configuration index (0 for homogeneous)
    inhom_group_id: str | None = None  # Stable group id for batching/averaging

    max_workers: int = 1
    sim_type: str = "1d"
    signal_types: List[str] = field(default_factory=lambda: ["rephasing"])

    def __post_init__(self) -> None:
        # Enforce RWA for Paper_eqs
        if self.ode_solver == "Paper_eqs" and not self.rwa_sl:
            warnings.warn(
                "rwa_sl forced True for Paper_eqs solver.",
                category=UserWarning,
                stacklevel=2,
            )
            self.rwa_sl = True

    def summary(self) -> str:
        return (
            "SimulationConfig Summary:\n"
            "-------------------------------\n"
            f"{self.sim_type} ELECTRONIC SPECTROSCOPY SIMULATION\n"
            f"Signal Type        : {self.signal_types}\n"
            "Time Parameters:\n"
            f"Coherence Time     : {self.t_coh} fs\n"
            f"Wait Time          : {self.t_wait} fs\n"
            f"Max Det. Time      : {self.t_det_max} fs\n\n"
            f"Time Step (dt)     : {self.dt} fs\n"
            "-------------------------------\n"
            f"Solver Type        : {self.ode_solver}\n"
            f"Use rwa_sl         : {self.rwa_sl}\n\n"
            "-------------------------------\n"
            f"Phase Cycles       : {self.n_phases}\n"
            f"Inhom. Points      : {self.n_inhomogen}\n"
            f"Inhom Enabled      : {self.inhom_enabled}\n"
            f"Inhom Averaged     : {self.inhom_averaged}\n"
            f"Inhom Index        : {self.inhom_index}\n"
            f"Max Workers        : {self.max_workers}\n"
            "-------------------------------\n"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover simple repr
        return self.summary()
