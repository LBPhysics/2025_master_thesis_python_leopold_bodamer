"""Simulation configuration data structures."""

from __future__ import annotations

from typing import List
from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    """Primary configuration object for simulations.

    Focused immutable configuration object; no legacy compatibility paths.
    """

    ode_solver: str = "ME"
    # Solver and pulse/detection options
    solver_options: dict[str, float | int] = field(default_factory=lambda: {})
    rwa_sl: bool = True

    dt: float = 0.1
    t_coh: float = 0.0
    t_wait: float = 0.0
    t_det_max: float = 100.0

    # potentially do 2d inhomogeneous broadened
    sim_type: str = "1d"
    n_inhomogen: int = 1

    n_phases: int = 4

    # Sampling / batching metadata
    inhom_averaged: bool = False  # True if data represent an average over inhom configs

    max_workers: int = 1
    signal_types: List[str] = field(default_factory=lambda: ["rephasing"])

    def __post_init__(self) -> None:
        # Enforce RWA for Paper_eqs
        if self.ode_solver == "Paper_eqs" and not self.rwa_sl:
            print(
                "⚠️  Warning: rwa_sl forced True for Paper_eqs solver.",
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
            f"Inhom Samples      : {self.n_inhomogen}\n"
            f"Inhom Averaged     : {self.inhom_averaged}\n"
            f"Max Workers        : {self.max_workers}\n"
            "-------------------------------\n"
        )

    def to_dict(self) -> dict:
        result = {
            "ode_solver": self.ode_solver,
            "rwa_sl": self.rwa_sl,
            "sim_type": self.sim_type,
            "max_workers": self.max_workers,
            "signal_types": self.signal_types,
            "t_det_max": self.t_det_max,
            "dt": self.dt,
            "t_wait": self.t_wait,
        }
        if self.t_coh is not None:
            result["t_coh"] = self.t_coh
        if self.solver_options:
            result["solver_options"] = self.solver_options
        result["n_inhomogen"] = self.n_inhomogen
        if self.n_inhomogen != 1:
            result["inhom_averaged"] = self.inhom_averaged
        if self.n_phases != 4:
            result["n_phases"] = self.n_phases
        return result

    def __str__(self) -> str:
        return self.summary()
