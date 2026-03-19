"""Simulation configuration data structures."""

from __future__ import annotations

from typing import Any, List
import warnings
from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    """Primary configuration object for simulations.

    Focused immutable configuration object; no legacy compatibility paths.
    """

    ode_solver: str = "lindblad"
    # Solver and pulse/detection options
    solver_options: dict[str, Any] = field(default_factory=lambda: {})
    rwa_sl: bool = True

    t_det: float = 100.0
    t_coh: float = 0.0
    t_wait: float = 0.0
    dt: float = 0.1
    pulse_fwhm_fs: float = 10.0

    # Runtime/adaptive attributes
    initial_state: str = "ground"

    # potentially do 2d inhomogeneous broadened
    sim_type: str = "1d"
    n_inhomogen: int = 1

    n_phases: int = 4

    # Sampling / batching metadata
    inhom_averaged: bool = False  # True if data represent an average over inhom configs

    max_workers: int = 1
    signal_types: List[str] = field(default_factory=lambda: ["rephasing"])

    def __post_init__(self) -> None:
        # Enforce RWA for paper_eqs
        if self.ode_solver == "paper_eqs" and not self.rwa_sl:
            warnings.warn(
                "rwa_sl forced True for paper_eqs solver.",
                category=UserWarning,
                stacklevel=2,
            )
            self.rwa_sl = True

    def summary(self) -> str:
        lines = [
            "SimulationConfig Summary:\n",
            "-------------------------------\n",
            f"{self.sim_type} ELECTRONIC SPECTROSCOPY SIMULATION\n",
            f"Signal Type        : {self.signal_types}\n",
            "Time Parameters:\n",
            f"Coherence Time     : {self.t_coh} fs\n",
            f"Wait Time          : {self.t_wait} fs\n",
            f"Max Det. Time      : {self.t_det} fs\n\n",
            f"Time Step (dt)     : {self.dt} fs\n",
            f"Pulse FWHM         : {self.pulse_fwhm_fs} fs\n",
            "-------------------------------\n",
            f"Solver Type        : {self.ode_solver}\n",
            f"Use rwa_sl         : {self.rwa_sl}\n\n",
            "-------------------------------\n",
            f"Phase Cycles       : {self.n_phases}\n",
            f"Inhom Samples      : {self.n_inhomogen}\n",
            f"Inhom Averaged     : {self.inhom_averaged}\n",
            f"Max Workers        : {self.max_workers}\n",
            "-------------------------------\n",
        ]
        return "".join(lines)

    def to_dict(self) -> dict:
        result = {
            "ode_solver": self.ode_solver,
            "rwa_sl": self.rwa_sl,
            "sim_type": self.sim_type,
            "max_workers": self.max_workers,
            "t_det": self.t_det,
            "dt": self.dt,
            "t_wait": self.t_wait,
            "t_coh": self.t_coh,
            "pulse_fwhm_fs": self.pulse_fwhm_fs,
        }
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
