"""Runtime simulation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SimulationConfig:
    """Concrete runtime configuration used by the simulation layer only."""

    ode_solver: str = "lindblad"
    solver_options: dict[str, Any] = field(default_factory=dict)
    solver_run_kwargs: dict[str, Any] = field(default_factory=dict)
    rwa_sl: bool = True

    t_det: float = 100.0
    t_coh: float = 0.0
    t_wait: float = 0.0
    dt: float = 0.1
    pulse_fwhm_fs: float = 10.0
    carrier_freq_cm: float = 16000.0
    envelope_type: str = "gaussian"

    initial_state: str = "ground"
    sim_type: str = "1d"
    n_inhomogen: int = 1
    n_phases: int = 4
    inhom_averaged: bool = False
    max_workers: int = 1
    signal_types: tuple[str, ...] = ("rephasing",)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ode_solver": self.ode_solver,
            "solver_options": dict(self.solver_options),
            "solver_run_kwargs": dict(self.solver_run_kwargs),
            "rwa_sl": self.rwa_sl,
            "t_det": self.t_det,
            "t_coh": self.t_coh,
            "t_wait": self.t_wait,
            "dt": self.dt,
            "pulse_fwhm_fs": self.pulse_fwhm_fs,
            "carrier_freq_cm": self.carrier_freq_cm,
            "envelope_type": self.envelope_type,
            "initial_state": self.initial_state,
            "sim_type": self.sim_type,
            "n_inhomogen": self.n_inhomogen,
            "n_phases": self.n_phases,
            "inhom_averaged": self.inhom_averaged,
            "max_workers": self.max_workers,
            "signal_types": list(self.signal_types),
        }
