"""Thin simulation assembly object."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from qutip import BosonicEnvironment, Qobj, QobjEvo, ket2dm, mesolve

from ..atomic_system import AtomicSystem
from ..bath_coupling import BathCoupling, lindblad_decay_channels, redfield_decay_channels
from ..laser_system import LaserPulseSequence, e_pulses, epsilon_pulses
from .sim_config import SimulationConfig


def split_solver_options(config: SimulationConfig) -> tuple[dict, dict]:
    key = str(config.ode_solver)
    src = dict(config.solver_options)
    option_keys = ("method", "atol", "rtol", "nsteps", "max_step", "min_step", "progress_bar")
    options = {k: src.pop(k) for k in option_keys if k in src}
    if key == "redfield" and "sec_cutoff" in src:
        return {"sec_cutoff": src.pop("sec_cutoff")}, options
    if key in {"lindblad", "paper_eqs", "redfield"}:
        return {}, options
    raise ValueError(f"Unsupported solver '{key}'.")


def build_initial_state(config: SimulationConfig, system: AtomicSystem, bath: BosonicEnvironment) -> Qobj:
    init_choice = getattr(config, "initial_state", "ground")
    if init_choice == "ground":
        return system.to_eigenbasis(system.ground_state_dm())
    if init_choice != "thermal":
        raise ValueError(f"Unsupported initial_state '{init_choice}'")

    temperature = getattr(bath, "T", None)
    if temperature is None or temperature <= 0:
        return system.to_eigenbasis(system.ground_state_dm())

    tlist = np.linspace(0.0, 10000.0, 100)
    result = mesolve(
        H=system.hamiltonian,
        rho0=system.ground_state_dm(),
        tlist=tlist,
        c_ops=lindblad_decay_channels(system),
        options={"store_states": False, "store_final_state": True},
    )
    return system.to_eigenbasis(result.final_state)


def build_observable_ops(system: AtomicSystem) -> list[Qobj]:
    eigenstates = system.eigenstates[1]
    operators = [ket2dm(state) for state in eigenstates]
    dim = system.dimension
    if dim > 1:
        operators.append(sum(eigenstates[0] * eigenstates[index].dag() for index in range(1, dim)))
    if dim > system.n_atoms + 1:
        operators.append(sum(eigenstates[0] * eigenstates[index].dag() for index in range(system.n_atoms + 1, dim)))
        operators.append(
            sum(
                eigenstates[e] * eigenstates[f].dag()
                for e in range(1, system.n_atoms + 1)
                for f in range(system.n_atoms + 1, dim)
            )
        )
    return operators


def build_observable_labels(system: AtomicSystem) -> list[str]:
    labels = [f"pop_{index}" for index in range(system.dimension)]
    if system.dimension > 1:
        labels.append(r"\text{coh}_{\mathrm{ge}}")
    if system.dimension > system.n_atoms + 1:
        labels.append(r"\text{coh}_{\mathrm{gf}}")
        labels.append(r"\text{coh}_{\mathrm{ef}}")
    return labels


@dataclass
class SimulationModuleOQS:
    simulation_config: SimulationConfig
    system: AtomicSystem
    laser: LaserPulseSequence
    bath: BosonicEnvironment
    bath_coupling: BathCoupling = field(init=False)

    def __post_init__(self) -> None:
        self.bath_coupling = BathCoupling(self.system, self.bath)
        if self.simulation_config.t_coh_current is None:
            self.simulation_config.t_coh_current = float(self.simulation_config.t_coh_max)

    def _solver_split(self) -> tuple[dict, dict]:
        return split_solver_options(self.simulation_config)

    @property
    def evo_obj(self) -> Qobj | QobjEvo:
        if self.simulation_config.ode_solver == "paper_eqs":
            from .paper_solver import matrix_ODE_paper

            return QobjEvo(lambda t: matrix_ODE_paper(t, self))
        return QobjEvo(self.H_total_t)

    @property
    def decay_channels(self) -> list[Qobj] | list[tuple[Qobj, BosonicEnvironment]]:
        solver = self.simulation_config.ode_solver
        if solver == "lindblad":
            return lindblad_decay_channels(self.system)
        if solver == "redfield":
            return redfield_decay_channels(self.system, self.bath)
        if solver == "paper_eqs":
            return []
        raise ValueError(f"Unsupported solver '{solver}'.")

    @property
    def sb_coupling(self) -> BathCoupling:
        """Backward-compat alias for bath_coupling."""
        return self.bath_coupling

    @property
    def initial_state(self) -> Qobj:
        return build_initial_state(self.simulation_config, self.system, self.bath)

    @property
    def H0_diagonalized(self) -> Qobj:
        energies, _ = self.system.eigenstates
        hamiltonian = Qobj(np.diag(energies), dims=self.system.hamiltonian.dims)
        if self.simulation_config.rwa_sl:
            hamiltonian -= self.laser.carrier_freq_fs * self.system.to_eigenbasis(self.system.number_op)
        return hamiltonian

    def H_int_sl(self, t: float) -> Qobj:
        lowering_op = self.system.to_eigenbasis(self.system.lowering_op)
        if self.simulation_config.rwa_sl:
            field_plus = e_pulses(t, self.laser)
            return -(lowering_op * np.conj(field_plus) + lowering_op.dag() * field_plus)
        dipole_op = lowering_op + lowering_op.dag()
        field_plus = epsilon_pulses(t, self.laser)
        return -dipole_op * (field_plus + np.conj(field_plus))

    def H_total_t(self, t: float) -> Qobj:
        return self.H0_diagonalized + self.H_int_sl(t)

    @cached_property
    def observable_ops(self) -> list[Qobj]:
        return build_observable_ops(self.system)

    @cached_property
    def observable_strs(self) -> list[str]:
        return build_observable_labels(self.system)

    def update_delays(self, t_coh: float, t_wait: float | None = None) -> None:
        if t_coh is None:
            raise TypeError("t_coh must be provided to update_delays and cannot be None")
        wait_time = float(self.simulation_config.t_wait if t_wait is None else t_wait)
        self.simulation_config.t_wait = wait_time
        self.laser.pulse_delays = [float(t_coh), wait_time]
        self.simulation_config.t_coh_current = float(t_coh)
        if t_coh > self.simulation_config.t_coh_max:
            self.simulation_config.t_coh_max = float(t_coh)


__all__ = [
    "SimulationModuleOQS",
    "build_initial_state",
    "build_observable_labels",
    "build_observable_ops",
    "split_solver_options",
]
