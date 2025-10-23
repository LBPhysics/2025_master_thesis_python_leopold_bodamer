"""Core simulation builders (model assembly & interaction Hamiltonians)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import numpy as np
from qutip import Qobj, QobjEvo, ket2dm
from typing import List, Tuple, Union
from qutip import BosonicEnvironment

from .sim_config import SimulationConfig
from ..atomic_system import AtomicSystem
from ..laser_system import e_pulses, epsilon_pulses, LaserPulseSequence
from ..atom_bath_class import AtomBathCoupling


@dataclass
class SimulationModuleOQS:
    simulation_config: SimulationConfig

    system: AtomicSystem
    laser: LaserPulseSequence
    bath: BosonicEnvironment

    sb_coupling: AtomBathCoupling = field(init=False)

    def __post_init__(self) -> None:
        self.sb_coupling = AtomBathCoupling(self.system, self.bath)

    # --- Deferred solver-dependent initialization ---------------------------------
    @property
    def evo_obj(self) -> Union[Qobj, QobjEvo]:
        solver = self.simulation_config.ode_solver
        if solver == "Paper_eqs":
            evo_obj = QobjEvo(self.paper_eqs_evo)
        elif solver == "ME" or solver == "BR":
            evo_obj = QobjEvo(self.H_total_t)
        else:  # Fallback: create evolution without lasers
            evo_obj = self.H0_diagonalized
        return evo_obj

    @property
    def decay_channels(self) -> list[Qobj] | list[tuple[Qobj, BosonicEnvironment]]:
        solver = self.simulation_config.ode_solver
        if solver == "ME":
            decay_channels = self.sb_coupling.me_decay_channels
        elif solver == "BR":
            decay_channels = self.sb_coupling.br_decay_channels
        else:  # for Paper_eqs & Fallback: create generic evolution with no decay channels.
            decay_channels = []
        return decay_channels

    # --- Hamiltonians & Evolutions -------------------------------------------------
    @property
    def H0_diagonalized(self) -> Qobj:
        """Return diagonal Hamiltonian (optionally shifted by laser frequency under RWA)."""
        Es, _ = self.system.eigenstates
        H_diag = Qobj(np.diag(Es), dims=self.system.hamiltonian.dims)
        return H_diag

    def paper_eqs_evo(self, t: float) -> Qobj:
        """Global helper for 'Paper_eqs' solver evolution.

        Kept at module scope so partial(paper_eqs_evo, sim) remains pickleable.
        Lazy import inside to avoid circular import at module load.
        """
        from qspectro2d.core.simulation.liouvillian_paper import (
            matrix_ODE_paper as _matrix_ODE_paper,
        )

        return _matrix_ODE_paper(t, self)

    def H_int_sl(self, t: float) -> Qobj:
        """
        Interaction Hamiltonian:
        With
            H_int = -(σ- E⁺(t) + σ+ E⁻(t))
            where   E⁺(t) = positive-frequency component of E(t), e.g. E * exp(-iφ)
                    σ⁻ is THE LOWERING OPERATOR / also the positive frequency part of the dipole operator
        Without RWA (full field):
            H_int(t) = -[E⁺(t) + E⁻(t)] ⊗ (σ⁺ + σ⁻)
                    = -E(t) (σ⁺ + σ⁻)
        With RWA (rotating frame transformation):
            Compute the full H_int(t), then transform to rotating frame:
            H_rot = W†(t) H_int(t) W(t) - iℏ W†(t) W˙(t)
            Since ℏ=1, and W˙ = i ω N W, so -i W† W˙ = ω N
            Thus H_rot = W† H_int W + ω N
        """
        lowering_op = self.system.lowering_op  # rotates with exp(+iwt) in RWA
        lowering_op = self.system.to_eigenbasis(lowering_op)  # rotates with exp(-iwt) in RWA

        if self.simulation_config.rwa_sl:
            # Compute full H_int(t)
            dipole_op = lowering_op + lowering_op.dag()
            E_plus = epsilon_pulses(t, self.laser)
            H_full = -dipole_op * (E_plus + np.conj(E_plus))

            # Transform to rotating frame
            from qspectro2d.utils.rwa_utils import rotating_frame_unitary, _excitation_number_vector

            omega_L = self.laser.carrier_freq_fs
            n_atoms = self.system.n_atoms
            W = rotating_frame_unitary(H_full, t, n_atoms, omega_L)
            N_vec = _excitation_number_vector(H_full.shape[0], n_atoms)
            N = Qobj(np.diag(N_vec), dims=H_full.dims)
            H_int = W.dag() * H_full * W + omega_L * N
            return H_int

        dipole_op = lowering_op + lowering_op.dag()
        E_plus = epsilon_pulses(t, self.laser)
        H_int = -dipole_op * (E_plus + np.conj(E_plus))
        return H_int

    def H_total_t(self, t: float) -> Qobj:
        """Return total Hamiltonian H0 + H_int(t) at time t."""
        H_total = self.H0_diagonalized + self.H_int_sl(t)
        return H_total

    def time_dep_eigenstates(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalues & eigenstates."""
        return self.H_total_t(t).eigenstates()

    def time_dep_omega_ij(self, i: int, j: int, t: float) -> float:
        """Return energy difference (frequency) between instantaneous eigenstates i and j in fs^-1."""
        Es, _ = self.time_dep_eigenstates(t)
        return Es[i] - Es[j]

    # --- Observables ---------------------------------------------------------------
    @cached_property
    def observable_ops(self) -> List[Qobj]:
        """in the eigenbasis of H0 (diagonalized system Hamiltonian)."""
        sys = self.system
        n = sys.n_atoms

        eigenstates = sys.eigenstates[1]
        ops = [ket2dm(state) for state in eigenstates]  # populations

        # Add coherences: |g><e|, |g><f|, |e><f|
        dim = sys.dimension
        if dim > 1:
            # |g><e| for all  for e (1, ..., n_atoms)
            ops.append(sum(eigenstates[0] * eigenstates[e].dag() for e in range(1, dim)))
        if dim > n + 1:
            # |g><f| for f (n_atoms+1, ..., dim)
            ops.append(sum(eigenstates[0] * eigenstates[f].dag() for f in range(n + 1, dim)))
            # |e><f| for e (1, ..., n_atoms) and f (n_atoms+1, ..., dim)
            ops.append(
                sum(
                    eigenstates[e] * eigenstates[f].dag()
                    for e in range(1, n + 1)
                    for f in range(n + 1, dim)
                )
            )

        return ops

    @cached_property
    def observable_strs(self) -> List[str]:
        sys = self.system
        n = sys.n_atoms
        dim = sys.dimension
        strs = []
        # Populations
        strs.extend([f"pop_{i}" for i in range(dim)])
        # Coherences
        if dim > 1:
            strs.append(r"\text{coh}_{\text{ge}}")
        if dim > n + 1:
            strs.append(r"\text{coh}_{\text{gf}}")
            strs.append(r"\text{coh}_{\text{ef}}")
        return strs

    # --- Time grids ----------------------------------------------------------------
    def update_delays(self, t_coh: float = None, t_wait: float = None) -> None:
        """Update laser pulse delays from current simulation config.
        If t_coh or t_wait are not provided, keep existing values."""

        if t_coh is not None:
            self.simulation_config.t_coh = t_coh
        else:
            t_coh = self.simulation_config.t_coh

        if t_wait is not None:
            self.simulation_config.t_wait = t_wait
        else:
            t_wait = self.simulation_config.t_wait

        self.laser.pulse_delays = [t_coh, t_wait]
        self.reset_times_local()

    @property
    def times_local(self):
        if hasattr(self, "_times_local_manual"):
            return self._times_local_manual

        cfg = self.simulation_config
        t0 = -2 * self.laser.carrier_fwhm_fs - cfg.t_coh - cfg.t_wait
        dt = cfg.dt
        # Compute number of steps to cover from t0 to t_det_max with step dt
        n_steps = int(np.floor((cfg.t_det_max - t0) / dt)) + 1
        # Generate time grid: [t0, t0 + dt, ..., t_det_max]
        times = t0 + dt * np.arange(n_steps, dtype=float)
        return times

    @times_local.setter
    def times_local(self, times: np.ndarray):
        self._times_local_manual = np.asarray(times, dtype=float).reshape(-1)

    def reset_times_local(self):
        if hasattr(self, "_times_local_manual"):
            delattr(self, "_times_local_manual")
        self.times_local  # Recompute based on config

    @cached_property
    def t_det(self):
        # Detection time grid with exact spacing dt starting at 0.
        dt = self.simulation_config.dt
        t_det_max = self.simulation_config.t_det_max
        n_steps = int(np.floor(t_det_max / dt)) + 1
        t_det = dt * np.arange(n_steps, dtype=float)
        # Ensure it doesn't exceed t_det_max
        t_det = t_det[t_det <= t_det_max]
        return t_det
