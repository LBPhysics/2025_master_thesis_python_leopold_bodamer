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
    def decay_channels(self):
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
        if self.simulation_config.rwa_sl:
            omega_L = self.laser.carrier_freq_fs
            # Determine excitation number for each eigenstate
            # Based on index: 0 -> 0 excitations, 1..N -> 1, N+1..end -> 2
            H_diag -= omega_L * self.system.number_op  # is the same in both bases
        return H_diag

    def paper_eqs_evo(self, t: float) -> Qobj:  # pragma: no cover simple wrapper
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
        """
        lowering_op = self.system.lowering_op
        lowering_op = self.system.to_eigenbasis(lowering_op)
        if self.simulation_config.rwa_sl:
            E_plus_RWA = e_pulses(t, self.laser)
            H_int = -(lowering_op * E_plus_RWA + lowering_op.dag() * np.conj(E_plus_RWA))
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

        t0 = -1 * self.laser.pulse_fwhms[0]
        cfg = self.simulation_config
        t_max_curr = cfg.t_coh + cfg.t_wait + cfg.t_det_max
        dt = cfg.dt
        # Compute number of steps to cover from t0 to t_max_curr with step dt
        n_steps = int(np.floor((t_max_curr - t0) / dt)) + 1
        # Generate time grid: [t0, t0 + dt, ..., t_max_curr]
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
        # Detection time grid with exact spacing dt starting at the first time >0 in times_local.
        dt = self.simulation_config.dt
        t_det_max = self.simulation_config.t_det_max
        # Compute the first time > 0
        times_local = self.times_local
        t_start = times_local[times_local >= 0][0]
        n_steps = int(np.floor(t_det_max / dt)) + 1
        if t_start + dt * (n_steps - 1) > t_det_max:
            # Cap it to avoid overshooting t_det_max and times_local
            n_steps = int(np.floor((t_det_max - t_start) / dt)) + 1
            n_steps = min(n_steps, times_local[times_local >= 0].size)
        return t_start + dt * np.arange(n_steps, dtype=float)

    @property
    def t_det_actual(self):
        cfg = self.simulation_config
        t_det0 = cfg.t_coh + cfg.t_wait
        return self.t_det + t_det0

    # --- Helper functions -----------------------------------------------------------
    # MAKE THEM TIME DEP HERE

    # only for the paper solver
    def time_dep_paper_gamma_ij(self, i: int, j: int, t: float) -> float:
        """
        Calculate the population relaxation rates. for the dimer system, analogous to the gamma_ij in the paper.

        Parameters:
            i (int): Index of the first state.
            j (int): Index of the second state.

        Returns:
            float: Population relaxation rate.
        """
        w_ij = self.time_dep_omega_ij(i, j, t)
        """
        from qspectro2d.core.bath_system.bath_fcts import (
            power_spectrum_func_paper,
            extract_bath_parameters,
        )

        args = extract_bath_parameters(self.bath)
        
        args["alpha"] = args["alpha"] * args["wc"]  # rescale coupling for paper eqs
        P_wij = power_spectrum_func_paper(w_ij, **args)
        """
        P_wij = self.bath.power_spectrum(w_ij)
        result = np.sin(2 * self.sb_coupling.theta) ** 2 * P_wij
        # Handle NaN/inf values
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    def time_dep_paper_Gamma_ij(self, i: int, j: int, t: float) -> float:
        """
        Calculate the pure dephasing rates. for the dimer system, analogous to the gamma_ij in the paper.

        Parameters:
            i (int): Index of the first state.
            j (int): Index of the second state.

        Returns:
            float: Pure dephasing rate.
        """
        # Pure dephasing rates helper
        """
        from qspectro2d.core.bath_system.bath_fcts import (
            power_spectrum_func_paper,
            extract_bath_parameters,
        )

        args = extract_bath_parameters(self.bath)
        args["alpha"] = args["alpha"] * args["wc"]  # rescale coupling for paper eqs
        P_0 = power_spectrum_func_paper(0, **args)
        """
        P_0 = self.bath.power_spectrum(0)
        P_0 = np.nan_to_num(P_0, nan=0.0, posinf=0.0, neginf=0.0)  # Handle NaN/inf

        Gamma_t_ab = 2 * np.cos(2 * self.sb_coupling.theta) ** 2 * P_0  # tilde
        Gamma_t_a0 = (1 - 0.5 * np.sin(2 * self.sb_coupling.theta) ** 2) * P_0
        Gamma_11 = self.time_dep_paper_gamma_ij(2, 1, t)
        Gamma_22 = self.time_dep_paper_gamma_ij(1, 2, t)
        Gamma_abar_0 = 2 * P_0
        Gamma_abar_a = Gamma_abar_0  # holds for dimer

        result = 0.0
        if i == 1:
            if j == 0:
                result = Gamma_t_a0 + 0.5 * self.time_dep_paper_gamma_ij(2, i, t)
            elif j == 1:
                result = Gamma_11
            elif j == 2:
                gamma_ij = self.time_dep_paper_gamma_ij(i, j, t)
                gamma_ji = self.time_dep_paper_gamma_ij(j, i, t)
                result = Gamma_t_ab + 0.5 * (gamma_ij + gamma_ji)
        elif i == 2:
            if j == 0:
                result = Gamma_t_a0 + 0.5 * self.time_dep_paper_gamma_ij(1, i, t)
            elif j == 1:
                gamma_ij = self.time_dep_paper_gamma_ij(i, j, t)
                gamma_ji = self.time_dep_paper_gamma_ij(j, i, t)
                result = Gamma_t_ab + 0.5 * (gamma_ij + gamma_ji)
            elif j == 2:
                result = Gamma_22
        elif i == 3:
            if j == 0:
                result = Gamma_abar_0
            elif j == 1:
                result = Gamma_abar_a + 0.5 * (self.time_dep_paper_gamma_ij(2, j, t))
            elif j == 2:
                result = Gamma_abar_a + 0.5 * (self.time_dep_paper_gamma_ij(1, j, t))
        else:
            raise ValueError("Invalid indices for i and j.")

        # Handle NaN/inf values in the final result
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
