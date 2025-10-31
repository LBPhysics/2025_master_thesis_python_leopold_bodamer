"""Core simulation builders (model assembly & interaction Hamiltonians)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import numpy as np
from qutip import (
    Qobj,
    QobjEvo,
    ket2dm,
    brmesolve,
    liouvillian,
)
from qutip.core.blochredfield import bloch_redfield_tensor
from typing import List, Union
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
        if self.simulation_config.t_coh_current is None:
            self.simulation_config.t_coh_current = float(self.simulation_config.t_coh_max)

    # --- Deferred solver-dependent initialization ---------------------------------
    @property
    def evo_obj(self) -> Union[Qobj, QobjEvo]:
        solver = self.simulation_config.ode_solver
        if solver == "Paper_eqs":
            from qspectro2d.core.simulation.liouvillian_paper import matrix_ODE_paper

            evo_obj = QobjEvo(lambda t: matrix_ODE_paper(t, self))
        elif solver == "ME":
            H_evo = QobjEvo(self.H_total_t)
            c_ops = self.sb_coupling.me_decay_channels
            evo_obj = liouvillian(H_evo, c_ops)
        elif solver == "BR":
            H_evo = QobjEvo(self.H_total_t)
            solver_opts = self.simulation_config.solver_options or {}
            sec_cutoff = solver_opts.get("sec_cutoff", 0.1)
            if sec_cutoff is None:
                sec_cutoff = 0.1
            sec_cutoff = float(sec_cutoff)
            tensor_type = (
                solver_opts.get("br_computation_method")
                or solver_opts.get("tensor_type")
                or "sparse"
            )
            tensor_type = str(tensor_type)
            a_ops = self.sb_coupling.br_decay_channels
            evo_obj = bloch_redfield_tensor(
                H_evo,
                a_ops=a_ops,
                sec_cutoff=sec_cutoff,
                fock_basis=True,
                br_computation_method=tensor_type,
            )
        else:  # Fallback: create evolution without lasers
            evo_obj = liouvillian(self.H0_diagonalized)
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

    @property
    def initial_state(self) -> Qobj:
        """Density matrix used as the solver's initial condition.
        TODO For now only 'ground' is supported. Thermal state behaves weird for 4 level system"""
        init_choice = getattr(self.simulation_config, "initial_state", "ground")
        if init_choice == "ground":
            initial_state = self.system.ground_state_dm()
        if init_choice == "thermal":
            initial_state = self._thermal_state()

        if not initial_state:
            raise ValueError(
                f"Unsupported initial_state '{init_choice}'. Expected 'ground' or 'thermal'."
            )
        return self.system.to_eigenbasis(initial_state)

    def _thermal_state(self) -> Qobj:
        """Return the Gibbs state associated with the system Hamiltonian and bath temperature."""
        temperature = getattr(self.bath, "T", None)
        if temperature is None or temperature <= 0:
            return self.system.ground_state_dm()

        # Use Bloch-Redfield evolution to relax into the steady state.
        tlist = np.linspace(0.0, 10000.0, 100)
        H = self.H0_diagonalized
        a_ops = self.decay_channels
        if not a_ops:  # No BLOCH REDFIELD decay channels; return ground state
            return self.system.ground_state_dm()

        rho0 = self.system.ground_state_dm()
        solver_opts = (self.simulation_config.solver_options or {}).copy()
        sec_cutoff = solver_opts.pop("sec_cutoff", 0.1)
        if sec_cutoff is None:
            sec_cutoff = 0.1
        solver_opts.pop("br_computation_method", None)
        solver_opts.pop("tensor_type", None)
        solver_opts.update({"store_states": False, "store_final_state": True})

        res = brmesolve(
            H,
            rho0,
            tlist,
            a_ops=a_ops,
            sec_cutoff=sec_cutoff,
            options=solver_opts,
        )
        rho_ss = getattr(res, "final_state", None)
        if rho_ss is None:
            raise ValueError("Thermal state computation failed: solver returned no state.")
        trace = rho_ss.tr()
        if np.isclose(trace, 0.0):
            raise ValueError("Thermal state computation failed: steady state has zero trace.")
        return rho_ss / trace

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
            N_eig = self.system.to_eigenbasis(self.system.number_op)
            H_diag -= omega_L * N_eig
        return H_diag

    def H_int_sl(self, t: float) -> Qobj:
        """
        Interaction Hamiltonian:
        With
            H_int = -(σ- E⁻(t) + σ+ E⁺(t))
            where   E⁺(t) = positive-frequency component of E_i(t), e.g. E_i^0 * exp(-iφ_i-wL*t_i)
                    σ⁻ is THE LOWERING OPERATOR / also the positive frequency part of the dipole operator
        Without RWA (full field):
            H_int(t) = -[E⁺(t) + E⁻(t)] ⊗ (σ⁺ + σ⁻)
        """
        lowering_op = self.system.to_eigenbasis(
            self.system.lowering_op
        )  # oscillates as exp(+i ω_L t) in RWA frame
        if self.simulation_config.rwa_sl:
            E_plus_RWA = e_pulses(t, self.laser)  # oscillates as exp(-i ω_L t) in lab frame
            E_minus_RWA = np.conj(E_plus_RWA)
            H_int = -(lowering_op * E_minus_RWA + lowering_op.dag() * E_plus_RWA)
            return H_int
        dipole_op = lowering_op + lowering_op.dag()
        E_plus = epsilon_pulses(t, self.laser)
        H_int = -dipole_op * (E_plus + np.conj(E_plus))
        return H_int

    def H_total_t(self, t: float) -> Qobj:
        """Return total Hamiltonian H0 + H_int(t) at time t."""
        H_total = self.H0_diagonalized + self.H_int_sl(t)
        return H_total

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
    def update_delays(self, t_coh: float, t_wait: float | None = None) -> None:
        """Update laser pulse delays.
        t_coh must be provided. Optionally provide a new t_wait;
        """
        # Enforce explicit t_coh (no None allowed)
        if t_coh is None:
            raise TypeError("t_coh must be provided to update_delays and cannot be None")

        # Update wait time if provided, otherwise keep current
        if t_wait is not None:
            self.simulation_config.t_wait = float(t_wait)
        else:
            t_wait = float(self.simulation_config.t_wait)

        # Apply to laser pulse delays and invalidate cached time properties
        self.laser.pulse_delays = [float(t_coh), float(t_wait)]
        self.simulation_config.t_coh_current = float(t_coh)

        if t_coh > self.simulation_config.t_coh_max:
            self.simulation_config.t_coh_max = t_coh
