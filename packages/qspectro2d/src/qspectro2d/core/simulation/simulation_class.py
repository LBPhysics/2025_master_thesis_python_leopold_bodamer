"""Core simulation builders (model assembly & interaction Hamiltonians)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from functools import cached_property
from collections.abc import Mapping
import numpy as np
from qutip import Qobj, QobjEvo, ket2dm, mesolve, brmesolve, liouvillian, BosonicEnvironment
from qutip.core.blochredfield import bloch_redfield_tensor
from typing import Any, Dict, List, Optional, Tuple, Union

from sympy import solve

from ..atomic_system import AtomicSystem
from ..laser_system import e_pulses, epsilon_pulses, LaserPulseSequence
from ..atom_bath_class import AtomBathCoupling
from .sim_config import SimulationConfig


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

    # --- tiny, strict splitter (incl. HEOM) ---------------------------------------
    def _solver_split(self) -> tuple[dict, dict, str]:
        """
        Split self.simulation_config.solver_options into:
        run_kwargs  -> solver call
        options     -> ODE options (passed as options=...)
        solver_key  -> {'linblad','redfield','montecarlo','heom'}
        """
        key = str(self.simulation_config.ode_solver).lower().strip()
        src = dict(self.simulation_config.solver_options or {})

        # universal ODE options
        opt_keys = ("method", "atol", "rtol", "nsteps", "max_step", "min_step", "progress_bar")
        options = {k: src.pop(k) for k in opt_keys if k in src}

        if key in {"linblad", "lindblad", "mesolve"}:
            key = "linblad"
            return {}, options, key

        if key in {"redfield", "brmesolve", "blochredfield"}:
            run = {}
            if "sec_cutoff" in src:
                run["sec_cutoff"] = src.pop("sec_cutoff")
            return run, options, "redfield"

        if key in {"montecarlo", "mcsolve"}:
            run = {}
            if "ntraj" in src:
                run["ntraj"] = int(src.pop("ntraj"))
            return run, options, "montecarlo"

        if key == "heom":
            # only runtime kw that goes to heomsolve; bath fit handled below
            run = {}
            if "max_depth" in src:
                run["max_depth"] = int(src.pop("max_depth"))
            # keep remaining keys in src for HEOM fit (_collect_heom_inputs consumes them)
            self._heom_src_cache = src  # stash local leftovers for the collector
            return run, options, "heom"

        raise ValueError(f"unknown solver '{key}'")

    def _collect_heom_inputs(self) -> tuple[int, list[Qobj], Any, dict, dict]:
        """
        Build HEOM inputs:
        return max_depth, coupling_ops, bath_env, options, run_kwargs
        """
        # pull remaining solver keys captured in _solver_split
        src = getattr(self, "_heom_src_cache", {}) or dict(
            self.simulation_config.solver_options or {}
        )

        coupling_ops = self.sb_coupling.heom_coupling_ops()

        # frequency window
        w_min = float(src.pop("w_min"))
        w_max_factor = float(src.pop("w_max_factor"))
        cutoff_val = (
            self.bath.wc
            if getattr(self.bath, "tag", "") == "ohmic"
            else getattr(self.bath, "gamma", 1.0)
        )
        w_max = float(cutoff_val) * float(w_max_factor)
        if w_max <= w_min:
            w_max = w_min * 10.0

        # time grid for env approximation
        t_max = float(src.pop("t_max"))
        n_t = int(src.pop("n_t"))
        tlist = np.linspace(0.0, t_max, n_t, dtype=float)

        approx_kwargs = dict(
            method=str(src.pop("approx_method")),
            tlist=tlist,
            Nr=int(src.pop("Nr")),
            Ni=int(src.pop("Ni")),
            separate=bool(src.pop("separate")),
            combine=bool(src.pop("combine")),
        )
        tag = src.pop("tag", None)
        if tag is not None:
            approx_kwargs["tag"] = tag

        bath_env, _fit = self.bath.approximate(**approx_kwargs)

        # max_depth is passed via run_kwargs; default to 1 if not set earlier
        max_depth = int(src.pop("max_depth", 1))
        run_kwargs = {"max_depth": max_depth}
        options = {}  # ODE options come from _solver_split
        return max_depth, coupling_ops, bath_env, options, run_kwargs

    @property
    def evo_obj(self) -> Union[Qobj, QobjEvo, Any]:
        solver = self.simulation_config.ode_solver
        if solver == "paper_eqs":
            from qspectro2d.core.simulation.liouvillian_paper import matrix_ODE_paper

            evo_obj = QobjEvo(lambda t: matrix_ODE_paper(t, self))
        else:  # solver == "linblad", "redfield", "montecarlo" or "heom":
            evo_obj = QobjEvo(self.H_total_t)
        return evo_obj

    @property
    def decay_channels(self) -> list[Qobj] | list[tuple[Qobj, BosonicEnvironment]]:
        solver = self.simulation_config.ode_solver
        if solver in {"linblad", "montecarlo"}:
            decay_channels = self.sb_coupling.me_decay_channels
        elif solver == "redfield":
            decay_channels = self.sb_coupling.br_decay_channels
        elif solver == "heom":
            decay_channels = self.sb_coupling.heom_coupling_ops()
        else:  # for paper_eqs & Fallback: create generic evolution with no decay channels.
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

        return self.system.to_eigenbasis(initial_state)

    def _thermal_state(self) -> Qobj:
        """Return the Gibbs state associated with the system Hamiltonian and bath temperature.
        TODO implement properly"""
        temperature = getattr(self.bath, "T", None)
        if temperature is None or temperature <= 0:
            return self.system.ground_state_dm()

        tlist = np.linspace(0.0, 10000.0, 100)
        H = self.system.hamiltonian
        rho0 = self.system.ground_state_dm()

        all_ops = {}
        all_ops.update({"store_states": False, "store_final_state": True})
        res = mesolve(
            H=H,
            rho0=rho0,
            tlist=tlist,
            c_ops=self.me_decay_channels,
            options=all_ops,
        )
        return res.final_state

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
