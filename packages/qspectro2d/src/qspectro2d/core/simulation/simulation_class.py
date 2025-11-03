"""Core simulation builders (model assembly & interaction Hamiltonians)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from functools import cached_property
from collections.abc import Mapping
import numpy as np
from qutip import Qobj, QobjEvo, ket2dm, mesolve, brmesolve, liouvillian, BosonicEnvironment
from qutip.core.blochredfield import bloch_redfield_tensor
from qutip.solver.heom import HEOMSolver
from typing import Any, Dict, List, Optional, Tuple, Union

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
    _heom_run_kwargs: Dict[str, Any] = field(init=False, default_factory=dict, repr=False)
    _heom_bath_fit_info: Optional[Dict[str, Any]] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.sb_coupling = AtomBathCoupling(self.system, self.bath)
        if self.simulation_config.t_coh_current is None:
            self.simulation_config.t_coh_current = float(self.simulation_config.t_coh_max)

    def _collect_heom_inputs(self) -> Tuple[int, List[Qobj], Any, Dict[str, Any], Dict[str, Any]]:
        """Gather HEOM parameters using a lightweight, dictionary-based layout."""

        solver_opts = self.simulation_config.solver_options
        if solver_opts is not None and not isinstance(solver_opts, Mapping):
            raise TypeError("SimulationConfig.solver_options must be a mapping when using HEOM.")

        if isinstance(solver_opts, Mapping):
            nested_cfg = solver_opts.get("heom")
            if isinstance(nested_cfg, Mapping):
                working_cfg: Dict[str, Any] = dict(nested_cfg)
            else:
                working_cfg = dict(solver_opts)
        else:
            working_cfg = {}

        def pop_key(target: Dict[str, Any], key: str, default: Any = None) -> Any:
            if key in target:
                return target.pop(key)
            return default

        max_depth = int(pop_key(working_cfg, "max_depth", 0))
        if max_depth < 0:
            raise ValueError("HEOM solver requires max_depth >= 0.")

        coupling_ops = self.sb_coupling.heom_coupling_ops()
        if not coupling_ops:
            raise ValueError("HEOM configuration produced no coupling operators.")

        bath_cfg_raw = pop_key(working_cfg, "bath")
        if bath_cfg_raw is None:
            bath_cfg_raw = {}
        if not isinstance(bath_cfg_raw, Mapping):
            raise TypeError("solver_options['bath'] must be a mapping when provided for HEOM.")
        bath_cfg = dict(bath_cfg_raw)

        method = str(bath_cfg.get("approx_method", "prony"))
        if method not in {"sd", "prony"}:
            raise ValueError(f"Unsupported HEOM bath approximation method '{method}'.")

        w_min = float(pop_key(working_cfg, "w_min", 0.0))
        w_max_factor = float(pop_key(working_cfg, "w_max_factor", 10.0))

        if self.bath.tag == "ohmic":
            cutoff_val = self.bath.wc
        elif self.bath.tag == "drudelorentz":
            cutoff_val = self.bath.gamma
        else:
            raise ValueError(f"Unsupported bath type: {self.bath.tag}")
        w_max = float(cutoff_val) * float(w_max_factor)

        if w_max <= w_min:
            w_max = w_min * 10.0

        n_points = max(int(pop_key(working_cfg, "n_points", 200)), 2)
        n_exp = max(int(pop_key(working_cfg, "n_exp", 3)), 1)

        if method == "sd":
            wlist = np.linspace(w_min, w_max, n_points, dtype=float)
            Nk = max(int(pop_key(bath_cfg, "Nk", n_exp)), 1)
            approx_kwargs: Dict[str, Any] = {
                "method": method,
                "wlist": wlist,
                "Nmax": n_exp,
                "Nk": Nk,
                "combine": bool(pop_key(bath_cfg, "combine", True)),
            }
            target_rmse = pop_key(bath_cfg, "target_rmse")
            if target_rmse is not None:
                approx_kwargs["target_rmse"] = float(target_rmse)
            sigma = pop_key(bath_cfg, "sigma")
            if sigma is not None:
                approx_kwargs["sigma"] = sigma
        else:  # method == "prony"
            Nr = max(int(pop_key(bath_cfg, "Nr", n_exp)), 1)
            Ni = max(int(pop_key(bath_cfg, "Ni", n_exp)), 1)
            t_max = float(pop_key(bath_cfg, "t_max", 10.0 / cutoff_val))
            n_t = max(int(pop_key(bath_cfg, "n_t", 1000)), 2)
            tlist = np.linspace(0.0, t_max, n_t, dtype=float)
            approx_kwargs = {
                "method": method,
                "tlist": tlist,
                "Nr": Nr,
                "Ni": Ni,
                "separate": bool(pop_key(bath_cfg, "separate", True)),
                "combine": bool(pop_key(bath_cfg, "combine", True)),
            }
            target_rmse = pop_key(bath_cfg, "target_rmse")
            if target_rmse is not None:
                approx_kwargs["target_rmse"] = float(target_rmse)

        tag = pop_key(bath_cfg, "tag")
        if tag is not None:
            approx_kwargs["tag"] = tag

        bath_env, fit_info = self.bath.approximate(**approx_kwargs)
        self._heom_bath_fit_info = fit_info

        options = {}

        # Extract ODE options from working_cfg
        ode_keys = [
            "atol",
            "rtol",
            "nsteps",
            "method",
            "max_step",
            "min_step",
        ]
        options = {key: working_cfg.pop(key) for key in ode_keys if key in working_cfg}

        # Map method to method for HEOMSolver
        if "method" in options:
            options["method"] = options.pop("method")

        dt = float(self.simulation_config.dt)
        if "max_step" not in options:
            options["max_step"] = dt
        if "min_step" in options:
            if options["min_step"] is None:
                options.pop("min_step")

        run_kwargs = {}

        return max_depth, coupling_ops, bath_env, options, run_kwargs

    def _build_heom_solver(self) -> Tuple[Any, Dict[str, Any]]:
        max_depth, coupling_ops, bath_env, options, run_kwargs = self._collect_heom_inputs()

        bath_specs = [(bath_env, op) for op in coupling_ops]

        H_evo = QobjEvo(self.H_total_t)

        solver = HEOMSolver(
            H=H_evo,
            bath=bath_specs,
            max_depth=max_depth,
            options=options,
        )
        return solver, run_kwargs

    # --- Deferred solver-dependent initialization ---------------------------------
    @property
    def evo_obj(self) -> Union[Qobj, QobjEvo, Any]:
        solver = self.simulation_config.ode_solver
        if solver == "paper_eqs":
            from qspectro2d.core.simulation.liouvillian_paper import matrix_ODE_paper

            evo_obj = QobjEvo(lambda t: matrix_ODE_paper(t, self))
        elif solver == "linblad":
            H_evo = QobjEvo(self.H_total_t)
            c_ops = self.sb_coupling.me_decay_channels
            evo_obj = liouvillian(H_evo, c_ops)
        elif solver == "redfield":
            H_evo = QobjEvo(self.H_total_t)
            solver_opts = (self.simulation_config.solver_options or {}).copy()
            sec_cutoff = solver_opts.pop("sec_cutoff", 0.1)
            if sec_cutoff is None:
                sec_cutoff = 0.1
            sec_cutoff = float(sec_cutoff)
            method_hint = solver_opts.pop("br_computation_method", "sparse")
            a_ops = self.sb_coupling.br_decay_channels
            tensor_kwargs = {"fock_basis": True, "br_computation_method": method_hint}
            evo_obj = bloch_redfield_tensor(
                H_evo,
                a_ops=a_ops,
                sec_cutoff=sec_cutoff,
                **tensor_kwargs,
            )
        elif solver == "montecarlo":
            evo_obj = QobjEvo(self.H_total_t)
        elif solver == "heom":
            heom_solver, run_kwargs = self._build_heom_solver()
            self._heom_run_kwargs = run_kwargs
            evo_obj = heom_solver
        else:  # Fallback: create evolution without lasers
            evo_obj = liouvillian(self.H0_diagonalized)
        return evo_obj

    @property
    def decay_channels(self) -> list[Qobj] | list[tuple[Qobj, BosonicEnvironment]]:
        solver = self.simulation_config.ode_solver
        if solver in {"linblad", "montecarlo"}:
            decay_channels = self.sb_coupling.me_decay_channels
        elif solver == "redfield":
            decay_channels = self.sb_coupling.br_decay_channels
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
        decay_channels = self.decay_channels
        if not decay_channels:  # No bath present -> return ground state
            return self.system.ground_state_dm()

        rho0 = self.system.ground_state_dm()

        solver_opts = dict(self.simulation_config.solver_options or {})
        solver_opts.pop("heom", None)
        solver_opts.update({"store_states": False, "store_final_state": True})

        ode_solver = self.simulation_config.ode_solver or "linblad"

        if ode_solver == "redfield":
            sec_cutoff = solver_opts.pop("sec_cutoff", 0.1)
            if sec_cutoff is None:
                sec_cutoff = 0.1
            method_hint = solver_opts.pop("br_computation_method", None)
            if method_hint is not None and not (
                "br_computation_method" in inspect.signature(bloch_redfield_tensor).parameters
            ):
                print(
                    "⚠️  bloch_redfield_tensor() does not support 'br_computation_method' for this QuTiP version."
                )

            res = brmesolve(
                H,
                rho0,
                tlist,
                a_ops=decay_channels,
                sec_cutoff=float(sec_cutoff),
                options=solver_opts,
            )
        elif ode_solver == "heom":
            raise NotImplementedError(
                "Thermal state preparation via HEOM is not implemented. "
                "Set initial_state='ground' or provide a custom density matrix."
            )
        else:
            # Remove redfield-only knobs before calling mesolve
            solver_opts.pop("sec_cutoff", None)
            solver_opts.pop("br_computation_method", None)

            res = mesolve(
                H,
                rho0,
                tlist,
                c_ops=decay_channels,
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
