"""Core simulation builders (model assembly & interaction Hamiltonians)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from functools import cached_property
import numpy as np
from qutip import (
    Qobj,
    QobjEvo,
    ket2dm,
    mesolve,
    brmesolve,
    liouvillian,
)
from qutip.core.blochredfield import bloch_redfield_tensor
from typing import Any, Dict, List, Optional, Tuple, Union
from qutip import BosonicEnvironment

from .sim_config import SimulationConfig
from ..atomic_system import AtomicSystem
from ..laser_system import e_pulses, epsilon_pulses, LaserPulseSequence
from ..atom_bath_class import AtomBathCoupling
from qutip.solver.heom import HEOMSolver

from .heom_defaults import (
    HEOM_DEFAULT_INCLUDE_DOUBLE,
    HEOM_DEFAULT_MAX_DEPTH,
    HEOM_DEFAULT_METHOD,
    HEOM_DEFAULT_N_EXP,
    HEOM_DEFAULT_N_POINTS,
    HEOM_DEFAULT_W_MAX_FACTOR,
    HEOM_DEFAULT_W_MIN,
)

_BRT_SUPPORTS_METHOD = (
    "br_computation_method" in inspect.signature(bloch_redfield_tensor).parameters
)


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

    def _build_heom_solver(self) -> Tuple[Any, Dict[str, Any]]:
        heom_cfg, run_kwargs = self._resolve_heom_config()

        max_depth_val = heom_cfg.get("max_depth", HEOM_DEFAULT_MAX_DEPTH)
        try:
            max_depth = int(max_depth_val)
        except (TypeError, ValueError) as exc:  # pragma: no cover - invalid user config
            raise ValueError("HEOM solver requires integer 'max_depth'.") from exc
        if max_depth < 0:
            raise ValueError("HEOM solver requires max_depth >= 0.")

        bath_cfg = dict(heom_cfg["bath"])
        bath_env = self._approximate_heom_environment(bath_cfg)

        include_double = bool(heom_cfg.get("include_double", HEOM_DEFAULT_INCLUDE_DOUBLE))
        coupling_ops = self._resolve_heom_couplings(heom_cfg.get("sites"), include_double)
        if not coupling_ops:
            raise ValueError("HEOM configuration produced no coupling operators.")
        bath_specs = [(bath_env, op) for op in coupling_ops]

        solver_options = self._normalize_heom_options(heom_cfg.get("options"))

        H_evo = QobjEvo(self.H_total_t)
        solver = HEOMSolver(H_evo, bath_specs, max_depth, options=solver_options)
        return solver, run_kwargs

    def _resolve_heom_config(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        solver_opts = self.simulation_config.solver_options or {}
        heom_section = solver_opts.get("heom")
        if heom_section is None:
            heom_section = {}
        elif not isinstance(heom_section, dict):
            raise TypeError("solver_options['heom'] must be a dict when using the HEOM solver.")

        cfg = dict(heom_section)

        bath_cfg = cfg.get("bath")
        if bath_cfg is None:
            bath_cfg = {}
        elif not isinstance(bath_cfg, dict):
            raise TypeError("solver_options['heom']['bath'] must be a dict when provided.")
        cfg["bath"] = dict(bath_cfg)

        run_kwargs: Dict[str, Any] = {}
        args_value = cfg.pop("args", None)
        if args_value is not None:
            if not isinstance(args_value, dict):
                raise TypeError("solver_options['heom']['args'] must be a dict when provided.")
            run_kwargs["args"] = args_value

        run_kwargs.setdefault("progress_bar", None)

        return cfg, run_kwargs

    def _approximate_heom_environment(self, cfg: Dict[str, Any]):
        method = str(cfg.get("method", HEOM_DEFAULT_METHOD)).lower()
        if method != HEOM_DEFAULT_METHOD:
            raise ValueError(f"Only HEOM bath method '{HEOM_DEFAULT_METHOD}' is supported.")

        w_min = float(cfg.get("w_min", HEOM_DEFAULT_W_MIN))
        w_max_factor = float(cfg.get("w_max_factor", HEOM_DEFAULT_W_MAX_FACTOR))
        wc = getattr(self.bath, "wc", None)
        if wc is None:
            freq_ref = getattr(self.system, "frequencies_fs", None)
            wc = float(np.max(freq_ref)) if freq_ref is not None else 1.0
        w_max = float(cfg.get("w_max", wc * w_max_factor))
        if w_max <= w_min:
            w_max = w_min * 10.0

        n_points = int(cfg.get("n_points", HEOM_DEFAULT_N_POINTS))
        if n_points < 2:
            raise ValueError("HEOM bath 'n_points' must be >= 2.")
        n_exp = int(cfg.get("n_exp", HEOM_DEFAULT_N_EXP))
        if n_exp <= 0:
            raise ValueError("HEOM bath 'n_exp' must be positive.")

        wlist = np.linspace(w_min, w_max, n_points, dtype=float)
        approx_kwargs = {"wlist": wlist, "Nmax": n_exp}

        bath_env, fit_info = self.bath.approximate(method, **approx_kwargs)
        self._heom_bath_fit_info = fit_info
        return bath_env

    def _resolve_heom_couplings(self, sites_cfg: Any, include_double: bool) -> List[Qobj]:
        if sites_cfg is None:
            site_indices = list(range(1, self.system.n_atoms + 1))
        else:
            try:
                site_indices = [int(idx) for idx in sites_cfg]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "solver_options['heom']['sites'] must be a sequence of integers."
                ) from exc

        return self.sb_coupling.heom_coupling_ops(
            sites=site_indices,
            include_double_manifold=include_double,
        )

    def _normalize_heom_options(self, raw: Any) -> Dict[str, Any]:
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise TypeError("HEOM solver options must be provided as a dict.")

        options = dict(raw)
        options.setdefault("store_states", True)
        options.setdefault("store_final_state", True)
        options.setdefault("progress_bar", None)
        return options

    # --- Deferred solver-dependent initialization ---------------------------------
    @property
    def evo_obj(self) -> Union[Qobj, QobjEvo, Any]:
        solver = self.simulation_config.ode_solver
        solver_upper = solver.upper() if isinstance(solver, str) else solver
        if solver == "Paper_eqs":
            from qspectro2d.core.simulation.liouvillian_paper import matrix_ODE_paper

            evo_obj = QobjEvo(lambda t: matrix_ODE_paper(t, self))
        elif solver_upper == "ME":
            H_evo = QobjEvo(self.H_total_t)
            c_ops = self.sb_coupling.me_decay_channels
            evo_obj = liouvillian(H_evo, c_ops)
        elif solver_upper == "BR":
            H_evo = QobjEvo(self.H_total_t)
            solver_opts = (self.simulation_config.solver_options or {}).copy()
            sec_cutoff = solver_opts.pop("sec_cutoff", 0.1)
            if sec_cutoff is None:
                sec_cutoff = 0.1
            sec_cutoff = float(sec_cutoff)
            method_hint = solver_opts.pop("br_computation_method", None)
            if method_hint is None and "tensor_type" in solver_opts:
                method_hint = solver_opts.pop("tensor_type")
            tensor_type = str(method_hint or "sparse")
            a_ops = self.sb_coupling.br_decay_channels
            tensor_kwargs = {"fock_basis": True}
            if _BRT_SUPPORTS_METHOD:
                tensor_kwargs["br_computation_method"] = tensor_type
            elif method_hint is not None:
                print(
                    "⚠️  bloch_redfield_tensor() does not support 'br_computation_method' for this QuTiP version."
                )
            evo_obj = bloch_redfield_tensor(
                H_evo,
                a_ops=a_ops,
                sec_cutoff=sec_cutoff,
                **tensor_kwargs,
            )
        elif solver_upper == "HEOM":
            heom_solver, run_kwargs = self._build_heom_solver()
            self._heom_run_kwargs = run_kwargs
            evo_obj = heom_solver
            self._heom_run_kwargs = run_kwargs
            evo_obj = heom_solver
        else:  # Fallback: create evolution without lasers
            evo_obj = liouvillian(self.H0_diagonalized)
        return evo_obj

    @property
    def decay_channels(self) -> list[Qobj] | list[tuple[Qobj, BosonicEnvironment]]:
        solver = self.simulation_config.ode_solver
        solver_upper = solver.upper() if isinstance(solver, str) else solver
        if solver_upper == "ME":
            decay_channels = self.sb_coupling.me_decay_channels
        elif solver_upper == "BR":
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
        decay_channels = self.decay_channels
        if not decay_channels:  # No bath present -> return ground state
            return self.system.ground_state_dm()

        rho0 = self.system.ground_state_dm()

        solver_opts = dict(self.simulation_config.solver_options or {})
        solver_opts.pop("heom", None)
        solver_opts.update({"store_states": False, "store_final_state": True})

        ode_solver = (self.simulation_config.ode_solver or "ME").upper()

        if ode_solver == "BR":
            sec_cutoff = solver_opts.pop("sec_cutoff", 0.1)
            if sec_cutoff is None:
                sec_cutoff = 0.1
            method_hint = solver_opts.pop("br_computation_method", None)
            if method_hint is None and "tensor_type" in solver_opts:
                method_hint = solver_opts.pop("tensor_type")
            if method_hint is not None and not _BRT_SUPPORTS_METHOD:
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
        elif ode_solver == "HEOM":
            raise NotImplementedError(
                "Thermal state preparation via HEOM is not implemented. "
                "Set initial_state='ground' or provide a custom density matrix."
            )
        else:
            # Remove BR-only knobs before calling mesolve
            solver_opts.pop("sec_cutoff", None)
            solver_opts.pop("br_computation_method", None)
            solver_opts.pop("tensor_type", None)

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
