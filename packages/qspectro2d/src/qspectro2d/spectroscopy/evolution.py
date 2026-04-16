"""Compute 1D emitted electric-field building blocks via phase-cycled third-order polarisation.

This module mirrors the physics flow used in the older implementation:

Steps:
- E_ks(t) is built from the phase-cycled polarisation component P_ks(t).
- P_{l,m}(t) = sum_{phi1} sum_{phi2} P_{phi1,phi2}(t) * exp(-i(l phi1 + m phi2 + n PHI_DET)).
- P_{phi1,phi2}(t) = P_total(t) - sum_i P_i(t), with P_total using all pulses and P_i
    with only pulse i active.
- P(t) is the complex/analytical polarisation evaluated from the dipole component used in
    qspectro2d.spectroscopy.polarisation.

Supports lindblad, redfield, and paper_eqs solvers via the internals of
SimulationModuleOQS.
"""

from __future__ import annotations
from qutip import Qobj, brmesolve, mesolve
import numpy as np
from qutip.solver.integrator.integrator import IntegratorException

from ..core.simulation import SimulationModuleOQS
from ..core.simulation.time_axes import compute_t_det, compute_times_local
from ..diagnostics.solver_inputs import log_lindblad_solver_debug, log_redfield_solver_debug

__all__ = [
    "compute_evolution",
    "compute_polarisation_over_window",
]


def _validate_external_time_grid(tlist: np.ndarray) -> np.ndarray:
    tlist = np.asarray(tlist, dtype=float)
    if tlist.ndim != 1:
        raise ValueError("solver_times must be a one-dimensional array")
    if len(tlist) < 2:
        raise ValueError("solver_times must contain at least two points")
    if not np.all(np.isfinite(tlist)):
        raise ValueError("solver_times must contain only finite values")
    if not np.all(np.diff(tlist) > 0):
        raise ValueError("solver_times must be strictly increasing")
    return tlist


def slice_polarisation_to_window(
    times: np.ndarray,
    polarisation: list,
    window: np.ndarray,
) -> np.ndarray:
    """Slice the polarisation list to the requested detection-window times."""
    times = np.asarray(times)
    window = np.asarray(window)
    polarisation_array = np.asarray(polarisation, dtype=np.complex128)

    indices = np.searchsorted(times, window, side="left")
    indices = np.clip(indices, 0, len(times) - 1)

    selected_indices = np.zeros_like(indices)
    for position, (index, time_value) in enumerate(zip(indices, window)):
        candidates = [index]
        if index > 0:
            candidates.append(index - 1)
        if index < len(times) - 1:
            candidates.append(index + 1)

        selected_indices[position] = min(
            candidates,
            key=lambda candidate: abs(times[candidate] - time_value),
        )

    return polarisation_array[selected_indices]


def compute_evolution(
    sim_oqs: SimulationModuleOQS,
    e_ops=None,
    *,
    solver_times: np.ndarray | None = None,
    initial_state: Qobj | None = None,
    field_free: bool = False,
    log_solver_debug_on_error: bool = False,
    **override_options: dict,
) -> tuple[np.ndarray, list]:
    """Return the simulation time grid together with expectations or states.

    Notes
    -----
    - If ``e_ops is None``, the returned data are solver states.
    - If ``e_ops`` is provided and has length 1, the returned data are a single
      expectation-value array.
    - If ``e_ops`` has length > 1, the returned data are a list of arrays.
    """
    if solver_times is None:
        t_list = compute_times_local(sim_oqs.simulation_config)
    else:
        t_list = _validate_external_time_grid(solver_times)

    state0 = sim_oqs.initial_state if initial_state is None else initial_state
    hamiltonian = sim_oqs.evo_obj

    solver = sim_oqs.simulation_config.ode_solver
    run_kwargs = sim_oqs.simulation_config.solver_run_kwargs.copy()
    options = sim_oqs.simulation_config.solver_options.copy()
    options.update(override_options or {})
    options.setdefault("progress_bar", False)

    if e_ops is not None:
        options.setdefault("store_states", False)
        options.setdefault("store_final_state", True)
    else:
        options.setdefault("store_states", True)

    if field_free:
        if solver == "paper_eqs":
            from ..core.simulation.paper_solver import paper_liouvillian_l0

            hamiltonian = paper_liouvillian_l0(sim_oqs)
        else:
            hamiltonian = sim_oqs.H0_diagonalized

    if solver == "redfield":
        try:
            result = brmesolve(
                H=hamiltonian,
                psi0=state0,
                tlist=t_list,
                a_ops=sim_oqs.decay_channels,
                e_ops=e_ops,
                options=options,
                **run_kwargs,
            )
        except Exception as exc:
            if isinstance(exc, IntegratorException) and str(options.get("method", "")).lower() == "lsoda":
                retry_options = options.copy()
                retry_options["method"] = "bdf"
                try:
                    result = brmesolve(
                        H=hamiltonian,
                        psi0=state0,
                        tlist=t_list,
                        a_ops=sim_oqs.decay_channels,
                        e_ops=e_ops,
                        options=retry_options,
                        **run_kwargs,
                    )
                except Exception:
                    if log_solver_debug_on_error:
                        log_redfield_solver_debug(sim_oqs, t_list, run_kwargs, retry_options)
                    raise
            else:
                if log_solver_debug_on_error:
                    log_redfield_solver_debug(sim_oqs, t_list, run_kwargs, options)
                raise
    elif solver == "lindblad":
        try:
            result = mesolve(
                H=hamiltonian,
                rho0=state0,
                tlist=t_list,
                c_ops=sim_oqs.decay_channels,
                e_ops=e_ops,
                options=options,
                **run_kwargs,
            )
        except Exception as exc:
            if isinstance(exc, IntegratorException) and str(options.get("method", "")).lower() == "lsoda":
                retry_options = options.copy()
                retry_options["method"] = "bdf"
                try:
                    result = mesolve(
                        H=hamiltonian,
                        rho0=state0,
                        tlist=t_list,
                        c_ops=sim_oqs.decay_channels,
                        e_ops=e_ops,
                        options=retry_options,
                        **run_kwargs,
                    )
                except Exception:
                    if log_solver_debug_on_error:
                        log_lindblad_solver_debug(sim_oqs, t_list, run_kwargs, retry_options)
                    raise
            else:
                if log_solver_debug_on_error:
                    log_lindblad_solver_debug(sim_oqs, t_list, run_kwargs, options)
                raise
    elif solver == "paper_eqs":
        result = mesolve(
            H=hamiltonian,
            rho0=state0,
            tlist=t_list,
            c_ops=sim_oqs.decay_channels,
            e_ops=e_ops,
            options=options,
            **run_kwargs,
        )
    else:
        raise ValueError(f"Unsupported solver '{solver}'.")

    if e_ops is None:
        data = result.states
    elif len(result.expect) == 1:
        data = np.asarray(result.expect[0], dtype=np.complex128)
    else:
        data = [np.asarray(values, dtype=np.complex128) for values in result.expect]

    return np.asarray(t_list, dtype=float), data


def compute_polarisation_over_window(
    sim_oqs: SimulationModuleOQS,
    window=None,
    *,
    solver_times=None,
    log_solver_debug_on_error: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical positive-frequency signal on the requested detection axis.

    For ``rwa_sl=False`` this is the lab-frame positive-frequency polarisation.
    For ``rwa_sl=True`` this is the rotating-frame envelope consistent with the
    paper convention used by ``fields.e_pulses``.
    """
    if window is None:
        window = compute_t_det(sim_oqs.simulation_config)
    window = np.asarray(window, dtype=float)

    dipole_op = sim_oqs.dipole_op_eigenbasis
    mu_plus = Qobj(np.tril(dipole_op.full(), k=-1), dims=dipole_op.dims)

    times, polarisation_t = compute_evolution(
        sim_oqs,
        e_ops=[mu_plus],
        solver_times=solver_times,
        log_solver_debug_on_error=log_solver_debug_on_error,
    )
    return window, slice_polarisation_to_window(times, polarisation_t, window)
