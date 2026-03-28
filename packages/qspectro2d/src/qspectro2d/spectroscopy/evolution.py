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

from copy import deepcopy
import os

import numpy as np
from qutip import Qobj, brmesolve, mesolve

from ..core.simulation import SimulationModuleOQS
from ..core.simulation.time_axes import compute_t_det, compute_times_local
from ..diagnostics.solver_inputs import log_redfield_solver_debug
from ..utils.rwa_utils import from_rotating_frame_list
from .polarisation import complex_polarisation, time_dependent_polarisation_rwa

__all__ = [
    "compute_evolution",
    "compute_polarisation_over_window",
    "simulation_with_pulses",
]


def _redfield_debug_enabled() -> bool:
    """Return whether verbose Redfield diagnostics are enabled explicitly."""
    value = os.getenv("QSPECTRO2D_REDFIELD_DEBUG", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


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

    return np.array([polarisation[int(index)] for index in selected_indices], dtype=complex)


def compute_evolution(
    sim_oqs: SimulationModuleOQS,
    e_ops=None,
    *,
    solver_times: np.ndarray | None = None,
    initial_state: Qobj | None = None,
    field_free: bool = False,
    **override_options: dict,
) -> tuple[np.ndarray, list]:
    """Return the simulation time grid together with expectations or states."""
    if solver_times is None:
        t_list = compute_times_local(sim_oqs.simulation_config)
    else:
        t_list = _validate_external_time_grid(solver_times)

    state0 = sim_oqs.initial_state if initial_state is None else initial_state
    hamiltonian = sim_oqs.evo_obj

    solver = sim_oqs.simulation_config.ode_solver
    run_kwargs, options = sim_oqs._solver_split()
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
        except Exception:
            if _redfield_debug_enabled():
                log_redfield_solver_debug(sim_oqs, t_list, run_kwargs, options)
            raise
    elif solver in {"lindblad", "paper_eqs"}:
        result = mesolve(
            H=hamiltonian,
            rho0=state0,
            tlist=t_list,
            c_ops=sim_oqs.decay_channels,
            e_ops=e_ops,
            options=options,
        )
    else:
        raise ValueError(f"Unsupported solver '{solver}'.")

    data = result.expect[0] if e_ops is not None else result.states
    if e_ops is None and not field_free and sim_oqs.simulation_config.rwa_sl and len(t_list):
        data = from_rotating_frame_list(
            data,
            np.array(t_list, float),
            sim_oqs.system.n_atoms,
            sim_oqs.laser.carrier_freq_fs,
        )

    return np.array(t_list, float), data


def compute_polarisation_over_window(
    sim_oqs: SimulationModuleOQS,
    window: np.ndarray | list | None = None,
    *,
    solver_times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evolve once with current laser settings and return (t_det, P(t_det)),
    where P is computed on the requested detection-window times.
    """
    if window is None:
        window = compute_t_det(sim_oqs.simulation_config)
    window = np.asarray(window, dtype=float)

    dipole_op = sim_oqs.dipole_op_eigenbasis

    def polarisation_density(state):
        rho = state.extract(0) if hasattr(state, "extract") else state
        if hasattr(rho, "isket") and rho.isket:
            rho = rho.proj()
        return rho

    if sim_oqs.simulation_config.rwa_sl:

        def polarisation(t, state):
            return time_dependent_polarisation_rwa(
                dipole_op,
                polarisation_density(state),
                t,
                sim_oqs.system.n_atoms,
                sim_oqs.laser.carrier_freq_fs,
            )

    else:

        def polarisation(_t, state):
            return complex_polarisation(dipole_op, polarisation_density(state))

    times, polarisation_t = compute_evolution(
        sim_oqs,
        e_ops=[polarisation],
        solver_times=solver_times,
    )

    return window, slice_polarisation_to_window(times, polarisation_t, window)


def simulation_with_pulses(
    sim_oqs: SimulationModuleOQS,
    active_indices: list[int],
) -> SimulationModuleOQS:
    """Return a deep-copied simulation with only the selected pulses active."""
    sim_copy = deepcopy(sim_oqs)
    sim_copy.laser = sim_copy.laser.subset(active_indices)
    sim_copy.refresh_cache()
    return sim_copy
