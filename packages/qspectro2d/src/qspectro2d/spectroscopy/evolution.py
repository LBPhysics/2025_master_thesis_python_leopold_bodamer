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

import numpy as np
from qutip import brmesolve, mesolve

from ..core.simulation import SimulationModuleOQS
from ..core.simulation.time_axes import compute_t_det, compute_times_local
from ..utils.rwa_utils import from_rotating_frame_list
from .polarisation import complex_polarisation, time_dependent_polarisation_rwa

__all__ = [
    "compute_evolution",
    "compute_polarisation_over_window",
    "simulation_with_pulses",
]


def slice_polarisation_to_window(
    times: np.ndarray,
    polarisation: list,
    window: np.ndarray,
) -> np.ndarray:
    """Slice the polarisation list to the requested detection-window times.

    Returns:
        Array of polarisation values corresponding to the window time points.
    """
    times = np.asarray(times)
    window = np.asarray(window)

    # Nearest index for each window time.
    indices = np.searchsorted(times, window, side="left")
    indices = np.clip(indices, 0, len(times) - 1)

    selected_indices = np.zeros_like(indices)
    for position, (index, time_value) in enumerate(zip(indices, window)):
        # Candidates: index-1, index, index+1 (if within bounds).
        candidates = [index]
        if index > 0:
            candidates.append(index - 1)
        if index < len(times) - 1:
            candidates.append(index + 1)
        # Choose the candidate with minimal absolute difference.
        selected_indices[position] = min(candidates, key=lambda candidate: abs(times[candidate] - time_value))

    return np.array([polarisation[int(index)] for index in selected_indices], dtype=complex)


def compute_evolution(
    sim_oqs: SimulationModuleOQS,
    e_ops=None,
    **override_options: dict,
) -> tuple[np.ndarray, list]:
    """Return the simulation time grid together with expectations or states.

    Returns:
        (tlist, data), where data is expectations if e_ops is given and states otherwise.
    """
    t_list = compute_times_local(sim_oqs.simulation_config)
    state0 = sim_oqs.initial_state
    hamiltonian = sim_oqs.evo_obj

    solver = sim_oqs.simulation_config.ode_solver
    run_kwargs, options = sim_oqs._solver_split()
    options.update(override_options or {})
    # Silence QuTiP solver progress output unless explicitly overridden.
    options.setdefault("progress_bar", False)

    if e_ops is not None:
        options.setdefault("store_states", False)
        options.setdefault("store_final_state", True)
    else:
        options.setdefault("store_states", True)

    if solver == "redfield":
        result = brmesolve(
            H=hamiltonian,
            psi0=state0,
            tlist=t_list,
            a_ops=sim_oqs.decay_channels,
            e_ops=e_ops,
            options=options,
            **run_kwargs,
        )
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
    if e_ops is None and sim_oqs.simulation_config.rwa_sl and len(t_list):
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evolve once with current laser settings and return (t_det, P(t_det)),
    where P is computed on the requested detection-window times.
    """
    if window is None:
        window = compute_t_det(sim_oqs.simulation_config)
    window = np.asarray(window, dtype=float)

    # Build the dipole operator in the energy eigenbasis. The polarisation helper then selects
    # the spectroscopic component consistently for both RWA and non-RWA branches.
    dipole_op = sim_oqs.system.to_eigenbasis(sim_oqs.system.dipole_op)

    def polarisation_density(state):
        rho = state.extract(0) if hasattr(state, "extract") else state
        if hasattr(rho, "isket") and rho.isket:
            rho = rho.proj()
        return rho

    if sim_oqs.simulation_config.rwa_sl:

        # Rotate the states back to the lab frame before taking the polarisation expectation value.
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

    # Get the polarisation on the global time grid.
    times, polarisation_t = compute_evolution(sim_oqs, e_ops=[polarisation])

    # Select the polarisation at the desired window times (nearest matches).
    return window, slice_polarisation_to_window(times, polarisation_t, window)


def simulation_with_pulses(
    sim_oqs: SimulationModuleOQS,
    active_indices: list[int],
) -> SimulationModuleOQS:
    """Return a deep-copied simulation with only the selected pulses active.

    Notes:
        - Deep-copy is used to avoid mutating the input and to be process-safe.
        - Phases and timings remain unchanged.
    """
    sim_copy = deepcopy(sim_oqs)
    sim_copy.laser = sim_copy.laser.subset(active_indices)
    return sim_copy