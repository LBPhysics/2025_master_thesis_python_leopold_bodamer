# TODO rename this module
"""Compute 1D emitted electric field E_ks(t_det) via phase-cycled third-order polarization.

This module provides a clean, focused API that mirrors the physics and flow you described:

Steps:
- E_ks(t) ∝ i P_ks(t)
- P_{l,m}(t) = Σ_{phi1} Σ_{phi2} P_{phi1,phi2}(t) * exp(-i(l phi1 + m phi2 + n PHI_DET))
- P_{phi1,phi2}(t) = P_total(t) - Σ_i P_i(t), with P_total using all pulses and P_i with only pulse i active
- P(t) is the complex/analytical polarization: P(t) = ⟨μ_+⟩(t), using the positive-frequency part of μ

Supports ME and BR solvers via the internals of SimulationModuleOQS.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from copy import deepcopy
from qutip import Qobj, Result, mesolve, brmesolve

from .polarization import complex_polarization
from ..core.simulation.simulation_class import SimulationModuleOQS
from ..core.laser_system.laser_class import LaserPulse, LaserPulseSequence
from ..config.default_simulation_params import (
    PHASE_CYCLING_PHASES,
    COMPONENT_MAP,
    DETECTION_PHASE,
)
from qspectro2d.utils.rwa_utils import from_rotating_frame_list


__all__ = [
    "parallel_compute_1d_e_comps",
]


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------
def slice_states_to_window(res: Result, window: np.ndarray) -> List[Qobj]:
    """Slice the list of states in `res` to only keep the detection window portion."""
    # Assuming res_times is sorted and equally spaced
    times = np.asarray(res.times)
    if len(times) < 2:
        raise ValueError("res.times must have at least 2 elements to determine dt.")
    window = np.asarray(window)
    start_idx = np.argmin(np.abs(times - window[0]))
    idxs = start_idx + np.arange(len(window))
    idxs = np.clip(idxs, 0, len(times) - 1)  # stay in bounds
    return [res.states[i] for i in idxs]


def compute_evolution(
    sim_oqs: SimulationModuleOQS,
    **solver_options: dict,
) -> Result:
    """
    Compute the evolution of the quantum system for a given pulse sequence.

    Parameters
    ----------
    sim_oqs : SimulationModuleOQS
        Prepared simulation object containing:
    **solver_options : dict
        User overrides for solver options (highest precedence).
        Examples: store_states=True, atol=1e-8, rtol=1e-6

    Returns
    -------
    Result
        QuTiP Result object containing:
        - states: List of density matrices at each time point (if store_states=True)
        - times: Array of time points corresponding to states
        - final_state: Final density matrix (if store_final_state=True)

    Notes
    -----
    The evolution is computed using either mesolve (master equation) or brmesolve
    (Bloch-Redfield master equation) depending on the ode_solver setting.
    Evolution starts from the beginning of the first pulse's active window and fills
    vacant beginning slots with the initial state.
    """
    from qspectro2d.config.default_simulation_params import SOLVER_OPTIONS

    options: dict = SOLVER_OPTIONS
    if solver_options:
        options.update(solver_options)

    # Get the starting point of the first pulse's active window
    if sim_oqs.laser.pulses:
        pulse_start_time = sim_oqs.laser.pulses[0].active_time_range[0]
        # print(f"Pulse start time: {pulse_start_time} fs")
    else:
        pulse_start_time = sim_oqs.times_local[0]  # fallback if no pulses

    # Find the index in times_local that corresponds to pulse_start_time
    times_array = np.asarray(sim_oqs.times_local)
    start_idx = np.argmin(np.abs(times_array - pulse_start_time))

    # Create modified time list starting from pulse active window
    modified_times = times_array[start_idx:]

    ode_solver = sim_oqs.simulation_config.ode_solver
    if ode_solver == "BR":
        # Optional Bloch-Redfield secular cutoff passthrough TODO add to options
        # sec_cutoff = options.get("sec_cutoff", None)
        res = brmesolve(
            H=sim_oqs.evo_obj,
            psi0=sim_oqs.system.psi_ini,
            tlist=modified_times,
            a_ops=sim_oqs.decay_channels,
            options=options,
            # sec_cutoff=sec_cutoff,
        )
    else:
        res = mesolve(
            H=sim_oqs.evo_obj,
            rho0=sim_oqs.system.psi_ini,
            tlist=modified_times,
            c_ops=sim_oqs.decay_channels,
            options=options,
        )

    # Fill vacant beginning slots with initial state if needed
    if start_idx > 0:
        # Create states list with initial state for the vacant beginning slots
        initial_states = [sim_oqs.system.psi_ini] * start_idx
        if hasattr(res, "states") and res.states is not None:
            res.states = initial_states + res.states

        # Restore original time list
        res.times = sim_oqs.times_local

    return res


def compute_polarization_over_window(
    sim: SimulationModuleOQS,
    window: np.ndarray | List = None,
    *,
    store_states: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evolve the system once (with current laser settings) and return (t_det, P(t_det)).

    - Uses `compute_seq_evolution` which dispatches ME/BR according to sim.simulation_config.
    - Extracts complex/analytical polarization over the detection window only.
    """
    # Ensure we store states to extract polarization
    res: Result = compute_evolution(sim, store_states=store_states)

    # print length of state-indices that are NOT THE GROUND STATE
    non_ground_state_indices = [
        i for i, state in enumerate(res.states) if not (state == sim.system.psi_ini)
    ]
    """    print("Length of non-ground state indices:", len(non_ground_state_indices))
    if len(non_ground_state_indices) != 0:
        print("found other states than ground state")
    else:
        print("no states other than ground state were stored")
    """
    window_states = slice_states_to_window(res, window)

    if sim.simulation_config.rwa_sl:
        # States are stored in the rotating frame; convert back to lab for polarization
        # Use RELATIVE times w.r.t. the start of the simulation window to avoid
        # imprinting a t_coh-dependent global phase across traces.
        window_rel = np.asarray(window) - float(res.times[0])
        window_states = from_rotating_frame_list(
            window_states, window_rel, sim.system.n_atoms, sim.laser.carrier_freq_fs
        )

    # Analytical polarization using positive-frequency part of dipole operator
    mu_op = sim.system.to_eigenbasis(sim.system.dipole_op)
    P_t = complex_polarization(mu_op, window_states)  # np.ndarray[complex]
    return window, P_t


def sim_with_only_pulses(
    sim: SimulationModuleOQS, active_indices: List[int]
) -> SimulationModuleOQS:
    """Return a deep-copied sim where only pulses in active_indices are active (others have amplitude 0).

    Notes:
    - Deep-copy is used to avoid mutating the input and to be process/thread-safe.
    - Phases and timings remain unchanged.
    """
    sim_i = deepcopy(sim)
    # Build a one-pulse sequence matching the i-th pulse timing and phase
    sim_i.laser.select_pulses(active_indices)
    return sim_i


def _compute_P_phi1_phi2(
    sim: SimulationModuleOQS, phi1: float, phi2: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute third-order P_{phi1,phi2}(t_det) as P_total - Σ_i P_i with probe phase fixed at 0.

    Uses: P^(3) ≈ P_total - Σ_i P_i
    Returns (t_det, P_phi1_phi2(t_det)).
    """
    sim_work = deepcopy(sim)
    sim_work.laser.pulse_phases = [
        phi1,
        phi2,
    ]  # NOTE: last pulse phase fixed to the DETECTION_PHASE (0)

    # Total signal with all pulses
    t_det_actual = sim_work.t_det_actual
    t_det_a, P_total = compute_polarization_over_window(sim_work, t_det_actual)

    # Linear signals: only pulse i active
    P_linear_sum = np.zeros_like(P_total, dtype=np.complex128)
    for i in range(len(sim_work.laser.pulses)):
        sim_i = sim_with_only_pulses(sim_work, [i])
        _, P_i = compute_polarization_over_window(sim_i, t_det_actual)
        P_linear_sum += P_i

    P_phi = P_total - P_linear_sum
    return t_det_a, P_phi


def _worker_P_phi_pair(
    sim_template: SimulationModuleOQS, phi1: float, phi2: float
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Process worker: compute (phi1, phi2, t_det, P_{phi1,phi2})."""
    sim_local = deepcopy(sim_template)
    t_det_a, P_phi = _compute_P_phi1_phi2(sim_local, phi1, phi2)
    return phi1, phi2, t_det_a, P_phi


def phase_cycle_component(
    phases: Sequence[float],
    P_grid: np.ndarray,
    *,
    lmn: Tuple[int, int, int] = (0, 0, 0),
    phi_det: float = 0.0,
    normalize: bool = False,
) -> np.ndarray:
    """Extract P_{l,m,n}(t) from a grid P^3[phi1,phi2,t].

    P_{l,m,n}(t) = Σ_{phi1} Σ_{phi2} P^3_{phi1,phi2}(t) exp(-i(l phi1 + m phi2 + n phi_det))
    """
    l, m, n = lmn
    P_out = np.zeros(P_grid.shape[-1], dtype=np.complex128)
    for i, phi1 in enumerate(phases):
        for j, phi2 in enumerate(phases):
            phase = -1j * (l * phi1 + m * phi2 + n * phi_det)
            P_out += P_grid[i, j, :] * np.exp(phase)
            # check if all P_grid entries are zero and warn

    if normalize:
        P_out /= len(phases) ** 2
    return P_out


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def parallel_compute_1d_e_comps(
    sim_oqs: SimulationModuleOQS,
    *,
    phases: Optional[Sequence[float]] = None,
    lmn: Optional[Tuple[int, int, int]] = None,
    phi_det: Optional[float] = None,
    time_cut: Optional[float] = None,
    parallel: bool = True,  # NOTE good for debugging: False
) -> List[np.ndarray]:
    """Compute 1D electric field components E_kS(t_det) with phase cycling only.

    This simplified function assumes the provided `sim_oqs` already encodes a single
    inhomogeneous realization (i.e., system frequencies are already set). No internal
    sampling or averaging over inhomogeneity is performed. Use external batching if needed.

    Parameters
    ----------
    sim_oqs : SimulationModuleOQS
        Prepared simulation (system, laser sequence, solver config).
    phases : Optional[Sequence[float]]
        Phase grid for (phi1, phi2). If None, use PHASE_CYCLING_PHASES truncated to n_phases.
    lmn : Optional[Tuple[int,int,int]]
        Component to extract; if None, derive from signal types via COMPONENT_MAP.
    phi_det : Optional[float]
        Detection phase; if None, use DETECTION_PHASE.
    time_cut : Optional[float]
        Truncate detection times after this value [fs] (soft mask applied).

    Returns
    -------
    List[np.ndarray]
        List of complex E-components, one per entry in `sim_oqs.simulation_config.signal_types`.
        Each array has length len(sim_oqs.t_det). A soft time_cut is applied by zeroing beyond cutoff.
    """
    # Determine phases from config defaults if not provided
    n_ph = sim_oqs.simulation_config.n_phases
    phases_src = phases if phases is not None else PHASE_CYCLING_PHASES
    phases_eff = tuple(float(x) for x in phases_src[:n_ph])

    # Prepare grid and helpers
    n_t = len(sim_oqs.t_det)
    sig_types = sim_oqs.simulation_config.signal_types
    phi_det_val = phi_det if phi_det is not None else float(DETECTION_PHASE)
    if len(sim_oqs.laser.pulses) < 3:
        raise ValueError("3 pulses (pump, pump, probe) are required.")

    # Optional time mask (keep length constant)
    # NOTE:
    # - `time_cut` is returned by solver validation in ABSOLUTE simulation time.
    # - For 2D runs (varying t_coh), the detection window is shifted by
    #   t0_det = t_coh + t_wait. Applying an absolute cutoff directly to
    #   `t_det_actual` would incorrectly truncate traces once t0_det > time_cut
    #   (observed as a diagonal "cut" in 2D maps around a given t_coh).
    # - Convert the absolute cutoff to a RELATIVE cutoff with respect to the
    #   start of the detection window so all t_coh share the same usable span.
    t_mask = None
    if time_cut is not None and np.isfinite(time_cut):
        t0_det = float(sim_oqs.simulation_config.t_coh + sim_oqs.simulation_config.t_wait)
        cut_rel = float(time_cut) - t0_det  # cutoff measured from start of detection window
        # Build mask over the local detection axis; negative cut -> all zeros
        t_mask = (sim_oqs.t_det <= cut_rel).astype(np.float64)

    # Compute P_{phi1,phi2} grid once for this realization (probe phase fixed to 0)
    P_grid = np.zeros((len(phases_eff), len(phases_eff), n_t), dtype=np.complex128)
    futures = []
    # Respect configured/SLURM CPU allocation to avoid oversubscription on HPC
    max_workers = sim_oqs.simulation_config.max_workers
    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for phi1 in phases_eff:
                for phi2 in phases_eff:
                    futures.append(ex.submit(_worker_P_phi_pair, deepcopy(sim_oqs), phi1, phi2))
            temp_results: Dict[Tuple[float, float], np.ndarray] = {}
            for fut in as_completed(futures):
                phi1_v, phi2_v, _, P_phi = fut.result()
                temp_results[(phi1_v, phi2_v)] = P_phi

        for i, phi1 in enumerate(phases_eff):
            for j, phi2 in enumerate(phases_eff):
                P_grid[i, j, :] = temp_results[(phi1, phi2)]
    else:
        for i, phi1 in enumerate(phases_eff):
            for j, phi2 in enumerate(phases_eff):
                _, _, _, P_phi = _worker_P_phi_pair(deepcopy(sim_oqs), phi1, phi2)
                P_grid[i, j, :] = P_phi
    # Extract components for this realization
    E_list: List[np.ndarray] = []
    for sig in sig_types:
        lmn_tuple = COMPONENT_MAP[sig] if lmn is None else lmn
        P_comp = phase_cycle_component(
            phases_eff,
            P_grid,
            lmn=lmn_tuple,
            phi_det=phi_det_val,
            normalize=True,
        )
        E_comp = 1j * P_comp
        if t_mask is not None:
            E_comp = E_comp * t_mask
        E_list.append(E_comp)

    return E_list
