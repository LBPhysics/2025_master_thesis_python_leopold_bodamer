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

from typing import List, Sequence, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from qutip import Qobj, Result, mesolve

from ..core.simulation.simulation_class import SimulationModuleOQS
from ..core.simulation.time_axes import compute_times_local, compute_t_det
from ..config.default_simulation_params import (
    PHASE_CYCLING_PHASES,
    COMPONENT_MAP,
)
from qspectro2d.utils.rwa_utils import from_rotating_frame_list

__all__ = [
    "parallel_compute_1d_e_comps",
]


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------
def slice_states_to_window(res: Result, window: np.ndarray) -> List[Qobj]:
    """Slice the list of states in `res` to only keep the detection window portion.

    Returns:
        List of states corresponding to the window time points.
    """
    # Assuming res_times is sorted and equally spaced
    times = np.asarray(res.times)
    window = np.asarray(window)
    # nearest index for each window time
    idxs = np.searchsorted(times, window, side="left")
    idxs = np.clip(idxs, 0, len(times) - 1)

    # For each window time, find the best matching time index
    selected_idxs = np.zeros_like(idxs)
    for k, (idx, w) in enumerate(zip(idxs, window)):
        # Candidates: idx-1, idx, idx+1 (if within bounds)
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        candidates.append(idx)
        if idx < len(times) - 1:
            candidates.append(idx + 1)

        # Choose the candidate with minimal absolute difference
        best_idx = min(candidates, key=lambda i: abs(times[i] - w))
        selected_idxs[k] = best_idx

    return [res.states[int(i)] for i in selected_idxs]


def slice_P_to_window(times: np.ndarray, P_t: list, window: np.ndarray) -> np.ndarray:
    """Slice the list of polarization values P_t to only keep the detection window portion.

    Returns:
        Array of polarization values corresponding to the window time points.
    """
    # Assuming times are sorted and equally spaced
    times = np.asarray(times)
    window = np.asarray(window)
    # nearest index for each window time
    idxs = np.searchsorted(times, window, side="left")
    idxs = np.clip(idxs, 0, len(times) - 1)

    # For each window time, find the best matching time index
    selected_idxs = np.zeros_like(idxs)
    for k, (idx, w) in enumerate(zip(idxs, window)):
        # Candidates: idx-1, idx, idx+1 (if within bounds)
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        candidates.append(idx)
        if idx < len(times) - 1:
            candidates.append(idx + 1)

        # Choose the candidate with minimal absolute difference
        best_idx = min(candidates, key=lambda i: abs(times[i] - w))
        selected_idxs[k] = best_idx

    return np.array([P_t[int(i)] for i in selected_idxs], dtype=complex)

def compute_evolution(
    sim_oqs: SimulationModuleOQS,
    e_ops=None,
    **solver_options: dict,
) -> tuple[np.ndarray, list]:
    """
    Compute the evolution of the quantum system for a given pulse sequence, handling overlapping pulses.
    Returns (times, data) where data is list of expectation values if e_ops provided, else list of states.
    """
    options: dict = sim_oqs.simulation_config.solver_options.copy()
    options.update(solver_options)
    # Remove Bloch-Redfield specific knobs that QuTiP's mesolve does not understand.
    options.pop("sec_cutoff", None)
    options.pop("br_computation_method", None)
    options.pop("tensor_type", None)
    if e_ops is not None:
        options["store_states"] = False
        options["store_final_state"] = True
        solver_e_ops = e_ops
    else:
        options["store_states"] = True
        solver_e_ops = []

    times_array = np.asarray(compute_times_local(sim_oqs.simulation_config))

    # Build event grid
    current_state = sim_oqs.initial_state

    res = mesolve(
            H=sim_oqs.evo_obj,
            rho0=current_state,
            tlist=times_array,
            e_ops=solver_e_ops,
            options=options,
        )

    data = res.expect[0] if e_ops is not None else res.states

    # If returning states and RWA was used, convert to lab frame
    if res.expect is None and sim_oqs.simulation_config.rwa_sl and times_array:
        data = from_rotating_frame_list(
            data, np.array(times_array), sim_oqs.system.n_atoms, sim_oqs.laser.carrier_freq_fs
        )

    return np.array(times_array, dtype=float), data


def compute_polarization_over_window(
    sim_oqs: SimulationModuleOQS,
    window: np.ndarray | list | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evolve once with current laser settings and return (t_det, P(t_det)),
    where P = ⟨μ_+⟩ computed on the requested detection window times.
    """
    if window is None:
        window = compute_t_det(sim_oqs.simulation_config)
    window = np.asarray(window, dtype=float)

    # Build μ_+ in energy eigenbasis (lower triangular part), which oscillates as exp(-i ω_L t) in RWA frame
    mu_pos = sim_oqs.system.to_eigenbasis(sim_oqs.system.lowering_op.dag())
    if sim_oqs.simulation_config.rwa_sl:
        # rotate the states back to lab frame before expectation value
        from qspectro2d.spectroscopy.polarization import time_dependent_polarization_rwa
        polarization = lambda t, state: time_dependent_polarization_rwa(mu_pos, state, t, sim_oqs.system.n_atoms, sim_oqs.laser.carrier_freq_fs)
    else:
        polarization = mu_pos

    # Get polarization on the global time grid
    times, P_t = compute_evolution(sim_oqs, e_ops=[polarization])

    # Select the polarization at the desired window times (nearest matches)
    P_on_window = slice_P_to_window(times, P_t, window)
    return window, P_on_window


def sim_with_only_pulses(
    sim_oqs: SimulationModuleOQS, active_indices: List[int]
) -> SimulationModuleOQS:
    """Return a deep-copied sim_oqs where only pulses in active_indices are active (others have amplitude 0).

    Notes:
    - Deep-copy is used to avoid mutating the input and to be process/thread-safe.
    - Phases and timings remain unchanged.
    """
    from copy import deepcopy

    sim_i = deepcopy(sim_oqs)
    # Build a one-pulse sequence matching the i-th pulse timing and phase
    sim_i.laser.select_pulses(active_indices)
    return sim_i


def _worker_P_phi_pair(
    config_path: str, t_coh: float, freq_vector: List[float], phi1: float, phi2: float
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Process worker: compute (phi1, phi2, t_det, P_{phi1,phi2})."""
    from qspectro2d.config.create_sim_obj import load_simulation

    sim_oqs = load_simulation(config_path)
    sim_oqs.update_delays(t_coh=t_coh)
    sim_oqs.system.update_frequencies_cm(freq_vector)
    sim_oqs.laser.pulse_phases = [
        phi1,
        phi2,
        0.0,
    ]  # NOTE: last pulse phase fixed to the DETECTION_PHASE (0)

    # Compute third-order P_{phi1,phi2}(t_det) as P_total - Σ_i P_i with probe phase fixed at 0.
    # Uses: P^(3) ≈ P_total - Σ_i P_i

    # Total signal with all pulses
    t_det = compute_t_det(sim_oqs.simulation_config)
    t_det_a, P_total = compute_polarization_over_window(sim_oqs, t_det)

    # Subtract signals from all subsets of size 1 and 2 pulses active
    P_sub_sum = np.zeros_like(P_total, dtype=np.complex128)
    n_pulses = len(sim_oqs.laser.pulses)
    import itertools

    for k in range(1, 2):  # subsets of size 1 only
        for combo in itertools.combinations(range(n_pulses), k):
            sim_sub = sim_with_only_pulses(sim_oqs, list(combo))
            _, P_sub = compute_polarization_over_window(sim_sub, t_det)
            P_sub_sum += P_sub

    P_phi = P_total - P_sub_sum
    return phi1, phi2, t_det_a, P_phi


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def parallel_compute_1d_e_comps(
    config_path: str,
    t_coh: float,
    freq_vector: List[float],
    *,
    phases: Optional[Sequence[float]] = None,
    lm: Optional[Tuple[int, int]] = None,
    time_cut: Optional[float] = None, # TODO for now just to see evolution -> later implement again
) -> List[np.ndarray]:
    """Compute 1D electric field components E_kS(t_det) with phase cycling only.

    loads config from YAML, assuming a single
    inhomogeneous realization (system frequencies set externally). No internal
    sampling or averaging over inhomogeneity is performed. Use external batching if needed.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.
    t_coh : float
        Coherence time.
    phases : Optional[Sequence[float]]
        Phase grid for (phi1, phi2). If None, use PHASE_CYCLING_PHASES truncated to n_phases.
    lm : Optional[Tuple[int,int]]
        Component to extract; if None, derive from signal types via COMPONENT_MAP.
    time_cut : Optional[float]
        Truncate detection times after this value [fs] (soft mask applied).

    Returns
    -------
    List[np.ndarray]
        List of complex E-components, one per entry in config.signal_types.
        Each array has length len(t_det). A soft time_cut is applied by zeroing beyond cutoff.
    """
    from qspectro2d.config.create_sim_obj import load_simulation_config

    config = load_simulation_config(config_path)
    t_coh_val = float(t_coh)
    config.t_coh_current = t_coh_val
    if t_coh_val > float(config.t_coh_max):
        config.t_coh_max = t_coh_val
    # Determine phases from config defaults if not provided
    n_ph = config.n_phases
    phases_src = phases if phases is not None else PHASE_CYCLING_PHASES
    phases_eff = tuple(float(x) for x in phases_src[:n_ph])

    # Prepare grid and helpers
    t_det = compute_t_det(config)
    n_t = len(t_det)
    sig_types = config.signal_types
    # Assume 3 pulses as per the function doc

    # Optional time mask (keep length constant)
    #t_mask = None
    #if time_cut is not None and np.isfinite(time_cut):
    #    t_mask = (t_det <= time_cut).astype(np.float64)

    # Accumulate P components as results arrive
    P_acc = {sig: np.zeros(n_t, dtype=np.complex128) for sig in sig_types}
    futures = []
    # Respect configured/SLURM CPU allocation to avoid oversubscription on HPC
    max_workers = config.max_workers

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for phi1 in phases_eff:
            for phi2 in phases_eff:
                futures.append(
                    ex.submit(_worker_P_phi_pair, config_path, t_coh, freq_vector, phi1, phi2)
                )
        for fut in as_completed(futures):
            phi1_v, phi2_v, _, P_phi = fut.result()
            for sig in sig_types:
                lm_tuple = COMPONENT_MAP[sig] if lm is None else lm
                l, m = lm_tuple
                phase_factor = np.exp(-1j * (l * phi1_v + m * phi2_v))
                P_acc[sig] += phase_factor * P_phi
                if np.all(P_phi == 0):
                    print("All zero P_phi detected!")

    # Extract components for this realization
    dphi = np.diff(phases_eff).mean() if len(phases_eff) > 1 else 1.0

    E_list: List[np.ndarray] = []
    for sig in sig_types:
        P_comp = P_acc[sig] * dphi * dphi  # normalization
        E_comp = -1j * P_comp
        # if t_mask is not None:
        #     E_comp = E_comp * t_mask
        E_list.append(E_comp)
    return E_list
