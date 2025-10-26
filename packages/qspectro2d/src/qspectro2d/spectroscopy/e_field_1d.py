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
from qutip import Qobj, Result, mesolve, brmesolve, expect
from copy import deepcopy

from ..core.simulation.simulation_class import SimulationModuleOQS
from ..core.simulation.time_axes import compute_times_global, compute_t_det
from ..config.default_simulation_params import (
    PHASE_CYCLING_PHASES,
    COMPONENT_MAP,
)
from qspectro2d.utils.rwa_utils import from_rotating_frame_list, from_rotating_frame_op

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


def compute_evolution(
    sim_oqs: SimulationModuleOQS,
    mu_op=None,
    use_eops=False,
    **solver_options: dict,
) -> Result | np.ndarray:
    """
    Compute the evolution of the quantum system for a given pulse sequence, handling overlapping pulses.

    Parameters
    ----------
    sim_oqs : SimulationModuleOQS
        Prepared simulation object.
    **solver_options : dict
        Optional solver arguments to override defaults.

    Returns
    -------
    Result
        QuTiP Result object containing states, times, and final_state.
    """
    from qspectro2d.config.default_simulation_params import SOLVER_OPTIONS

    options: dict = SOLVER_OPTIONS.copy()
    options.update(solver_options)
    if use_eops:
        options["store_states"] = False
    else:
        options.setdefault("store_states", True)

    times_array = np.asarray(compute_times_global(sim_oqs.simulation_config))
    pulses = sim_oqs.laser.pulses
    ode_solver = sim_oqs.simulation_config.ode_solver

    e_ops = [mu_op] if use_eops and mu_op is not None else []
    all_expect = [] if use_eops else None

    def run_solver(H, psi0_rho0, tlist, e_ops):
        if len(tlist) < 2:
            return None  # No evolution needed
        if ode_solver == "BR":
            return brmesolve(
                H=H,
                psi0=psi0_rho0,
                tlist=tlist,
                a_ops=sim_oqs.decay_channels,
                e_ops=e_ops,
                options=options,
                sec_cutoff=-1,
            )
        else:
            return mesolve(
                H=H,
                rho0=psi0_rho0,
                tlist=tlist,
                c_ops=sim_oqs.decay_channels,
                e_ops=e_ops,
                options=options,
            )

    # Helper to get active pulses at time t
    def get_active_pulse_at_interval(i):
        t0 = event_times[i]
        t1 = event_times[i + 1]
        for p in pulses:
            p_start, p_end = p.active_time_range
            if p_start <= t1 and p_end >= t0:
                return True
        return False

    # Get all event times: pulse starts, ends, and simulation boundaries
    event_times = [p.active_time_range[0] for p in pulses] + [
        p.active_time_range[1] for p in pulses
    ]
    event_times = [times_array[0]] + event_times + [times_array[-1]]
    event_times = sorted(set(event_times))  # Unique and sorted

    # Precompute indices for efficient slicing
    event_i0 = np.searchsorted(times_array, event_times, side="left")
    event_i1 = np.searchsorted(times_array, event_times, side="right")

    # Initialize
    all_states = []
    all_times = []
    current_state = sim_oqs.system.psi_ini

    # Evolve over each interval where active pulses are constant
    for i in range(len(event_times) - 1):
        active_pulses = get_active_pulse_at_interval(i)

        # Determine Hamiltonian
        if active_pulses:
            H = sim_oqs.evo_obj
        else:
            H = sim_oqs.H0_diagonalized

        # Evolve over this interval
        i0 = event_i0[i]
        i1 = event_i1[i + 1]
        t_slice = times_array[i0:i1]
        if len(t_slice) >= 1:
            if len(t_slice) == 1:
                # Single point, no evolution needed
                if not (len(all_times) > 0 and abs(t_slice[0] - all_times[-1]) < 1e-12):
                    # Not a duplicate, add the point
                    if use_eops:
                        # For expectation values, we need to compute at this point
                        # But since single point, perhaps interpolate or something, but for now, skip or handle
                        pass  # Assuming single points are not critical for e_ops
                    else:
                        all_states.append(current_state)
                        all_times.append(t_slice[0])
            else:
                res = run_solver(H, current_state, t_slice, e_ops)
                if res:
                    if use_eops:
                        all_expect += list(res.expect[0])
                    else:
                        if len(all_times) > 0 and abs(t_slice[0] - all_times[-1]) < 1e-12:
                            # Drop the first point only if it would duplicate the previous segment's last time
                            all_states += res.states[1:]
                            all_times += list(t_slice[1:])
                        else:
                            all_states += res.states
                            all_times += list(t_slice)
                        current_state = res.states[-1]

    if use_eops:
        return np.array(all_times), np.array(all_expect)
    else:
        res.states = all_states
        res.times = np.array(all_times)
        return res


def compute_polarization_over_window(
    sim: SimulationModuleOQS,
    window: np.ndarray | List = None,
    *,
    store_states: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evolve the system once (with current laser settings) and return (t_det, P(t_det)).
    usually window = t_det

    - Uses `compute_seq_evolution` which dispatches ME/BR according to sim.simulation_config.
    - Extracts complex/analytical polarization over the detection window only.
    """
    if window is None:
        window = compute_t_det(sim.simulation_config)
    # Positive-frequency part for this codebase's basis ordering corresponds to
    # the strictly UPPER-triangular portion (i < j) in the energy eigenbasis.
    # ~ sigma^- e^[+iwt]
    mu_op = sim.system.to_eigenbasis(sim.system.dipole_op)
    dipole_op_pos = Qobj(np.triu(mu_op.full(), k=1), dims=mu_op.dims)
    if not store_states:
        if sim.simulation_config.rwa_sl:

            def e_ops_callable(t, state):
                state_lab = from_rotating_frame_op(
                    state, t, sim.system.n_atoms, sim.laser.carrier_freq_fs
                )
                return expect(dipole_op_pos, state_lab)

            P_t = compute_evolution(sim, mu_op=e_ops_callable, use_eops=True)
        else:
            P_t = compute_evolution(sim, mu_op=dipole_op_pos, use_eops=True)
        return window, P_t
    else:
        # Ensure we store states to extract polarization
        res: Result = compute_evolution(sim, store_states=store_states)

        window_states = slice_states_to_window(res, window)

        if sim.simulation_config.rwa_sl:
            # States are stored in the rotating frame; convert back to lab for polarization
            # Use times relative to the start of the simulation to preserve phase accumulation
            window_states = from_rotating_frame_list(
                window_states, window, sim.system.n_atoms, sim.laser.carrier_freq_fs
            )

        # Analytical polarization using positive-frequency part of dipole operator
        P_t = np.array([expect(dipole_op_pos, state) for state in window_states])
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


def _worker_P_phi_pair(
    config_path: str, t_coh: float, freq_vector: List[float], phi1: float, phi2: float
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Process worker: compute (phi1, phi2, t_det, P_{phi1,phi2})."""
    from qspectro2d.config.create_sim_obj import load_simulation

    sim = load_simulation(config_path)
    sim.update_delays(t_coh=t_coh)
    sim.system.update_frequencies_cm(freq_vector)
    sim.laser.pulse_phases = [
        phi1,
        phi2,
        0.0,
    ]  # NOTE: last pulse phase fixed to the DETECTION_PHASE (0)

    # Compute third-order P_{phi1,phi2}(t_det) as P_total - Σ_i P_i with probe phase fixed at 0.
    # Uses: P^(3) ≈ P_total - Σ_i P_i
    sim_work = deepcopy(sim)
    sim_work.laser.pulse_phases = [
        phi1,
        phi2,
        0.0,
    ]  # NOTE: last pulse phase fixed to the DETECTION_PHASE (0)

    # Total signal with all pulses
    t_det = compute_t_det(sim_work.simulation_config)
    t_det_a, P_total = compute_polarization_over_window(sim_work, t_det)

    # Subtract signals from all subsets of size 1 and 2 pulses active
    P_sub_sum = np.zeros_like(P_total, dtype=np.complex128)
    n_pulses = len(sim_work.laser.pulses)
    import itertools

    for k in range(1, 2):  # subsets of size 1 only
        for combo in itertools.combinations(range(n_pulses), k):
            sim_sub = sim_with_only_pulses(sim_work, list(combo))
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
    time_cut: Optional[float] = None,
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
    # Determine phases from config defaults if not provided
    n_ph = config.n_phases
    phases_src = phases if phases is not None else PHASE_CYCLING_PHASES
    phases_eff = tuple(float(x) for x in phases_src[:n_ph])

    # Prepare grid and helpers
    n_t = len(compute_t_det(config))
    sig_types = config.signal_types
    # Assume 3 pulses as per the function doc

    # Optional time mask (keep length constant)
    t_mask = None
    if time_cut is not None and np.isfinite(time_cut):
        t_det = compute_t_det(config)
        t_mask = (t_det <= time_cut).astype(np.float64)

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

    # Extract components for this realization
    dphi = np.diff(phases_eff).mean() if len(phases_eff) > 1 else 1.0

    E_list: List[np.ndarray] = []
    for sig in sig_types:
        P_comp = P_acc[sig] * dphi * dphi  # normalization
        E_comp = 1j * P_comp
        if t_mask is not None:
            E_comp = E_comp * t_mask
        E_list.append(E_comp)

    return E_list
