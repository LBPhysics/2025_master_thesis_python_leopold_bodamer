"""
Solver validation and diagnostics utilities.

This module provides `check_the_solver`, which stress-tests the configured
quantum solver and validates density matrix properties over an extended
time window. It is factored out of the previous calculations module to keep concerns
separate and simplify maintenance.
"""

from __future__ import annotations

# STANDARD LIBRARY IMPORTS
from copy import deepcopy
from typing import List, TYPE_CHECKING

# THIRD-PARTY IMPORTS
import numpy as np
from qutip import Qobj, Result

# LOCAL IMPORTS
from ..core.simulation import SimulationModuleOQS
from .one_d_field import compute_evolution


__all__ = ["check_the_solver"]


def _validate_simulation_input(sim_oqs: SimulationModuleOQS) -> None:
    """Validate simulation input parameters."""
    if not isinstance(sim_oqs.system.psi_ini, Qobj):
        raise TypeError("psi_ini must be a Qobj")
    if not isinstance(sim_oqs.times_local, np.ndarray):
        raise TypeError("times_local must be a numpy.ndarray")
    if not isinstance(sim_oqs.observable_ops, list) or not all(
        isinstance(op, Qobj) for op in sim_oqs.observable_ops
    ):
        raise TypeError("observable_ops must be a list of Qobj")
    if len(sim_oqs.times_local) < 2:
        raise ValueError("times_local must have at least two elements")


def _log_system_diagnostics(sim_oqs: SimulationModuleOQS) -> None:
    """Log diagnostic information about the quantum system."""
    print("=== SYSTEM DIAGNOSTICS ===")
    psi_ini = sim_oqs.system.psi_ini
    print(
        f"Initial state type, shape, is hermitian, trace: {type(psi_ini)}, {psi_ini.shape}, {psi_ini.isherm}, {psi_ini.tr():.6f}"
    )

    if psi_ini.type == "oper":  # density matrix
        ini_eigvals = psi_ini.eigenenergies()
        print(f"Initial eigenvalues range: [{ini_eigvals.min():.6f}, {ini_eigvals.max():.6f}]")
        print(f"Initial min eigenvalue: {ini_eigvals.min():.10f}")

    # System Hamiltonian diagnostics
    try:
        if hasattr(sim_oqs, "evo_obj") and sim_oqs.evo_obj is not None:
            H_tot_t = sim_oqs.evo_obj
            if hasattr(H_tot_t, "dims"):
                print(f"Total Hamiltonian dims: {H_tot_t.dims}")
            print(f"Total Hamiltonian type: {type(H_tot_t)}")

        if hasattr(sim_oqs, "decay_channels") and sim_oqs.decay_channels:
            print(f"Number of decay channels: {len(sim_oqs.decay_channels)}")
    except Exception as e:
        print(f"Could not analyze Hamiltonian: {e}")


def _check_density_matrix_properties(
    states: List[Qobj], times: np.ndarray
) -> tuple[List[str], float]:
    """
    Check density matrix properties for numerical stability.

    Returns:
        tuple: (error_messages, time_cut)
    """
    error_messages = []
    time_cut = np.inf
    from qspectro2d.config.default_simulation_params import (
        NEGATIVE_EIGVAL_THRESHOLD,
        TRACE_TOLERANCE,
    )

    check_interval = max(1, len(states) // 10)  # Check every 10% of states

    for index, state in enumerate(states):
        time = times[index]

        # Sample state analysis
        # if index % check_interval == 0 or index < 5:
        #    print(
        #        f"State {index} (t={time:.3f}): trace={state.tr():.6f}, Hermitian={state.isherm}"
        #    )

        # Check Hermiticity
        if not state.isherm:
            error_messages.append(f"Density matrix is not Hermitian after t = {time}")
            print(f"Non-Hermitian density matrix at t = {time}")
            print(f"  State details: trace={state.tr():.6f}, shape={state.shape}")

        # Check positive semidefiniteness
        eigvals = state.eigenenergies()
        min_eigval = eigvals.min()

        if not np.all(eigvals >= NEGATIVE_EIGVAL_THRESHOLD):
            error_messages.append(
                f"Density matrix is not positive semidefinite after t = {time}: "
                f"The lowest eigenvalue is {min_eigval}"
            )
            print(f"NEGATIVE EIGENVALUE DETECTED:")
            print(f"  Time: {time:.6f}")
            print(f"  Min eigenvalue: {min_eigval:.12f}")
            print(f"  Threshold: {NEGATIVE_EIGVAL_THRESHOLD}")
            print(f"  All eigenvalues: {eigvals[:5]}...")
            print(f"  State trace: {state.tr():.10f}")
            print(f"  State index: {index}/{len(states)}")

            if index > 0:
                prev_state = states[index - 1]
                prev_eigvals = prev_state.eigenenergies()
                print(f"  Previous state min eigval: {prev_eigvals.min():.12f}")
                print(f"  Eigenvalue change: {min_eigval - prev_eigvals.min():.12f}")

            time_cut = time

        # Check trace preservation
        trace_val = state.tr()

        if not np.isclose(trace_val, 1.0, atol=TRACE_TOLERANCE):
            error_messages.append(
                f"Density matrix is not trace-preserving after t = {time}: "
                f"The trace is {trace_val}"
            )
            print(f"TRACE VIOLATION:")
            print(f"  Time: {time:.6f}")
            print(f"  Trace: {trace_val:.10f}")
            print(f"  Deviation from 1: {abs(trace_val - 1.0):.10f}")
            print(f"  Tolerance: {TRACE_TOLERANCE}")

            time_cut = min(time_cut, time)

        # Break on first error for detailed analysis
        if error_messages:
            print("=== FIRST ERROR ANALYSIS ===")
            print(f"Stopping analysis at first error (state {index}, t={time:.6f})")
            print("Density matrix validation failed: " + "; ".join(error_messages))
            break

    return error_messages, time_cut


def check_the_solver(sim_oqs: SimulationModuleOQS) -> tuple[Result, float]:
    """
    Validate the quantum solver by running a test evolution and checking density matrix properties.

    This function performs a comprehensive validation of the solver by:
    1. Running a test evolution with extended time range
    2. Checking density matrix properties (Hermiticity, trace preservation, positive semidefiniteness)
    3. Applying RWA phase factors if needed
    4. Logging detailed diagnostics throughout the process

    Parameters
    ----------
    sim_oqs : SimulationModuleOQS
        Simulation object containing system parameters, laser pulses, and configuration.
        A deep copy is made internally to avoid modifying the original object.

    Returns
    -------
    tuple[Result, float]
        result : Result
            QuTiP Result object from the test evolution.
        time_cut : float
            Time after which numerical instabilities were detected, or np.inf if all checks passed.

    Notes
    -----
    The function uses extended time parameters (2x t_max, 10x dt) to stress-test the solver.
    It checks for common numerical issues in quantum simulations:
    - Non-Hermitian density matrices
    - Negative eigenvalues (non-physical states)
    - Trace deviation from 1.0

    If RWA (Rotating Wave Approximation) is enabled, phase factors are applied to the states
    before validation to account for the rotating frame transformation.
    """
    # print(f"Checking '{sim_oqs.simulation_config.ode_solver}' solver")
    copy_sim_oqs = deepcopy(sim_oqs)
    t_max = 2 * sim_oqs.times_local[-1]
    dt = 10 * copy_sim_oqs.simulation_config.dt
    t0 = -2 * copy_sim_oqs.laser.pulse_fwhms[0]
    times = np.linspace(t0, t_max, int((t_max - t0) / dt) + 1)
    copy_sim_oqs.times_local = times
    copy_sim_oqs.laser.pulse_phases = [1.0] * len(copy_sim_oqs.laser.pulses)

    # DETAILED SYSTEM DIAGNOSTICS

    print(f"=== SOLVER DIAGNOSTICS ===")
    print(f"Solver: {copy_sim_oqs.simulation_config.ode_solver}")
    print(f"Time range: t0={t0:.3f}, t_max={t_max:.3f}, dt={dt:.6f}")
    print(f"Number of time points: {len(times)}")
    print(f"RWA enabled: {getattr(copy_sim_oqs.simulation_config, 'rwa_sl', False)}")

    _log_system_diagnostics(copy_sim_oqs)

    # INPUT VALIDATION
    _validate_simulation_input(copy_sim_oqs)

    result = compute_evolution(copy_sim_oqs, **{"store_states": True})
    states = result.states

    # CHECK THE RESULT
    if not isinstance(result, Result):
        raise TypeError("Result must be a Result object")
    if list(result.times) != list(times):
        raise ValueError("Result times do not match input times")
    if len(result.states) != len(times):
        raise ValueError("Number of output states does not match number of time points")

    # CHECK DENSITY MATRIX PROPERTIES
    # Apply RWA phase factors if needed
    if getattr(copy_sim_oqs.simulation_config, "rwa_sl", False):
        n_atoms = copy_sim_oqs.system.n_atoms
        omega_laser = copy_sim_oqs.laser.carrier_freq_fs
        print(f"Applying RWA phase factors: n_atoms={n_atoms}, omega_laser={omega_laser} [fs^-1]")
        # Lazy import here to avoid triggering package-level imports during module import
        from ..utils.rwa_utils import from_rotating_frame_list

        states = from_rotating_frame_list(states, times, n_atoms, omega_laser)

    # Enhanced state checking with more diagnostics
    print("=== STATE-BY-STATE ANALYSIS ===")
    error_messages, time_cut = _check_density_matrix_properties(states, times)

    if not error_messages:
        print("✅ Checks passed. DM remains Hermitian and positive.")
        print(f"Final state trace: {states[-1].tr():.6f}")
        print(f"Final state min eigenvalue: {states[-1].eigenenergies().min():.10f}")

    return result, time_cut
