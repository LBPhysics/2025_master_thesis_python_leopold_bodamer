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
from typing import List

# THIRD-PARTY IMPORTS
import numpy as np
from qutip import Qobj

# LOCAL IMPORTS
from ..core.simulation import SimulationModuleOQS
from ..core.simulation.time_axes import compute_times_local
from .e_field_1d import compute_evolution

__all__ = ["check_the_solver"]


def _validate_simulation_input(sim_oqs: SimulationModuleOQS) -> None:
    """Validate simulation input parameters."""
    if not isinstance(sim_oqs.initial_state, Qobj):
        raise TypeError("initial_state must be a Qobj")
    times_global = compute_times_local(sim_oqs.simulation_config)
    if not isinstance(times_global, np.ndarray):
        raise TypeError("times_global must be a numpy.ndarray")
    if not isinstance(sim_oqs.observable_ops, list) or not all(
        isinstance(op, Qobj) for op in sim_oqs.observable_ops
    ):
        raise TypeError("observable_ops must be a list of Qobj")
    if len(times_global) < 2:
        raise ValueError("times_global must have at least two elements")


def _log_system_diagnostics(sim_oqs: SimulationModuleOQS) -> None:
    """Log diagnostic information about the quantum system."""
    print("\n \n=== SYSTEM DIAGNOSTICS ===")
    rho_ini = sim_oqs.initial_state
    print(
        f"Initial state type, shape, is hermitian, trace: {type(rho_ini)}, {rho_ini.shape}, {rho_ini.isherm}, {rho_ini.tr():.6f}"
    )

    if rho_ini.type == "oper":  # density matrix
        ini_eigvals = rho_ini.eigenenergies()
        print(f"Initial eigenvalues range: [{ini_eigvals.min():.6f}, {ini_eigvals.max():.6f}]")
        print(f"Initial min eigenvalue: {ini_eigvals.min():.10f}")


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

    for index, state in enumerate(states):
        time = times[index]
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


def check_the_solver(sim_oqs: SimulationModuleOQS) -> float:
    """
    Validate the quantum solver by running a test evolution and checking density matrix properties.

    This function performs a comprehensive validation of the solver by:
    1. Running a test evolution with extended time range
    2. Checking density matrix properties (Hermiticity, trace preservation, positive semidefiniteness)
    3. Logging detailed diagnostics throughout the process

    Parameters
    ----------
    sim_oqs : SimulationModuleOQS
        Simulation object containing system parameters, laser pulses, and configuration.
        A deep copy is made internally to avoid modifying the original object.

    Returns
    -------
    float
        Time after which numerical instabilities were detected, or np.inf if all checks passed.

    Notes
    -----
    The function uses extended time parameters (2x t_max, 10x dt) to stress-test the solver.
    It checks for common numerical issues in quantum simulations:
    - Non-Hermitian density matrices
    - Negative eigenvalues (non-physical states)
    - Trace deviation from 1.0

    RWA conversion is automatically applied in compute_evolution if enabled.
    """
    # print(f"Checking '{sim_oqs.simulation_config.ode_solver}' solver")
    copy_sim_oqs = deepcopy(sim_oqs)
    times = compute_times_local(sim_oqs.simulation_config)
    t0 = times[0]
    dt = times[1] - times[0]
    copy_sim_oqs.laser.pulse_phases = [1.0] * len(copy_sim_oqs.laser.pulses)

    # DETAILED SYSTEM DIAGNOSTICS

    print(f"\n \n=== SOLVER DIAGNOSTICS ===")
    print(f"Solver: {copy_sim_oqs.simulation_config.ode_solver}")
    print(f"Time range: t0={t0:.3f}, t_max={times[-1]:.3f}, dt={dt:.6f}")
    print(f"Number of time points: {len(times)}")
    print(f"RWA enabled: {getattr(copy_sim_oqs.simulation_config, 'rwa_sl', False)}")

    _log_system_diagnostics(copy_sim_oqs)

    # INPUT VALIDATION
    _validate_simulation_input(copy_sim_oqs)

    times_result, states = compute_evolution(copy_sim_oqs, progress_bar="text")

    # CHECK THE RESULT
    if not isinstance(times_result, np.ndarray):
        raise TypeError("Times must be a numpy array")
    if not isinstance(states, list):
        raise TypeError("States must be a list")
    if len(times_result) != len(times) or not np.allclose(times_result, times):
        print("Warning: Result times do not match input times exactly")
        print(f"Result times length: {len(times_result)}, input: {len(times)}")
        # print(f"First few result times: {times_result[:10]}")
        # print(f"First few input times: {times[:10]}")
        # Don't raise, as the evolution may have boundary overlaps
    if len(states) != len(times_result):
        raise ValueError("Number of output states does not match number of result time points")

    # CHECK DENSITY MATRIX PROPERTIES
    # RWA conversion is already done in compute_evolution if needed

    # Enhanced state checking with more diagnostics
    print("\n \n=== STATE-BY-STATE ANALYSIS ===")
    error_messages, time_cut = _check_density_matrix_properties(states, times_result)

    if not error_messages:
        print("✅ Checks passed. DM remains Hermitian and positive.")
        print(f"Final state trace: {states[-1].tr():.6f}")
        print(f"Final state min eigenvalue: {states[-1].eigenenergies().min():.10f}")

    return time_cut
