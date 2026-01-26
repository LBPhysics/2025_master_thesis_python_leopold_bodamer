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
from qutip import Qobj, mesolve, brmesolve

# LOCAL IMPORTS
from ..core.simulation import SimulationModuleOQS
from ..core.simulation.time_axes import compute_times_local
from qspectro2d.utils.rwa_utils import from_rotating_frame_op

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


def _check_density_matrix_state(
    state: Qobj,
    time: float,
    index: int,
    total: int,
    prev_state: Qobj | None = None,
) -> tuple[List[str], float]:
    """
    Check density matrix properties for numerical stability at a single time step.

    Returns:
        tuple: (error_messages, time_cut)
    """
    error_messages = []
    time_cut = np.inf
    from qspectro2d.config.signal_processing import NEGATIVE_EIGVAL_THRESHOLD, TRACE_TOLERANCE

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
        print("NEGATIVE EIGENVALUE DETECTED:")
        print(f"  Time: {time:.6f}")
        print(f"  Min eigenvalue: {min_eigval:.12f}")
        print(f"  Threshold: {NEGATIVE_EIGVAL_THRESHOLD}")
        print(f"  All eigenvalues: {eigvals[:5]}...")
        print(f"  State trace: {state.tr():.10f}")
        print(f"  State index: {index}/{total}")

        if prev_state is not None:
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
        print("TRACE VIOLATION:")
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

    return error_messages, time_cut


def _evolve_single_step(
    sim_oqs: SimulationModuleOQS,
    state: Qobj,
    t_prev: float,
    t_curr: float,
    options: dict,
    run_kwargs: dict,
) -> Qobj:
    """Evolve one time step and return the updated state (lab frame if RWA enabled)."""
    t_list = [t_prev, t_curr]
    H = sim_oqs.evo_obj
    solver = sim_oqs.simulation_config.ode_solver

    if solver == "redfield":
        res = brmesolve(
            H=H,
            psi0=state,
            tlist=t_list,
            a_ops=sim_oqs.decay_channels,
            e_ops=None,
            options=options,
            **run_kwargs,
        )
    elif solver in {"lindblad", "paper_eqs"}:
        res = mesolve(
            H=H,
            rho0=state,
            tlist=t_list,
            c_ops=sim_oqs.decay_channels,
            e_ops=None,
            options=options,
        )
    else:
        raise ValueError(f"Unsupported solver '{solver}'.")

    next_state = res.states[-1]
    if sim_oqs.simulation_config.rwa_sl:
        next_state = from_rotating_frame_op(
            next_state,
            t_curr,
            sim_oqs.system.n_atoms,
            sim_oqs.laser.carrier_freq_fs,
        )
    return next_state


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

    RWA conversion is applied per step to keep diagnostics in the lab frame.
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

    # Prepare solver options for step-by-step evolution
    run_kwargs, options = copy_sim_oqs._solver_split()
    options.setdefault("progress_bar", False)
    options.setdefault("store_states", True)
    options.setdefault("store_final_state", True)

    # Step-by-step evolution with immediate checks
    print("\n \n=== STATE-BY-STATE ANALYSIS ===")
    prev_state = None
    current_state = copy_sim_oqs.initial_state
    time_cut = np.inf
    error_messages: List[str] = []

    for index, time in enumerate(times):
        if index == 0:
            state_to_check = (
                from_rotating_frame_op(
                    current_state,
                    float(time),
                    copy_sim_oqs.system.n_atoms,
                    copy_sim_oqs.laser.carrier_freq_fs,
                )
                if copy_sim_oqs.simulation_config.rwa_sl
                else current_state
            )
        else:
            current_state = _evolve_single_step(
                copy_sim_oqs,
                current_state,
                float(times[index - 1]),
                float(time),
                options,
                run_kwargs,
            )
            state_to_check = current_state

        error_messages, time_cut = _check_density_matrix_state(
            state_to_check,
            float(time),
            index,
            len(times),
            prev_state=prev_state,
        )

        if error_messages:
            break

        prev_state = state_to_check

    if not error_messages and prev_state is not None:
        print("✅ Checks passed. DM remains Hermitian and positive.")
        print(f"Final state trace: {prev_state.tr():.6f}")
        print(f"Final state min eigenvalue: {prev_state.eigenenergies().min():.10f}")

    return time_cut
