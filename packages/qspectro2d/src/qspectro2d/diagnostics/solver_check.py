"""Solver validation and diagnostics utilities."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from qutip import Qobj, brmesolve, mesolve

from ..config.defaults import NEGATIVE_EIGVAL_THRESHOLD, TRACE_TOLERANCE
from ..core.simulation import SimulationModuleOQS
from ..core.simulation.time_axes import compute_times_local
from ..utils.rwa_utils import from_rotating_frame_list

__all__ = ["check_the_solver"]


def _validate_simulation_input(sim_oqs: SimulationModuleOQS) -> None:
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
    print("\n \n=== SYSTEM DIAGNOSTICS ===")
    rho_ini = sim_oqs.initial_state
    print(
        f"Initial state type, shape, is hermitian, trace: {type(rho_ini)}, {rho_ini.shape}, {rho_ini.isherm}, {rho_ini.tr():.6f}"
    )
    if rho_ini.type == "oper":
        initial_eigenvalues = rho_ini.eigenenergies()
        print(
            "Initial eigenvalues range: "
            f"[{initial_eigenvalues.min():.6f}, {initial_eigenvalues.max():.6f}]"
        )
        print(f"Initial min eigenvalue: {initial_eigenvalues.min():.10f}")


def _check_density_matrix_state(
    state: Qobj,
    time: float,
    index: int,
    total: int,
    prev_state: Qobj | None = None,
) -> tuple[list[str], float]:
    error_messages: list[str] = []
    time_cut = np.inf

    if not state.isherm:
        error_messages.append(f"Density matrix is not Hermitian after t = {time}")
        print(f"Non-Hermitian density matrix at t = {time}")
        print(f"  State details: trace={state.tr():.6f}, shape={state.shape}")

    eigenvalues = state.eigenenergies()
    min_eigenvalue = eigenvalues.min()
    if not np.all(eigenvalues >= NEGATIVE_EIGVAL_THRESHOLD):
        error_messages.append(
            f"Density matrix is not positive semidefinite after t = {time}: The lowest eigenvalue is {min_eigenvalue}"
        )
        print("NEGATIVE EIGENVALUE DETECTED:")
        print(f"  Time: {time:.6f}")
        print(f"  Min eigenvalue: {min_eigenvalue:.12f}")
        print(f"  Threshold: {NEGATIVE_EIGVAL_THRESHOLD}")
        print(f"  All eigenvalues: {eigenvalues[:5]}...")
        print(f"  State trace: {state.tr():.10f}")
        print(f"  State index: {index}/{total}")
        if prev_state is not None:
            prev_eigenvalues = prev_state.eigenenergies()
            print(f"  Previous state min eigval: {prev_eigenvalues.min():.12f}")
            print(f"  Eigenvalue change: {min_eigenvalue - prev_eigenvalues.min():.12f}")
        time_cut = time

    trace_value = state.tr()
    if not np.isclose(trace_value, 1.0, atol=TRACE_TOLERANCE):
        error_messages.append(
            f"Density matrix is not trace-preserving after t = {time}: The trace is {trace_value}"
        )
        print("TRACE VIOLATION:")
        print(f"  Time: {time:.6f}")
        print(f"  Trace: {trace_value:.10f}")
        print(f"  Deviation from 1: {abs(trace_value - 1.0):.10f}")
        print(f"  Tolerance: {TRACE_TOLERANCE}")
        time_cut = min(time_cut, time)

    if error_messages:
        print("=== FIRST ERROR ANALYSIS ===")
        print(f"Stopping analysis at first error (state {index}, t={time:.6f})")
        print("Density matrix validation failed: " + "; ".join(error_messages))

    return error_messages, time_cut


def check_the_solver(sim_oqs: SimulationModuleOQS) -> float:
    """Stress-test the configured solver and return the first failing time, if any."""
    sim_copy = deepcopy(sim_oqs)
    times = compute_times_local(sim_oqs.simulation_config)
    t0 = times[0]
    dt = times[1] - times[0]
    sim_copy.laser.pulse_phases = [1.0] * len(sim_copy.laser.pulses)

    print("\n \n=== SOLVER DIAGNOSTICS ===")
    print(f"Solver: {sim_copy.simulation_config.ode_solver}")
    print(f"Time range: t0={t0:.3f}, t_max={times[-1]:.3f}, dt={dt:.6f}")
    print(f"Number of time points: {len(times)}")
    print(f"RWA enabled: {getattr(sim_copy.simulation_config, 'rwa_sl', False)}")

    _log_system_diagnostics(sim_copy)
    _validate_simulation_input(sim_copy)

    run_kwargs, options = sim_copy._solver_split()
    options.setdefault("progress_bar", False)
    options.setdefault("store_states", True)
    options.setdefault("store_final_state", True)

    solver = sim_copy.simulation_config.ode_solver
    hamiltonian = sim_copy.evo_obj
    rho0 = sim_copy.initial_state

    print("\n \n=== STATE-BY-STATE ANALYSIS (single-shot solver run) ===")
    if solver == "redfield":
        result = brmesolve(
            H=hamiltonian,
            psi0=rho0,
            tlist=times,
            a_ops=sim_copy.decay_channels,
            e_ops=None,
            options=options,
            **run_kwargs,
        )
    elif solver in {"lindblad", "paper_eqs"}:
        result = mesolve(
            H=hamiltonian,
            rho0=rho0,
            tlist=times,
            c_ops=sim_copy.decay_channels,
            e_ops=None,
            options=options,
        )
    else:
        raise ValueError(f"Unsupported solver '{solver}'.")

    states = result.states
    if sim_copy.simulation_config.rwa_sl and len(states):
        states = from_rotating_frame_list(
            states,
            np.asarray(times, dtype=float),
            sim_copy.system.n_atoms,
            sim_copy.laser.carrier_freq_fs,
        )

    prev_state = None
    time_cut = np.inf
    error_messages: list[str] = []
    for index, (time, state_to_check) in enumerate(zip(times, states)):
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
        print("Checks passed. DM remains Hermitian and positive.")
        print(f"Final state trace: {prev_state.tr():.6f}")
        print(f"Final state min eigenvalue: {prev_state.eigenenergies().min():.10f}")

    return time_cut