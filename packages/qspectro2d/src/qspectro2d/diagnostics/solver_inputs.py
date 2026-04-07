"""Shared solver-input preparation, validation, and solver diagnostics."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..core.simulation import SimulationModuleOQS

__all__ = ["log_lindblad_solver_debug", "log_redfield_solver_debug"]


def _sample_times_for_debug(tlist: np.ndarray) -> list[float]:
    sample_times: list[float] = [float(tlist[0])]
    if len(tlist) > 1:
        sample_times.append(float(tlist[1]))
    sample_times.append(0.0)

    unique_times: list[float] = []
    for time_value in sample_times:
        if time_value not in unique_times:
            unique_times.append(time_value)
    return unique_times


def _preview_array(values: np.ndarray, *, limit: int = 5) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size <= limit:
        return [float(x) for x in arr.tolist()]
    return [float(x) for x in arr[:limit].tolist()]


def _log_common_solver_debug(
    header: str,
    sim_oqs: SimulationModuleOQS,
    tlist: np.ndarray,
    run_kwargs: Mapping[str, Any],
    options: Mapping[str, Any],
) -> None:
    solver = sim_oqs.simulation_config.ode_solver
    dt_values = np.diff(tlist)
    print(f"\n=== {header} ===")
    print("solver:", solver)
    print("run_kwargs:", dict(run_kwargs))
    print("max_workers:", sim_oqs.simulation_config.max_workers)
    print("options:", dict(options))
    print("t_wait:", float(sim_oqs.simulation_config.t_wait))
    print("t_coh:", float(sim_oqs.simulation_config.t_coh))
    print(
        "t_wait + t_coh:",
        float(sim_oqs.simulation_config.t_wait + sim_oqs.simulation_config.t_coh),
    )
    print("tlist len:", len(tlist))
    print("t0:", float(tlist[0]))
    print("tlist first few:", _preview_array(tlist[:5]))
    print("tlist last few:", _preview_array(tlist[-5:]))
    print("tlist strictly increasing:", bool(np.all(dt_values > 0)))
    print("tlist finite:", bool(np.all(np.isfinite(tlist))))
    print("dt min/max:", float(np.min(dt_values)), float(np.max(dt_values)))

    from ..config.defaults import ALLOWED_SOLVER_OPTIONS

    for key in ALLOWED_SOLVER_OPTIONS.get(solver, []):
        value = options.get(key)
        print(f"{key} =", value, type(value))


def _log_hamiltonian_debug(sim_oqs: SimulationModuleOQS, tlist: np.ndarray) -> None:
    for sample_time in _sample_times_for_debug(tlist):
        hamiltonian = sim_oqs.H_total_t(float(sample_time))
        matrix = hamiltonian.full()
        print(f"H(t={sample_time}) finite:", bool(np.all(np.isfinite(matrix))))
        print(f"H(t={sample_time}) max abs:", float(np.max(np.abs(matrix))) if matrix.size else 0.0)


def _log_initial_state_debug(sim_oqs: SimulationModuleOQS) -> None:
    state0 = sim_oqs.initial_state
    matrix = state0.full()
    print("rho0 type:", state0.type)
    print("rho0 shape:", state0.shape)
    print("rho0 isherm:", bool(state0.isherm))
    print("rho0 trace:", complex(state0.tr()))
    print("rho0 finite:", bool(np.all(np.isfinite(matrix))))
    print("rho0 max abs:", float(np.max(np.abs(matrix))) if matrix.size else 0.0)


def _log_pulse_debug(sim_oqs: SimulationModuleOQS) -> None:
    print("pulse peak times:", [float(pulse.pulse_peak_time) for pulse in sim_oqs.laser.pulses])
    print(
        "pulse active windows:",
        [
            [float(start), float(end)]
            for start, end in (pulse.active_time_range for pulse in sim_oqs.laser.pulses)
        ],
    )
    print("pulse phases:", [float(pulse.pulse_phase) for pulse in sim_oqs.laser.pulses])
    print("pulse amplitudes:", [float(value) for value in sim_oqs.laser.pulse_amplitudes])


def _log_lindblad_channels_debug(sim_oqs: SimulationModuleOQS) -> None:
    channels = list(sim_oqs.decay_channels)
    print("collapse operators:", len(channels))
    for index, operator in enumerate(channels):
        matrix = operator.full()
        print(f"c_op[{index}] finite:", bool(np.all(np.isfinite(matrix))))
        print(f"c_op[{index}] shape:", operator.shape)
        print(f"c_op[{index}] max abs:", float(np.max(np.abs(matrix))) if matrix.size else 0.0)


def log_redfield_solver_debug(
    sim_oqs: SimulationModuleOQS,
    tlist: np.ndarray,
    run_kwargs: Mapping[str, Any],
    options: Mapping[str, Any],
) -> None:
    """Print the exact Redfield/LSODA inputs and a few finite-value checks."""
    _log_common_solver_debug("PRE-BRMESOLVE DEBUG", sim_oqs, tlist, run_kwargs, options)
    _log_hamiltonian_debug(sim_oqs, tlist)

    for index, channel in enumerate(sim_oqs.decay_channels):
        if not isinstance(channel, tuple) or len(channel) != 2:
            continue
        operator, bath = channel
        matrix = operator.full()
        hermitian_error = (operator - operator.dag()).norm()
        print(f"a_op[{index}] finite:", bool(np.all(np.isfinite(matrix))))
        print(f"a_op[{index}] hermitian error:", float(hermitian_error))
        print(f"bath type[{index}]:", type(bath))


def log_lindblad_solver_debug(
    sim_oqs: SimulationModuleOQS,
    tlist: np.ndarray,
    run_kwargs: Mapping[str, Any],
    options: Mapping[str, Any],
) -> None:
    """Print the exact Lindblad/mesolve inputs and finite-value checks."""
    _log_common_solver_debug("PRE-MESOLVE LINDBLAD DEBUG", sim_oqs, tlist, run_kwargs, options)
    _log_initial_state_debug(sim_oqs)
    _log_pulse_debug(sim_oqs)
    _log_hamiltonian_debug(sim_oqs, tlist)
    _log_lindblad_channels_debug(sim_oqs)
