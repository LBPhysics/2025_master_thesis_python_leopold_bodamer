"""Shared solver-input preparation, validation, and Redfield diagnostics."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..core.simulation import SimulationModuleOQS

__all__ = ["log_redfield_solver_debug"]


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


def log_redfield_solver_debug(
    sim_oqs: SimulationModuleOQS,
    tlist: np.ndarray,
    run_kwargs: Mapping[str, Any],
    options: Mapping[str, Any],
) -> None:
    """Print the exact Redfield/LSODA inputs and a few finite-value checks."""
    dt_values = np.diff(tlist)
    print("\n=== PRE-BRMESOLVE DEBUG ===")
    print("solver:", sim_oqs.simulation_config.ode_solver)
    print("run_kwargs:", dict(run_kwargs))
    print("max_workers:", sim_oqs.simulation_config.max_workers)
    print("options:", dict(options))
    print("tlist len:", len(tlist))
    print("tlist first/last:", float(tlist[0]), float(tlist[-1]))
    print("tlist strictly increasing:", bool(np.all(dt_values > 0)))
    print("tlist finite:", bool(np.all(np.isfinite(tlist))))
    print("dt min/max:", float(np.min(dt_values)), float(np.max(dt_values)))

    for key in ["atol", "rtol", "nsteps", "max_step", "min_step", "method"]:
        value = options.get(key)
        print(f"{key} =", value, type(value))

    for sample_time in _sample_times_for_debug(tlist):
        hamiltonian = sim_oqs.H_total_t(float(sample_time))
        matrix = hamiltonian.full()
        print(f"H(t={sample_time}) finite:", bool(np.all(np.isfinite(matrix))))
        print(f"H(t={sample_time}) max abs:", float(np.max(np.abs(matrix))) if matrix.size else 0.0)

    for index, channel in enumerate(sim_oqs.decay_channels):
        if not isinstance(channel, tuple) or len(channel) != 2:
            continue
        operator, bath = channel
        matrix = operator.full()
        hermitian_error = (operator - operator.dag()).norm()
        print(f"a_op[{index}] finite:", bool(np.all(np.isfinite(matrix))))
        print(f"a_op[{index}] hermitian error:", float(hermitian_error))
        print(f"bath type[{index}]:", type(bath))
