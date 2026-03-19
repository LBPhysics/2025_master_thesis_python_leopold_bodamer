"""Time axis computation utilities."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sim_config import SimulationConfig

from ..laser_system.laser import DEFAULT_ACTIVE_WINDOW_NFWHM


def _validate_solver_time_grid(tlist: np.ndarray) -> None:
    """Validate the solver time grid invariants at construction time."""
    if tlist.ndim != 1:
        raise ValueError("tlist must be a one-dimensional array")
    if len(tlist) < 2:
        raise ValueError("tlist must contain at least two time points")
    if not np.all(np.isfinite(tlist)):
        raise ValueError("tlist must contain only finite values")

    dt_values = np.diff(tlist)
    if not np.all(dt_values > 0):
        raise ValueError("tlist must be strictly increasing")


def _solver_start_time(cfg: "SimulationConfig", t_coh: float) -> float:
    # Keep one consistent definition of the left solver boundary.
    return float(np.floor(-(cfg.t_wait + t_coh + DEFAULT_ACTIVE_WINDOW_NFWHM * cfg.pulse_fwhm_fs)))


def _first_non_negative_grid_time(t0: float, dt: float) -> float:
    k = int(np.floor((-t0) / dt))
    x = t0 + k * dt
    if x < 0:
        x += dt
    return float(x)


def _validated_sim_type(cfg: "SimulationConfig") -> str:
    sim_type = str(getattr(cfg, "sim_type", "1d"))
    if sim_type not in {"0d", "1d", "2d"}:
        raise ValueError(f"Unsupported sim_type: {sim_type}")
    return sim_type


def compute_times_local(
    cfg: "SimulationConfig",
    *,
    t_coh_override: float | None = None,
) -> np.ndarray:
    """Compute the solver-local time grid.

    By default, this uses cfg.t_coh_current as the active coherence delay.
    For special cases (for example, constructing a 2D sweep axis) callers can
    provide t_coh_override to force a specific coherence delay.
    """
    dt = float(cfg.dt)
    t_det_max = float(cfg.t_det_max)
    if t_coh_override is None and getattr(cfg, "t_coh_current", None) is None:
        raise ValueError("cfg.t_coh_current must be set before computing local time grid")
    t_coh = float(cfg.t_coh_current if t_coh_override is None else t_coh_override)
    t0 = _solver_start_time(cfg, t_coh)

    n_steps = int(np.floor((t_det_max - t0) / dt)) + 1
    times_local = t0 + dt * np.arange(n_steps, dtype=float)
    _validate_solver_time_grid(times_local)
    return times_local


def compute_t_det(cfg: "SimulationConfig") -> np.ndarray:
    """Detection-time axis.

    Behavior depends on simulation type:
    - '0d': return a single detection time equal to t_det_max (as a 1-element array)
    - otherwise: return the usual grid starting from the first non-negative time in
        times_local with spacing dt up to t_det_max.
    """
    sim_type = _validated_sim_type(cfg)

    # 0d: treat everything as a single detection sample at t_det_max
    if sim_type == "0d":
        return np.asarray([float(cfg.t_det_max)], dtype=float)

    dt = float(cfg.dt)
    t_det_max = float(cfg.t_det_max)
    times_local = compute_times_local(cfg)

    x = _first_non_negative_grid_time(float(times_local[0]), dt)

    # Ensure x is within bounds
    if x > t_det_max:
        return np.array([])

    n_steps = int(np.floor((t_det_max - x) / dt)) + 1
    t_det = x + dt * np.arange(n_steps, dtype=float)
    return t_det


def compute_t_coh(cfg: "SimulationConfig") -> np.ndarray:
    """Coherence-time axis.

    Behavior depends on simulation type:
    - '0d': single value (cfg.t_coh_current)
    - '1d': single coherence value (cfg.t_coh_current)
    - '2d': return array aligned to the local time grid from ~0 to cfg.t_coh_max
    """
    sim_type = _validated_sim_type(cfg)
    dt = float(cfg.dt)

    if sim_type in {"0d", "1d"}:
        # 0d/1d run a single coherence-time value.
        t_coh_value = float(cfg.t_coh_current)
        return np.asarray([t_coh_value], dtype=float)

    if sim_type == "2d":
        t_coh_max = float(cfg.t_coh_max)
        times_local = compute_times_local(cfg, t_coh_override=t_coh_max)
        x = _first_non_negative_grid_time(float(times_local[0]), dt)
        # Ensure x is within bounds
        if x > t_coh_max:
            return np.array([])

        n_steps = int(np.floor((t_coh_max - x) / dt)) + 1
        t_coh_axis = x + dt * np.arange(n_steps, dtype=float)
        return t_coh_axis

    raise ValueError(f"Unsupported sim_type: {sim_type}")


def compute_global_t_det(cfg: "SimulationConfig") -> np.ndarray:
    """Compute the GLOBAL detection-time grid (using t_coh_max).

    This is used for consistent output across all t_coh sweeps.
    All signals are padded/cropped to match this grid.
    """
    sim_type = _validated_sim_type(cfg)

    if sim_type == "0d":
        return np.asarray([float(cfg.t_det_max)], dtype=float)

    dt = float(cfg.dt)
    t_det_max = float(cfg.t_det_max)

    t0 = _solver_start_time(cfg, float(cfg.t_coh_max))
    x = _first_non_negative_grid_time(t0, dt)

    if x > t_det_max:
        return np.array([])

    n_steps = int(np.floor((t_det_max - x) / dt)) + 1
    t_det = x + dt * np.arange(n_steps, dtype=float)
    return t_det
