"""Time axis computation utilities."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sim_config import SimulationConfig


def compute_times_global(cfg: "SimulationConfig") -> np.ndarray:
    """Compute the global time grid for the simulation."""
    dt = cfg.dt
    t0 = -2 * cfg.pulse_fwhm_fs - cfg.t_coh_max - cfg.t_wait

    # Compute number of steps to cover from t0 to t_det_max with step dt
    n_steps = int(np.floor((cfg.t_det_max - t0) / dt)) + 1
    # Generate time grid: [t0, t0 + dt, ..., t_det_max]
    times_global = t0 + dt * np.arange(n_steps, dtype=float)
    return times_global


def compute_t_det(cfg: "SimulationConfig") -> np.ndarray:
    """Detection-time axis.

    Behavior depends on simulation type:
    - '0d': return a single detection time equal to t_det_max (as a 1-element array)
    - otherwise: return the usual grid starting from the first non-negative time in
      times_global with spacing dt up to t_det_max.
    """
    # 0d: treat everything as a single detection sample at t_det_max
    if getattr(cfg, "sim_type", "1d") == "0d":
        return np.asarray([float(cfg.t_det_max)], dtype=float)

    # default behaviour for 1d/2d
    dt = cfg.dt
    t_det_max = cfg.t_det_max
    times_global = compute_times_global(cfg)

    # Find the smallest time in times_global that is >= 0
    t0 = times_global[0]
    k = int(np.ceil(-t0 / dt))
    x = t0 + k * dt

    # Ensure x is within bounds
    if x > t_det_max:
        return np.array([])

    # Generate t_det starting from x with step dt, up to <= t_det_max
    n_steps = int(np.floor((t_det_max - x) / dt)) + 1
    t_det = x + dt * np.arange(n_steps, dtype=float)
    return t_det


def compute_t_coh(cfg: "SimulationConfig") -> np.ndarray:
    """Coherence-time axis.

    Behavior depends on simulation type:
    - '0d': single value (cfg.t_coh_max)
    - '1d': single coherence value (cfg.t_coh_max)
    - '2d': return array aligned to times_global grid from ~0 to cfg.t_coh_max
    """
    sim_type = getattr(cfg, "sim_type", "1d")
    t_coh_max = float(cfg.t_coh_max)
    dt = float(cfg.dt)

    if sim_type == "0d" or sim_type == "1d":
        # For 0d/1d return a 1-element array for consistent consumers
        return np.asarray([t_coh_max], dtype=float)

    # 2d: build axis aligned to times_global, starting from grid point closest to 0
    if sim_type == "2d":
        times_global = compute_times_global(cfg)
        t0 = times_global[0]
        # Find the grid point at or immediately after 0 (same logic as t_det)
        k = int(np.ceil((0 - t0) / dt))
        x = t0 + k * dt

        # Ensure x is within bounds
        if x > t_coh_max:
            return np.array([])

        n_steps = int(np.floor((t_coh_max - x) / dt)) + 1
        t_coh_axis = x + dt * np.arange(n_steps, dtype=float)
        return t_coh_axis

    # fallback: return single value
    return np.asarray([t_coh_max], dtype=float)
