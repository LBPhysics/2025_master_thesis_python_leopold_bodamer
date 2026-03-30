"""Time axis computation utilities.

Canonical rule
--------------
- The solver-local grid may start before the first pulse.
- Public detection and coherence axes always start at 0.
- dt controls spacing only, never a hidden origin shift.
"""

from __future__ import annotations
import numpy as np

from .sim_config import SimulationConfig
from ..laser_system.laser import DEFAULT_ACTIVE_WINDOW_NFWHM


def _validate_solver_time_grid(tlist: np.ndarray) -> None:
    if tlist.ndim != 1:
        raise ValueError("tlist must be one-dimensional")
    if tlist.size < 2:
        raise ValueError("tlist must contain at least two points")
    if not np.all(np.isfinite(tlist)):
        raise ValueError("tlist must contain only finite values")
    if not np.all(np.diff(tlist) > 0):
        raise ValueError("tlist must be strictly increasing")


def _validated_sim_type(cfg: "SimulationConfig") -> str:
    sim_type = str(getattr(cfg, "sim_type", "1d"))
    if sim_type not in {"0d", "1d", "2d"}:
        raise ValueError(f"Unsupported sim_type: {sim_type}")
    return sim_type


def _solver_start_time(cfg: "SimulationConfig", t_coh: float) -> float:
    dt = float(cfg.dt)
    envelope_type = str(getattr(cfg, "envelope_type", "gaussian"))

    if envelope_type == "delta":
        left_edge = -(float(cfg.t_wait) + float(t_coh)) - dt
    else:
        left_edge = -(
            float(cfg.t_wait)
            + float(t_coh)
            + DEFAULT_ACTIVE_WINDOW_NFWHM * float(cfg.pulse_fwhm_fs)
        )

    return float(dt * np.floor(left_edge / dt))


def _uniform_axis(stop: float, dt: float) -> np.ndarray:
    stop = float(stop)
    dt = float(dt)

    if dt <= 0:
        raise ValueError("dt must be positive")
    if stop < 0:
        raise ValueError("axis stop must be non-negative")

    n = int(np.floor(stop / dt)) + 1
    return dt * np.arange(n, dtype=float)


def compute_times_local(
    cfg: "SimulationConfig",
    *,
    t_coh_override: float | None = None,
) -> np.ndarray:
    """Return the full solver-local time grid used for propagation."""
    dt = float(cfg.dt)
    t_det = float(cfg.t_det)
    t_coh_value = float(cfg.t_coh if t_coh_override is None else t_coh_override)

    t0 = _solver_start_time(cfg, t_coh_value)
    n_steps = int(np.floor((t_det - t0) / dt)) + 1
    times_local = t0 + dt * np.arange(n_steps, dtype=float)

    _validate_solver_time_grid(times_local)
    return times_local


def compute_t_det(
    cfg: "SimulationConfig",
) -> np.ndarray:
    """Return the public detection-time axis."""
    sim_type = _validated_sim_type(cfg)

    if sim_type == "0d":
        return np.asarray([float(cfg.t_det)], dtype=float)

    return _uniform_axis(float(cfg.t_det), float(cfg.dt))


def compute_t_coh(cfg: "SimulationConfig") -> np.ndarray:
    """Return the public coherence-time axis."""
    sim_type = _validated_sim_type(cfg)

    if sim_type in {"0d", "1d"}:
        return np.asarray([float(cfg.t_coh)], dtype=float)

    return _uniform_axis(float(cfg.t_coh), float(cfg.dt))
