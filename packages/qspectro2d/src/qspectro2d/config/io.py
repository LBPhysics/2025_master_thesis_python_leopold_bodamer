"""Load and merge configuration into one concrete config dict."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from .defaults import ALLOWED_SOLVER_OPTIONS, SOLVER_OPTIONS, get_defaults


def _read_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise TypeError("Top-level YAML must be a mapping/dict")
    return data


def _merge_dict(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _normalize_solver_options(solver: str, solver_options: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(SOLVER_OPTIONS.get(solver, {}))
    merged.update(dict(solver_options))
    allowed_keys = set(ALLOWED_SOLVER_OPTIONS.get(solver, []))

    normalized: dict[str, Any] = {}
    for key, value in merged.items():
        if key not in allowed_keys:
            continue
        if isinstance(value, str):
            text = value.strip()
            try:
                numeric_value = float(text)
            except ValueError:
                normalized[key] = value
            else:
                if numeric_value.is_integer() and "e" not in text.lower() and "." not in text:
                    normalized[key] = int(numeric_value)
                else:
                    normalized[key] = numeric_value
        else:
            normalized[key] = value
    return normalized


def get_max_workers() -> int:
    """Use SLURM allocation if available, otherwise local CPU count."""
    try:
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    except ValueError:
        slurm_cpus = 0

    local_cpus = os.cpu_count() or 1
    return slurm_cpus if slurm_cpus > 0 else int(local_cpus)


def merge_config(user_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Merge user config onto defaults and resolve derived values once."""
    cfg = get_defaults()
    if user_cfg:
        _merge_dict(cfg, user_cfg)

    laser_cfg = cfg["laser"]
    sim_cfg = cfg["config"]

    laser_cfg["pulse_fwhm_fs"] = float(laser_cfg["pulse_fwhm_fs"])

    sim_cfg["t_det_max"] = float(sim_cfg["t_det_max"])
    sim_cfg["t_coh_max"] = (
        sim_cfg["t_det_max"] if sim_cfg.get("t_coh_max") is None else float(sim_cfg["t_coh_max"])
    )
    sim_cfg["t_coh"] = None if sim_cfg.get("t_coh") is None else float(sim_cfg["t_coh"])
    if sim_cfg["sim_type"] in {"0d", "1d"} and sim_cfg["t_coh"] is None:
        sim_cfg["t_coh"] = sim_cfg["t_coh_max"]
    sim_cfg["max_workers"] = (
        get_max_workers() if sim_cfg.get("max_workers") is None else int(sim_cfg["max_workers"])
    )
    solver = str(sim_cfg["solver"])
    sim_cfg["solver"] = solver
    sim_cfg["solver_options"] = _normalize_solver_options(
        solver,
        sim_cfg.get("solver_options", {}),
    )
    if solver == "redfield" and sim_cfg["solver_options"].get("max_step") is None:
        sim_cfg["solver_options"]["max_step"] = float(laser_cfg["pulse_fwhm_fs"]) / 4.0

    return cfg


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML once and return one merged config dict."""
    if path is None:
        return merge_config()
    return merge_config(_read_yaml(Path(path)))


__all__ = ["get_max_workers", "load_config", "merge_config"]
