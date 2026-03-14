"""Load and merge configuration into one concrete config dict."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from .defaults import ALLOWED_SOLVER_OPTIONS, SOLVER_OPTIONS, default_pulse_fwhm_fs, get_defaults


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
    user_laser_cfg: Mapping[str, Any] = {}
    if isinstance(user_cfg, Mapping):
        laser_section = user_cfg.get("laser", {})
        if isinstance(laser_section, Mapping):
            user_laser_cfg = laser_section
    if user_cfg:
        _merge_dict(cfg, user_cfg)

    atomic_cfg = cfg["atomic"]
    laser_cfg = cfg["laser"]
    bath_cfg = cfg["bath"]
    sim_cfg = cfg["config"]

    n_atoms = int(atomic_cfg["n_atoms"])
    atomic_cfg["n_atoms"] = n_atoms
    atomic_cfg["n_chains"] = int(atomic_cfg["n_chains"])
    atomic_cfg["frequencies_cm"] = [float(value) for value in atomic_cfg["frequencies_cm"]]
    atomic_cfg["dip_moments"] = [float(value) for value in atomic_cfg["dip_moments"]]
    atomic_cfg["coupling_cm"] = float(atomic_cfg["coupling_cm"])
    atomic_cfg["max_excitation"] = int(atomic_cfg["max_excitation"])
    atomic_cfg["n_inhomogen"] = int(atomic_cfg["n_inhomogen"])
    atomic_cfg["delta_inhomogen_cm"] = float(atomic_cfg["delta_inhomogen_cm"])
    atomic_cfg["deph_rate_fs"] = float(atomic_cfg["deph_rate_fs"])
    atomic_cfg["down_rate_fs"] = float(atomic_cfg["down_rate_fs"])
    atomic_cfg["up_rate_fs"] = float(atomic_cfg["up_rate_fs"])

    laser_cfg["pulse_fwhm_fs"] = (
        default_pulse_fwhm_fs(n_atoms)
        if laser_cfg.get("pulse_fwhm_fs") is None
        else float(laser_cfg["pulse_fwhm_fs"])
    )
    base_amplitude = float(laser_cfg.get("base_amplitude", 0.01))
    if "pulse_amplitudes" in user_laser_cfg:
        laser_cfg["pulse_amplitudes"] = [float(value) for value in laser_cfg["pulse_amplitudes"]]
    else:
        # Legacy behavior: first two pulses at E0, probe pulse at 10% of E0.
        laser_cfg["pulse_amplitudes"] = [base_amplitude, base_amplitude, 0.1 * base_amplitude]
    laser_cfg["carrier_freq_cm"] = float(laser_cfg["carrier_freq_cm"])
    laser_cfg["rwa_sl"] = bool(laser_cfg["rwa_sl"])

    bath_cfg["temperature"] = float(bath_cfg["temperature"])
    bath_cfg["cutoff"] = float(bath_cfg["cutoff"])
    bath_cfg["coupling"] = float(bath_cfg["coupling"])
    bath_cfg["s"] = None if bath_cfg.get("s") is None else float(bath_cfg["s"])
    bath_cfg["wmax_factor"] = float(bath_cfg["wmax_factor"])
    bath_cfg["peak_strength"] = float(bath_cfg["peak_strength"])
    bath_cfg["peak_width"] = float(bath_cfg["peak_width"])
    bath_cfg["peak_center"] = float(bath_cfg["peak_center"])

    sim_cfg["solver"] = str(sim_cfg["solver"])
    sim_cfg["t_det_max"] = float(sim_cfg["t_det_max"])
    sim_cfg["t_coh_max"] = (
        sim_cfg["t_det_max"] if sim_cfg.get("t_coh_max") is None else float(sim_cfg["t_coh_max"])
    )
    sim_cfg["t_coh"] = None if sim_cfg.get("t_coh") is None else float(sim_cfg["t_coh"])
    if sim_cfg["sim_type"] in {"0d", "1d"} and sim_cfg["t_coh"] is None:
        sim_cfg["t_coh"] = sim_cfg["t_coh_max"]
    sim_cfg["t_wait"] = float(sim_cfg["t_wait"])
    sim_cfg["dt"] = float(sim_cfg["dt"])
    sim_cfg["n_phases"] = int(sim_cfg["n_phases"])
    sim_cfg["signal_types"] = list(sim_cfg["signal_types"])
    sim_cfg["max_workers"] = (
        get_max_workers() if sim_cfg.get("max_workers") is None else int(sim_cfg["max_workers"])
    )
    sim_cfg["solver_options"] = _normalize_solver_options(
        sim_cfg["solver"],
        sim_cfg.get("solver_options", {}),
    )

    return cfg


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML once and return one merged config dict."""
    if path is None:
        return merge_config()
    return merge_config(_read_yaml(Path(path)))


__all__ = ["get_max_workers", "load_config", "merge_config"]
