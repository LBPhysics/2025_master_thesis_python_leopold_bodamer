"""Load, merge, normalize, and validate configuration into one concrete config dict."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .defaults import (
    ALLOWED_SOLVER_OPTIONS,
    COMPONENT_MAP,
    SOLVER_OPTIONS,
    SUPPORTED_BATHS,
    SUPPORTED_ENVELOPES,
    SUPPORTED_SIM_TYPES,
    SUPPORTED_SOLVERS,
    get_defaults,
)


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

    sim_cfg["t_det"] = float(sim_cfg["t_det"])
    sim_cfg["t_coh"] = float(sim_cfg["t_coh"])
    sim_cfg["t_wait"] = float(sim_cfg["t_wait"])
    sim_cfg["dt"] = float(sim_cfg["dt"])
    sim_cfg["solver"] = str(sim_cfg["solver"])
    sim_cfg["sim_type"] = str(sim_cfg["sim_type"])

    sim_cfg["max_workers"] = (
        get_max_workers() if sim_cfg.get("max_workers") is None else int(sim_cfg["max_workers"])
    )

    solver = sim_cfg["solver"]
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


def validate_config(cfg: Mapping[str, Any]) -> None:
    atomic_cfg = cfg["atomic"]
    laser_cfg = cfg["laser"]
    bath_cfg = cfg["bath"]
    sim_cfg = cfg["config"]

    ode_solver = str(sim_cfg["solver"])
    bath_type = str(bath_cfg["bath_type"])
    frequencies_cm = atomic_cfg["frequencies_cm"]
    n_atoms = int(atomic_cfg["n_atoms"])
    dip_moments = atomic_cfg["dip_moments"]
    bath_temp = float(bath_cfg["temperature"])
    bath_cutoff = float(bath_cfg["cutoff"])
    bath_coupling = float(bath_cfg["coupling"])
    bath_s = float(bath_cfg["s"])
    n_phases = int(sim_cfg["n_phases"])
    max_excitation = int(atomic_cfg["max_excitation"])
    n_chains = int(atomic_cfg["n_chains"])
    pulse_amplitudes = laser_cfg["pulse_amplitudes"]
    rwa_sl = bool(laser_cfg["rwa_sl"])
    carrier_freq_cm = float(laser_cfg["carrier_freq_cm"])
    pulse_fwhm_fs = float(laser_cfg["pulse_fwhm_fs"])
    envelope_type = str(laser_cfg["envelope_type"])
    coupling_cm = float(atomic_cfg["coupling_cm"])
    delta_inhomogen_cm = float(atomic_cfg["delta_inhomogen_cm"])
    solver_options = sim_cfg["solver_options"]
    sim_type = str(sim_cfg["sim_type"])
    max_workers = int(sim_cfg["max_workers"])
    t_det = float(sim_cfg["t_det"])
    t_coh = float(sim_cfg["t_coh"])
    dt = float(sim_cfg["dt"])
    t_wait = float(sim_cfg["t_wait"])
    n_inhomogen = int(atomic_cfg["n_inhomogen"])
    signal_types = list(sim_cfg["signal_types"])
    wmax_factor = float(bath_cfg["wmax_factor"])
    peak_strength = float(bath_cfg["peak_strength"])
    peak_width = float(bath_cfg["peak_width"])
    peak_center = float(bath_cfg["peak_center"])

    if bath_type not in SUPPORTED_BATHS:
        raise ValueError(f"bath_type '{bath_type}' not in {SUPPORTED_BATHS}")

    if bath_type in {"ohmic", "ohmic+lorentzian"} and float(bath_s) <= 0:
        raise ValueError("bath.s must be > 0 for Ohmic-family baths")

    if ode_solver not in SUPPORTED_SOLVERS:
        raise ValueError(f"Invalid solver '{ode_solver}'. Supported: {sorted(SUPPORTED_SOLVERS)}")

    if dt <= 0:
        raise ValueError("dt must be > 0")
    if t_coh < 0:
        raise ValueError("t_coh must be >= 0")
    if t_wait < 0:
        raise ValueError("t_wait must be >= 0")
    if t_det <= 0:
        raise ValueError("t_det must be > 0")

    if pulse_fwhm_fs <= 0:
        raise ValueError("pulse_fwhm_fs must be > 0")
    if envelope_type not in SUPPORTED_ENVELOPES:
        raise ValueError(f"envelope_type '{envelope_type}' not in {SUPPORTED_ENVELOPES}")

    if coupling_cm < 0:
        raise ValueError("coupling_cm must be >= 0")
    if delta_inhomogen_cm < 0:
        raise ValueError("delta_inhomogen_cm must be >= 0")

    if n_phases <= 0:
        raise ValueError("n_phases must be > 0")
    if n_inhomogen <= 0:
        raise ValueError("n_inhomogen must be > 0")

    if len(frequencies_cm) != n_atoms:
        raise ValueError(f"frequencies_cm length ({len(frequencies_cm)}) != n_atoms ({n_atoms})")
    if len(dip_moments) != n_atoms:
        raise ValueError(f"dip_moments length ({len(dip_moments)}) != n_atoms ({n_atoms})")

    if bath_temp < 0:
        raise ValueError("bath.temperature must be >= 0")
    if bath_cutoff <= 0:
        raise ValueError("bath.cutoff must be > 0")
    if bath_coupling <= 0:
        raise ValueError("bath.coupling must be > 0")
    if wmax_factor <= 0:
        raise ValueError("bath.wmax_factor must be > 0")

    if bath_type.endswith("+lorentzian"):
        if peak_strength < 0:
            raise ValueError("bath.peak_strength must be >= 0")
        if peak_width <= 0:
            raise ValueError("bath.peak_width must be > 0")
        if peak_center < 0:
            raise ValueError("bath.peak_center must be >= 0")

    if max_excitation not in (1, 2):
        raise ValueError("max_excitation must be 1 or 2")

    if n_chains < 1:
        raise ValueError("n_chains must be >= 1")
    if n_atoms > 2 and n_atoms % n_chains != 0:
        raise ValueError(
            f"n_chains ({n_chains}) does not divide n_atoms ({n_atoms}) for cylindrical geometry"
        )

    if len(pulse_amplitudes) != 3:
        raise ValueError("laser.pulse_amplitudes must have exactly 3 elements")
    if any(amplitude <= 0 for amplitude in pulse_amplitudes):
        raise ValueError("All laser.pulse_amplitudes entries must be > 0")

    if not isinstance(solver_options, dict):
        raise TypeError("solver_options must be a dict")

    atol = solver_options.get("atol")
    rtol = solver_options.get("rtol")
    nsteps = solver_options.get("nsteps")
    max_step = solver_options.get("max_step")
    min_step = solver_options.get("min_step")
    if atol is not None and atol <= 0:
        raise ValueError("solver_options.atol must be > 0")
    if rtol is not None and rtol <= 0:
        raise ValueError("solver_options.rtol must be > 0")
    if nsteps is not None and nsteps <= 0:
        raise ValueError("solver_options.nsteps must be > 0")
    if min_step is not None and min_step < 0:
        raise ValueError("solver_options.min_step must be >= 0")
    if max_step is not None and max_step < 0:
        raise ValueError("solver_options.max_step must be >= 0")
    if min_step is not None and max_step is not None and min_step > max_step:
        raise ValueError("solver_options.min_step must be <= solver_options.max_step")

    allowed_keys = set(ALLOWED_SOLVER_OPTIONS.get(ode_solver, []))
    unknown_keys = set(solver_options) - allowed_keys
    if unknown_keys:
        raise ValueError(
            f"solver_options includes unsupported keys for {ode_solver}: {sorted(unknown_keys)}"
        )

    if sim_type not in SUPPORTED_SIM_TYPES:
        raise ValueError(f"sim_type '{sim_type}' not in {SUPPORTED_SIM_TYPES}")
    if max_workers <= 0:
        raise ValueError("max_workers must be >= 1")

    unknown_signal_types = [
        signal_type for signal_type in signal_types if signal_type not in COMPONENT_MAP
    ]
    if unknown_signal_types:
        raise ValueError(f"Unsupported signal_types: {unknown_signal_types}")

    if rwa_sl:
        freqs_array = np.asarray(frequencies_cm, dtype=float)
        max_detuning = np.max(np.abs(freqs_array - carrier_freq_cm))
        rel_detuning = max_detuning / carrier_freq_cm if carrier_freq_cm != 0 else np.inf
        if rel_detuning > 1e-2:
            print(
                (
                    "WARNING: RWA probably not valid, since relative detuning: "
                    f"{rel_detuning} is too large"
                ),
                flush=True,
            )


def validate_defaults() -> None:
    validate_config(load_config())


def resolve_config(config_or_path: Mapping[str, Any] | str | Path | None = None) -> dict[str, Any]:
    """Resolve config from path or dict, merging and validating."""
    cfg: dict[str, Any]
    if config_or_path is None or isinstance(config_or_path, (str, Path)):
        cfg = load_config(config_or_path)
    else:
        cfg = merge_config(config_or_path)
    validate_config(cfg)
    return cfg


__all__ = [
    "get_max_workers",
    "load_config",
    "merge_config",
    "resolve_config",
    "validate_config",
    "validate_defaults",
]
