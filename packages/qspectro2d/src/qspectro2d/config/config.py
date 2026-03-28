"""Load, merge, normalize, and validate configuration into one concrete config dict."""

from __future__ import annotations

import os
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from .defaults import (
    ALLOWED_SOLVER_OPTIONS,
    COMPONENT_MAP,
    N_PULSES,
    SUPPORTED_BATHS,
    SUPPORTED_ENVELOPES,
    SUPPORTED_SIM_TYPES,
    SUPPORTED_SOLVERS,
    get_defaults,
)

Section = dict[str, Any]
Coercer = Callable[..., Any]


# -----------------------------------------------------------------------------
# YAML loading / merging
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Type coercion helpers
# -----------------------------------------------------------------------------


def _coerce_float(value: Any, *, field_name: str) -> float:
    """Parse numeric config values, including safe fraction strings like '77/23024'."""
    if isinstance(value, bool):
        raise TypeError(
            f"Expected numeric value for {field_name}, got {type(value).__name__}: {value!r}"
        )
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            try:
                return float(Fraction(text))
            except (ValueError, ZeroDivisionError) as exc:
                raise ValueError(
                    f"Could not parse numeric value for {field_name}: {value!r}"
                ) from exc
    raise TypeError(
        f"Expected numeric value for {field_name}, got {type(value).__name__}: {value!r}"
    )


def _coerce_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(
            f"Expected integer value for {field_name}, got {type(value).__name__}: {value!r}"
        )
    if isinstance(value, int):
        return value

    numeric_value = _coerce_float(value, field_name=field_name)
    if not numeric_value.is_integer():
        raise ValueError(f"Expected integer value for {field_name}, got non-integer: {value!r}")
    return int(numeric_value)


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"Expected boolean-like value for {field_name}, got: {value!r}")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Expected boolean-like string for {field_name}, got: {value!r}")
    raise TypeError(
        f"Expected boolean value for {field_name}, got {type(value).__name__}: {value!r}"
    )


def _coerce_float_list(value: Any, *, field_name: str) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Expected a list for {field_name}, got {type(value).__name__}: {value!r}")
    return [
        _coerce_float(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value)
    ]


def _coerce_str_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Expected a list for {field_name}, got {type(value).__name__}: {value!r}")
    return [str(item) for item in value]


def _coerce_str(value: Any, *, field_name: str) -> str:
    del field_name
    return str(value)


SECTION_SCHEMAS: dict[str, dict[str, Coercer]] = {
    "atomic": {
        "n_atoms": _coerce_int,
        "n_chains": _coerce_int,
        "max_excitation": _coerce_int,
        "n_inhomogen": _coerce_int,
        "coupling_cm": _coerce_float,
        "delta_inhomogen_cm": _coerce_float,
        "deph_rate_fs": _coerce_float,
        "down_rate_fs": _coerce_float,
        "up_rate_fs": _coerce_float,
        "frequencies_cm": _coerce_float_list,
        "dip_moments": _coerce_float_list,
    },
    "laser": {
        "pulse_fwhm_fs": _coerce_float,
        "carrier_freq_cm": _coerce_float,
        "pulse_amplitudes": _coerce_float_list,
        "envelope_type": _coerce_str,
        "rwa_sl": _coerce_bool,
    },
    "bath": {
        "bath_temperature": _coerce_float,
        "sb_coupling": _coerce_float,
        "bath_cutoff": _coerce_float,
        "s": _coerce_float,
        "wmax_factor": _coerce_float,
        "peak_strength": _coerce_float,
        "peak_width": _coerce_float,
        "peak_center": _coerce_float,
    },
    "config": {
        "solver": _coerce_str,
        "sim_type": _coerce_str,
        "t_det": _coerce_float,
        "t_coh": _coerce_float,
        "t_wait": _coerce_float,
        "dt": _coerce_float,
        "n_phases": _coerce_int,
        "signal_types": _coerce_str_list,
        "initial_state": _coerce_str,
    },
}


def _apply_schema(section: Section, schema: Mapping[str, Coercer], prefix: str) -> None:
    for key, coercer in schema.items():
        if key in section:
            section[key] = coercer(section[key], field_name=f"{prefix}.{key}")


def _normalize_solver_options(solver: str, solver_options: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(solver_options, Mapping):
        raise TypeError("config.solver_options must be a mapping/dict")

    allowed_keys = set(ALLOWED_SOLVER_OPTIONS.get(solver, []))
    filtered = {k: v for k, v in solver_options.items() if k in allowed_keys}

    if "nsteps" in filtered:
        filtered["nsteps"] = _coerce_int(
            filtered["nsteps"], field_name="config.solver_options.nsteps"
        )
    for key in {"atol", "rtol", "sec_cutoff", "max_step"} & filtered.keys():
        filtered[key] = _coerce_float(filtered[key], field_name=f"config.solver_options.{key}")
    if "method" in filtered:
        filtered["method"] = str(filtered["method"])

    return filtered


# -----------------------------------------------------------------------------
# Post-merge normalization
# -----------------------------------------------------------------------------


def get_max_workers() -> int:
    """Use SLURM allocation if available, otherwise local CPU count."""
    try:
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    except ValueError:
        slurm_cpus = 0

    local_cpus = os.cpu_count() or 1
    return slurm_cpus if slurm_cpus > 0 else int(local_cpus)


def _normalize_solver_state_constraints(cfg: dict[str, Any]) -> None:
    sim_cfg = cfg["config"]
    laser_cfg = cfg["laser"]

    solver = str(sim_cfg["solver"])
    rwa_sl = bool(laser_cfg["rwa_sl"])
    initial_state = str(sim_cfg.get("initial_state", "ground"))

    if solver == "paper_eqs" and not rwa_sl:
        warnings.warn(
            "solver='paper_eqs' requires laser.rwa_sl=True; forcing rwa_sl=True.",
            category=UserWarning,
            stacklevel=2,
        )
        laser_cfg["rwa_sl"] = True

    if initial_state == "thermal" and solver != "redfield":
        warnings.warn(
            "config.initial_state='thermal' is only supported for solver='redfield'; "
            "forcing initial_state='ground'.",
            category=UserWarning,
            stacklevel=2,
        )
        sim_cfg["initial_state"] = "ground"


def _inject_default_max_step(cfg: dict[str, Any]) -> None:
    """Handle max_step internally for time-dependent solvers."""
    sim_cfg = cfg["config"]
    laser_cfg = cfg["laser"]
    solver = str(sim_cfg["solver"])

    if solver not in {"lindblad", "redfield"}:
        return

    max_step = float(laser_cfg["pulse_fwhm_fs"]) / 10.0
    dt = float(sim_cfg["dt"])
    sim_cfg["solver_options"]["max_step"] = max(max_step, dt)


# -----------------------------------------------------------------------------
# Public config API
# -----------------------------------------------------------------------------


def merge_config(user_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = get_defaults()
    if user_cfg:
        _merge_dict(cfg, user_cfg)

    for section_name, schema in SECTION_SCHEMAS.items():
        _apply_schema(cfg[section_name], schema, section_name)

    sim_cfg = cfg["config"]
    sim_cfg["max_workers"] = (
        get_max_workers()
        if sim_cfg.get("max_workers") is None
        else _coerce_int(sim_cfg["max_workers"], field_name="config.max_workers")
    )
    sim_cfg["solver_options"] = _normalize_solver_options(
        str(sim_cfg["solver"]),
        sim_cfg.get("solver_options", {}),
    )

    _normalize_solver_state_constraints(cfg)
    _inject_default_max_step(cfg)
    return cfg


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return merge_config()
    return merge_config(_read_yaml(Path(path)))


def validate_config(cfg: Mapping[str, Any], *, emit_runtime_warnings: bool = True) -> None:
    atomic_cfg = cfg["atomic"]
    laser_cfg = cfg["laser"]
    bath_cfg = cfg["bath"]
    sim_cfg = cfg["config"]

    ode_solver = sim_cfg["solver"]
    bath_type = str(bath_cfg["bath_type"])
    frequencies_cm = atomic_cfg["frequencies_cm"]
    n_atoms = atomic_cfg["n_atoms"]
    dip_moments = atomic_cfg["dip_moments"]
    bath_temp = bath_cfg["bath_temperature"]
    bath_cutoff = bath_cfg["bath_cutoff"]
    bath_coupling = bath_cfg["sb_coupling"]
    bath_s = bath_cfg["s"]
    n_phases = sim_cfg["n_phases"]
    max_excitation = atomic_cfg["max_excitation"]
    n_chains = atomic_cfg["n_chains"]
    pulse_amplitudes = laser_cfg["pulse_amplitudes"]
    rwa_sl = laser_cfg["rwa_sl"]
    carrier_freq_cm = laser_cfg["carrier_freq_cm"]
    pulse_fwhm_fs = laser_cfg["pulse_fwhm_fs"]
    envelope_type = laser_cfg["envelope_type"]
    coupling_cm = atomic_cfg["coupling_cm"]
    delta_inhomogen_cm = atomic_cfg["delta_inhomogen_cm"]
    solver_options = sim_cfg["solver_options"]
    sim_type = sim_cfg["sim_type"]
    initial_state = sim_cfg["initial_state"]
    max_workers = sim_cfg["max_workers"]
    t_det = sim_cfg["t_det"]
    t_coh = sim_cfg["t_coh"]
    dt = sim_cfg["dt"]
    t_wait = sim_cfg["t_wait"]
    n_inhomogen = atomic_cfg["n_inhomogen"]
    signal_types = sim_cfg["signal_types"]
    wmax_factor = bath_cfg["wmax_factor"]
    peak_strength = bath_cfg["peak_strength"]
    peak_width = bath_cfg["peak_width"]
    peak_center = bath_cfg["peak_center"]

    if bath_type not in SUPPORTED_BATHS:
        raise ValueError(f"bath_type '{bath_type}' not in {SUPPORTED_BATHS}")
    if bath_type in {"ohmic", "ohmic+lorentzian"} and bath_s <= 0:
        raise ValueError("bath.s must be > 0 for Ohmic-family baths")

    if ode_solver not in SUPPORTED_SOLVERS:
        raise ValueError(f"Invalid solver '{ode_solver}'. Supported: {sorted(SUPPORTED_SOLVERS)}")

    if initial_state not in {"ground", "thermal"}:
        raise ValueError("config.initial_state must be 'ground' or 'thermal'")
    if initial_state == "thermal" and ode_solver != "redfield":
        raise ValueError("config.initial_state='thermal' is only allowed for solver='redfield'")

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
        raise ValueError("bath.bath_temperature must be >= 0")
    if bath_cutoff <= 0:
        raise ValueError("bath.bath_cutoff must be > 0")
    if bath_coupling <= 0:
        raise ValueError("bath.sb_coupling must be > 0")
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

    if len(pulse_amplitudes) != N_PULSES:
        raise ValueError(f"laser.pulse_amplitudes must have exactly {N_PULSES} elements")
    if any(amplitude <= 0 for amplitude in pulse_amplitudes):
        raise ValueError("All laser.pulse_amplitudes entries must be > 0")

    if not isinstance(solver_options, dict):
        raise TypeError("solver_options must be a dict")

    atol = solver_options.get("atol")
    rtol = solver_options.get("rtol")
    nsteps = solver_options.get("nsteps")
    max_step = solver_options.get("max_step")
    if atol is not None and atol <= 0:
        raise ValueError("solver_options.atol must be > 0")
    if rtol is not None and rtol <= 0:
        raise ValueError("solver_options.rtol must be > 0")
    if nsteps is not None and nsteps <= 0:
        raise ValueError("solver_options.nsteps must be > 0")
    if max_step is not None and max_step < 0:
        raise ValueError("solver_options.max_step must be >= 0")

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

    if rwa_sl and emit_runtime_warnings:
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


def resolve_config(
    config_or_path: Mapping[str, Any] | str | Path | None = None,
    *,
    emit_runtime_warnings: bool = True,
) -> dict[str, Any]:
    """Resolve config from path or dict, merging and validating."""
    cfg: dict[str, Any]
    if config_or_path is None or isinstance(config_or_path, (str, Path)):
        cfg = load_config(config_or_path)
    else:
        cfg = merge_config(config_or_path)
    validate_config(cfg, emit_runtime_warnings=emit_runtime_warnings)
    return cfg


__all__ = [
    "get_max_workers",
    "load_config",
    "merge_config",
    "resolve_config",
    "validate_config",
    "validate_defaults",
]
