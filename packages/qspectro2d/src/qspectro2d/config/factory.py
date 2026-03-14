"""Build simulation objects from one merged config dict."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from qutip import DrudeLorentzEnvironment, OhmicEnvironment
from qutip.core.environment import BosonicEnvironment

from ..core.atomic_system.system import AtomicSystem
from ..core.laser_system.laser import LaserPulseSequence
from ..core.simulation.sim_config import SimulationConfig
from ..core.simulation.simulation import SimulationModuleOQS
from ..utils.constants import convert_cm_to_fs
from .defaults import SUPPORTED_BATHS
from .validate import ensure_config, validate_config


def _ohmic_exponent(raw_value: float | None) -> float:
    if raw_value is not None:
        return float(raw_value)
    return 1.0


def _as_config(source: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if isinstance(source, Path):
        return ensure_config(str(source))
    return ensure_config(source)


def load_simulation_config(source: Mapping[str, Any] | str | Path | None = None) -> SimulationConfig:
    cfg = _as_config(source)
    atomic_cfg = cfg["atomic"]
    laser_cfg = cfg["laser"]
    sim_cfg = cfg["config"]

    return SimulationConfig(
        ode_solver=str(sim_cfg["solver"]),
        solver_options=dict(sim_cfg["solver_options"]),
        rwa_sl=bool(laser_cfg["rwa_sl"]),
        dt=float(sim_cfg["dt"]),
        t_coh_max=float(sim_cfg["t_coh_max"]),
        t_coh_current=sim_cfg["t_coh"],
        t_wait=float(sim_cfg["t_wait"]),
        t_det_max=float(sim_cfg["t_det_max"]),
        pulse_fwhm_fs=float(laser_cfg["pulse_fwhm_fs"]),
        n_phases=int(sim_cfg["n_phases"]),
        n_inhomogen=int(atomic_cfg["n_inhomogen"]),
        signal_types=list(sim_cfg["signal_types"]),
        sim_type=str(sim_cfg["sim_type"]),
        max_workers=int(sim_cfg["max_workers"]),
        initial_state=str(sim_cfg["initial_state"]),
    )


def load_simulation_laser(source: Mapping[str, Any] | str | Path | None = None) -> LaserPulseSequence:
    cfg = _as_config(source)
    laser_cfg = cfg["laser"]
    sim_cfg = cfg["config"]

    t_coh_delay = sim_cfg["t_coh"] if sim_cfg["t_coh"] is not None else sim_cfg["t_coh_max"]

    return LaserPulseSequence.from_pulse_delays(
        pulse_delays=[float(t_coh_delay), float(sim_cfg["t_wait"])],
        pulse_fwhm_fs=float(laser_cfg["pulse_fwhm_fs"]),
        carrier_freq_cm=float(laser_cfg["carrier_freq_cm"]),
        envelope_type=str(laser_cfg["envelope_type"]),
        pulse_amplitudes=list(laser_cfg["pulse_amplitudes"]),
        phases=[0.0, 0.0, 0.0],
    )


def load_simulation_atomic_system(source: Mapping[str, Any] | str | Path | None = None) -> AtomicSystem:
    cfg = _as_config(source)
    atomic_cfg = cfg["atomic"]

    return AtomicSystem(
        n_atoms=int(atomic_cfg["n_atoms"]),
        n_chains=int(atomic_cfg["n_chains"]),
        frequencies_cm=list(atomic_cfg["frequencies_cm"]),
        dip_moments=list(atomic_cfg["dip_moments"]),
        coupling_cm=float(atomic_cfg["coupling_cm"]),
        delta_inhomogen_cm=float(atomic_cfg["delta_inhomogen_cm"]),
        max_excitation=int(atomic_cfg["max_excitation"]),
    )


def load_simulation_bath(source: Mapping[str, Any] | str | Path | None = None) -> BosonicEnvironment:
    cfg = _as_config(source)
    atomic_cfg = cfg["atomic"]
    bath_cfg = cfg["bath"]

    freqs_cm = list(atomic_cfg["frequencies_cm"])
    if not freqs_cm:
        raise ValueError("Missing atomic.frequencies_cm; cannot determine bath scaling.")

    freqs_fs = np.asarray(convert_cm_to_fs(freqs_cm), dtype=float)
    w0_bar_fs = float(np.mean(freqs_fs))

    bath_type = str(bath_cfg["bath_type"])
    temperature = float(bath_cfg["temperature"]) * w0_bar_fs
    cutoff = float(bath_cfg["cutoff"]) * w0_bar_fs
    coupling = float(bath_cfg["coupling"]) * w0_bar_fs
    bath_s = _ohmic_exponent(bath_cfg.get("s"))
    w_max = float(bath_cfg["wmax_factor"]) * cutoff

    peak_strength = float(bath_cfg["peak_strength"]) * coupling
    peak_width = float(bath_cfg["peak_width"]) * w0_bar_fs
    peak_center = float(bath_cfg["peak_center"]) * w0_bar_fs

    if bath_type == "ohmic":
        return OhmicEnvironment(T=temperature, alpha=coupling, wc=cutoff, s=bath_s, tag=bath_type)

    if bath_type == "drudelorentz":
        return DrudeLorentzEnvironment(
            T=temperature,
            gamma=cutoff,
            lam=coupling * cutoff / 2,
            tag=bath_type,
        )

    if bath_type in {
        "ohmic+lorentzian",
        "drudelorentz+lorentzian",
    }:
        if bath_type == "ohmic+lorentzian":
            bath_base = OhmicEnvironment(
                T=temperature,
                alpha=coupling,
                wc=cutoff,
                s=bath_s,
                tag=bath_type.split("+", 1)[0],
            )
        else:
            bath_base = DrudeLorentzEnvironment(
                T=temperature,
                gamma=cutoff,
                lam=coupling * cutoff / 2,
                tag="drudelorentz",
            )

        def j_lorentz_peak(w, center=peak_center, gamma=peak_width, strength=peak_strength):
            w_arr = np.asarray(w, dtype=float)
            result = np.zeros_like(w_arr)
            positive = w_arr > 0
            wp = w_arr[positive]
            result[positive] = strength * wp * (gamma**2) / ((wp - center) ** 2 + gamma**2)
            return float(result) if np.isscalar(w) else result

        def j_base_plus_peak(w):
            return bath_base.spectral_density(w) + j_lorentz_peak(w)

        bath_env = BosonicEnvironment.from_spectral_density(
            j_base_plus_peak,
            T=temperature,
            wMax=w_max,
            tag=bath_type,
        )
        if bath_type == "ohmic+lorentzian":
            bath_env.wc = cutoff
            bath_env.alpha = coupling
            bath_env.s = bath_s
        else:
            bath_env.gamma = cutoff
            bath_env.lam = coupling * cutoff / 2
        return bath_env

    raise ValueError(f"Unsupported bath_type: {bath_type}. Supported: {SUPPORTED_BATHS}")


def load_simulation(
    source: Mapping[str, Any] | str | Path | None = None,
    run_validation: bool = False,
) -> SimulationModuleOQS:
    cfg = _as_config(source)
    if run_validation:
        validate_config(cfg)

    return SimulationModuleOQS(
        simulation_config=load_simulation_config(cfg),
        system=load_simulation_atomic_system(cfg),
        laser=load_simulation_laser(cfg),
        bath=load_simulation_bath(cfg),
    )


def create_base_sim_oqs(
    config_path: Path | None = None,
) -> tuple[SimulationModuleOQS, float]:
    sim = load_simulation(config_path, run_validation=True)

    print("Base simulation created from config.")

    time_cut = -np.inf
    t_max = sim.simulation_config.t_det_max
    print("Validating solver...")
    try:
        from qspectro2d.diagnostics import check_the_solver

        time_cut = check_the_solver(sim)
        print("#" * 60)
        print(
            "Solver validation worked: Evolution becomes unphysical at "
            f"({time_cut / t_max:.2f} x t_max)"
        )
    except Exception as exc:
        print(f"WARNING: Solver validation failed: {exc}")

    if time_cut < t_max:
        print(
            f"WARNING: Time cut {time_cut} is less than the last time point {t_max}. "
            "This may affect the simulation results.",
            flush=True,
        )

    return sim, time_cut


__all__ = [
    "create_base_sim_oqs",
    "load_simulation",
    "load_simulation_atomic_system",
    "load_simulation_bath",
    "load_simulation_config",
    "load_simulation_laser",
]