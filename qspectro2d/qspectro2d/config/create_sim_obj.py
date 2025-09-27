"""Simplified configuration → simulation factory.

Purpose:
    Load a YAML config (or fall back to defaults) and directly build a
    `SimulationModuleOQS` instance with the core objects:
        - AtomicSystem
        - LaserPulseSequence
        - BosonicEnvironment (qutip)
        - SimulationConfig

Usage:
    from qspectro2d.config.create_sim_obj import load_simulation
    sim = load_simulation("scripts/config.yaml")  # or None for defaults

YAML Schema (example of all options):
"""

from __future__ import annotations

import os
import hashlib
from re import T
import numpy as np
import psutil
from pathlib import Path
from typing import Any, Mapping, Optional
from qutip import OhmicEnvironment
import yaml

from ..core.simulation.simulation_class import SimulationModuleOQS
from . import default_simulation_params as dflt

__all__ = ["load_simulation", "create_base_sim_oqs", "get_max_workers"]


# HELPERS


def _read_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Add 'PyYAML' to requirements.")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise TypeError("Top-level YAML must be a mapping/dict")
    return data


def _get_section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    sec = cfg.get(name, {})
    return sec if isinstance(sec, Mapping) else {}


def load_simulation(
    path: Optional[str | Path] = None,
    validate: bool = True,
) -> SimulationModuleOQS:
    """Create a `SimulationModuleOQS` directly from a YAML file or defaults.

    Overrides (if provided) take precedence over YAML/defaults and are applied
    BEFORE constructing the laser sequence & simulation config so that all
    derived internal time arrays are consistent. This avoids the need to
    rebuild the `SimulationModuleOQS` later and prevents mismatches (e.g.
    insufficient evolution states) when large coherence pulse_delays were first
    baked in and then changed afterwards.

    Parameters
    ----------
    path: str | Path | None
        YAML configuration file. If None, defaults from
        `default_simulation_params` are used.
    validate: bool
        If True (default) run physics validation via `default_simulation_params.validate`.
    t_coh_override, t_wait_override, t_det_max_override, dt_override, ode_solver_override:
        Optional scalar overrides for timing / solver settings.
    """
    # Import here to avoid circular import
    from qspectro2d.core.atomic_system.system_class import AtomicSystem
    from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
    from qspectro2d.core.simulation.simulation_class import SimulationModuleOQS
    from qspectro2d.core.simulation.sim_config import SimulationConfig

    # LOAD / FALLBACK
    # If no path is provided, use an empty config so all values fall back to defaults in `dflt`.
    if path is None:
        cfg_root = {}
    else:
        cfg_root = _read_yaml(Path(path))

    # ATOMIC SYSTEM
    atomic_cfg = _get_section(cfg_root, "atomic")
    n_atoms = int(atomic_cfg.get("n_atoms", dflt.N_ATOMS))
    n_chains = int(atomic_cfg.get("n_chains", dflt.N_CHAINS))
    freqs_cm = list(atomic_cfg.get("frequencies_cm", dflt.FREQUENCIES_CM))
    dip_moments = list(atomic_cfg.get("dip_moments", dflt.DIP_MOMENTS))
    coupling_cm = float(atomic_cfg.get("coupling_cm", dflt.COUPLING_CM))
    delta_inhomogen_cm = float(atomic_cfg.get("delta_inhomogen_cm", dflt.DELTA_INHOMOGEN_CM))
    max_excitation = int(atomic_cfg.get("max_excitation", dflt.MAX_EXCITATION))

    atomic_system = AtomicSystem(
        n_atoms=n_atoms,
        n_chains=n_chains,
        frequencies_cm=freqs_cm,
        dip_moments=dip_moments,
        coupling_cm=coupling_cm,
        delta_inhomogen_cm=delta_inhomogen_cm,
        max_excitation=max_excitation,
    )

    # LASER / PULSES
    laser_cfg = _get_section(cfg_root, "laser")
    pulse_fwhm_fs = float(laser_cfg.get("pulse_fwhm_fs", dflt.PULSE_FWHM_FS))
    base_amp = float(laser_cfg.get("base_amplitude", dflt.BASE_AMPLITUDE))
    envelope = str(laser_cfg.get("envelope_type", dflt.ENVELOPE_TYPE))
    carrier_cm = float(laser_cfg.get("carrier_freq_cm", dflt.CARRIER_FREQ_CM))
    rwa_sl = bool(laser_cfg.get("rwa_sl", dflt.RWA_SL))

    pulses_cfg = _get_section(cfg_root, "pulses")
    relative_e0s = list(pulses_cfg.get("relative_e0s", dflt.RELATIVE_E0S))

    # synthesize 3-pulse sequence from time window
    window_cfg = _get_section(cfg_root, "window")
    t_coh = float(window_cfg.get("t_coh", dflt.T_COH))
    t_wait = float(window_cfg.get("t_wait", dflt.T_WAIT))
    dt = float(window_cfg.get("dt", dflt.DT))
    t_det_max = float(window_cfg.get("t_det_max", dflt.T_DET_MAX))

    # Apply early overrides (timing relevant for pulse pulse_delays)
    pulse_delays = [t_coh, t_wait]  # -> 3 pulses
    phases = [0.0, 0.0, dflt.DETECTION_PHASE]  # last phase is detection phase

    laser_sequence = LaserPulseSequence.from_pulse_delays(
        pulse_delays=pulse_delays,
        base_amplitude=base_amp,
        pulse_fwhm_fs=pulse_fwhm_fs,
        carrier_freq_cm=carrier_cm,
        envelope_type=envelope,
        relative_E0s=relative_e0s,
        phases=phases,
    )

    # BATH (qutip BosonicEnvironment)
    bath_cfg = _get_section(cfg_root, "bath")
    temperature = float(bath_cfg.get("temperature", dflt.BATH_TEMP))
    cutoff = float(bath_cfg.get("cutoff", dflt.BATH_CUTOFF))
    coupling = float(bath_cfg.get("coupling", dflt.BATH_COUPLING))
    bath_type = str(bath_cfg.get("bath_type", dflt.BATH_TYPE))

    # TODO extend to BsosonicEnvironment
    bath_env = OhmicEnvironment(
        T=temperature,
        alpha=coupling,  # / cutoff,  # TODO make this is exactly the paper implementation
        wc=cutoff,
        s=1.0,
        tag=bath_type,
    )

    # simulation config
    config_cfg = _get_section(cfg_root, "config")
    n_phases = int(config_cfg.get("n_phases", dflt.N_PHASES))
    ode_solver = str(config_cfg.get("solver", dflt.ODE_SOLVER))
    signal_types = list(config_cfg.get("signal_types", dflt.SIGNAL_TYPES))
    sim_type = str(config_cfg.get("sim_type", dflt.SIM_TYPE))
    n_inhomogen = int(atomic_cfg.get("n_inhomogen", dflt.N_INHOMOGEN))
    max_workers = get_max_workers()
    print(
        f"🔧 Configured to use max_workers={max_workers} for parallel tasks.",
        flush=True,
    )

    # -----------------
    # VALIDATION (physics-level) BEFORE FINAL ASSEMBLY
    # -----------------
    if validate:
        params = {
            "solver": ode_solver,
            "bath_type": bath_type,
            "frequencies_cm": freqs_cm,
            "n_atoms": n_atoms,
            "dip_moments": dip_moments,
            "temperature": temperature,
            "cutoff": cutoff,
            "coupling": coupling,
            "n_phases": n_phases,
            "max_excitation": max_excitation,
            "n_chains": n_chains,
            "relative_e0s": relative_e0s,
            "rwa_sl": rwa_sl,
            "carrier_freq_cm": carrier_cm,
            "signal_types": signal_types,
            "t_det_max": t_det_max,
            "dt": dt,
            "t_coh": t_coh,
            "t_wait": t_wait,
            "n_inhomogen": n_inhomogen,
            # Newly added for extended validation
            "pulse_fwhm_fs": pulse_fwhm_fs,
            "base_amplitude": base_amp,
            "envelope_type": envelope,
            "coupling_cm": coupling_cm,
            "delta_inhomogen_cm": delta_inhomogen_cm,
            "solver_options": dflt.SOLVER_OPTIONS,  # TODO add those to the sim_config class
            "sim_type": sim_type,  # factory defaults (could expose later)
            "max_workers": max_workers,
        }
        dflt.validate(params)

    # --------------------------------------------------
    # Inhomogeneity bookkeeping flags
    # --------------------------------------------------
    inhom_enabled = n_inhomogen > 1 and delta_inhomogen_cm > 0.0

    # Stable group id so that all individual inhomogeneous configuration runs can be matched and averaged.
    # Hash only the physically relevant parameters that define the distribution so the same set up maps to same group id.
    inhom_hash_inputs = (
        str(n_atoms),
        str(freqs_cm),
        f"{delta_inhomogen_cm:.6g}",
        f"{n_inhomogen}",
        f"{coupling_cm:.6g}",
    )
    hash_basis = "|".join(inhom_hash_inputs).encode("utf-8")
    inhom_group_id = f"inhom_{hashlib.sha1(hash_basis).hexdigest()[:10]}"

    sim_config = SimulationConfig(
        ode_solver=ode_solver,
        rwa_sl=rwa_sl,
        dt=dt,
        t_coh=t_coh,
        t_wait=t_wait,
        t_det_max=t_det_max,
        n_phases=n_phases,
        n_inhomogen=n_inhomogen,
        signal_types=signal_types,
        sim_type=sim_type,
        max_workers=max_workers,
        inhom_enabled=inhom_enabled,
        inhom_index=0,
        inhom_group_id=inhom_group_id,
    )

    # -----------------
    # ASSEMBLE
    # -----------------
    simulation = SimulationModuleOQS(
        simulation_config=sim_config,
        system=atomic_system,
        laser=laser_sequence,
        bath=bath_env,
    )

    return simulation


def create_base_sim_oqs(
    config_path: Path | None = None,
) -> tuple[SimulationModuleOQS, float]:
    """Create base simulation instance and perform solver validation once.

    Parameters:
        config_path: Optional path to YAML config (None -> defaults)

    Returns:
        tuple: (SimulationModuleOQS instance, time_cut from solver validation)
    """
    # Gather overrides from args once and pass into loader (done earlier now)
    sim = load_simulation(
        config_path,
        validate=True,
    )

    print("🔧 Base simulation created from config (overrides applied early).")

    # -----------------
    # SOLVER VALIDATION
    # -----------------
    time_cut = -np.inf
    t_max = sim.times_local[-1]
    print("🔍 Validating solver...")
    try:
        from qspectro2d.spectroscopy.solver_check import check_the_solver

        _, time_cut = check_the_solver(sim)
        print("#" * 60)
        print(
            f"✅ Solver validation worked: Evolution becomes unphysical at "
            f"({time_cut / t_max:.2f} × t_max)"
        )
    except Exception as e:  # pragma: no cover
        print(f"⚠️  WARNING: Solver validation failed: {e}")

    if time_cut < t_max:
        print(
            f"⚠️  WARNING: Time cut {time_cut} is less than the last time point "
            f"{t_max}. This may affect the simulation results.",
            flush=True,
        )

    return sim, time_cut


def get_max_workers() -> int:
    """Get the maximum number of workers for parallel processing."""
    # Use SLURM environment variable if available, otherwise detect automatically
    try:
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    except ValueError:
        slurm_cpus = 0

    local_cpus = psutil.cpu_count(logical=True) or 1
    return slurm_cpus if slurm_cpus > 0 else local_cpus
