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
import numpy as np
from pathlib import Path
from typing import Any, Mapping, Optional
from qutip import OhmicEnvironment, DrudeLorentzEnvironment
from qutip.core.environment import BosonicEnvironment
import yaml

from ..core.simulation.simulation_class import SimulationModuleOQS
from ..core.simulation.sim_config import SimulationConfig
from ..core.laser_system.laser_class import LaserPulseSequence
from ..core.atomic_system.system_class import AtomicSystem
from ..utils.constants import convert_cm_to_fs
from .atomic_system import (
    COUPLING_CM,
    DELTA_INHOMOGEN_CM,
    DIP_MOMENTS,
    FREQUENCIES_CM,
    MAX_EXCITATION,
    N_ATOMS,
    N_CHAINS,
    N_INHOMOGEN,
)
from .bath_system import BATH_COUPLING, BATH_CUTOFF, BATH_TEMP, BATH_TYPE
from .defaults import DT, INITIAL_STATE, SUPPORTED_BATHS, T_DET_MAX, T_WAIT
from .laser_system import (
    BASE_AMPLITUDE,
    CARRIER_FREQ_CM,
    ENVELOPE_TYPE,
    PULSE_FWHM_FS,
    RWA_SL,
)
from .signal_processing import N_PHASES, RELATIVE_E0S, SIGNAL_TYPES
from .simulation import ALLOWED_SOLVER_OPTIONS, ODE_SOLVER, SIM_TYPE, SOLVER_OPTIONS
from .validation import validate

__all__ = [
    "load_simulation",
    "create_base_sim_oqs",
    "load_simulation_config",
    "load_simulation_laser",
    "load_simulation_atomic_system",
    "load_simulation_bath",
]


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


def load_simulation_config(
    path: Optional[str | Path] = None,
) -> SimulationConfig:
    """Load only the SimulationConfig from a YAML file or defaults."""
    # LOAD / FALLBACK
    if path is None:
        cfg_root = {}
    else:
        cfg_root = _read_yaml(Path(path))

    # Extract sections
    atomic_cfg = _get_section(cfg_root, "atomic")
    laser_cfg = _get_section(cfg_root, "laser")
    config_cfg = _get_section(cfg_root, "config")

    # Extract values
    pulse_fwhm_fs = float(laser_cfg.get("pulse_fwhm_fs", PULSE_FWHM_FS))
    t_det_max = float(config_cfg.get("t_det_max", T_DET_MAX))
    t_coh_max = float(config_cfg.get("t_coh_max", t_det_max))
    t_coh_value = config_cfg.get("t_coh")
    t_coh_current = float(t_coh_value) if t_coh_value is not None else None
    t_wait = float(config_cfg.get("t_wait", T_WAIT))
    dt = float(config_cfg.get("dt", DT))
    n_phases = int(config_cfg.get("n_phases", N_PHASES))
    n_inhomogen = int(atomic_cfg.get("n_inhomogen", N_INHOMOGEN))
    ode_solver = str(config_cfg.get("solver", ODE_SOLVER))
    signal_types = list(config_cfg.get("signal_types", SIGNAL_TYPES))
    sim_type = str(config_cfg.get("sim_type", SIM_TYPE))
    if sim_type in {"0d", "1d"} and t_coh_current is None:
        t_coh_current = t_coh_max
    rwa_sl = bool(laser_cfg.get("rwa_sl", RWA_SL))
    initial_state = str(config_cfg.get("initial_state", INITIAL_STATE))
    solver_options_cfg = config_cfg.get("solver_options", {})
    solver_options = dict(SOLVER_OPTIONS.get(ode_solver))
    if isinstance(solver_options_cfg, Mapping):
        solver_options.update(solver_options_cfg)
    # Normalize CLI/YAML numeric strings (e.g. "1e-6") into numbers for validation downstream.
    normalized_solver_opts: dict[str, Any] = {}
    for key, value in solver_options.items():
        if isinstance(value, str):
            text = value.strip()
            try:
                numeric_val = float(text)
            except ValueError:
                normalized_solver_opts[key] = value
            else:
                if numeric_val.is_integer() and text.lower().find("e") == -1 and "." not in text:
                    normalized_solver_opts[key] = int(numeric_val)
                else:
                    normalized_solver_opts[key] = numeric_val
        else:
            normalized_solver_opts[key] = value
    solver_options = normalized_solver_opts

    # Filter to only allowed options for the current solver
    allowed_keys = ALLOWED_SOLVER_OPTIONS.get(ode_solver, [])
    solver_options = {k: v for k, v in solver_options.items() if k in allowed_keys}

    max_workers = get_max_workers()

    return SimulationConfig(
        ode_solver=ode_solver,
        solver_options=solver_options,
        rwa_sl=rwa_sl,
        dt=dt,
        t_coh_max=t_coh_max,
        t_coh_current=t_coh_current,
        t_wait=t_wait,
        t_det_max=t_det_max,
        pulse_fwhm_fs=pulse_fwhm_fs,
        n_phases=n_phases,
        n_inhomogen=n_inhomogen,
        signal_types=signal_types,
        sim_type=sim_type,
        max_workers=max_workers,
        initial_state=initial_state,
    )


def load_simulation_laser(
    path: Optional[str | Path] = None,
) -> LaserPulseSequence:
    """Load LaserPulseSequence from a YAML file or defaults, with delays set to t_coh_max."""
    # LOAD / FALLBACK
    if path is None:
        cfg_root = {}
    else:
        cfg_root = _read_yaml(Path(path))

    # Extract sections
    laser_cfg = _get_section(cfg_root, "laser")
    config_cfg = _get_section(cfg_root, "config")

    # Extract values
    pulse_fwhm_fs = float(laser_cfg.get("pulse_fwhm_fs", PULSE_FWHM_FS))
    base_amp = float(laser_cfg.get("base_amplitude", BASE_AMPLITUDE))
    envelope = str(laser_cfg.get("envelope_type", ENVELOPE_TYPE))
    carrier_cm = float(laser_cfg.get("carrier_freq_cm", CARRIER_FREQ_CM))
    relative_e0s = RELATIVE_E0S
    t_wait = float(config_cfg.get("t_wait", T_WAIT))

    # Create laser with initial delays
    t_det_max = float(config_cfg.get("t_det_max", T_DET_MAX))
    t_coh_max = float(config_cfg.get("t_coh_max", t_det_max))
    t_coh_value = config_cfg.get("t_coh")
    t_coh_current = float(t_coh_value) if t_coh_value is not None else None
    t_coh_delay = t_coh_current if t_coh_current is not None else t_coh_max
    pulse_delays = [t_coh_delay, t_wait]  # -> 3 pulses
    phases = [0.0, 0.0, 0.0]  # last phase is detection phase

    laser = LaserPulseSequence.from_pulse_delays(
        pulse_delays=pulse_delays,
        base_amplitude=base_amp,
        pulse_fwhm_fs=pulse_fwhm_fs,
        carrier_freq_cm=carrier_cm,
        envelope_type=envelope,
        relative_E0s=relative_e0s,
        phases=phases,
    )

    return laser


def load_simulation_atomic_system(
    path: Optional[str | Path] = None,
) -> AtomicSystem:
    """Load AtomicSystem from a YAML file or defaults."""

    # LOAD / FALLBACK
    if path is None:
        cfg_root = {}
    else:
        cfg_root = _read_yaml(Path(path))

    # ATOMIC SYSTEM
    atomic_cfg = _get_section(cfg_root, "atomic")
    n_atoms = int(atomic_cfg.get("n_atoms", N_ATOMS))
    n_chains = int(atomic_cfg.get("n_chains", N_CHAINS))
    freqs_cm = list(atomic_cfg.get("frequencies_cm", FREQUENCIES_CM))
    dip_moments = list(atomic_cfg.get("dip_moments", DIP_MOMENTS))
    coupling_cm = float(atomic_cfg.get("coupling_cm", COUPLING_CM))
    delta_inhomogen_cm = float(atomic_cfg.get("delta_inhomogen_cm", DELTA_INHOMOGEN_CM))
    max_excitation = int(atomic_cfg.get("max_excitation", MAX_EXCITATION))

    atomic_system = AtomicSystem(
        n_atoms=n_atoms,
        n_chains=n_chains,
        frequencies_cm=freqs_cm,
        dip_moments=dip_moments,
        coupling_cm=coupling_cm,
        delta_inhomogen_cm=delta_inhomogen_cm,
        max_excitation=max_excitation,
    )

    return atomic_system


def load_simulation_bath(
    path: Optional[str | Path] = None,
) -> OhmicEnvironment:
    """Load BosonicEnvironment from a YAML file or defaults.

    Unit convention
    ---------------
    This codebase stores input transition frequencies in `atomic.frequencies_cm` (cm⁻¹),
    but the `AtomicSystem` converts them internally to fs⁻¹ for the dynamics.

    To keep YAML configs simple and comparable across systems, bath parameters are
    always interpreted as *dimensionless multiples* of
        ω0̄ = mean(system transition frequencies)
    (computed in the same internal units as the Hamiltonian, i.e. fs⁻¹).

    Concretely:
        - `bath.temperature` means T/ω0̄
        - `bath.cutoff` means ωc/ω0̄
        - `bath.coupling` means coupling/ω0̄
    """
    # LOAD / FALLBACK
    if path is None:
        cfg_root = {}
    else:
        cfg_root = _read_yaml(Path(path))

    # BATH (qutip BosonicEnvironment)
    bath_cfg = _get_section(cfg_root, "bath")
    temperature = float(bath_cfg.get("temperature", BATH_TEMP))
    cutoff = float(bath_cfg.get("cutoff", BATH_CUTOFF))
    coupling = float(bath_cfg.get("coupling", BATH_COUPLING))
    bath_type = str(bath_cfg.get("bath_type", BATH_TYPE))

    # Controls the internal frequency cutoff used by QuTiP when constructing
    # BosonicEnvironment objects from a spectral density.
    # In internal units, we use: wMax = wmax_factor * cutoff.
    wmax_factor = float(bath_cfg.get("wmax_factor", 10.0))
    if wmax_factor <= 0:
        raise ValueError("bath.wmax_factor must be > 0")

    # Optional Lorentzian peak parameters (only used for "+lorentzian" bath types)
    # Normalized (dimensionless) YAML inputs:
    #   peak_width:    gamma / w0_bar
    #   peak_strength: strength / coupling
    #   peak_center:   omega_center / w0_bar
    peak_strength = float(bath_cfg.get("peak_strength", 0.0))
    peak_width = float(bath_cfg.get("peak_width", 1.0))
    peak_center = float(bath_cfg.get("peak_center", 0.0))

    # Scale bath parameters by ω0̄ (mean system frequency) in internal fs⁻¹ units.
    atomic_cfg = _get_section(cfg_root, "atomic")
    freqs_cm = list(atomic_cfg.get("frequencies_cm", FREQUENCIES_CM))
    if not freqs_cm:
        raise ValueError("Missing `atomic.frequencies_cm`; cannot determine ω0̄ for bath scaling.")
    freqs_fs = np.asarray(convert_cm_to_fs(freqs_cm), dtype=float)
    w0_bar_fs = float(np.mean(freqs_fs))

    temperature *= w0_bar_fs
    cutoff *= w0_bar_fs
    coupling *= w0_bar_fs

    w_max = wmax_factor * cutoff

    # Scale Lorentzian peak parameters into internal units (fs^-1 and the same coupling units as the base bath)
    peak_width = peak_width * w0_bar_fs
    peak_center = peak_center * w0_bar_fs
    peak_strength = peak_strength * coupling

    if bath_type == "ohmic":
        bath_env = OhmicEnvironment(
            T=temperature,
            alpha=coupling,
            wc=cutoff,
            s=1.0,
            tag=bath_type,
        )
    # To match the Ohmic bath at low frequencies, adjust lam for Drude-Lorentz:
    # For Ohmic s=1: J(ω) ≈ α ω (at low ω)
    # For Drude-Lorentz: J(ω) ≈ 2 λ ω / γ (at low ω)
    # So set λ = α γ / 2 to approximate the same low-ω behavior
    elif bath_type == "drudelorentz":
        bath_env = DrudeLorentzEnvironment(
            T=temperature,
            gamma=cutoff,
            lam=coupling * cutoff / 2,
            tag=bath_type,
        )

    elif bath_type in {"ohmic+lorentzian", "drudelorentz+lorentzian"}:
        if peak_strength < 0:
            raise ValueError("bath.peak_strength must be >= 0")
        if peak_width <= 0:
            raise ValueError("bath.peak_width must be > 0")
        if peak_center < 0:
            raise ValueError("bath.peak_center must be >= 0")

        # Base environment (used for its spectral density definition).
        if bath_type.startswith("ohmic"):
            bath_base = OhmicEnvironment(
                T=temperature,
                alpha=coupling,
                wc=cutoff,
                s=1.0,
                tag="ohmic",
            )
        else:
            bath_base = DrudeLorentzEnvironment(
                T=temperature,
                gamma=cutoff,
                lam=coupling * cutoff / 2,
                tag="drudelorentz",
            )

        def J_lorentz_peak(w, center=peak_center, gamma=peak_width, strength=peak_strength):
            """Low-frequency Lorentzian-weighted contribution to J(ω) (ω>0 only).

            Important: for bosonic baths, making J(0) finite leads to a divergent power
            spectrum S(ω) as ω→0 at finite temperature. To boost pure dephasing while
            keeping S(0) finite, this term scales ~ω near ω=0.
            """
            w_arr = np.asarray(w, dtype=float)
            J = np.zeros_like(w_arr)
            pos = w_arr > 0
            wp = w_arr[pos]
            J[pos] = strength * wp * (gamma**2) / ((wp - center) ** 2 + gamma**2)
            return float(J) if np.isscalar(w) else J

        def J_base_plus_peak(w):
            return bath_base.spectral_density(w) + J_lorentz_peak(w)

        bath_env = BosonicEnvironment.from_spectral_density(
            J_base_plus_peak,
            T=temperature,
            wMax=w_max,
            tag=bath_type,
        )

        # Attach base parameters so downstream validation and reporting can reuse them.
        if bath_type.startswith("ohmic"):
            bath_env.wc = cutoff
            bath_env.alpha = coupling
        else:
            bath_env.gamma = cutoff
            bath_env.lam = coupling * cutoff / 2

    else:
        raise ValueError(f"Unsupported bath_type: {bath_type}. Supported: {SUPPORTED_BATHS}")

    return bath_env


def load_simulation(
    path: Optional[str | Path] = None,
    run_validation: bool = False,
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
        YAML configuration file. If None, module defaults are used.
    run_validation: bool
        If True run physics validation via `qspectro2d.config.validation.validate`.
    t_coh_override, t_wait_override, t_det_max_override, dt_override, ode_solver_override:
        Optional scalar overrides for timing / solver settings.
    """
    # Load components
    sim_config = load_simulation_config(path)
    atomic_system = load_simulation_atomic_system(path)
    bath_env = load_simulation_bath(path)
    laser_sequence = load_simulation_laser(path)

    # -----------------
    # VALIDATION (physics-level) BEFORE FINAL ASSEMBLY
    # -----------------
    if run_validation:
        # Handle bath-specific attributes
        if bath_env.tag in {"ohmic", "ohmic+lorentzian"}:
            cutoff_val = bath_env.wc
            coupling_val = bath_env.alpha
        elif bath_env.tag in {"drudelorentz", "drudelorentz+lorentzian"}:
            cutoff_val = bath_env.gamma
            coupling_val = bath_env.lam
        else:
            raise ValueError(f"Unsupported bath_type for validation: {bath_env.tag}")

        params = {
            "solver": sim_config.ode_solver,
            "bath_type": bath_env.tag,
            "frequencies_cm": atomic_system.frequencies_cm,
            "n_atoms": atomic_system.n_atoms,
            "dip_moments": atomic_system.dip_moments,
            "temperature": bath_env.T,
            "cutoff": cutoff_val,
            "coupling": coupling_val,
            "n_phases": sim_config.n_phases,
            "max_excitation": atomic_system.max_excitation,
            "n_chains": atomic_system.n_chains,
            "relative_e0s": RELATIVE_E0S,
            "rwa_sl": sim_config.rwa_sl,
            "carrier_freq_cm": laser_sequence.carrier_freq_cm,
            "signal_types": sim_config.signal_types,
            "t_det_max": sim_config.t_det_max,
            "dt": sim_config.dt,
            "t_coh_max": sim_config.t_coh_max,
            "t_coh_current": sim_config.t_coh_current,
            "t_wait": sim_config.t_wait,
            "n_inhomogen": sim_config.n_inhomogen,
            "solver_options": sim_config.solver_options,
            # Newly added for extended validation
            "pulse_fwhm_fs": sim_config.pulse_fwhm_fs,
            "base_amplitude": laser_sequence.E0,
            "envelope_type": laser_sequence.carrier_type,
            "coupling_cm": atomic_system.coupling_cm,
            "delta_inhomogen_cm": atomic_system.delta_inhomogen_cm,
            "sim_type": sim_config.sim_type,
            "max_workers": sim_config.max_workers,
        }
        validate(params)

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
        run_validation=True,
    )

    print("🔧 Base simulation created from config (overrides applied early).")

    # -----------------
    # SOLVER VALIDATION
    # -----------------
    time_cut = -np.inf
    t_max = sim.simulation_config.t_det_max
    print("🔍 Validating solver...")
    try:
        from qspectro2d.spectroscopy.solver_check import check_the_solver

        time_cut = check_the_solver(sim)
        print("#" * 60)
        print(
            f"✅ Solver validation worked: Evolution becomes unphysical at "
            f"({time_cut / t_max:.2f} × t_max)"
        )
    except Exception as e:
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

    local_cpus = os.cpu_count() or 1
    return slurm_cpus if slurm_cpus > 0 else int(local_cpus)
