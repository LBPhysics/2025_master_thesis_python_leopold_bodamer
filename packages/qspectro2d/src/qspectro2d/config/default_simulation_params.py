"""
Default simulation parameters for qspectro2d.

This module contains default values for simulation parameters used across
the project. Centralizing these constants makes them easier to maintain
and reduces code duplication.
"""

import numpy as np
import warnings

from ..core.simulation.heom_defaults import (
    HEOM_DEFAULT_INCLUDE_DOUBLE,
    HEOM_DEFAULT_MAX_DEPTH,
    HEOM_DEFAULT_METHOD,
    HEOM_DEFAULT_N_EXP,
    HEOM_DEFAULT_N_POINTS,
    HEOM_DEFAULT_W_MAX_FACTOR,
    HEOM_DEFAULT_W_MIN,
)

INITIAL_STATE = "ground"
# === signal processing / phase cycling ===
# PHASE_CYCLING_PHASES = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
N_PHASES = 4  # Number of phase cycles for the simulation
DPHI = 2 * np.pi / N_PHASES
PHASE_CYCLING_PHASES = DPHI * np.arange(N_PHASES)
SIGNAL_TYPES = ["rephasing"]  # Default signal == photon echo to simulate
COMPONENT_MAP: dict[str, tuple[int, int]] = {
    "average": (0, 0),  # special case for just averaging all phases
    "rephasing": (-1, 1),  # photon echo is extracted here
    "nonrephasing": (1, -1),
    "doublequantum": (1, 1),
}  # represents the (k1, k2) phase factors for each signal type [k3 doenst matter]
# last pulse is 10% of the first two to ensure probing character
RELATIVE_E0S = [1.0, 1.0, 0.1]

# Validation thresholds for physics checks
NEGATIVE_EIGVAL_THRESHOLD = -1e-3
TRACE_TOLERANCE = 1e-6

# supported solvers and bath models
SUPPORTED_SOLVERS = ["ME", "BR", "Paper_eqs", "HEOM"]
SUPPORTED_BATHS = ["ohmic"]  # , "dl" NOTE: not yet implemented
SUPPORTED_ENVELOPES = ["gaussian", "cos2"]
SUPPORTED_SIM_TYPES = ["0d", "1d", "2d"]


# === ATOMIC SYSTEM DEFAULTS ===
N_ATOMS = 1
N_CHAINS = 1  # defaults to linear chain (single chain layout)
FREQUENCIES_CM = [16000.0] * N_ATOMS  # Number of frequency components in the system
DIP_MOMENTS = [1.0] * N_ATOMS  # Dipole moments for each atom
COUPLING_CM = 0.0  # Coupling strength [cm⁻¹]
DELTA_INHOMOGEN_CM = 0.0  # Inhomogeneous broadening [cm⁻¹]
MAX_EXCITATION = 1  # 1 -> ground+single manifold, 2 -> add double-excitation manifold
N_INHOMOGEN = 1  # 1 == no inhomogeneous broadening


# === LASER SYSTEM DEFAULTS ===
# TODO add N_PULSES -> 3 -< check that RELATIVE_E0S, phases has correct length (N_PULSES, N_PULSES-1)
PULSE_FWHM_FS = 15.0 if N_ATOMS == 1 else 5.0  # Pulse FWHM in fs
BASE_AMPLITUDE = 0.01  # -> such that for 1 atom the |exe| < 1%
ENVELOPE_TYPE = "gaussian"  # Type of pulse envelope # gaussian or cos2
CARRIER_FREQ_CM = 16000.0  # np.mean(FREQUENCIES_CM)  # Carrier frequency of the laser
RWA_SL = True


# === SIMULATION DEFAULTS ===
ODE_SOLVER = "BR"  # ODE solver to use
SIM_TYPE = "1d"
SOLVER_OPTIONS = {
    #    "nsteps": 200000,
    #    "atol": 1e-5,
    #    "rtol": 1e-3,
    #    "method": "bdf",  # Changed to bdf for stiff ODE systems
}
# === BATH SYSTEM DEFAULTS ===
# frequencies = [convert_cm_to_fs(freq_cm) for freq_cm in FREQUENCIES_CM]
BATH_TYPE = "ohmic"  # TODO at the moment only ohmic baths are supported
BATH_CUTOFF = 1e2  # * frequencies[0]  # Cutoff frequency in cm⁻¹
BATH_TEMP = 1e-3  # * frequencies[0] / BOLTZMANN
BATH_COUPLING = 1e-4  # * frequencies[0]


# === 2D SIMULATION DEFAULTS ===
T_DET_MAX = 20.0  # Maximum detection time in fs
T_COH_MAX = T_DET_MAX  # Coherence time in fs
DT = 0.1  # Spacing between t_coh, and of also t_det values in fs
# -> very good resolution
T_WAIT = 0.0  # Waiting time in fs


# VALIDATION AND SANITY CHECKS
def validate(params: dict) -> None:
    """Validate that a parameter dictionary is consistent and sensible."""
    # Extract parameters with defaults fallback
    ode_solver = params.get("solver", ODE_SOLVER)
    bath_type = params.get("bath_type", BATH_TYPE)
    frequencies_cm = params.get("frequencies_cm", FREQUENCIES_CM)
    n_atoms = params.get("n_atoms", N_ATOMS)
    dip_moments = params.get("dip_moments", DIP_MOMENTS)
    bath_temp = params.get("temperature", BATH_TEMP)
    bath_cutoff = params.get("cutoff", BATH_CUTOFF)
    bath_coupling = params.get("coupling", BATH_COUPLING)
    n_phases = params.get("n_phases", N_PHASES)
    max_excitation = params.get("max_excitation", MAX_EXCITATION)
    n_chains = params.get("n_chains", N_CHAINS)
    relative_e0s = params.get("relative_e0s", RELATIVE_E0S)
    rwa_sl = params.get("rwa_sl", RWA_SL)
    carrier_freq_cm = params.get("carrier_freq_cm", CARRIER_FREQ_CM)
    # Newly exposed / previously unvalidated
    pulse_fwhm_fs = params.get("pulse_fwhm_fs", PULSE_FWHM_FS)
    base_amplitude = params.get("base_amplitude", BASE_AMPLITUDE)
    envelope_type = params.get("envelope_type", ENVELOPE_TYPE)
    coupling_cm = params.get("coupling_cm", COUPLING_CM)
    delta_inhomogen_cm = params.get("delta_inhomogen_cm", DELTA_INHOMOGEN_CM)
    solver_options = params.get("solver_options", SOLVER_OPTIONS)
    sim_type = params.get("sim_type", SIM_TYPE)
    max_workers = params.get("max_workers", 1)
    # Time/grid parameters
    t_det_max = params.get("t_det_max", T_DET_MAX)
    dt = params.get("dt", DT)
    t_coh_max = params.get("t_coh_max", T_COH_MAX)
    t_coh_current = params.get("t_coh_current")
    t_wait = params.get("t_wait", T_WAIT)

    # Sampling and signal types
    n_inhomogen = params.get("n_inhomogen", N_INHOMOGEN)
    signal_types = params.get("signal_types", SIGNAL_TYPES)

    # Validate solver
    if ode_solver not in SUPPORTED_SOLVERS:
        raise ValueError(f"ODE_SOLVER '{ode_solver}' not in {SUPPORTED_SOLVERS}")

    # Validate bath type
    if bath_type not in SUPPORTED_BATHS:
        raise ValueError(f"BATH_TYPE '{bath_type}' not in {SUPPORTED_BATHS}")

    # Improved solver validation
    if ode_solver not in SUPPORTED_SOLVERS:
        raise ValueError(
            f"Invalid ode_solver '{ode_solver}'. Supported: {sorted(SUPPORTED_SOLVERS)}"
        )

    # Basic time parameter checks
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if t_coh_max < 0:
        raise ValueError("t_coh_max must be >= 0")
    if t_coh_current is not None:
        if t_coh_current < 0:
            raise ValueError("config.t_coh must be >= 0")
    if t_wait < 0:
        raise ValueError("t_wait must be >= 0")
    if t_det_max <= 0:
        raise ValueError("t_det_max must be > 0")

    # Pulse / laser checks
    if pulse_fwhm_fs <= 0:
        raise ValueError("pulse_fwhm_fs must be > 0")
    if base_amplitude <= 0:
        raise ValueError("base_amplitude must be > 0")
    if envelope_type not in SUPPORTED_ENVELOPES:
        raise ValueError(f"envelope_type '{envelope_type}' not in {SUPPORTED_ENVELOPES}")

    # Atomic coupling / broadening checks
    if coupling_cm < 0:
        raise ValueError("coupling_cm must be >= 0")
    if delta_inhomogen_cm < 0:
        raise ValueError("delta_inhomogen_cm must be >= 0")

    # NOTE deleted dipole posititvity

    # Phase/frequency sampling checks
    if n_phases <= 0:
        raise ValueError("n_phases must be > 0")
    if n_inhomogen <= 0:
        raise ValueError("n_inhomogen must be > 0")

    # Validate atomic system consistency
    if len(frequencies_cm) != n_atoms:
        raise ValueError(f"FREQUENCIES_CM length ({len(frequencies_cm)}) != N_ATOMS ({n_atoms})")

    if len(dip_moments) != n_atoms:
        raise ValueError(f"DIP_MOMENTS length ({len(dip_moments)}) != N_ATOMS ({n_atoms})")

    # Validate positive values
    if bath_temp < 0:
        raise ValueError("BATH_TEMP must be positive")

    if bath_cutoff <= 0:
        raise ValueError("BATH_CUTOFF must be positive")

    if bath_coupling <= 0:
        raise ValueError("BATH_COUPLING must be positive")

    # Validate phases
    if n_phases <= 0:
        raise ValueError("N_PHASES must be positive")

    # Validate excitation truncation
    if max_excitation not in (1, 2):
        raise ValueError("MAX_EXCITATION must be 1 or 2")

    # Validate n_chains divisibility if provided and relevant
    if n_chains is not None and n_atoms > 2:
        if n_chains < 1:
            raise ValueError("N_CHAINS must be >=1 when specified")
        if n_atoms % n_chains != 0:
            raise ValueError(
                f"N_CHAINS ({n_chains}) does not divide N_ATOMS ({n_atoms}) for cylindrical geometry"
            )

    # Validate relative amplitudes
    if len(relative_e0s) != 3:
        raise ValueError("RELATIVE_E0S must have exactly 3 elements (3-pulse assumption)")
    if any(a <= 0 for a in relative_e0s):
        raise ValueError("All RELATIVE_E0S entries must be > 0")
    # Optional heuristic: last should be probe (smaller or equal)
    if not (relative_e0s[2] <= relative_e0s[0] and relative_e0s[2] <= relative_e0s[1]):
        warnings.warn(
            "RELATIVE_E0S third entry not smaller/equal to first two (probe heuristic)",
            stacklevel=2,
        )

    # Solver options sanity
    if isinstance(solver_options, dict):
        atol = solver_options.get("atol")
        rtol = solver_options.get("rtol")
        nsteps = solver_options.get("nsteps")
        if atol is not None and atol <= 0:
            raise ValueError("solver_options.atol must be > 0")
        if rtol is not None and rtol <= 0:
            raise ValueError("solver_options.rtol must be > 0")
        if nsteps is not None and nsteps <= 0:
            raise ValueError("solver_options.nsteps must be > 0")

        heom_opts = solver_options.get("heom")

        if ode_solver == "HEOM":
            if heom_opts is None:
                heom_opts = {}
            elif not isinstance(heom_opts, dict):
                raise TypeError(
                    "solver_options['heom'] must be a dict when provided."
                )

            allowed_keys = {"max_depth", "bath", "sites", "include_double", "options", "args"}
            unexpected = set(heom_opts) - allowed_keys
            if unexpected:
                raise ValueError(
                    f"Unsupported HEOM configuration keys: {sorted(unexpected)}. "
                    "See README for the accepted structure."
                )

            max_depth_val = heom_opts.get("max_depth", HEOM_DEFAULT_MAX_DEPTH)
            try:
                max_depth_int = int(max_depth_val)
            except (TypeError, ValueError) as exc:
                raise ValueError("HEOM solver requires an integer 'max_depth'.") from exc
            if max_depth_int < 0:
                raise ValueError("HEOM solver requires max_depth >= 0.")

            sites_val = heom_opts.get("sites")
            if sites_val is not None:
                if not isinstance(sites_val, (list, tuple)) or not all(
                    isinstance(idx, (int, np.integer)) for idx in sites_val
                ):
                    raise ValueError("solver_options['heom']['sites'] must be a list of integers if provided.")

            include_double = heom_opts.get("include_double", HEOM_DEFAULT_INCLUDE_DOUBLE)
            if not isinstance(include_double, bool):
                raise TypeError("solver_options['heom']['include_double'] must be boolean if provided.")

            bath_cfg = heom_opts.get("bath")
            if bath_cfg is None:
                bath_cfg = {}
            elif not isinstance(bath_cfg, dict):
                raise TypeError("solver_options['heom']['bath'] must be a dict when provided.")

            method = str(bath_cfg.get("method", HEOM_DEFAULT_METHOD)).lower()
            if method != HEOM_DEFAULT_METHOD:
                raise ValueError(f"HEOM bath method must be '{HEOM_DEFAULT_METHOD}'.")

            for key, default in (
                ("w_min", HEOM_DEFAULT_W_MIN),
                ("w_max_factor", HEOM_DEFAULT_W_MAX_FACTOR),
                ("n_points", HEOM_DEFAULT_N_POINTS),
                ("n_exp", HEOM_DEFAULT_N_EXP),
            ):
                value = bath_cfg.get(key, default)
                if key in {"n_points", "n_exp"}:
                    try:
                        int_value = int(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"HEOM bath '{key}' must be an integer.") from exc
                    if int_value <= 0:
                        raise ValueError(f"HEOM bath '{key}' must be positive.")
                else:
                    try:
                        float_value = float(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"HEOM bath '{key}' must be numeric.") from exc
                    if float_value <= 0:
                        raise ValueError(f"HEOM bath '{key}' must be positive.")

            options_cfg = heom_opts.get("options", {})
            if options_cfg is not None and not isinstance(options_cfg, dict):
                raise TypeError("solver_options['heom']['options'] must be a dict when provided.")

            args_cfg = heom_opts.get("args")
            if args_cfg is not None and not isinstance(args_cfg, dict):
                raise TypeError("solver_options['heom']['args'] must be a dict when provided.")
    else:
        raise TypeError("solver_options must be a dict")

    # Simulation type / workers
    if sim_type not in SUPPORTED_SIM_TYPES:
        raise ValueError(f"sim_type '{sim_type}' not in {SUPPORTED_SIM_TYPES}")
    if max_workers <= 0:
        raise ValueError("max_workers must be >= 1")

    if sim_type in ("0d", "1d") and t_coh_current is None:
        raise ValueError("config.t_coh must be provided for 0d/1d simulations")

    if rwa_sl:
        freqs_array = np.array(frequencies_cm)
        max_detuning = np.max(np.abs(freqs_array - carrier_freq_cm))
        rel_detuning = max_detuning / carrier_freq_cm if carrier_freq_cm != 0 else np.inf
        if rel_detuning > 1e-2:
            print(
                f"WARNING: RWA probably not valid, since relative detuning: {rel_detuning} is too large",
                flush=True,
            )


def validate_defaults():
    """Validate that all default values are consistent and sensible."""
    validate({})
