"""Validation and sanity checks for qspectro2d."""

import warnings

import numpy as np

from .signal_processing import N_PHASES, RELATIVE_E0S, SIGNAL_TYPES
from .defaults import (
    SUPPORTED_SOLVERS,
    SUPPORTED_BATHS,
    SUPPORTED_ENVELOPES,
    SUPPORTED_SIM_TYPES,
    T_DET_MAX,
    DT,
    T_COH_MAX,
    T_WAIT,
)
from .atomic_system import (
    N_ATOMS,
    N_CHAINS,
    FREQUENCIES_CM,
    DIP_MOMENTS,
    COUPLING_CM,
    DELTA_INHOMOGEN_CM,
    MAX_EXCITATION,
    N_INHOMOGEN,
)
from .laser_system import (
    PULSE_FWHM_FS,
    BASE_AMPLITUDE,
    ENVELOPE_TYPE,
    CARRIER_FREQ_CM,
    RWA_SL,
)
from .simulation import (
    ODE_SOLVER,
    SIM_TYPE,
    SOLVER_OPTIONS,
    ALLOWED_SOLVER_OPTIONS,
)
from .bath_system import (
    BATH_TYPE,
    BATH_CUTOFF,
    BATH_TEMP,
    BATH_COUPLING,
)



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
    bath_s = params.get("bath_s")
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
    solver_defaults = SOLVER_OPTIONS.get(ode_solver, {})
    solver_options = params.get("solver_options")
    if solver_options is None:
        solver_options = solver_defaults
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

    # Validate bath type
    if bath_type not in SUPPORTED_BATHS:
        raise ValueError(f"BATH_TYPE '{bath_type}' not in {SUPPORTED_BATHS}")

    # Validate Ohmic-family exponent (dimensionless). Only enforced when provided.
    if bath_type in {
        "ohmic",
        "subohmic",
        "superohmic",
        "ohmic+lorentzian",
        "subohmic+lorentzian",
        "superohmic+lorentzian",
    }:
        if bath_s is not None and float(bath_s) <= 0:
            raise ValueError("bath.s (bath_s) must be > 0")

    # Validate solver
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

        if ode_solver == "heom":
            bath_cfg = solver_options.get("bath")
            if bath_cfg is not None and not isinstance(bath_cfg, dict):
                raise TypeError("solver_options['bath'] must be a dict when provided.")

        allowed_keys = set(ALLOWED_SOLVER_OPTIONS.get(ode_solver, []))
        unknown_keys = set(solver_options) - allowed_keys
        if unknown_keys:
            raise ValueError(
                f"solver_options includes unsupported keys for {ode_solver}: {sorted(unknown_keys)}"
            )
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
